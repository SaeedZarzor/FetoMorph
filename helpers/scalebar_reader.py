"""Automatic scale-bar reading.

Finds the scale bar burned into a 2-D slice image, OCRs its label
(e.g. ``"20 mm"``) and turns the pair into a pixel size, so the user does
not have to drag a line over the bar by hand.

The OCR is a self-contained template matcher over the OpenCV Hershey fonts
rather than an external engine: FetoMorph draws its own bars with
``cv2.putText(..., FONT_HERSHEY_SIMPLEX, ...)``
(:func:`functions.nifti_to_image.draw_new_scale_bar`) and with a Qt sans
face (:func:`helpers.helpers.add_scalebar`), and the glyph set that matters
is tiny -- digits, ``.``, and the handful of letters that spell a length
unit. That keeps the feature free of a tesseract/easyocr dependency.

Public entry point: :func:`read_scalebar`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import logging
import re

import cv2
import numpy as np

logger = logging.getLogger("fetomorph")

# Glyphs the label may contain. Units are matched as whole words afterwards,
# so the alphabet only needs the letters that appear in the supported units.
_DIGITS = "0123456789"
_LETTERS = "mcunµ"
_PUNCT = ".,"
_ALPHABET = _DIGITS + _LETTERS + _PUNCT

# Hershey faces used to build templates. FetoMorph's own bars use SIMPLEX;
# the others widen tolerance for labels drawn by Qt or by other tools.
_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_COMPLEX,
)

# Every glyph is normalized into this box before matching.
_TPL = 28

# Length units we can resolve, mapped to millimetres.
_UNIT_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "um": 0.001,
    "µm": 0.001,
    "nm": 1e-6,
}

# Unit spellings that OCR may produce, normalized to a key of _UNIT_MM.
_UNIT_ALIASES = {
    "mm": "mm", "nm": "nm", "cm": "cm", "m": "m",
    "um": "um", "µm": "um", "un": "um",
}

# Sanity bounds on the result. A misread digit or unit ("1 cm" coming back as
# "16 m") produces an absurd pixel size rather than a subtly wrong one, so a
# generous plausibility window turns almost every OCR slip into an honest
# "could not read" instead of a silently wrong calibration.
_MIN_BAR_MM, _MAX_BAR_MM = 0.05, 500.0
_MIN_MM_PER_PX, _MAX_MM_PER_PX = 1e-5, 5.0


@dataclass
class ScalebarReading:
    """One successful scale-bar read.

    Attributes:
        bar_px: Measured bar length in pixels.
        value: Physical length printed on the label, in :attr:`unit`.
        unit: Unit as written on the label (``"mm"``, ``"cm"``, ...).
        mm_per_px: Pixel size in millimetres, ``value_in_mm / bar_px``.
        unit_per_px: Pixel size expressed in :attr:`unit`.
        text: The OCR'd label text.
        bar_box: Bar bounding box ``(x, y, w, h)`` in image pixels.
        text_box: Label bounding box ``(x, y, w, h)``, or ``None``.
        confidence: Mean per-glyph match score in ``[0, 1]``.
        polarity: ``"dark"`` for dark ink on a light background, else ``"light"``.
    """

    bar_px: float
    value: float
    unit: str
    mm_per_px: float
    unit_per_px: float
    text: str
    bar_box: tuple[int, int, int, int]
    text_box: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    polarity: str = "dark"
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable description of the reading."""
        return (f"{self.bar_px:.1f} px = {self.value:g} {self.unit} "
                f"→ {self.unit_per_px:.6f} {self.unit}/pixel "
                f"(label “{self.text}”, confidence {self.confidence:.0%})")


# --------------------------------------------------------------------------
# Glyph templates
# --------------------------------------------------------------------------

_TEMPLATE_CACHE: dict[str, list[tuple[str, np.ndarray]]] = {}

# Every glyph is rendered and probed inside a "line box": the strip running
# from the cap height down to the baseline, _TPL rows tall. Keeping the
# vertical placement (rather than centring each glyph in its own box) is what
# separates "." from "o", and the x-height letters "m"/"c"/"u" from digits.


def _template_line_box(ch: str, font: int, thickness: int) -> np.ndarray | None:
    """Render *ch* into a ``_TPL``-row line box, cropped to its ink columns."""
    # Choose the scale from a capital-height reference so every face lands on
    # the same cap height, then place the baseline at the bottom of the box.
    (_dw, cap_h), _cap_base = cv2.getTextSize("0", font, 1.0, thickness)
    if cap_h <= 0:
        return None
    scale = _TPL / float(cap_h)
    (tw, th), base = cv2.getTextSize(ch, font, scale, thickness)
    if tw <= 0:
        return None
    pad = _TPL
    canvas = np.zeros((_TPL + 2 * pad, tw + 2 * pad), np.uint8)
    # Baseline sits on the last row of the line box.
    cv2.putText(canvas, ch, (pad, pad + _TPL), font, scale, 255,
                thickness, cv2.LINE_AA)
    box = canvas[pad:pad + _TPL, :]
    cols = np.nonzero(box.any(axis=0))[0]
    if cols.size == 0:
        return None
    return box[:, cols.min():cols.max() + 1].astype(np.float32) / 255.0


def _build_templates() -> list[tuple[str, np.ndarray]]:
    """Render every glyph in every Hershey face into line-box templates."""
    cached = _TEMPLATE_CACHE.get("all")
    if cached is not None:
        return cached

    templates: list[tuple[str, np.ndarray]] = []
    for ch in _ALPHABET:
        for font in _FONTS:
            for thickness in (1, 2, 3):
                tpl = _template_line_box(ch, font, thickness)
                if tpl is not None and tpl.any():
                    templates.append((ch, tpl))
    _TEMPLATE_CACHE["all"] = templates
    return templates


def _match_glyph(line_mask: np.ndarray) -> tuple[str, float]:
    """Best-matching character for one glyph given as a line-box strip.

    *line_mask* must already span cap-height to baseline vertically (see
    :func:`_to_line_box`); only its width is glyph-specific.
    """
    ys, xs = np.nonzero(line_mask)
    if xs.size == 0:
        return "", 0.0
    probe_full = line_mask[:, xs.min():xs.max() + 1].astype(np.float32)
    pw = probe_full.shape[1]

    best_ch, best_score = "", -1.0
    for ch, tpl in _build_templates():
        tw = tpl.shape[1]
        # Compare at a common width so a narrow "1" cannot masquerade as "0":
        # the aspect mismatch is paid for in the union term.
        width = max(pw, tw)
        probe = cv2.resize(probe_full, (width, _TPL), interpolation=cv2.INTER_AREA)
        cand = cv2.resize(tpl, (width, _TPL), interpolation=cv2.INTER_AREA)
        inter = float(np.minimum(probe, cand).sum())
        union = float(np.maximum(probe, cand).sum())
        score = (inter / union) if union > 0 else 0.0
        # Penalise gross aspect disagreement, which IoU alone tolerates once
        # both are stretched to a common width.
        ratio = min(pw, tw) / float(max(pw, tw))
        score *= 0.55 + 0.45 * ratio
        if score > best_score:
            best_ch, best_score = ch, score
    return best_ch, best_score


def _to_line_box(ink: np.ndarray, top: int, base: int) -> np.ndarray:
    """Crop soft *ink* to the text line ``top..base`` and scale to ``_TPL`` rows."""
    top = max(0, int(top))
    base = min(ink.shape[0], int(base))
    if base - top < 2:
        return np.zeros((_TPL, max(1, ink.shape[1])), np.float32)
    strip = np.ascontiguousarray(ink[top:base, :], dtype=np.float32)
    scale = _TPL / float(base - top)
    width = max(1, int(round(strip.shape[1] * scale)))
    # Upscaling a 7-px-tall label is where the soft edges pay off, so use a
    # smooth interpolation rather than INTER_AREA (which is for shrinking).
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(strip, (width, _TPL), interpolation=interp)
    return np.clip(resized, 0.0, 1.0)


# --------------------------------------------------------------------------
# Ink masks
# --------------------------------------------------------------------------

def _ink_masks(img_bgr: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return candidate ``(polarity, mask, ink)`` triples for bar/label ink.

    Scale bars are drawn in a neutral colour -- black on the white slice
    exports, white on dark backdrops -- so both polarities are tried and the
    caller keeps whichever yields a plausible bar.

    *mask* is the hard mask used for shape analysis; *ink* is the matching
    continuous ink coverage in ``[0, 1]``. Keeping the soft version matters
    for the OCR: on a small export the label is only a handful of pixels
    tall, and the anti-aliased edges carry most of the glyph's identity.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    v = V.astype(np.float32) / 255.0
    dark = ((S < 90) & (V < 110)).astype(np.uint8) * 255
    light = ((S < 60) & (V > 200)).astype(np.uint8) * 255
    return [("dark", dark, np.clip(1.0 - v, 0.0, 1.0)),
            ("light", light, np.clip(v, 0.0, 1.0))]


def _components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Connected components with stats, 8-connected."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return num, labels, stats


# --------------------------------------------------------------------------
# Bar detection
# --------------------------------------------------------------------------

def _find_bar(mask: np.ndarray, shape: tuple[int, int],
              *, region_top: float, limit: int = 6
              ) -> list[tuple[tuple[int, int, int, int], float]]:
    """Rank the bar-like components in *mask*, best first.

    A scale bar is a solid, strongly horizontal rectangle, usually the
    longest one near the bottom-right. Several candidates are returned
    because a letter stroke can look bar-like on its own; the caller
    disambiguates by requiring a readable label underneath.
    """
    h, w = shape
    roi = np.zeros_like(mask)
    roi[int(region_top * h):, :] = 255
    cand = cv2.bitwise_and(mask, roi)
    # Bridge the 1-px gaps anti-aliasing leaves along a thin bar without
    # merging the bar into the label underneath it.
    cand = cv2.morphologyEx(
        cand, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=1)

    num, labels, stats = _components(cand)
    if num <= 1:
        return []

    scored: list[tuple[tuple[int, int, int, int], float]] = []
    for i in range(1, num):
        x, y, cw, ch = (int(stats[i, 0]), int(stats[i, 1]),
                        int(stats[i, 2]), int(stats[i, 3]))
        if cw < max(12, 0.02 * w) or ch < 1:
            continue
        if cw / float(ch) < 2.0:
            continue
        # On a small export the label sits close enough to the bar that the two
        # touch and arrive as one component, so the bar is extracted from the
        # component's row profile rather than trusting its bounding box.
        sub = (labels[y:y + ch, x:x + cw] == i)
        rect = _solid_row_band(sub)
        if rect is None:
            continue
        rx, ry, rw, rh = rect
        if rh > max(1.0, 0.06 * h) or rw < max(12, 0.02 * w):
            continue
        aspect = rw / float(rh)
        if aspect < 4.0:
            continue
        # Length dominates: the scale bar is the longest solid horizontal run
        # in the search region, well ahead of any letter stroke.
        score = 2.0 * (rw / float(w)) + min(aspect, 60.0) / 120.0
        if (x + rx + rw / 2.0) > 0.5 * w:
            score += 0.2
        if (y + ry) > 0.7 * h:
            score += 0.2
        scored.append(((x + rx, y + ry, rw, rh), score))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:limit]


def _solid_row_band(sub: np.ndarray) -> tuple[int, int, int, int] | None:
    """Extract the solid horizontal bar inside a binary component.

    Returns the ``(x, y, w, h)`` of the tallest run of rows that are inked
    across nearly their whole width -- the bar itself -- measured within
    *sub* (component-local coordinates), or ``None`` if there is no such run.
    """
    ch, cw = sub.shape
    if ch == 0 or cw == 0:
        return None
    coverage = sub.sum(axis=1) / float(cw)
    solid = coverage >= 0.75
    if not solid.any():
        return None

    best_run = (0, -1)
    start = None
    for r in range(ch + 1):
        if r < ch and solid[r]:
            if start is None:
                start = r
        elif start is not None:
            if (r - start) > (best_run[1] - best_run[0] + 1):
                best_run = (start, r - 1)
            start = None
    r0, r1 = best_run
    if r1 < r0:
        return None

    # Trim to the bar's own horizontal extent: the longest run of columns that
    # carry ink on those rows, so a label overhanging the bar is excluded.
    cols = sub[r0:r1 + 1, :].any(axis=0)
    best_cols, c_start = (0, -1), None
    for c in range(cw + 1):
        if c < cw and cols[c]:
            if c_start is None:
                c_start = c
        elif c_start is not None:
            if (c - c_start) > (best_cols[1] - best_cols[0] + 1):
                best_cols = (c_start, c - 1)
            c_start = None
    c0, c1 = best_cols
    if c1 < c0:
        return None
    return c0, r0, c1 - c0 + 1, r1 - r0 + 1


# --------------------------------------------------------------------------
# Label detection + OCR
# --------------------------------------------------------------------------

#: Score charged for every glyph the decoder emits, in column units. Without
#: it the dynamic program happily shreds one wide glyph into several narrow
#: ones that each match some sliver of it very well.
_GLYPH_PENALTY = 0.2 * _TPL


def _width_bank() -> dict[int, tuple[np.ndarray, list[str], np.ndarray]]:
    """Templates grouped by candidate width, ready for vectorized matching.

    Returns ``{width: (stack[T, _TPL, width], chars, aspect_factor)}``. Each
    template contributes a few stretched widths around its natural one, which
    is all that is needed because templates are rendered at the same cap
    height the probe line is normalized to.
    """
    cached = _WIDTH_BANK_CACHE.get("bank")
    if cached is not None:
        return cached

    by_width: dict[int, list[tuple[str, np.ndarray, float]]] = {}
    min_w = max(4, int(round(0.16 * _TPL)))
    for ch, arr in _build_templates():
        tw = arr.shape[1]
        for stretch in (0.8, 0.9, 1.0, 1.12, 1.3):
            w = max(min_w, int(round(tw * stretch)))
            if w > 3 * _TPL:
                continue
            # Disagreement with the glyph's natural width is suspicious even
            # when the stretched shapes overlap well.
            ratio = min(w, tw) / float(max(w, tw))
            by_width.setdefault(w, []).append(
                (ch, cv2.resize(arr, (w, _TPL), interpolation=cv2.INTER_AREA),
                 0.6 + 0.4 * ratio))

    bank: dict[int, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for w, entries in by_width.items():
        stack = np.stack([e[1] for e in entries]).astype(np.float32)
        chars = [e[0] for e in entries]
        factors = np.asarray([e[2] for e in entries], dtype=np.float32)
        bank[w] = (stack, chars, factors)
    _WIDTH_BANK_CACHE["bank"] = bank
    return bank


_WIDTH_BANK_CACHE: dict[str, dict[int, tuple[np.ndarray, list[str], np.ndarray]]] = {}


def _decode_line(line: np.ndarray) -> tuple[str, float]:
    """Read a normalized text line with a left-to-right dynamic program.

    Glyphs touch constantly in small anti-aliased labels -- ``mm`` fuses into
    one blob, and so does ``2.5`` around its decimal point -- so no cut points
    are chosen up front. Instead every way of covering the strip with glyph
    templates is scored, and the best-scoring cover wins. ``cost[x]`` is the
    best total score for the first *x* columns.
    """
    height, width = line.shape
    if width < 3 or height != _TPL:
        return "", 0.0

    bank = _width_bank()
    col_ink = line.sum(axis=0)
    blank = col_ink < 0.06 * height
    ink_prefix = np.concatenate([[0.0], np.cumsum(col_ink)])

    neg = -1e18
    cost = np.full(width + 1, neg, dtype=np.float64)
    cost[0] = 0.0
    back: list[tuple[int, str, float] | None] = [None] * (width + 1)

    for x in range(1, width + 1):
        # An empty column costs nothing to skip and earns near-full credit, so
        # inter-glyph gaps do not drag the confidence down.
        if blank[x - 1] and cost[x - 1] > neg:
            value = cost[x - 1] + 0.85
            if value > cost[x]:
                cost[x] = value
                back[x] = (x - 1, "", 0.0)
        for w, (stack, chars, factors) in bank.items():
            if w > x or cost[x - w] <= neg:
                continue
            # Skip covers that would place a glyph over blank canvas.
            if (ink_prefix[x] - ink_prefix[x - w]) < 0.05 * height * w:
                continue
            seg = line[np.newaxis, :, x - w:x]
            inter = np.minimum(stack, seg).sum(axis=(1, 2))
            union = np.maximum(stack, seg).sum(axis=(1, 2))
            scores = np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0) * factors
            k = int(np.argmax(scores))
            value = cost[x - w] + float(scores[k]) * w - _GLYPH_PENALTY
            if value > cost[x]:
                cost[x] = value
                back[x] = (x - w, chars[k], float(scores[k]))

    if cost[width] <= neg:
        return "", 0.0

    # Walk the best cover back to front, turning wide gaps into spaces.
    pieces: list[tuple[int, int, str, float]] = []
    x = width
    while x > 0:
        step = back[x]
        if step is None:
            break
        prev, ch, score = step
        pieces.append((prev, x, ch, score))
        x = prev
    pieces.reverse()

    chars: list[str] = []
    weighted, covered = 0.0, 0
    gap = 0
    for start, end, ch, score in pieces:
        if not ch:
            gap += end - start
            continue
        if chars and gap > 0.3 * height:
            chars.append(" ")
        gap = 0
        chars.append(ch)
        weighted += score * (end - start)
        covered += end - start

    # Confidence is the ink-weighted mean glyph match, not the penalized DP
    # total, so it stays comparable between labels of different lengths.
    conf = (weighted / covered) if covered else 0.0
    return "".join(chars).strip(), float(conf)


def _read_label(mask: np.ndarray, ink: np.ndarray,
                bar_box: tuple[int, int, int, int],
                shape: tuple[int, int]) -> tuple[str, float, tuple[int, int, int, int] | None]:
    """OCR the text that belongs to the bar at *bar_box*.

    Returns ``(text, mean_confidence, text_box)``. The label is looked for
    below the bar first (FetoMorph's own layout) and then above it.
    """
    h, w = shape
    bx, by, bw, bh = bar_box

    best: tuple[str, float, tuple[int, int, int, int] | None] = ("", 0.0, None)
    for side in ("below", "above"):
        if side == "below":
            y0, y1 = by + bh, min(h, by + bh + max(int(6.0 * max(bh, 3)), int(0.12 * h)))
        else:
            y1, y0 = by, max(0, by - max(int(6.0 * max(bh, 3)), int(0.12 * h)))
        if y1 - y0 < 4:
            continue
        # Allow the text to overhang the bar on both sides (it is centred, and
        # "20 mm" is usually wider than a short bar).
        pad = int(0.6 * bw) + 10
        x0, x1 = max(0, bx - pad), min(w, bx + bw + pad)
        band = mask[y0:y1, x0:x1]
        band_ink = ink[y0:y1, x0:x1]
        if not band.any():
            continue

        num, _labels, stats = _components(band)
        blobs = []
        for i in range(1, num):
            gx, gy, gw, gh_, area = (int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]),
                                     int(stats[i, 3]), int(stats[i, 4]))
            if area < 2 or gh_ < 2 or gw < 1:
                continue
            if gh_ > 0.9 * (y1 - y0):
                continue
            blobs.append((gx, gy, gw, gh_))
        if not blobs:
            continue

        # Locate the text line. Every glyph in the alphabet sits on the
        # baseline and none descends below it, so the baseline is the modal
        # bottom edge and the cap height is the tallest glyph standing on it.
        base_y = float(np.median([g[1] + g[3] for g in blobs]))
        tallest = max(g[3] for g in blobs)
        on_line = [g for g in blobs if abs((g[1] + g[3]) - base_y) <= 0.3 * tallest]
        if not on_line:
            continue
        # The cap height must come from the tallest glyph, not a median: a
        # median over "20 mm" lands between the digits' cap height and the
        # x-height of "m", which truncates every digit. For the same reason a
        # decimal point must not pull it down.
        cell = float(max(g[3] for g in on_line))
        if cell < 4:
            continue
        cap_y = base_y - cell

        # Everything that sits on this line, including punctuation.
        line_blobs = [g for g in blobs
                      if g[1] >= cap_y - 0.35 * cell and (g[1] + g[3]) <= base_y + 0.35 * cell]
        if not line_blobs:
            continue
        lx0 = min(g[0] for g in line_blobs)
        lx1 = max(g[0] + g[2] for g in line_blobs)
        # Keep the soft ink of the glyphs on this line, masked to their boxes
        # so nothing else in the band leaks into the strip.
        strip = np.zeros(band.shape, np.float32)
        for gx, gy, gw, gh_ in line_blobs:
            box_ink = band_ink[gy:gy + gh_, gx:gx + gw]
            strip[gy:gy + gh_, gx:gx + gw] = np.maximum(
                strip[gy:gy + gh_, gx:gx + gw], box_ink)

        # Normalize the whole line to a fixed cap-to-baseline box, so glyph
        # templates can be matched with their vertical placement intact.
        line = _to_line_box(strip[:, lx0:lx1], int(round(cap_y)), int(round(base_y)))
        if not line.any():
            continue

        text, conf = _decode_line(line)
        if text and conf > best[1]:
            box = (x0 + lx0, y0 + int(cap_y),
                   max(1, lx1 - lx0), max(1, int(round(base_y - cap_y))))
            best = (text, conf, box)

    return best


_NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)")


def _parse_label(text: str) -> tuple[float, str] | None:
    """Turn an OCR'd label such as ``"20 mm"`` into ``(20.0, "mm")``."""
    cleaned = text.replace(" ", "")
    m = _NUM_RE.search(cleaned)
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", "."))
    except ValueError:
        return None
    if value <= 0:
        return None

    tail = cleaned[m.end():].lower()
    # Letters only; OCR sometimes reads the "m" of "mm" as "n" or "u".
    tail = "".join(c for c in tail if c.isalpha() or c == "µ")
    unit = _UNIT_ALIASES.get(tail)
    if unit is None and tail:
        # Fall back to the longest known spelling that prefixes the tail.
        for cand in sorted(_UNIT_ALIASES, key=len, reverse=True):
            if tail.startswith(cand):
                unit = _UNIT_ALIASES[cand]
                break
    if unit is None:
        return None
    return value, unit


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def read_scalebar(img_bgr: np.ndarray, *,
                  min_confidence: float = 0.45,
                  region_top: float = 0.55) -> ScalebarReading | None:
    """Detect the scale bar in *img_bgr* and read its label.

    Args:
        img_bgr: Image in BGR colour space, as returned by ``cv2.imread``.
        min_confidence: Lowest mean glyph score accepted for the label.
        region_top: Fraction of the image height above which the bar is not
            looked for. ``0.55`` searches the bottom 45%.

    Returns:
        A :class:`ScalebarReading`, or ``None`` when no bar-and-label pair
        could be resolved.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    h, w = img_bgr.shape[:2]
    best: ScalebarReading | None = None

    for polarity, mask, ink in _ink_masks(img_bgr):
        # Widen the search if the bottom strip holds nothing readable; some
        # exports park the bar mid-frame.
        for top in (region_top, 0.0):
            for bar_box, _bar_score in _find_bar(mask, (h, w), region_top=top):
                text, conf, text_box = _read_label(mask, ink, bar_box, (h, w))
                if not text or conf < min_confidence:
                    continue
                parsed = _parse_label(text)
                if parsed is None:
                    continue
                value, unit = parsed
                bar_px = float(bar_box[2])
                if bar_px <= 1:
                    continue
                bar_mm = value * _UNIT_MM[unit]
                mm_per_px = bar_mm / bar_px
                if not (_MIN_BAR_MM <= bar_mm <= _MAX_BAR_MM):
                    logger.debug("Scalebar label %r implies an implausible bar "
                                 "length (%g mm); ignoring.", text, bar_mm)
                    continue
                if not (_MIN_MM_PER_PX <= mm_per_px <= _MAX_MM_PER_PX):
                    logger.debug("Scalebar label %r implies an implausible pixel "
                                 "size (%g mm/px); ignoring.", text, mm_per_px)
                    continue
                reading = ScalebarReading(
                    bar_px=bar_px,
                    value=value,
                    unit=unit,
                    mm_per_px=mm_per_px,
                    unit_per_px=value / bar_px,
                    text=text,
                    bar_box=bar_box,
                    text_box=text_box,
                    confidence=conf,
                    polarity=polarity,
                )
                if best is None or reading.confidence > best.confidence:
                    best = reading
                break
            if best is not None:
                break

    return best


def read_scalebar_file(path: str, **kwargs) -> ScalebarReading | None:
    """:func:`read_scalebar` for an image on disk. ``None`` if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Scalebar reader could not open %s", path)
        return None
    return read_scalebar(img, **kwargs)


@dataclass
class ScalebarContours:
    """How much contour area the drawn scale bar contributes to an image.

    The bar and its label are ink on the slice, so the brain segmentation
    picks them up as contours and every area / perimeter measurement
    includes them. Raising the contour-area threshold above
    :attr:`scalebar_max_px2` drops them again.

    Attributes:
        scalebar_max_px2: Area of the largest scale-bar contour, in px².
        scalebar_total_px2: Combined area of all scale-bar contours, in px².
        largest_other_px2: Area of the largest contour that is *not* part of
            the scale bar -- normally the brain, and the value a suggested
            threshold must stay well below.
        count: How many contours were attributed to the scale bar.
    """

    scalebar_max_px2: float
    scalebar_total_px2: float
    largest_other_px2: float
    count: int

    def threshold_px2(self, margin: float = 1.15) -> float:
        """Contour-area cut that just excludes the bar, in px²."""
        return self.scalebar_max_px2 * float(margin)

    def is_safe(self, margin: float = 1.15, headroom: float = 2.0) -> bool:
        """True when excluding the bar leaves the brain contour untouched.

        The suggested cut must sit at least *headroom* times below the
        largest non-scale-bar contour, otherwise raising the threshold could
        discard the anatomy along with the bar.
        """
        if self.largest_other_px2 <= 0:
            return False
        return self.threshold_px2(margin) * float(headroom) <= self.largest_other_px2


def measure_scalebar_contours(img_bgr: np.ndarray,
                              reading: ScalebarReading,
                              *, pad: int = 6) -> ScalebarContours | None:
    """Measure the contour area the scale bar adds to *img_bgr*.

    The image is binarised exactly the way the measurement pipeline does it,
    so the areas reported here are the ones the contour-area threshold will
    be compared against. Contours whose centre falls inside the (padded) bar
    or label box are attributed to the scale bar.

    Returns ``None`` when the image yields no contours at all.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    # Imported lazily: this module is otherwise free of the measurement stack.
    from functions.measurements_image import _binarise_brain

    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    try:
        bw = _binarise_brain(img_bgr, cv2.COLOR_RGB2GRAY)
    except Exception as ex:
        logger.warning("Could not binarise image for scalebar contours: %s", ex)
        return None

    contours, _hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    boxes = [reading.bar_box]
    if reading.text_box:
        boxes.append(reading.text_box)

    def _belongs_to_scalebar(contour) -> bool:
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w / 2.0, y + h / 2.0
        for bx, by, bw_, bh_ in boxes:
            if (bx - pad) <= cx <= (bx + bw_ + pad) and \
               (by - pad) <= cy <= (by + bh_ + pad):
                return True
        return False

    bar_areas, other_areas = [], []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        (bar_areas if _belongs_to_scalebar(contour) else other_areas).append(area)

    if not bar_areas:
        return None
    return ScalebarContours(
        scalebar_max_px2=max(bar_areas),
        scalebar_total_px2=float(sum(bar_areas)),
        largest_other_px2=max(other_areas) if other_areas else 0.0,
        count=len(bar_areas),
    )


def annotate_reading(img_bgr: np.ndarray, reading: ScalebarReading) -> np.ndarray:
    """Return a copy of *img_bgr* with the detected bar and label outlined.

    Used by the confirmation dialog so the user can see what was measured
    before the scale is applied.
    """
    out = img_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    x, y, bw, bh = reading.bar_box
    pad = max(2, int(round(0.004 * max(out.shape[:2]))))
    cv2.rectangle(out, (x - pad, y - pad), (x + bw + pad, y + bh + pad),
                  (0, 200, 0), max(1, pad // 2 + 1))
    if reading.text_box:
        tx, ty, tw, th = reading.text_box
        # Orange (BGR) for the label, against the green of the measured bar.
        cv2.rectangle(out, (tx - pad, ty - pad), (tx + tw + pad, ty + th + pad),
                      (0, 140, 255), max(1, pad // 2 + 1))
    return out
