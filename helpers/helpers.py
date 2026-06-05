"""Shared helper functions for FetoMorph measurement pipelines.

Provides utilities for OpenCV text sizing, morphological kernels, convexity-
defect depth conversion (pixel ↔ mm), red-cube scale calibration, scalebar
drawing, and PyVista slice geometry.
"""

from __future__ import annotations

from deps import *
from functions.nifti_to_image import draw_new_scale_bar
from constants import (
    SULCUS_PRIMARY_MIN_FRACTION,
    SULCUS_PRIMARY_MAX_FRACTION,
    SULCUS_SECONDARY_MIN_FRACTION,
    SULCUS_SECONDARY_MAX_FRACTION,
    SULCUS_TERTIARY_MIN_FRACTION,
    SULCUS_TERTIARY_MAX_FRACTION,
)

logger = logging.getLogger(__name__)


def _get_viz():
    """Lazy import to break the helpers ↔ managers circular dependency at load time."""
    from managers.visualization_settings import get_active
    return get_active()


# Default marker colours (BGR) for classified sulci. The live values come from
# VisualizationSettings — these defaults are used only when settings aren't yet
# initialised. Use ``sulcus_color(class_name)`` to look up the active value.
SULCUS_CLASS_COLORS_DEFAULTS = {
    "primary":      (255,   0,   0),
    "secondary":    (  0, 215, 255),
    "tertiary":     (255, 255,   0),
    "unclassified": (200, 200, 200),
}


def sulcus_color(class_name: str) -> tuple[int, int, int]:
    """Return the BGR colour for one sulcus class from live VisualizationSettings."""
    vs = _get_viz()
    return {
        "primary":      tuple(vs.sulcus_primary_color_bgr),
        "secondary":    tuple(vs.sulcus_secondary_color_bgr),
        "tertiary":     tuple(vs.sulcus_tertiary_color_bgr),
        "unclassified": tuple(vs.sulcus_unclassified_color_bgr),
    }[class_name]


class _SulcusClassColorsLive(dict):
    """Drop-in replacement for the legacy dict that reads from live settings."""

    def __getitem__(self, key):
        return sulcus_color(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        return iter(SULCUS_CLASS_COLORS_DEFAULTS)

    def keys(self):
        return SULCUS_CLASS_COLORS_DEFAULTS.keys()

    def values(self):
        return [self[k] for k in self.keys()]

    def items(self):
        return [(k, self[k]) for k in self.keys()]


# Back-compat: callers that still index ``SULCUS_CLASS_COLORS["primary"]`` now
# transparently read from VisualizationSettings.
SULCUS_CLASS_COLORS = _SulcusClassColorsLive()


# Reference palette used to label arbitrary BGR colours with a human-readable
# name (nearest-Euclidean match in RGB). Order is irrelevant; ties are unlikely
# with these well-spaced anchors.
_NAMED_COLORS_RGB = {
    "red":        (255,   0,   0),
    "green":      (  0, 128,   0),
    "lime":       (  0, 255,   0),
    "blue":       (  0,   0, 255),
    "yellow":     (255, 255,   0),
    "cyan":       (  0, 255, 255),
    "magenta":    (255,   0, 255),
    "white":      (255, 255, 255),
    "black":      (  0,   0,   0),
    "gray":       (128, 128, 128),
    "light gray": (200, 200, 200),
    "dark gray":  ( 64,  64,  64),
    "gold":       (255, 215,   0),
    "orange":     (255, 165,   0),
    "purple":     (128,   0, 128),
    "pink":       (255, 192, 203),
    "brown":      (165,  42,  42),
    "navy":       (  0,   0, 128),
    "teal":       (  0, 128, 128),
    "olive":      (128, 128,   0),
}


def bgr_to_color_name(bgr) -> str:
    """Return the closest named colour for a BGR tuple from VisualizationSettings."""
    b, g, r = (int(x) for x in bgr)
    return min(
        _NAMED_COLORS_RGB.items(),
        key=lambda kv: (kv[1][0] - r) ** 2 + (kv[1][1] - g) ** 2 + (kv[1][2] - b) ** 2,
    )[0]


def classify_sulcus_depth(depth_value: float, slice_length: float) -> str:
    """Bin a sulcus depth into primary / secondary / tertiary by % of slice length.

    Returns ``"unclassified"`` when the depth falls outside every range.
    Only meaningful when ``slice_length`` is the longest side of the brain's
    bounding box (or equivalent physical extent) in the same unit as
    ``depth_value``.
    """
    if slice_length <= 0:
        return "unclassified"
    frac = depth_value / slice_length
    if SULCUS_PRIMARY_MIN_FRACTION <= frac <= SULCUS_PRIMARY_MAX_FRACTION:
        return "primary"
    if SULCUS_SECONDARY_MIN_FRACTION <= frac < SULCUS_SECONDARY_MAX_FRACTION:
        return "secondary"
    if SULCUS_TERTIARY_MIN_FRACTION <= frac <= SULCUS_TERTIARY_MAX_FRACTION:
        return "tertiary"
    return "unclassified"


def empty_depth_sets() -> dict:
    """Return a fresh ``{primary, secondary, tertiary, unclassified}`` dict of lists."""
    return {"primary": [], "secondary": [], "tertiary": [], "unclassified": []}


def flatten_depth_sets(depth_sets: dict) -> list:
    """Sort each class set descending and return one combined sorted list."""
    for _k in depth_sets:
        depth_sets[_k].sort(reverse=True)
    flat = (
        depth_sets["primary"]
        + depth_sets["secondary"]
        + depth_sets["tertiary"]
        + depth_sets["unclassified"]
    )
    flat.sort(reverse=True)
    return flat


def format_sulcus_class_summary(depth_sets: dict) -> str:
    """One-line console summary of per-class sulcus counts.

    Colour labels are derived from the current :class:`VisualizationSettings`
    so the printed names stay in sync with whatever colours the user picked.
    """
    return (
        "Sulci classification: "
        f"primary({bgr_to_color_name(sulcus_color('primary'))})={len(depth_sets['primary'])}, "
        f"secondary({bgr_to_color_name(sulcus_color('secondary'))})={len(depth_sets['secondary'])}, "
        f"tertiary({bgr_to_color_name(sulcus_color('tertiary'))})={len(depth_sets['tertiary'])}, "
        f"unclassified({bgr_to_color_name(sulcus_color('unclassified'))})={len(depth_sets['unclassified'])}"
    )


# ----------------------------------------------------------------------
# Shared per-class sulcus export helpers
# ----------------------------------------------------------------------

SULCUS_CLASSES = ("primary", "secondary", "tertiary", "unclassified")


def sulcus_export_columns(unit: str = "mm") -> list:
    """Return the per-class sulcus column headers for an Excel export.

    For each class (primary/secondary/tertiary/unclassified) we export:
    a count, three raw value cells (``v1``..``v3``), and three summary
    stats (``min``/``max``/``mean``). Whether the raw cells or the
    summary cells are populated for a given row is decided per-row by
    :func:`sulcus_export_cells` based on a 3-value threshold.
    """
    cols: list = []
    for k in SULCUS_CLASSES:
        prefix = k.capitalize()
        cols.extend([
            f"{prefix}_count",
            f"{prefix}_v1_{unit}",
            f"{prefix}_v2_{unit}",
            f"{prefix}_v3_{unit}",
            f"{prefix}_min_{unit}",
            f"{prefix}_max_{unit}",
            f"{prefix}_mean_{unit}",
        ])
    return cols


def sulcus_export_cells(depth_sets: dict) -> list:
    """Build the per-class cells for one row of an Excel export.

    For each class:
    - count cell holds ``len(values)``
    - if count ≤ 3: the values populate ``v1``..``v3`` (padded with ``None``)
      and the min/max/mean cells are ``None``
    - if count > 3: ``v1``..``v3`` are ``None`` and min/max/mean hold the
      summary stats
    """
    cells: list = []
    for k in SULCUS_CLASSES:
        vals = list(depth_sets.get(k, []))
        n = len(vals)
        cells.append(n)
        if n <= 3:
            padded = (vals + [None, None, None])[:3]
            cells.extend(padded)
            cells.extend([None, None, None])
        else:
            cells.extend([None, None, None])
            cells.extend([min(vals), max(vals), sum(vals) / n])
    return cells


def pad_row(row, width: int) -> list:
    """Pad a ragged summary row to a fixed DataFrame width with ``None``s."""
    return list(row) + [None] * (width - len(row))


def drop_empty_columns(df):
    """Drop any DataFrame column whose every cell is ``None`` / ``NaN`` / ``""``.

    ``0`` counts as a real value (e.g. a per-class ``_count`` of zero is
    informative). No column is special-cased: every column is checked.
    """
    keep_cols: list = []
    for col in df.columns:
        for v in df[col]:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if v == "":
                continue
            keep_cols.append(col)
            break
    return df[keep_cols]

def image_annotation_style(
    h: int,
    w: int | None = None,
    *,
    style: str = "bold",
    cap: int = 30,
) -> tuple[int, float, int]:
    """Return (thickness, font_scale, radius_px) based on image dimensions."""
    h = max(1, int(h))
    w = h if w is None else max(1, int(w))
    base = min(h, w)

    base_div = {"thin": 380, "regular": 320, "bold": 260}[style]
    thickness = max(1, min(int(round(h / base_div)), cap))
    font_scale = float(np.clip(base / 700.0, 0.8, 5.0))
    radius_px = int(np.clip(round(base / 100), 2.0, 30))

    vs = _get_viz()
    thickness = max(1, int(round(thickness * float(vs.contour_thickness_multiplier))))
    font_scale = float(font_scale) * float(vs.text_scale_multiplier)
    radius_px = max(1, int(round(radius_px * float(vs.marker_radius_multiplier))))
    return thickness, font_scale, radius_px

def draw_hallmarks_values_on_image(
    bgr: np.ndarray,
    thickness: int | None = None,
    font_scale: float | None = None,
    *,
    area: float | None = None,
    perimeter: float | None = None,
    perimeter_internal: float | None = None,
    curve: float | None = None,
    lgi: float | None = None,
    compactness: float | None = None,
    unit: str = "mm",
    box_position: str | None = "topleft",
    anchor_ratio: tuple[float, float] = (0.02, 0.05),
    anchor_px: tuple[int, int] | None = None,
    margin: int | None = None,
    margin_ratio: float = 0.012,
) -> np.ndarray:
    """Draw Area/Perimeter/LGI/Compactness as a non-overlapping text block.

    Args:
        box_position: Preset for textbox placement:
            "topleft", "topright", "bottomleft", "bottomright", "center".
            If unknown, placement falls back to anchor settings.
    """
    if bgr is None or not isinstance(bgr, np.ndarray) or bgr.ndim != 3:
        return bgr

    out = bgr.copy()
    h, w = out.shape[:2]
    if margin is None:
        margin = int(np.clip(round(w * float(margin_ratio)), 3, 24))
    else:
        margin = max(0, int(margin))
    if thickness is None or font_scale is None:
        t_auto, f_auto, _ = image_annotation_style(h, w, style="regular")
        if thickness is None:
            thickness = t_auto
        if font_scale is None:
            font_scale = f_auto
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = []
    if area is not None:
        lines.append(f"Area: {area:.2f} {unit}^2")
    if perimeter is not None:
        label = "Perimeter (ext)" if perimeter_internal is not None else "Perimeter"
        lines.append(f"{label}: {perimeter:.2f} {unit}")
    if perimeter_internal is not None:
        lines.append(f"Perimeter (int): {perimeter_internal:.2f} {unit}")
    if curve is not None:
        lines.append(f"Curve: {curve:.2f} {unit}")
    if lgi is not None:
        lines.append(f"lGI: {lgi:.2f}")
    if compactness is not None:
        lines.append(f"Compactness: {compactness:.2f}")
    if not lines:
        return out



    def _box_metrics(fs: float) -> tuple[int, int, int, int]:
        sizes = [cv2.getTextSize(t, font, fs, thickness) for t in lines]
        tw = max(s[0][0] for s in sizes)
        lh = max(s[0][1] + s[1] for s in sizes)
        lg = max(2, int(round(lh * 0.25)))
        bw = tw + 2 * margin
        bh = (lh * len(lines)) + (lg * (len(lines) - 1)) + 2 * margin
        return tw, lh, lg, bw, bh

    # Keep textbox width within 15%..30% of image width by scaling font size.
    min_ratio, max_ratio = 0.15, 0.30
    text_w, line_h, line_gap, box_w, box_h = _box_metrics(float(font_scale))
    # Skip overlay when image is extremely small relative to the text block.
    # Rule: if the box dimension is smaller than one-quarter of any image dimension.

    if w > 0:
        if box_w > w/2 or box_h > h/2:
            return out
        else:
            ratio = box_w / float(w)
            # Shrink/grow gradually until textbox width ratio fits the target band.
            n_iter = 0
            while ratio > max_ratio and font_scale > 0.3 and n_iter < 40:
                font_scale = max(0.3, float(font_scale) * 0.9)
                text_w, line_h, line_gap, box_w, box_h = _box_metrics(font_scale)
                ratio = box_w / float(w)
                n_iter += 1

            n_iter = 0
            while ratio < min_ratio and font_scale < 8.0 and n_iter < 40:
                font_scale = min(8.0, float(font_scale) * 1.1)
                text_w, line_h, line_gap, box_w, box_h = _box_metrics(font_scale)
                # Bail out if growing would dominate a very small image.
                if box_w > w/2 or box_h > h/2:
                    return out
                ratio = box_w / float(w)
                n_iter += 1


    if anchor_px is not None:
        x1, y1 = int(anchor_px[0]), int(anchor_px[1])
    elif isinstance(box_position, str):
        pos = box_position.strip().lower()
        if pos == "topleft":
            x1, y1 = margin, margin
        elif pos == "topright":
            x1, y1 = w - box_w - margin, margin
        elif pos == "bottomleft":
            x1, y1 = margin, h - box_h - margin
        elif pos == "bottomright":
            x1, y1 = w - box_w - margin, h - box_h - margin
        elif pos == "center":
            x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        else:
            rx = float(np.clip(anchor_ratio[0], 0.0, 1.0))
            ry = float(np.clip(anchor_ratio[1], 0.0, 1.0))
            x1, y1 = int(round(w * rx)), int(round(h * ry))
    else:
        rx = float(np.clip(anchor_ratio[0], 0.0, 1.0))
        ry = float(np.clip(anchor_ratio[1], 0.0, 1.0))
        x1, y1 = int(round(w * rx)), int(round(h * ry))
    x1 = min(max(0, x1), max(0, w - box_w))
    y1 = min(max(0, y1), max(0, h - box_h))

    cv2.rectangle(out, (x1, y1), (x1 + box_w, y1 + box_h), (0, 0, 0), -1)
    text_color = tuple(_get_viz().hallmark_text_color_bgr)
    y_base = y1 + margin + line_h
    for i, txt in enumerate(lines):
        y = y_base + i * (line_h + line_gap)
        cv2.putText(out, txt, (x1 + margin, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return out
    
    
def compute_kernel_convex(kernel_size: int) -> np.ndarray:
    """Create an elliptical morphological structuring element.

    Used by the GI (gyrification index) pipeline to morphologically close
    sulci before computing the "outer" (convex) perimeter.

    Args:
        kernel_size: Diameter of the ellipse in pixels.

    Returns:
        A ``uint8`` structuring element suitable for ``cv2.morphologyEx``.
    """
    kernel_convex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    return kernel_convex

def defect_mm_per_px_and_fixed(
    start: tuple[float, float],
    end:   tuple[float, float],
    far:   tuple[float, float],
    sx: float,   # mm per pixel in x
    sz: float,   # mm per pixel in y (your z)
) -> tuple[float | None, float | None]:
    """
    Returns:
        mm_per_px     : mm per 1 pixel along the defect's normal direction
        mm_per_fixed  : mm per 1 raw 'd' unit (remember: depth_px = d/256)
    """
    a = np.asarray(start, float); b = np.asarray(end, float); f = np.asarray(far, float)
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 == 0.0:
        return None, None  # degenerate edge

    # projection of 'far' onto the (infinite) line through start-end
    t = float(np.dot(f - a, ab) / ab2)
    p = a + t * ab

    # normal vector in pixel space (direction of the defect)
    v = f - p
    n = float(np.hypot(v[0], v[1]))
    if n == 0.0:
        return 0.0, 0.0  # zero-depth defect

    ux, uy = v[0]/n, v[1]/n  # unit normal (pixels)
    mm_per_px = math.hypot(sx * ux, sz * uy)
    mm_per_fixed = mm_per_px / 256.0  # because OpenCV stores d in 8.8 fixed-point
    return mm_per_px, mm_per_fixed

def contours_exclude(contours: list, excluded_space: np.ndarray, image_shape: tuple[int, int]) -> list:
    # Needed to remove the red reference-cube contour from the brain contours
    # so that it does not pollute area / perimeter measurements.
    """
    Filter contours by excluding any that overlap with 'excluded_space' (uint8 mask).
    """
    filtered = []
    for cnt in contours:
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if np.count_nonzero(cv2.bitwise_and(mask, excluded_space)) == 0:
            filtered.append(cnt)
    return filtered

def fill_section_polydata(section):
    """Return a filled (capped) cross-section to render as a solid colored face.

    ``mesh.slice()`` of a surface mesh (e.g. FreeSurfer pial) returns the
    boundary CURVE (line cells), which renders as a thin outline so the slice
    interior reads as background — the cross-section looks hollow. This
    triangulates the closed contour into filled polygons (``vtkStripper`` +
    ``vtkContourTriangulator``, even-odd rule) so the section renders as a solid
    silhouette the same way a sliced volumetric VTK dataset already does:
    concavities (sulci) are followed and genuine holes (enclosed voids) are
    preserved. If the section is already polygonal (a sliced volume) or
    triangulation fails/empties, the input is returned unchanged.
    """
    if section is None or getattr(section, "n_points", 0) == 0:
        return section
    try:
        if section.n_faces_strict > 0:
            return section  # already a filled face (e.g. sliced volume)
    except Exception:
        pass
    try:
        import vtk
        from pyvista import wrap
        strip = vtk.vtkStripper()
        strip.SetInputData(section)
        strip.Update()
        tri = vtk.vtkContourTriangulator()
        tri.SetInputData(strip.GetOutput())
        tri.Update()
        filled = wrap(tri.GetOutput())
        if filled is None or filled.n_cells == 0:
            return section
        return filled
    except Exception as ex:
        logger.warning("section fill failed; rendering outline: %s", ex)
        return section


def split_inner_and_internal_contours(
    contours, hierarchy, excluded_space, image_shape, min_area,
):
    """Split a RETR_CCOMP result into kept outer outlines and kept holes.

    - Top-level contours (hierarchy[i][3] == -1) are first filtered by the
      red-rect exclusion mask, then by ``min_area``.
    - A child contour is kept iff its parent is one of the kept top-level
      contours and its own area exceeds ``min_area``.
    Returns ``(inner_filtered, internal_filtered)``.
    """
    if hierarchy is None or not contours:
        return [], []

    hierarchy = np.asarray(hierarchy)
    if hierarchy.ndim == 3:
        hierarchy = hierarchy[0]

    inner_filtered = []
    kept_top_level = set()
    min_area = float(min_area)

    for idx, cnt in enumerate(contours):
        if hierarchy[idx][3] != -1:
            continue

        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if np.count_nonzero(cv2.bitwise_and(mask, excluded_space)) != 0:
            continue
        if cv2.contourArea(cnt) <= min_area:
            continue

        inner_filtered.append(cnt)
        kept_top_level.add(idx)

    if not kept_top_level:
        return inner_filtered, []

    internal_filtered = []
    for idx, cnt in enumerate(contours):
        parent_idx = int(hierarchy[idx][3])
        if parent_idx in kept_top_level and cv2.contourArea(cnt) > min_area:
            internal_filtered.append(cnt)

    return inner_filtered, internal_filtered
    
SCALE_CUBE_DETECTION_METHOD = "HSV red mask + morphology + minAreaRect"


def red_scale_cube_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Return a cleaned HSV red mask for the rendered scale cube."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_red = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    upper_red = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype("uint8")


def calc_scale_with_metadata(
    image_rgb: np.ndarray,
    cube_length: float,
    *,
    expected_px_range: tuple[float, float] | None = None,
    min_area_px: float = 25.0,
    max_aspect_error: float = 0.35,
) -> tuple[float | None, dict, np.ndarray]:
    """Detect the red scale cube and return ``(mm_per_px, metadata, mask)``.

    The visible cube face should be approximately square and parallel to the
    camera plane. A failed validation returns ``None`` with a clear status.
    """
    metadata = {
        "cube_len_mm": float(cube_length),
        "detected_cube_size_px": None,
        "computed_mm_per_px": None,
        "projection_mode": "parallel",
        "cube_detection_method": SCALE_CUBE_DETECTION_METHOD,
        "calibration_status": "failed",
        "calibration_error_percent": None,
    }
    mask = red_scale_cube_mask(image_rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        metadata["calibration_status"] = "failed: no red contour"
        return None, metadata, mask

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area_px:
        metadata["calibration_status"] = f"failed: red contour area {area:.1f} below {min_area_px:.1f}px"
        return None, metadata, mask

    rect = cv2.minAreaRect(contour)
    width_px, height_px = (float(rect[1][0]), float(rect[1][1]))
    if width_px <= 0 or height_px <= 0:
        metadata["calibration_status"] = "failed: fitted red rectangle has non-positive dimensions"
        return None, metadata, mask

    long_px = max(width_px, height_px)
    short_px = min(width_px, height_px)
    aspect = long_px / max(short_px, 1e-9)
    if aspect > (1.0 + max_aspect_error):
        metadata["detected_cube_size_px"] = long_px
        metadata["calibration_status"] = f"failed: red contour aspect ratio {aspect:.3f} is not square"
        return None, metadata, mask

    if expected_px_range is None:
        h, w = image_rgb.shape[:2]
        expected_px_range = (5.0, 0.75 * float(min(h, w)))
    lo, hi = expected_px_range
    if long_px < lo or long_px > hi:
        metadata["detected_cube_size_px"] = long_px
        metadata["calibration_status"] = f"failed: detected cube size {long_px:.1f}px outside [{lo:.1f}, {hi:.1f}]"
        return None, metadata, mask

    mm_per_px = float(cube_length) / long_px
    measured_cube_len = mm_per_px * long_px
    error_percent = abs(measured_cube_len - float(cube_length)) / max(float(cube_length), 1e-9) * 100.0
    metadata.update({
        "detected_cube_size_px": long_px,
        "computed_mm_per_px": mm_per_px,
        "calibration_status": "valid",
        "calibration_error_percent": error_percent,
    })
    return mm_per_px, metadata, mask


def calc_scale(image_rgb: np.ndarray, cube_length: float) -> float | None:
    """Compatibility wrapper returning only mm-per-pixel."""
    mm_per_px, metadata, _mask = calc_scale_with_metadata(image_rgb, cube_length)
    if mm_per_px is None:
        print(f"[Scale] Scale cube was not detected or failed validation. {metadata['calibration_status']}. Measurements for this slice were skipped.")
    return mm_per_px
    
def get_red_rect_offset(image_rgb: np.ndarray) -> np.ndarray:  # noqa: returning shape (2,)
    """
    Detect red rectangle and return its center (x,y) in pixels — used to zero-align contours.
    """
    coords = np.argwhere(red_scale_cube_mask(image_rgb) > 0)
    if coords.size == 0:
        return np.array([0, 0])
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return np.array([(x_min + x_max) // 2, (y_min + y_max) // 2])

def get_nifti_present_labels(path: str, cap: int = 5000) -> set[int] | None:
    """Return the set of unique integer labels present in a NIfTI file.

    Args:
        path: Path to the ``.nii`` / ``.nii.gz`` file.
        cap: Ignore label values above this threshold (avoids noise).

    Returns:
        A set of integer labels, or ``None`` on failure.
    """
    try:
        import nibabel as nib
        img = nib.load(path)
        data = img.get_fdata(dtype=float)

        arr_i = np.rint(data).astype(np.int32)
        uniq = np.unique(arr_i)
        uniq = uniq[(uniq >= 0) & (uniq <= cap)]
        uniq_list = set(uniq.tolist())
        print(f"Region labels: \n {uniq_list} \n")
        return uniq_list
    except Exception as ex:
        logger.warning("Could not detect labels: %s", ex)
        # Fall back to defaults
        return None
        
def _mm_per_pixel_x_for_axis(zooms: tuple, ax: int) -> float:
    """Return the in-plane mm/px for the displayed X axis given the slice axis."""
    # zooms is (z0,z1,z2) == voxel size along axes 0,1,2 in mm
    if ax == 0:               # slice is a[i, :, :]
        return float(zooms[2])  # X shows axis 2
    elif ax == 1:             # slice is a[:, i, :]
        return float(zooms[2])  # X shows axis 2
    else:                     # ax == 2: slice is a[:, :, i]
        return float(zooms[1])  # X shows axis 1

def add_scalebar(qimg: QImage, zooms: np.ndarray, ax: int) -> tuple[QImage, float, float]:
    """Draw a scalebar (mm) at the bottom-right of qimg and return it."""
    # QPainter needs a 32-bit RGB(A) surface for best compatibility
    if qimg.format() not in (QImage.Format_RGB32, QImage.Format_ARGB32):
        qimg = qimg.convertToFormat(QImage.Format_RGB32)

    w, h = qimg.width(), qimg.height()
    mm_per_px = _mm_per_pixel_x_for_axis(zooms, ax)

    # Pick a nice bar length (mm) that fits ~25% of the width
    max_px = int(w * 0.25)
    nice_lengths_mm = [100, 50, 25, 20, 10, 5]
    bar_mm = next((L for L in nice_lengths_mm if (L / mm_per_px) <= max_px and (L / mm_per_px) >= 30), None)
    if bar_mm is None:
        # fallback to whatever fits (at least 20 px)
        bar_mm = max(5, int(max_px * mm_per_px))
    bar_px = int(round(bar_mm / mm_per_px))

    vs = _get_viz()
    margin = max(6, int(round(0.03 * min(w, h))))
    bar_thick = max(4, int(round(0.008 * min(w, h) * float(vs.scalebar_thickness_multiplier))))
    label_h = max(12, int(round(0.028 * min(w, h))))

    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # Backdrop for contrast
    pad = 6
    rect_w = bar_px + 2 * pad
    rect_h = bar_thick + label_h + 3 * pad
    rect_x = w - margin - rect_w
    rect_y = h - margin - rect_h
    painter.fillRect(rect_x, rect_y, rect_w, rect_h, QColor(0, 0, 0, 160))

    # Scalebar line (white)
    y_bar = rect_y + pad + bar_thick // 2
    pen = QPen(QColor(255, 255, 255))
    pen.setWidth(bar_thick)
    painter.setPen(pen)
    x1 = rect_x + pad
    x2 = x1 + bar_px
    painter.drawLine(x1, y_bar, x2, y_bar)

    # Text (e.g., "20 mm")
    sb_b, sb_g, sb_r = vs.scalebar_text_color_bgr
    painter.setPen(QColor(int(sb_r), int(sb_g), int(sb_b)))
    font = painter.font()
    font.setPointSizeF(max(8.0, 0.9 * label_h))
    painter.setFont(font)
    text_rect = QRectF(x1, y_bar + pad, bar_px, label_h + pad)
    painter.drawText(text_rect, Qt.AlignCenter, f"{int(round(bar_mm))} mm")

    painter.end()
    return qimg, mm_per_px, bar_mm

def _add_scalebar_on_annotated(
    annotated: np.ndarray,
    pixel_size: float,
    unit: str,
    add_scalebar: bool = True,
) -> np.ndarray:
    """Optionally draw a physical scale bar on an annotated BGR image."""
    if not add_scalebar or pixel_size <= 0:
        return annotated

    image_width_phys = annotated.shape[1] * pixel_size
    target = image_width_phys * 0.2
    magnitude = 10 ** int(np.floor(np.log10(max(target, 1e-9))))
    bar_phys = next(
        (magnitude * n for n in [1, 2, 5, 10] if magnitude * n >= target * 0.7),
        magnitude * 10,
    )
    bar_px = max(1, int(round(bar_phys / pixel_size)))
    return draw_new_scale_bar(annotated, bar_px, text=f"{bar_phys:g} {unit}")


def get_max_slice_thickness(path: str) -> float | None:
    """Return the smallest bounding-box dimension of an STL/VTK mesh.

    This gives the maximum sensible slice thickness for the mesh — slicing
    along the smallest axis with a larger step would produce no slices.

    Args:
        path: Path to an ``.stl`` or ``.vtk`` mesh file.

    Returns:
        The smallest extent (mm) or ``None`` for unsupported formats.
    """
    ext = Path(path).suffix.lower()
    mesh = pv.read(str(path))

    if ext in (".stl", ".vtk"):
        # ensure we have polygonal surface
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()

        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        dims = [x_max - x_min, y_max - y_min, z_max - z_min]
        return min(dims)

    return None


def slice_at(mesh: pv.DataSet, Slice_direction: str, s: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute the slicing normal and origin for a PyVista mesh.

    Args:
        mesh: Any PyVista dataset (will be converted to PolyData if needed).
        Slice_direction: ``"X"``, ``"Y"``, or ``"Z"``.
        s: Position along the slice axis.

    Returns:
        Tuple of ``(normal, origin)`` ready for ``mesh.slice()``.
    """
    # ensure polygonal surface
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    c = mesh.center  # (cx, cy, cz)

    axis = {
        "X": ( (1.0, 0.0, 0.0), lambda s: (float(s), c[1], c[2]) ),
        "Y": ( (0.0, 1.0, 0.0), lambda s: (c[0], float(s), c[2]) ),
        "Z": ( (0.0, 0.0, 1.0), lambda s: (c[0], c[1], float(s)) ),
    }.get(Slice_direction)

    if axis is None:
        raise ValueError("Slice_direction must be 'X','Y','Z'.")

    normal, origin_fn = axis
    origin = origin_fn(s)
    return normal, origin

def make_scale_cube(Slice_direction: str, cube_len: float, origin, s: float, offset: float = 50.0) -> pv.PolyData:
    """Create a thin red reference cube for scale calibration in renders.

    The cube is placed next to the slice cross-section.  Its known side length
    (``cube_len``) is later detected in the screenshot to compute mm-per-pixel.

    Args:
        Slice_direction: ``"X"``, ``"Y"``, or ``"Z"``.
        cube_len: Side length of the cube face parallel to the slice plane.
        origin: Mesh center ``(x, y, z)``.
        s: Current slice position along the slice axis.
        offset: Translation away from the mesh centre to avoid overlap.

    Returns:
        A ``pv.PolyData`` cube positioned beside the slice.
    """
    offset = float(offset) + float(cube_len)
    thickness = max(float(cube_len) * 0.01, 1e-6)
    if Slice_direction == "X":
        c = (float(s), float(origin[1]) + offset, float(origin[2]) + offset)
        cube = pv.Cube(center=c, x_length=thickness, y_length=cube_len, z_length=cube_len)
    elif Slice_direction == "Y":
        c = (float(origin[0]) + offset, float(s), float(origin[2]) + offset)
        cube = pv.Cube(center=c, x_length=cube_len, y_length=thickness, z_length=cube_len)
    elif Slice_direction == "Z":
        c = (float(origin[0]) + offset, float(origin[1]) + offset, float(s))
        cube = pv.Cube(center=c, x_length=cube_len, y_length=cube_len, z_length=thickness)
    else:
        raise ValueError("Slice_direction must be 'X','Y','Z'.")

    return cube


def prepare_orthographic_slice_render(p, view_fn) -> None:
    """Apply reproducible camera settings before a PyVista slice screenshot."""
    view_fn()
    try:
        p.parallel_projection = True
    except Exception:
        pass
    try:
        p.enable_parallel_projection()
    except Exception:
        pass
    p.reset_camera()
    try:
        p.camera.zoom(0.95)
    except Exception:
        pass


def validate_scale_cube_sanity(
    p,
    scale_cube,
    cube_len_mm: float,
    view_fn,
    *,
    background: str,
) -> tuple[bool, dict]:
    """Render a reference-only calibration frame and validate detection."""
    p.clear()
    p.set_background(background)
    p.add_mesh(scale_cube, color="red", lighting=False)
    prepare_orthographic_slice_render(p, view_fn)
    img_rgb = p.screenshot(return_img=True)
    mm_per_px, metadata, _mask = calc_scale_with_metadata(img_rgb, cube_len_mm)
    if mm_per_px is None:
        return False, metadata
    detected_px = metadata.get("detected_cube_size_px")
    measured_cube_len = mm_per_px * float(detected_px)
    error_percent = abs(measured_cube_len - float(cube_len_mm)) / max(float(cube_len_mm), 1e-9) * 100.0
    metadata["calibration_error_percent"] = error_percent
    if error_percent > 1.0:
        metadata["calibration_status"] = f"failed: sanity error {error_percent:.3f}% exceeds 1%"
        return False, metadata
    metadata["calibration_status"] = "valid: sanity check passed"
    return True, metadata

def compactness_2D(area: float, perimeter: float) -> float:
    if perimeter == 0:
        return 0
    return (4 * 3.141592653589793 * area) / (perimeter ** 2)


def mask_perimeter_mm(
    mask,
    pixel_size_x,
    pixel_size_y,
    *,
    method="arc_length",
    simplify=False,
    epsilon=0.5,
):
    """Perimeter (mm) of the boundary of a binary tissue mask.

    ``method="arc_length"`` sums OpenCV contour lengths after scaling contour
    coordinates by the x/y pixel spacing. ``method="crofton"`` fills holes,
    resamples this 2-D mask to isotropic spacing, and runs scikit-image's
    4-direction Crofton estimator. If scikit-image is unavailable, Crofton
    falls back to the arc-length path.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0

    px_x = float(pixel_size_x)
    px_y = float(pixel_size_y)
    if not (np.isfinite(px_x) and np.isfinite(px_y) and px_x > 0 and px_y > 0):
        raise ValueError("pixel_size_x and pixel_size_y must be positive finite values")

    method = str(method or "arc_length").lower()
    binary = (arr > 0).astype(np.uint8)

    def _arc_length(mask_u8) -> float:
        contours, _hierarchy = cv2.findContours(
            mask_u8.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        total = 0.0
        for cnt in contours:
            work = cnt
            if simplify and float(epsilon) > 0 and len(work) >= 3:
                work = cv2.approxPolyDP(work, float(epsilon), True)
            cnt_mm = work.astype(np.float32) * np.array([px_x, px_y], dtype=np.float32)
            total += float(cv2.arcLength(cnt_mm, True))
        return total

    if method == "arc_length":
        return _arc_length(binary)
    if method != "crofton":
        raise ValueError(f"Unsupported perimeter method: {method!r}")

    try:
        from scipy.ndimage import binary_fill_holes, zoom
        from skimage.measure import perimeter_crofton
    except Exception as ex:
        logger.warning("Crofton perimeter unavailable; falling back to arc_length: %s", ex)
        return _arc_length(binary)

    filled = binary_fill_holes(binary.astype(bool))
    if not np.any(filled):
        return 0.0

    iso_px = min(px_x, px_y)
    scale_y = px_y / iso_px
    scale_x = px_x / iso_px
    iso_mask = zoom(
        filled.astype(np.uint8),
        zoom=(scale_y, scale_x),
        order=0,
        mode="nearest",
    ).astype(bool)
    return float(perimeter_crofton(iso_mask, directions=4) * iso_px)

def compactness_3D(volume: float, surface_area: float) -> float:
    if surface_area == 0:
        return 0
    comp = (36 * 3.141592653589793 * (volume ** 2)) / (surface_area ** 3)
    if comp > 1.0:
        print(f"[Compactness 3D] WARN: compactness {comp:.4f} > 1.0 — surface area is likely "
              f"underestimated relative to volume (check unit consistency).")
    return comp


def _simpson_axis_integral(samples) -> float:
    """``∫ y(h) dh`` over ``(position, value)`` samples via Simpson's rule.

    Sorts by position, merges duplicate positions (mean value), and returns
    ``0.0`` with fewer than two usable samples. A higher-order (≈ O(Δh⁴))
    quadrature than the ``Σ value × Δh`` rectangle sum; integrating the value
    directly means per-slice noise cannot inflate it (no frustum-style slant).
    """
    pts = [(float(h), float(v)) for (h, v) in samples
           if h is not None and v is not None
           and np.isfinite(h) and np.isfinite(v) and v >= 0]
    if len(pts) < 2:
        return 0.0
    pts.sort(key=lambda t: t[0])
    h = np.array([t[0] for t in pts], dtype=float)
    v = np.array([t[1] for t in pts], dtype=float)
    # Merge duplicate positions (mean value) so x is strictly increasing.
    uh, inv = np.unique(h, return_inverse=True)
    if uh.size < 2:
        return 0.0
    if uh.size != h.size:
        sums = np.zeros(uh.size)
        cnts = np.zeros(uh.size)
        np.add.at(sums, inv, v)
        np.add.at(cnts, inv, 1)
        v = sums / cnts
        h = uh
    try:
        from scipy.integrate import simpson
    except ImportError:  # SciPy < 1.6
        from scipy.integrate import simps as simpson
    return float(abs(simpson(v, x=h)))


def lateral_area_simpson(samples) -> float:
    """Lateral surface area = ``∫ perimeter(h) dh`` via Simpson's rule.

    ``samples`` is an iterable of ``(position, perimeter)`` in a consistent unit
    (result in that unit²). See :func:`_simpson_axis_integral`.
    """
    return _simpson_axis_integral(samples)


def volume_simpson(samples) -> float:
    """Volume = ``∫ cross_section_area(h) dh`` via Simpson's rule.

    ``samples`` is an iterable of ``(position, cross_section_area)`` in a
    consistent unit (result in that unit³). See :func:`_simpson_axis_integral`.
    """
    return _simpson_axis_integral(samples)


def total_surface_area_simpson(perimeter_samples, area_samples):
    """Total surface area = Simpson lateral ``∫ perimeter dh`` + end caps.

    The end caps are the cross-section areas at the first and last slice
    positions (the top + bottom faces of the stack). A single valid slice yields
    ``2 × area`` (a flat disk's two faces) and zero lateral.

    Args:
        perimeter_samples: iterable of ``(position, perimeter)``.
        area_samples: iterable of ``(position, cross_section_area)``.

    Returns:
        ``(total, lateral, caps)`` in the samples' unit².
    """
    lateral = lateral_area_simpson(perimeter_samples)
    pts = [(float(h), float(a)) for (h, a) in area_samples
           if h is not None and a is not None
           and np.isfinite(h) and np.isfinite(a) and a >= 0]
    if not pts:
        caps = 0.0
    else:
        pts.sort(key=lambda t: t[0])
        caps = 2.0 * pts[0][1] if len(pts) == 1 else (pts[0][1] + pts[-1][1])
    return lateral + caps, lateral, caps


def area_based_gi_3d(mesh, scale=None):
    """3-D area-based gyrification index = total mesh surface area ÷ convex-hull
    surface area, both measured on the same (optionally scaled) geometry.

    The convex hull is the smooth convex envelope of the mesh. For a closed,
    roughly star-convex surface (like a brain) the ratio is ≥ 1 — ≈ 1 when
    unfolded and larger when gyrified (a FreeSurfer-style 3-D GI proxy).
    Independent of the 2-D perimeter ``GI``. NOTE: for handle/hole topologies
    (e.g. a torus) the hull area can exceed the mesh area, giving a ratio < 1.

    Args:
        mesh: A PyVista mesh exposing ``points`` and ``extract_surface()``.
        scale: Optional per-axis ``(3,)`` physical scale (model → mm) applied to
            the points before measuring. Pass ``None`` for STL (already mm).

    Returns:
        ``(gi_3d, total_area, hull_area)``; any element may be ``None`` on
        failure (degenerate geometry, SciPy missing, etc.).
    """
    try:
        from scipy.spatial import ConvexHull
    except Exception as ex:  # pragma: no cover - SciPy is a hard dep elsewhere
        print(f"[Warning] area_based_gi_3d — SciPy unavailable: {ex}")
        return None, None, None
    try:
        scaled = mesh
        if scale is not None:
            scaled = mesh.copy(deep=True)
            scaled.points = np.asarray(scaled.points, dtype=float) * np.asarray(scale, dtype=float)
        surf = scaled.extract_surface().triangulate()
        total_area = float(surf.area)
        pts = np.asarray(scaled.points, dtype=float)
        if total_area <= 0 or pts.ndim != 2 or pts.shape[0] < 4:
            return None, (total_area if total_area > 0 else None), None
        hull_area = float(ConvexHull(pts).area)
        if hull_area <= 0:
            return None, total_area, None
        return total_area / hull_area, total_area, hull_area
    except Exception as ex:
        print(f"[Warning] area_based_gi_3d — {ex}")
        return None, None, None
