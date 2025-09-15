# save as scale_bar_tools.py (requires: pip install opencv-python numpy)

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# ---------- Background cleanup (preserve colored regions) ----------

def clean_background_keep_colored(
    img_bgr: np.ndarray,
    s_thresh: int = 60,
    v_thresh: int = 40,
    unify_color: Optional[Tuple[int, int, int]] = None  # (B, G, R)
) -> np.ndarray:
    """
    Return a copy of `img_bgr` where non-colored pixels become white.
    Optionally paint all colored pixels the same color.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]

    # "Colored" = sufficiently high saturation & not too dark.
    colored_mask = ((S > s_thresh) & (V > v_thresh))

    out = np.full_like(img_bgr, 255)  # white background
    out[colored_mask] = img_bgr[colored_mask]

    if unify_color is not None:
        out[colored_mask] = np.array(unify_color, dtype=np.uint8)

    return out


# ---------- Scale-bar detection & measurement ----------

def detect_scale_bar_length(img_bgr: np.ndarray) -> tuple[Optional[int], Optional[Tuple[int,int,int,int]]]:
    """
    Detect the original scale bar and return (length_in_pixels, bounding_box),
    or (None, None) if not found.
    Heuristics: near-white, bottom region, long & thin, right-biased.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]

    # near-white pixels
    white = ((S < 30) & (V > 200)).astype(np.uint8) * 255

    # restrict to bottom 40% of the image
    roi = np.zeros_like(white); roi[int(0.60 * h):, :] = 255
    cand = cv2.bitwise_and(white, roi)

    # remove specks, connect bar & digits
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), 2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if num <= 1:
        return None, None

    best = None
    best_score = -1.0
    for i in range(1, num):
        x, y, cw, ch, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        aspect = cw / max(ch, 1)

        # positional preferences: bottom/right
        right_bias = (x + cw / 2) > 0.55 * w
        bottom_bias = y > 0.55 * h

        # score balances shape, size, and location
        score = (aspect if aspect > 1 else 0) + 0.00001 * area
        if right_bias:  score += 0.2
        if bottom_bias: score += 0.2

        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return None, None

    x, y, cw, ch = best
    return int(cw), (x, y, cw, ch)


# ---------- Draw a new scale bar ----------

def draw_new_scale_bar(
    img_bgr: np.ndarray,
    length_px: int,
    *,
    where: str | Tuple[int, int] = "bottom_right",
    color: Tuple[int, int, int] = (0, 0, 0),  # black in BGR
    thickness_ratio: float = 0.007,
    margin_ratio: float = 0.08,
    text: Optional[str] = None,
    font_scale_ratio: float = 0.9,
    font_thickness: int = 2
) -> np.ndarray:
    """
    Draw a crisp horizontal bar with `length_px` pixels.
      - where='bottom_right' or (x_left, y_bottom) for custom position.
      - thickness & margins scale with image size.
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    base = min(h, w)
    thickness = max(1, int(thickness_ratio * base))
    margin = int(margin_ratio * base)

    if isinstance(where, str) and where == "bottom_right":
        x2 = w - margin
        x1 = x2 - length_px
        y1 = h - margin - thickness
        y2 = h - margin
    else:
        x1, y2 = where  # custom top-left baseline
        x2 = x1 + length_px
        y1 = y2 - thickness

    # clamp to image bounds
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=-1)

    if text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # roughly tie font size to bar thickness
        fscale = (thickness / 10.0) * font_scale_ratio + 0.3
        tx = max(5, x1)
        ty = min(h - 5, y2 + int(5 * thickness))
        cv2.putText(out, text, (tx, ty), font, fscale, color, font_thickness, cv2.LINE_AA)

    return out


# ---------- High-level helper ----------

def nifti_slice_to_image(
    in_path: str,
    out_path: str,
    *,
    unify_color: Optional[Tuple[int, int, int]] = None,    # (B,G,R), e.g., (255,0,0)
#    new_bar_color: Tuple[int, int, int] = None,
    label_text: Optional[str] = None,
    scale_bar:bool = True,
#    match_fraction_of_width: Optional[float] = None
    smooth: Optional[str] = "median",   # "gaussian", "median", "bilateral"
    smooth_strength: int = 5        # kernel size or strength parameter
) -> int:
    """
    1) Read the image (resolution preserved).
    2) Remove gray/black background -> white; optionally unify colored region.
    3) Measure the existing scale bar length (or set as a fraction of width).
    4) Draw a new scale bar of that length at the bottom-right with optional text.

    Returns the pixel length used for the new bar.
    """
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(in_path)
    
    cleaned = clean_background_keep_colored(img, unify_color=unify_color)
    
    length_px, _ = detect_scale_bar_length(img)
#    if match_fraction_of_width is not None:
#        length_px = int(img.shape[1] * float(match_fraction_of_width))
#    if not length_px or length_px <= 0:
#        # sensible fallback if detection fails
#        length_px = int(img.shape[1] * 0.10)

    # --- Apply optional smoothing ---
    if smooth:
        if smooth == "gaussian":
            # kernel size must be odd
            k = smooth_strength if smooth_strength % 2 == 1 else smooth_strength + 1
            cleaned = cv2.GaussianBlur(cleaned, (k, k), 0)
        elif smooth == "median":
            k = smooth_strength if smooth_strength % 2 == 1 else smooth_strength + 1
            cleaned = cv2.medianBlur(cleaned, k)
        elif smooth == "bilateral":
            # (diameter, sigmaColor, sigmaSpace)
            cleaned = cv2.bilateralFilter(cleaned, smooth_strength, 75, 75)
    
    if scale_bar:
        result = draw_new_scale_bar(
            cleaned,
            length_px,
            where="bottom_right",
#            color=(new_bar_color),
            text=label_text
        )
       
    

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, result)
    return length_px

#
#if __name__ == "__main__":
#    # Example: keep original colors, draw a black scale bar, add "25 mm"
#    length = nifti_slice_to_image(
#        in_path="seg.nii_view.png",
#        out_path="seg.nii_view_with_new_bar.png",
#        unify_color=None,              # or e.g. (255, 0, 0) to make regions pure red (BGR)
#        new_bar_color=(0, 0, 0),       # black bar on white background
#        label_text="25 mm",            # optional
#        match_fraction_of_width=None   # or e.g. 0.12 to force 12% of width
#    )
#    print("New scale bar length (px):", length)
