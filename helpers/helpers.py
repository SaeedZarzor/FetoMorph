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
        lines.append(f"Perimeter: {perimeter:.2f} {unit}")
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
    
def calc_scale(image_rgb: np.ndarray, cube_length: float) -> float | None:
    """
    Compute mm-per-pixel from a red reference cube drawn in the render.
    cube_length_mm: the real cube side length (x_length) in mm.
    """
    red_rect = np.where((image_rgb[:, :, 0] > 150) & (image_rgb[:, :, 1] < 50), 255, 0).astype("uint8")
    _, thresh_red = cv2.threshold(red_rect, 150, 255, 0)
    contours, _ = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[Scale] No red reference contour found; default scale 1.0 mm/px")
        return 1.0
    # Use the largest red blob as reference
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if float(w)>0:
        scale = (cube_length/float(w))
    else:
        print("[STL Scale] Cup length error: cup length not found!")
        return None
    return scale
    
def get_red_rect_offset(image_rgb: np.ndarray) -> np.ndarray:  # noqa: returning shape (2,)
    """
    Detect red rectangle and return its center (x,y) in pixels — used to zero-align contours.
    """
    red_mask = (image_rgb[:, :, 0] > 150) & (image_rgb[:, :, 1] < 50)
    coords = np.argwhere(red_mask)
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
    # choose thin axis and translation vector
    if Slice_direction == "X":
        c = (s, origin[1], origin[2])
        cube = pv.Cube(center = c, x_length=0.01, y_length=cube_len, z_length=cube_len)
        cube.translate((-0.05, offset, offset), inplace=True)
    elif Slice_direction == "Y":
        c = (origin[0],s, origin[2])
        cube = pv.Cube(center = c, x_length=cube_len, y_length=0.01, z_length=cube_len)
        cube.translate((offset,-0.05 , offset), inplace=True)
    elif Slice_direction == "Z":
        c = (origin[0], origin[1], s)
        cube = pv.Cube(center = c, x_length=cube_len, y_length=cube_len, z_length=0.01)
        cube.translate((offset, offset, -0.05), inplace=True)
    else:
        raise ValueError("Slice_direction must be 'X','Y','Z'.")

    return cube

def compactness_2D(area: float, perimeter: float) -> float:
    if perimeter == 0:
        return 0
    return (4 * 3.141592653589793 * area) / (perimeter ** 2)

def compactness_3D(volume: float, surface_area: float) -> float:
    if surface_area == 0:
        return 0
    return (36 * 3.141592653589793 * (volume ** 2)) / (surface_area ** 3)
