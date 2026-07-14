"""2-D image measurement functions for FetoMorph.

Operates on single brain-slice images (PNG / JPEG).  Each function
thresholds the image, extracts contours, and computes one or more
morphometric quantities (area, perimeter, GI, sulci depth).

All pixel → physical-unit conversions use ``pixel_size`` (mm/px).
"""

from __future__ import annotations

from deps import *
from helpers.helpers import (
    image_annotation_style,
    compute_kernel_convex,
    compactness_2D,
    mask_perimeter_mm,
    split_inner_and_internal_contours,
    threshold_binary,
    border_is_white,
    segment_pial_mask,
    _add_scalebar_on_annotated,
    draw_hallmarks_values_on_image,
    SULCUS_CLASS_COLORS,
    classify_sulcus_depth,
    sulcus_depth_min,
    empty_depth_sets,
    flatten_depth_sets,
    format_sulcus_class_summary,
)
from helpers.slice_kind_classifier import classify_slice_kind, SliceKind
from managers.visualization_settings import get_active as _get_viz
from constants import (
    BINARY_THRESHOLD_DEFAULT,
    DEFAULT_KERNEL_SIZE_MM,
    DEFAULT_PERIMETER_METHOD,
    DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    DEFECT_FIXED_POINT,
    MIN_PIXEL_SIZE_FOR_DEPTH_LABELS,
    SULCUS_PRIMARY_MAX_FRACTION,
    SULCUS_TERTIARY_MIN_FRACTION,
)


def _to_kernel_px(kernel_size_mm: float, pixel_size_mm: float) -> int:
    px = max(3, int(round(float(kernel_size_mm) / max(float(pixel_size_mm), 1e-9))))
    return px


def _binarise_brain(image: np.ndarray, gray_code: int = cv2.COLOR_RGB2GRAY) -> np.ndarray:
    """Binary brain mask (uint8, 255 = brain) with cropped-section handling.

    White-background colour label-map crops (e.g. ``Examples/cropped_slices``)
    are segmented with :func:`helpers.helpers.segment_pial_mask`: bright labels
    such as cyan are kept (not dropped by a brightness cut) and the inner white
    cortex ribbon is sealed into the ROI, so the folded boundary / LGI is not
    inflated. Dark-background full-slice renders keep the legacy Otsu inverse
    threshold, so full-slice behaviour is unchanged.
    """
    if border_is_white(image):
        return segment_pial_mask(image)
    gray = cv2.cvtColor(image, gray_code)
    return threshold_binary(gray, BINARY_THRESHOLD_DEFAULT, invert=True)


def _split_contours_for_mode(bw, cnt_threshold: float, pixel_size: float):
    """Detect brain-boundary and internal-hole contours for Contour Accounting.

    Uses ``RETR_CCOMP`` + :func:`split_inner_and_internal_contours` (the same helper
    the mesh paths use) to return ``(inner_filtered, internal_filtered)``:
    ``inner_filtered`` are the outer (brain-boundary) contours, ``internal_filtered``
    are the holes inside them. ``cnt_threshold`` is in mm²; the helper filters in
    px², so it is converted with ``pixel_size`` (mm/px).
    """
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return [], []
    px = float(pixel_size)
    min_area_px = float(cnt_threshold) / (px ** 2) if px > 0 else float(cnt_threshold)
    return split_inner_and_internal_contours(
        contours, hierarchy, np.zeros_like(bw), bw.shape, min_area_px)


def compute_image_allmarks(
    file_path: str,
    pixel_size: float = 0.01,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    contour_mode: str = "outer",
) -> tuple[float, float, float | None, float, float, float, list, dict, np.ndarray, SliceKind]:
    """Compute all hallmarks (area, perimeters, GI, sulci depths) from an image.

    Pipeline:
        1. Threshold → extract inner contours (brain boundary).
        2. Morphological close → extract outer contours (sulci filled).
        3. GI = exterior perimeter / closed-envelope perimeter.
        4. Convexity defects → sulci depths.

    Args:
        file_path: Path to a brain-slice image.
        pixel_size: Physical size of one pixel (mm/px).
        kernel_size_mm: Diameter of the elliptical kernel for morph close, in mm.
        cnt_threshold: Minimum contour area (pixels) to keep.
        unit: Label for output units.
        add_scalebar: If True, overlay a new scale bar on the annotated output.
        draw_hallmarks: If True, draw hallmark numeric values on the image.

    ``contour_mode`` (Contour Accounting) adjusts the reported **area**
    (``outer`` / ``subtract`` = brain minus holes / ``internal_only``) and, in
    ``subtract`` mode, also returns the interior-hole boundary as
    ``perimeter_internal``. GI and compactness stay on the brain boundary.

    Returns:
        Tuple of ``(area, perimeter, perimeter_internal, perimeter_outer_envelope, GI,
        compactness, depths, depth_sets, annotated_bgr, slice_kind)``.
    """
    margin = 6
    kernel_size_px = _to_kernel_px(kernel_size_mm, pixel_size)

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    print(f"[Image] {file_path} is processing")
    im_bw = _binarise_brain(image, cv2.COLOR_RGB2GRAY)
    # Brain-boundary (inner) + hole (internal) contours for Contour Accounting.
    inner_filtered, internal_filtered = _split_contours_for_mode(im_bw, cnt_threshold, pixel_size)
    filtered_contours = inner_filtered  # sulci defects + GI operate on the brain boundary

    annotated = image.copy()
    H, W = annotated.shape[:2]
    thickness, font_scale, radius_px = image_annotation_style(H, W, style="thin")

    # If the input is a full MRI slice (sagittal/coronal/axial), depth defects
    # are filtered as a fraction of the slice's largest in-plane physical
    # extent (mirrors the 0.5%-25% rule used in measurements_nifti.py:198).
    # Cropped sub-slice bands fall back to the original fixed-millimeter rule.
    slice_kind, slice_kind_conf = classify_slice_kind(image)
    use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
    print(f"[Image] Classified slice kind: {slice_kind} (confidence {slice_kind_conf:.2f}), using {'percent' if use_percent_filter else 'fixed'} filter for sulci depth.")
    # Slice length = longest side of the brain's bounding box (physical units),
    # not the raw image size. Falls back to image extent if no brain contour was found.
    if filtered_contours:
        _bx, _by, _bw_px, _bh_px = cv2.boundingRect(np.vstack(filtered_contours))
        slice_length = max(_bw_px, _bh_px) * pixel_size
    else:
        slice_length = max(W, H) * pixel_size

    if use_percent_filter:
        print(f"[Image] The sulcus depth filter thresholds are {(SULCUS_TERTIARY_MIN_FRACTION * slice_length):.2f} {unit} (min) and {(SULCUS_PRIMARY_MAX_FRACTION * slice_length):.2f} {unit} (max).")

    if filtered_contours:
        cv2.drawContours(annotated, filtered_contours, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)
    if internal_filtered and contour_mode != "outer":
        cv2.drawContours(annotated, internal_filtered, -1, tuple(_get_viz().contour_internal_color_bgr), thickness)

    # Area honours the Contour Accounting mode.
    inner_area_px = sum(cv2.contourArea(cnt) for cnt in inner_filtered)
    internal_area_px = sum(cv2.contourArea(cnt) for cnt in internal_filtered)
    if contour_mode == "subtract":
        area_sum = inner_area_px - internal_area_px
    elif contour_mode == "internal_only":
        area_sum = internal_area_px
    else:  # "outer"
        area_sum = inner_area_px

    area = area_sum * pixel_size**2
    inner_mask_only = np.zeros_like(im_bw)
    cv2.drawContours(inner_mask_only, filtered_contours, -1, 255, thickness=cv2.FILLED)
    # Exterior perimeter (brain boundary). GI/compactness use this.
    perimeter = mask_perimeter_mm(
        inner_mask_only, pixel_size, pixel_size,
        method=perimeter_method,
        simplify=simplify_contours_for_perimeter,
        epsilon=contour_simplify_epsilon,
    )
    # Interior (holes) perimeter — reported as a second value in subtract mode.
    if contour_mode == "subtract" and internal_filtered:
        internal_mask_only = np.zeros_like(im_bw)
        cv2.drawContours(internal_mask_only, internal_filtered, -1, 255, thickness=cv2.FILLED)
        perimeter_internal = mask_perimeter_mm(
            internal_mask_only, pixel_size, pixel_size,
            method=perimeter_method,
            simplify=simplify_contours_for_perimeter,
            epsilon=contour_simplify_epsilon,
        )
    else:
        perimeter_internal = None

    if filtered_contours and len(filtered_contours[0]) > 0:
        x1, y1 = filtered_contours[0][0][0]
    else:
        x1, y1 = 15, 40

    # --- Outer contour: morphological close fills sulci, giving the "convex" boundary.
    # GI (gyrification index) = exterior perimeter / closed-envelope perimeter.
    # Rebuild the source mask from ONLY the kept inner contours so noise blobs
    # the inner filter rejected cannot produce spurious outer components after
    # morph-close.
    closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size_px))
    convex_Contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_conv_contours = [cnt_conv for cnt_conv in convex_Contours if cv2.contourArea(cnt_conv) * (pixel_size ** 2) > cnt_threshold]

    if filtered_conv_contours:
        annotated = cv2.drawContours(annotated, filtered_conv_contours, -1, tuple(_get_viz().contour_outer_color_bgr), thickness)
        outer_mask_only = np.zeros_like(closed_mask)
        cv2.drawContours(outer_mask_only, filtered_conv_contours, -1, 255, thickness=cv2.FILLED)
        perimeter_outer_envelope = mask_perimeter_mm(
            outer_mask_only, pixel_size, pixel_size,
            method=perimeter_method,
            simplify=simplify_contours_for_perimeter,
            epsilon=contour_simplify_epsilon,
        )

    else:
        perimeter_outer_envelope = None

    perimeter_Rate = (perimeter / perimeter_outer_envelope
                      if perimeter_outer_envelope else None)  # GI ratio
    comp = (compactness_2D(area, perimeter)
            if perimeter_outer_envelope is not None else None)

    # --- Sulci depth via convexity defects ---
    # When use_percent_filter is True (full MRI slices), depths are split into
    # four named sets (primary / secondary / tertiary / unclassified). The flat
    # `depth` list returned to the dispatcher is the union of all four.
    depth_sets = empty_depth_sets()
    for cnt in filtered_contours:
        hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
        if hull is not None and len(hull) >= 3 and len(cnt) > 3 and np.all(np.diff(hull.ravel()) > 0):
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    # s = start index, e = end index, f = farthest point index
                    # d = depth in OpenCV 8.8 fixed-point (divide by 256 → pixels)
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    annotated = cv2.line(annotated, start, end, list(_get_viz().measurement_line_color_bgr), thickness)
                    # Convert fixed-point depth to physical units. Full MRI
                    # slices use a percent-of-slice-length window; cropped
                    # bands keep the original fixed-millimeter threshold.
                    depth_value = d * pixel_size / DEFECT_FIXED_POINT
                    if use_percent_filter:
                        keep = (SULCUS_TERTIARY_MIN_FRACTION * slice_length) < depth_value < (SULCUS_PRIMARY_MAX_FRACTION * slice_length) and depth_value > sulcus_depth_min(unit)
                    else:
                        keep = depth_value > sulcus_depth_min(unit)

                    # Classify by % of slice_length (full MRI slices only).
                    if keep:
                        if use_percent_filter:
                            sulcus_class = classify_sulcus_depth(depth_value, slice_length)
                        else:
                            sulcus_class = "unclassified"

                        marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                        depth_sets[sulcus_class].append(depth_value)
                        annotated = cv2.circle(annotated, far, radius_px, marker_color, -1)
                        # Skip numeric labels when pixel_size is small (dense
                        # images would otherwise be cluttered by overlapping
                        # text); the marker circle is still drawn.
                        if pixel_size <= MIN_PIXEL_SIZE_FOR_DEPTH_LABELS:
                            label = f"{depth_value:.2f} {unit}"
                            label_scale = font_scale * float(_get_viz().sulcus_label_scale_multiplier)
                            label_thickness = max(1, thickness - 1)
                            (label_w, label_h), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness
                            )
                            # Place the label on the same side of the image as the
                            # sulcus: right half -> text to the right of the marker,
                            # left half -> text to the left (grows outward).
                            if far[0] >= W / 2:
                                tx = int(far[0] + radius_px + 4)
                            else:
                                tx = int(far[0] - radius_px - 4 - label_w)
                            ty = int(far[1] - radius_px - 4)
                            # Clamp so the whole text box stays inside the image.
                            tx = min(max(0, tx), max(0, W - label_w))
                            ty = min(max(label_h, ty), max(label_h, H - 1))
                            cv2.putText(
                                annotated,
                                label,
                                (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                label_scale,
                                marker_color,
                                label_thickness,
                                cv2.LINE_AA,
                            )

    depth = flatten_depth_sets(depth_sets)
    if use_percent_filter:
        print(f"[Image] {format_sulcus_class_summary(depth_sets)}")

    if draw_hallmarks:
        annotated = draw_hallmarks_values_on_image(
            annotated,
            thickness=thickness,
            font_scale=font_scale,
            area=area,
            perimeter=perimeter,
            perimeter_internal=perimeter_internal,
            lgi=perimeter_Rate,
            compactness=comp,
            unit=unit,
            box_position="topleft",
            anchor_ratio=(0.02, 0.05),
        )
    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return area, perimeter, perimeter_internal, perimeter_outer_envelope ,perimeter_Rate, comp, depth, depth_sets, annotated, slice_kind  # BGR ndarray


def compute_image_perimeter(
    file_path: str,
    pixel_size: float = 0.01,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    contour_mode: str = "outer",
) -> tuple[float, float | None, np.ndarray, SliceKind]:
    """
    Compute foreground perimeter from a 2D image by thresholding & contour filtering.
    Returns ``(perimeter, perimeter_internal, annotated_bgr, slice_kind)``.

    ``contour_mode`` (Contour Accounting) selects which boundaries are reported:
    ``outer`` → exterior (brain boundary) only, ``perimeter_internal`` is ``None``;
    ``subtract`` → exterior as ``perimeter`` **and** the interior (holes) boundary
    as ``perimeter_internal``; ``internal_only`` → interior only (as ``perimeter``).

    No files are written here.
    """
    margin = 6

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    bw = _binarise_brain(image, cv2.COLOR_BGR2GRAY)

    inner_filtered, internal_filtered = _split_contours_for_mode(bw, cnt_threshold, pixel_size)

    annotated = image.copy()
    H, W = annotated.shape[:2]
    thickness, font_scale, _ = image_annotation_style(H, W, style="regular")

    if inner_filtered and contour_mode != "internal_only":
        cv2.drawContours(annotated, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)
    if internal_filtered and contour_mode != "outer":
        cv2.drawContours(annotated, internal_filtered, -1, tuple(_get_viz().contour_internal_color_bgr), thickness)

    def _perim(cnts):
        if not cnts:
            return 0.0
        m = np.zeros_like(bw)
        cv2.drawContours(m, cnts, -1, 255, thickness=cv2.FILLED)
        return mask_perimeter_mm(
            m, pixel_size, pixel_size,
            method=perimeter_method,
            simplify=simplify_contours_for_perimeter,
            epsilon=contour_simplify_epsilon,
        )

    ext_perim = _perim(inner_filtered)
    int_perim = _perim(internal_filtered)
    if contour_mode == "internal_only":
        perimeter, perimeter_internal = int_perim, None
    elif contour_mode == "subtract":
        perimeter, perimeter_internal = ext_perim, int_perim
    else:  # "outer"
        perimeter, perimeter_internal = ext_perim, None

    if draw_hallmarks:
        annotated = draw_hallmarks_values_on_image(
            annotated,
            thickness=thickness,
            font_scale=font_scale,
            perimeter=perimeter,
            perimeter_internal=perimeter_internal,
            lgi=None,
            unit=unit,
            box_position="topleft",
            anchor_ratio=(0.02, 0.05),
        )
    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return perimeter, perimeter_internal, annotated, slice_kind  # BGR ndarray


def compute_image_area(
    file_path: str,
    pixel_size: float = 0.01,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
    contour_mode: str = "outer",
) -> tuple[float, np.ndarray, SliceKind]:
    """
    Compute foreground area from a 2D image by thresholding & contour filtering.
    Returns ``(area, annotated_bgr, slice_kind)``.

    ``contour_mode`` (Contour Accounting): ``outer`` = brain-boundary area,
    ``subtract`` = brain area minus internal holes, ``internal_only`` = only the
    internal-hole area.

    No files are written here.
    """
    margin = 6

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    bw = _binarise_brain(image, cv2.COLOR_BGR2GRAY)

    inner_filtered, internal_filtered = _split_contours_for_mode(bw, cnt_threshold, pixel_size)

    annotated = image.copy()
    H, W = annotated.shape[:2]
    thickness, _, _ = image_annotation_style(H, W, style="regular")

    if inner_filtered and contour_mode != "internal_only":
        cv2.drawContours(annotated, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)
    if internal_filtered and contour_mode != "outer":
        cv2.drawContours(annotated, internal_filtered, -1, tuple(_get_viz().contour_internal_color_bgr), thickness)

    inner_area_px = float(sum(cv2.contourArea(c) for c in inner_filtered))
    internal_area_px = float(sum(cv2.contourArea(c) for c in internal_filtered))
    if contour_mode == "subtract":
        px_area_sum = inner_area_px - internal_area_px
    elif contour_mode == "internal_only":
        px_area_sum = internal_area_px
    else:  # "outer"
        px_area_sum = inner_area_px
    area_units2 = px_area_sum * (pixel_size ** 2)

    filtered = inner_filtered
    if filtered and len(filtered[0]) > 0:
        x1, y1 = filtered[0][0][0]
    else:
        x1, y1 = 15, 40


    if draw_hallmarks:
        annotated = draw_hallmarks_values_on_image(
            annotated,
            thickness=thickness,
            area=area_units2,
            lgi=None,
            unit=unit,
            box_position="topleft",
            anchor_ratio=(0.02, 0.05),
        )
    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return area_units2, annotated, slice_kind  # BGR ndarray

def compute_image_lGI(
    file_path: str,
    pixel_size: float,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
)  -> tuple[float, float, float, np.ndarray, SliceKind]:
    """Compute the local Gyrification Index from a 2-D brain-slice image.

    GI = exterior perimeter / closed-envelope perimeter, where the "closed-envelope" is derived by
    morphologically closing the binary mask (fills sulci).

    Args:
        file_path: Path to the image.
        pixel_size: mm per pixel.
        kernel_size_mm: Morph-close kernel diameter, in mm.
        cnt_threshold: Minimum contour area to keep (pixels).
        unit: Label for output units.

    Returns:
        Tuple of ``(GI_ratio, inner_perim_mm, outer_perim_mm,
        annotated_bgr, slice_kind)``.
    """
    margin =6
    kernel_size_px = _to_kernel_px(kernel_size_mm, pixel_size)

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    print(f"[Image] {file_path} is processing")
    im_bw = _binarise_brain(image, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) * (pixel_size ** 2) > cnt_threshold]
   
    annotated = image.copy()
    H, W = annotated.shape[:2]
    thickness, _, _ = image_annotation_style(H, W, style="regular")
    
    if filtered_contours:
        cv2.drawContours(annotated, filtered_contours, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)
    else:
        perimeter = 0

    # Outer contour: rebuild a mask from ONLY the kept inner contours so noise
    # rejected by the inner area filter cannot produce spurious outer components
    # after morph-close.
    inner_mask_only = np.zeros_like(im_bw)
    cv2.drawContours(inner_mask_only, filtered_contours, -1, 255, thickness=cv2.FILLED)
    perimeter_u = mask_perimeter_mm(
        inner_mask_only, pixel_size, pixel_size,
        method=perimeter_method,
        simplify=simplify_contours_for_perimeter,
        epsilon=contour_simplify_epsilon,
    )
    closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size_px))
    convex_Contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_conv_contours = [cnt_conv for cnt_conv in convex_Contours if cv2.contourArea(cnt_conv) * (pixel_size ** 2) > cnt_threshold]
    
    if filtered_conv_contours:
        annotated = cv2.drawContours(annotated, filtered_conv_contours, -1, tuple(_get_viz().contour_outer_color_bgr), thickness)
        outer_mask_only = np.zeros_like(closed_mask)
        cv2.drawContours(outer_mask_only, filtered_conv_contours, -1, 255, thickness=cv2.FILLED)
        perimeter_outer_envelope_u = mask_perimeter_mm(
            outer_mask_only, pixel_size, pixel_size,
            method=perimeter_method,
            simplify=simplify_contours_for_perimeter,
            epsilon=contour_simplify_epsilon,
        )
    
    else:
        perimeter_outer_envelope_u = None
        

    perimeter_Rate = (perimeter_u / perimeter_outer_envelope_u
                      if perimeter_outer_envelope_u else None)
    

    if filtered_contours and len(filtered_contours[0]) > 0:
        x1, y1 = filtered_contours[0][0][0]
    else:
        x1, y1 = 15, 40


#    text_lgi =   f"lGI:{perimeter_Rate:.2f}"
#    (tw, th), baseline = cv2.getTextSize(text_lgi, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#    bw, bh = tw + 2*margin, th + baseline + 2*margin  # box size
#    
#    inside = (0 <= x1 <= max(0, W - bw)) and (0 <= y1 <= max(0, H - bh))
#    x1 = min(max(0, x1), max(0, W - bw))
#    y1 = min(max(0, y1), max(0, H - bh))
#    cv2.putText(
#        annotated,
#        text_lgi,
#        (int(x1 + margin), int(y1 + margin + th)),
#        cv2.FONT_HERSHEY_SIMPLEX,
#        font_scale,
#        (255, 0, 200),
#        thickness,
#        cv2.LINE_AA,
#    )
    
    if draw_hallmarks:
        annotated = draw_hallmarks_values_on_image(
            annotated,
            thickness=thickness,
            lgi=perimeter_Rate,
            unit=unit,
            box_position="topleft",
            anchor_ratio=(0.02, 0.05),
        )
    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return perimeter_Rate, perimeter_u, perimeter_outer_envelope_u, annotated, slice_kind  # BGR ndarray

        
def compute_image_sulci_depth(
    file_path: str,
    pixel_size: float,
    cnt_threshold: float,
    unit: str = "mm",
    add_scalebar: bool | None = True,
) -> tuple[list, dict, np.ndarray, SliceKind]:
    """Compute sulci depths from convexity defects on a 2-D brain-slice image.

    For each contour, computes the convex hull and then identifies
    convexity defects (indentations).  Each defect's depth is converted
    from OpenCV 8.8 fixed-point to mm using ``pixel_size``.

    Args:
        file_path: Path to the image.
        pixel_size: mm per pixel.
        cnt_threshold: Minimum contour area to keep (pixels).
        unit: Label for output units.

    Returns:
        Tuple of ``(depth_list_mm, depth_sets, annotated_bgr, slice_kind)``.
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    print(f"[Image] {file_path} is processing")
    im_bw = _binarise_brain(image, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) * (pixel_size ** 2) > cnt_threshold]
   
    annotated = image.copy()
    H, W = annotated.shape[:2]
    thickness, font_scale, radius_px = image_annotation_style(H, W, style="bold")

    # See compute_image_allmarks: full MRI slices use a percent window;
    # cropped sub-slice bands keep the fixed-millimeter rule.
    slice_kind, slice_kind_conf = classify_slice_kind(image)
    use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
    # Slice length = longest side of the brain's bounding box (physical units),
    # not the raw image size. Falls back to image extent if no brain contour was found.
    if filtered_contours:
        _bx, _by, _bw_px, _bh_px = cv2.boundingRect(np.vstack(filtered_contours))
        slice_length = max(_bw_px, _bh_px) * pixel_size
    else:
        slice_length = max(W, H) * pixel_size

    print(f"[Image] Classified slice kind: {slice_kind} (confidence {slice_kind_conf:.2f}), using {'percent' if use_percent_filter else 'fixed'} filter for sulci depth.")  
    if use_percent_filter:
        print(f"[Image] The sulcus depth filter thresholds are {(SULCUS_TERTIARY_MIN_FRACTION * slice_length):.2f} {unit} (min) and {(SULCUS_PRIMARY_MAX_FRACTION * slice_length):.2f} {unit} (max).")

    if filtered_contours:
        cv2.drawContours(annotated, filtered_contours, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)


    # When use_percent_filter is True (full MRI slices), depths are split into
    # four named sets (primary / secondary / tertiary / unclassified). The flat
    # `depth` list returned to the dispatcher is the union of all four.
    depth_sets = empty_depth_sets()
    for cnt in filtered_contours:
        hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
        if hull is not None and len(hull) >= 3 and len(cnt) > 3 and np.all(np.diff(hull.ravel()) > 0):
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    # s = start index, e = end index, f = farthest point
                    # d = depth in 8.8 fixed-point (d / 256 → pixels)
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    annotated = cv2.line(annotated, start, end, list(_get_viz().measurement_line_color_bgr), thickness)
                    # Convert fixed-point depth to physical units. Full MRI
                    # slices use a percent-of-slice-length window; cropped
                    # bands keep the original fixed-millimeter threshold.
                    depth_value = d * pixel_size / DEFECT_FIXED_POINT
                    if use_percent_filter:
                        keep = (SULCUS_TERTIARY_MIN_FRACTION * slice_length) < depth_value < (SULCUS_PRIMARY_MAX_FRACTION * slice_length) and depth_value > sulcus_depth_min(unit)
                    else:
                        keep = depth_value > sulcus_depth_min(unit)
                    if keep:
                        # Classify by % of slice_length (full MRI slices only).
                        if use_percent_filter:
                            sulcus_class = classify_sulcus_depth(depth_value, slice_length)
                        else:
                            sulcus_class = "unclassified"
                        marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                        depth_sets[sulcus_class].append(depth_value)
                        annotated = cv2.circle(annotated, far, radius_px, marker_color, -1)
                        # Skip numeric labels when pixel_size is small (dense
                        # images would otherwise be cluttered by overlapping
                        # text); the marker circle is still drawn.
                        if pixel_size <= MIN_PIXEL_SIZE_FOR_DEPTH_LABELS:
                            label = f"{depth_value:.2f} {unit}"
                            label_scale = font_scale * float(_get_viz().sulcus_label_scale_multiplier)
                            label_thickness = max(1, thickness - 1)
                            (label_w, label_h), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness
                            )
                            # Place the label on the same side of the image as the
                            # sulcus: right half -> text to the right of the marker,
                            # left half -> text to the left (grows outward).
                            if far[0] >= W / 2:
                                tx = int(far[0] + radius_px + 4)
                            else:
                                tx = int(far[0] - radius_px - 4 - label_w)
                            ty = int(far[1] - radius_px - 4)
                            # Clamp so the whole text box stays inside the image.
                            tx = min(max(0, tx), max(0, W - label_w))
                            ty = min(max(label_h, ty), max(label_h, H - 1))
                            cv2.putText(
                                annotated,
                                label,
                                (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                label_scale,
                                marker_color,
                                label_thickness,
                                cv2.LINE_AA,
                            )

    depth = flatten_depth_sets(depth_sets)
    if use_percent_filter:
        print(f"[Image] {format_sulcus_class_summary(depth_sets)}")

    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return depth, depth_sets, annotated, slice_kind  # BGR ndarray


def compute_compactness_2D(file_path: str, cnt_threshold: float = 20.0, pixel_size: float = 0.01) -> tuple[float, np.ndarray, SliceKind]:
    margin = 6
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    bw = _binarise_brain(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) * (pixel_size ** 2) > cnt_threshold]

    annotated = image.copy()
    H, W = annotated.shape[:2]
    thickness, _, _ = image_annotation_style(H, W, style="regular")
    
    if filtered:
        cv2.drawContours(annotated, filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

    perimeter = sum(cv2.arcLength(cnt, True) for cnt in filtered)
    area = float(sum(cv2.contourArea(c) for c in filtered))
    compactness_2D_value = compactness_2D(area, perimeter)

    if filtered and len(filtered[0]) > 0:
        x1, y1 = filtered[0][0][0]
    else:
        x1, y1 = 15, 40
        
    annotated = draw_hallmarks_values_on_image(
        annotated,
        thickness=thickness,
        compactness=compactness_2D_value,
        box_position="topleft",
        anchor_ratio=(0.02, 0.05),
    )

    return compactness_2D_value, annotated, slice_kind  # BGR ndarray

def compute_image_curved_length(
    file_path: str,
    pixel_size: float = 0.01,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
    curvature_window: int = 7,
    straightness_threshold: float = 0.02,
) -> tuple[float, np.ndarray, SliceKind]:
    """Measure only the longest curved segment of the brain contour.

    Uses the same thresholding and contour extraction as
    ``compute_image_perimeter``, then classifies each contour point as
    *curved* or *straight* using local curvature over a sliding window.
    Consecutive curved points form segments; the longest one is reported.

    Args:
        file_path: Path to the image.
        pixel_size: mm per pixel.
        cnt_threshold: Minimum contour area to keep (pixels).
        unit: Label for output units.
        add_scalebar: If True, overlay a scale bar.
        draw_hallmarks: If True, draw the curved-length value on the image.
        curvature_window: Half-window size for local curvature estimation.
        straightness_threshold: Curvature below this value (1/px) is
            considered straight.

    Returns:
        Tuple of ``(curved_length, annotated_bgr, slice_kind)``.
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    bw = _binarise_brain(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) * (pixel_size ** 2) > cnt_threshold]

    annotated = image.copy()
    H, W = annotated.shape[:2]
    thickness, _, _ = image_annotation_style(H, W, style="regular")

    # if filtered:
    #     cv2.drawContours(annotated, filtered, -1, (0, 0, 255), thickness)

    best_curved_length_px = 0.0
    best_curved_pts: list[np.ndarray] = []

    for cnt in filtered:
        pts = cnt[:, 0, :]  # (N, 2)
        n = len(pts)
        if n < 2 * curvature_window + 1:
            continue

        is_curved = np.zeros(n, dtype=bool)
        for i in range(n):
            p_prev = pts[(i - curvature_window) % n].astype(float)
            p_curr = pts[i].astype(float)
            p_next = pts[(i + curvature_window) % n].astype(float)

            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 < 1e-6 or len2 < 1e-6:
                continue
            cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            curvature = angle / max(len1, len2)
            if curvature > straightness_threshold:
                is_curved[i] = True

        # Only exclude straight segments longer than a quarter of the
        # longest straight segment in this contour.
        straight_runs: list[tuple[int, int]] = []
        s_start = None
        for i in range(n):
            if not is_curved[i] and s_start is None:
                s_start = i
            elif is_curved[i] and s_start is not None:
                straight_runs.append((s_start, i))
                s_start = None
        if s_start is not None:
            straight_runs.append((s_start, n))

        straight_lengths = []
        for sr_start, sr_end in straight_runs:
            p0 = pts[sr_start % n].astype(float)
            p1 = pts[(sr_end - 1) % n].astype(float)
            straight_lengths.append(np.linalg.norm(p1 - p0))

        max_straight = max(straight_lengths) if straight_lengths else 0.0
        min_straight_len_px = max_straight / 4.0

        for idx_r, (sr_start, sr_end) in enumerate(straight_runs):
            if straight_lengths[idx_r] < min_straight_len_px:
                for i in range(sr_start, sr_end):
                    is_curved[i % n] = True

        # Find connected runs of curved points on the original (non-doubled)
        # array, then merge runs separated by small straight gaps.
        runs: list[tuple[int, int]] = []
        start = None
        for i in range(n):
            if is_curved[i] and start is None:
                start = i
            elif not is_curved[i] and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, n))

        # Wrap-around: if both the first and last runs touch the boundary,
        # merge them into one run that spans the wrap point.
        if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == n:
            wrap_run = (runs[-1][0], runs[0][1] + n)
            runs = runs[1:-1]
            runs.append(wrap_run)

        # Merge runs separated by small straight gaps (≤ merge_gap points).
        # Repeat until no more merges occur so that any number of nearby
        # curves get joined into a single continuous segment.
        merge_gap = 2
        changed = True
        while changed and len(runs) > 1:
            changed = False
            merged: list[tuple[int, int]] = [runs[0]]
            for rs, re in runs[1:]:
                prev_start, prev_end = merged[-1]
                gap = (rs - prev_end) % n
                if gap <= merge_gap:
                    merged[-1] = (prev_start, prev_end + gap + (re - rs))
                    changed = True
                else:
                    merged.append((rs, re))
            if len(merged) >= 2:
                gap = (merged[0][0] - merged[-1][1] % n) % n
                if gap <= merge_gap:
                    merged[-1] = (merged[-1][0], merged[-1][1] + gap + (merged[0][1] - merged[0][0]))
                    merged = merged[1:]
                    changed = True
            runs = merged

        for run_start, run_end in runs:
            run_len = run_end - run_start
            if run_len > n:
                run_len = n
            seg_length_px = 0.0
            seg_pts = []
            last_curved_idx = None
            for j in range(run_len):
                idx = (run_start + j) % n
                if is_curved[idx]:
                    seg_pts.append(pts[idx])
                    if last_curved_idx is not None:
                        seg_length_px += np.linalg.norm(
                            pts[idx].astype(float) - pts[last_curved_idx].astype(float)
                        )
                    last_curved_idx = idx
            if seg_length_px > best_curved_length_px:
                best_curved_length_px = seg_length_px
                best_curved_pts = seg_pts

    curved_length = best_curved_length_px * pixel_size

    if best_curved_pts:
        curve_arr = np.array(best_curved_pts, dtype=np.int32)
        cv2.polylines(annotated, [curve_arr], isClosed=False, color=(255, 0, 255), thickness=thickness )

    if draw_hallmarks:
        annotated = draw_hallmarks_values_on_image(
            annotated,
            thickness=thickness,
            curve=curved_length,
            unit=unit,
            box_position="topleft",
            anchor_ratio=(0.02, 0.05),
        )
    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return curved_length, annotated, slice_kind


def put_label_on_bgr(
    bgr: np.ndarray,
    text: str,
    pos: str | tuple[int, int] = "topleft",  # 'topleft'|'topright'|'bottomleft'|'bottomright' or (x, y)
    *,
    font_scale: float | None = None,     # auto if None
    thickness: int | None = None,        # auto if None
    margin: int = 6,
    box_color: tuple[int, int, int] = (0, 0, 0),      # BGR
    box_alpha: float = 0.55,                           # 0..1
    text_color: tuple[int, int, int] = (255, 255, 255) # BGR
) -> np.ndarray:
    """
    Draw `text` with a translucent background box on a BGR image and return the result (BGR).
    """
    if not (isinstance(bgr, np.ndarray) and bgr.ndim == 3 and bgr.shape[2] == 3 and bgr.dtype == np.uint8):
        raise ValueError("Expected uint8 BGR image of shape (H, W, 3).")

    out = bgr.copy()
    H, W = out.shape[:2]
    if not text:
        return out

    # Auto size to image height (looks good across sizes)
    if font_scale is None:
        font_scale = max(0.45, H / 800.0 * 0.9)
    if thickness is None:
        thickness = max(1, int(round(H / 400.0)))

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    bw, bh = tw + 2*margin, th + baseline + 2*margin  # box size

    # Anchor selection
    if isinstance(pos, tuple):
        x, y = int(pos[0]), int(pos[1])
    else:
        pos = str(pos).lower()
        if pos == "bottomleft":
            x, y = margin, H - bh - margin
        elif pos == "bottomright":
            x, y = W - bw - margin, H - bh - margin
        elif pos == "topright":
            x, y = W - bw - margin, margin
        else:  # "topleft"
            x, y = margin, margin

    # Clamp box fully inside image
    x = max(0, min(x, W - bw))
    y = max(0, min(y, H - bh))

    # Translucent background box
    overlay = out.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), box_color, -1)
    cv2.addWeighted(overlay, float(box_alpha), out, 1.0 - float(box_alpha), 0, out)

    # Text baseline inside the box
    tx, ty = x + margin, y + margin + th
    cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
    return out
