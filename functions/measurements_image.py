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
    _add_scalebar_on_annotated,
    draw_hallmarks_values_on_image,
    SULCUS_CLASS_COLORS,
    classify_sulcus_depth,
    empty_depth_sets,
    flatten_depth_sets,
    format_sulcus_class_summary,
)
from helpers.slice_kind_classifier import classify_slice_kind, SliceKind
from constants import (
    BINARY_THRESHOLD_DEFAULT,
    DEFECT_FIXED_POINT,
    MIN_PIXEL_SIZE_FOR_DEPTH_LABELS,
    SULCUS_PRIMARY_MAX_FRACTION,
    SULCUS_TERTIARY_MIN_FRACTION,
)


def compute_image_allmarks(
    file_path: str,
    pixel_size: float = 0.01,
    kernel_size: int = 5,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
) -> tuple[float, float, float, float, float, list, dict, np.ndarray, SliceKind]:
    """Compute all hallmarks (area, perimeters, GI, sulci depths) from an image.

    Pipeline:
        1. Threshold → extract inner contours (brain boundary).
        2. Morphological close → extract outer contours (sulci filled).
        3. GI = inner perimeter / outer perimeter.
        4. Convexity defects → sulci depths.

    Args:
        file_path: Path to a brain-slice image.
        pixel_size: Physical size of one pixel (mm/px).
        kernel_size: Diameter of the elliptical kernel for morph close.
        cnt_threshold: Minimum contour area (pixels) to keep.
        unit: Label for output units.
        add_scalebar: If True, overlay a new scale bar on the annotated output.
        draw_hallmarks: If True, draw hallmark numeric values on the image.

    Returns:
        Tuple of ``(area, perimeter, perimeter_convex, GI, compactness,
        depths, depth_sets, annotated_bgr, slice_kind)``.
    """
    margin = 6

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    print(file_path + " is processing")
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(im_bw, BINARY_THRESHOLD_DEFAULT, 255, 1)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > cnt_threshold]
   
    annotated = image.copy()
    W, H = annotated.shape[:2]
    thickness, font_scale, radius_px = image_annotation_style(H, W, style="thin")

    # If the input is a full MRI slice (sagittal/coronal/axial), depth defects
    # are filtered as a fraction of the slice's largest in-plane physical
    # extent (mirrors the 0.5%-25% rule used in measurements_nifti.py:198).
    # Cropped sub-slice bands fall back to the original fixed-millimeter rule.
    slice_kind, slice_kind_conf = classify_slice_kind(image)
    use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
    print(f"Classified slice kind: {slice_kind} (confidence {slice_kind_conf:.2f}), using {'percent' if use_percent_filter else 'fixed'} filter for sulci depth.")
    # Slice length = longest side of the brain's bounding box (physical units),
    # not the raw image size. Falls back to image extent if no brain contour was found.
    if filtered_contours:
        _bx, _by, _bw_px, _bh_px = cv2.boundingRect(np.vstack(filtered_contours))
        slice_length = max(_bw_px, _bh_px) * pixel_size
    else:
        slice_length = max(W, H) * pixel_size

    if use_percent_filter:
        print(f"The sulcus depth filter thresholds are {(SULCUS_TERTIARY_MIN_FRACTION * slice_length):.2f} {unit} (min) and {(SULCUS_PRIMARY_MAX_FRACTION * slice_length):.2f} {unit} (max).")

    if filtered_contours:
        cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)
        area_sum = sum(cv2.contourArea(cnt) for cnt in filtered_contours)
        perimeter_sum = sum(cv2.arcLength(cnt, True) for cnt in filtered_contours)
    else:
        area_sum = perimeter_sum = 0
        
    area = area_sum * pixel_size**2
    perimeter = perimeter_sum * pixel_size
    
    if filtered_contours and len(filtered_contours[0]) > 0:
        x1, y1 = filtered_contours[0][0][0]
    else:
        x1, y1 = 15, 40

    # --- Outer contour: morphological close fills sulci, giving the "convex" boundary.
    # GI (gyrification index) = inner perimeter / outer perimeter.
    # Rebuild the source mask from ONLY the kept inner contours so noise blobs
    # the inner filter rejected cannot produce spurious outer components after
    # morph-close.
    inner_mask_only = np.zeros_like(im_bw)
    cv2.drawContours(inner_mask_only, filtered_contours, -1, 255, thickness=cv2.FILLED)
    closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size))
    convex_Contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_conv_contours = [cnt_conv for cnt_conv in convex_Contours if cv2.contourArea(cnt_conv) > cnt_threshold]

    if filtered_conv_contours:
        annotated = cv2.drawContours(annotated, filtered_conv_contours, -1, (0, 255, 0), thickness)
        perimeter_convex_sum= sum( cv2.arcLength(convex_cnt, True) for convex_cnt in filtered_conv_contours)

    else:
        perimeter_convex_sum = 1  # fallback to avoid division by zero

    perimeter_convex = perimeter_convex_sum * pixel_size
    perimeter_Rate = perimeter / perimeter_convex  # GI ratio
    comp = compactness_2D(area, perimeter) 

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
                    annotated = cv2.line(annotated, start, end, [255, 0, 0], thickness)
                    # Convert fixed-point depth to physical units. Full MRI
                    # slices use a percent-of-slice-length window; cropped
                    # bands keep the original fixed-millimeter threshold.
                    depth_value = d * pixel_size / DEFECT_FIXED_POINT
                    if use_percent_filter:
                        keep = (SULCUS_TERTIARY_MIN_FRACTION * slice_length) < depth_value < (SULCUS_PRIMARY_MAX_FRACTION * slice_length)
                    else:
                        keep = depth_value > (0.5 if unit == "mm" else 0.05 if unit == "cm" else 0)

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
                            tx = min(max(0, int(far[0] + radius_px + 4)), max(0, W - 1))
                            ty = min(max(0, int(far[1] - radius_px - 4)), max(0, H - 1))
                            cv2.putText(
                                annotated,
                                label,
                                (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale * 0.75,
                                marker_color,
                                max(1, thickness - 1),
                                cv2.LINE_AA,
                            )

    depth = flatten_depth_sets(depth_sets)
    if use_percent_filter:
        print(format_sulcus_class_summary(depth_sets))

    if draw_hallmarks:
        annotated = draw_hallmarks_values_on_image(
            annotated,
            thickness=thickness,
            font_scale=font_scale,
            area=area,
            perimeter=perimeter,
            lgi=perimeter_Rate,
            compactness=comp,
            unit=unit,
            box_position="topleft",
            anchor_ratio=(0.02, 0.05),
        )
    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return area, perimeter, perimeter_convex ,perimeter_Rate, comp, depth, depth_sets, annotated, slice_kind  # BGR ndarray


def compute_image_perimeter(
    file_path: str,
    pixel_size: float = 0.01,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
) -> tuple[float, np.ndarray, SliceKind]:
    """
    Compute foreground perimeter from a 2D image by thresholding & contour filtering.
    Returns ``(perimeter, annotated_bgr, slice_kind)``.

    No files are written here.
    """
    margin = 6

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) > cnt_threshold]

    annotated = image.copy()
    W, H = annotated.shape[:2]
    thickness, font_scale, _ = image_annotation_style(H, W, style="regular")

    if filtered:
        cv2.drawContours(annotated, filtered, -1, (0, 0, 255), thickness)

    perimeter_sum = sum(cv2.arcLength(cnt, True) for cnt in filtered)
    perimeter = perimeter_sum * pixel_size

    if draw_hallmarks:
        annotated = draw_hallmarks_values_on_image(
            annotated,
            thickness=thickness,
            font_scale=font_scale,
            perimeter=perimeter,
            lgi=None,
            unit=unit,
            box_position="topleft",
            anchor_ratio=(0.02, 0.05),
        )
    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return perimeter, annotated, slice_kind  # BGR ndarray


def compute_image_area(
    file_path: str,
    pixel_size: float = 0.01,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
) -> tuple[float, np.ndarray, SliceKind]:
    """
    Compute foreground area from a 2D image by thresholding & contour filtering.
    Returns ``(area, annotated_bgr, slice_kind)``.

    No files are written here.
    """
    margin = 6

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) > cnt_threshold]

    annotated = image.copy()
    W, H = annotated.shape[:2]
    thickness, _, _ = image_annotation_style(H, W, style="regular")

    if filtered:
        cv2.drawContours(annotated, filtered, -1, (0, 0, 255), thickness)

    px_area_sum = float(sum(cv2.contourArea(c) for c in filtered))
    area_units2 = px_area_sum * (pixel_size ** 2)

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
    kernel_size: int = 5,
    cnt_threshold: float = 20,
    unit: str = "mm",
    add_scalebar: bool | None = True,
    draw_hallmarks: bool = True,
)  -> tuple[float, float, float, np.ndarray, SliceKind]:
    """Compute the local Gyrification Index from a 2-D brain-slice image.

    GI = inner perimeter / outer perimeter, where "outer" is derived by
    morphologically closing the binary mask (fills sulci).

    Args:
        file_path: Path to the image.
        pixel_size: mm per pixel.
        kernel_size: Morph-close kernel diameter.
        cnt_threshold: Minimum contour area to keep (pixels).
        unit: Label for output units.

    Returns:
        Tuple of ``(GI_ratio, inner_perim_mm, outer_perim_mm,
        annotated_bgr, slice_kind)``.
    """
    margin =6

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    print(file_path + " is processing")
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(im_bw, BINARY_THRESHOLD_DEFAULT, 255, 1)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > cnt_threshold]
   
    annotated = image.copy()
    W, H = annotated.shape[:2]
    thickness, _, _ = image_annotation_style(H, W, style="regular")
    
    if filtered_contours:
        cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in filtered_contours)
    else:
        perimeter = 0

    # Outer contour: rebuild a mask from ONLY the kept inner contours so noise
    # rejected by the inner area filter cannot produce spurious outer components
    # after morph-close.
    inner_mask_only = np.zeros_like(im_bw)
    cv2.drawContours(inner_mask_only, filtered_contours, -1, 255, thickness=cv2.FILLED)
    closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size))
    convex_Contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_conv_contours = [cnt_conv for cnt_conv in convex_Contours if cv2.contourArea(cnt_conv) > cnt_threshold]
    
    if filtered_conv_contours:
        annotated = cv2.drawContours(annotated, filtered_conv_contours, -1, (0, 255, 0), thickness)
        perimeter_convex= sum( cv2.arcLength(convex_cnt, True) for convex_cnt in filtered_conv_contours)
    
    else:
        perimeter_convex = 1
        

    perimeter_Rate = perimeter / perimeter_convex
    perimeter_u = perimeter * pixel_size
    perimeter_convex_u = perimeter_convex * pixel_size
    

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
    return perimeter_Rate, perimeter_u, perimeter_convex_u, annotated, slice_kind  # BGR ndarray

        
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

    print(file_path + " is processing")
    im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(im_bw, BINARY_THRESHOLD_DEFAULT, 255, 1)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > cnt_threshold]
   
    annotated = image.copy()
    W, H = annotated.shape[:2]
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

    print(f"Classified slice kind: {slice_kind} (confidence {slice_kind_conf:.2f}), using {'percent' if use_percent_filter else 'fixed'} filter for sulci depth.")  
    if use_percent_filter:
        print(f"The sulcus depth filter thresholds are {(SULCUS_TERTIARY_MIN_FRACTION * slice_length):.2f} {unit} (min) and {(SULCUS_PRIMARY_MAX_FRACTION * slice_length):.2f} {unit} (max).")

    if filtered_contours:
        cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)


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
                    annotated = cv2.line(annotated, start, end, [255, 0, 0], thickness)
                    # Convert fixed-point depth to physical units. Full MRI
                    # slices use a percent-of-slice-length window; cropped
                    # bands keep the original fixed-millimeter threshold.
                    depth_value = d * pixel_size / DEFECT_FIXED_POINT
                    if use_percent_filter:
                        keep = (SULCUS_TERTIARY_MIN_FRACTION * slice_length) < depth_value < (SULCUS_PRIMARY_MAX_FRACTION * slice_length)
                    else:
                        keep = depth_value > (0.5 if unit == "mm" else 0.05 if unit == "cm" else 0)
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
                            tx = min(max(0, int(far[0] + radius_px + 4)), max(0, W - 1))
                            ty = min(max(0, int(far[1] - radius_px - 4)), max(0, H - 1))
                            cv2.putText(
                                annotated,
                                label,
                                (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale * 0.75,
                                marker_color,
                                max(1, thickness - 1),
                                cv2.LINE_AA,
                            )

    depth = flatten_depth_sets(depth_sets)
    if use_percent_filter:
        print(format_sulcus_class_summary(depth_sets))

    annotated = _add_scalebar_on_annotated(annotated, pixel_size, unit, add_scalebar)
    return depth, depth_sets, annotated, slice_kind  # BGR ndarray


def compute_compactness_2D(file_path: str, cnt_threshold: float = 20.0) -> tuple[float, np.ndarray, SliceKind]:
    margin = 6
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image: {file_path}")

    slice_kind, _ = classify_slice_kind(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) > cnt_threshold]

    annotated = image.copy()
    W, H = annotated.shape[:2]
    thickness, _, _ = image_annotation_style(H, W, style="regular")
    
    if filtered:
        cv2.drawContours(annotated, filtered, -1, (0, 0, 255), thickness)

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
        return None

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
