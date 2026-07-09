"""Batch morphometric measurement of brain slice images.

Processes a directory of 2-D slice images, extracts contours,
computes area, perimeter, local Gyrification Index (LGI), and
convexity-defect depths, then writes annotated PNGs and an Excel
summary.
"""

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
    SULCUS_CLASS_COLORS,
    SULCUS_CLASSES,
    classify_sulcus_depth,
    empty_depth_sets,
    flatten_depth_sets,
    format_sulcus_class_summary,
    sulcus_export_columns,
    sulcus_export_cells,
    pad_row,
    drop_empty_columns,
)
from helpers.slice_kind_classifier import classify_slice_kind
from constants import (
    BINARY_THRESHOLD_DEFAULT,
    DEFAULT_KERNEL_SIZE_MM,
    DEFAULT_PERIMETER_METHOD,
    DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    DEFECT_FIXED_POINT,
    SULCUS_TERTIARY_MIN_FRACTION,
    SULCUS_PRIMARY_MAX_FRACTION,
)

def _to_kernel_px(kernel_size_mm: float, pixel_size_mm: float) -> int:
    px = max(3, int(round(float(kernel_size_mm) / max(float(pixel_size_mm), 1e-9))))
    return px


def _binarise_brain(image: np.ndarray, gray_code: int = cv2.COLOR_RGB2GRAY) -> np.ndarray:
    """Binary brain mask (uint8, 255 = brain) with cropped-section handling.

    White-background colour label-map crops (e.g. ``Examples/cropped_slices``)
    are segmented with :func:`helpers.helpers.segment_pial_mask`: bright labels
    such as cyan are kept and the inner white cortex ribbon is sealed into the
    ROI, so the folded boundary / LGI is not inflated. Dark-background full-slice
    renders keep the legacy Otsu inverse threshold, so full-slice behaviour is
    unchanged.
    """
    if border_is_white(image):
        return segment_pial_mask(image)
    gray = cv2.cvtColor(image, gray_code)
    return threshold_binary(gray, BINARY_THRESHOLD_DEFAULT, invert=True)


def process_on_images_batch(directory_path,
    out_dir,
    pixel_size: float = 0.01,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    cnt_threshold: float = 20,
    unit: str = "mm",
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    contour_mode: str = "outer"):
    """Run morphometric analysis on every image in a directory.

    For each image the function binarises it, finds external contours,
    computes area and perimeter in physical units, derives a convex-hull
    perimeter for LGI, and records convexity-defect depths.  Annotated
    images are saved and all measurements are collected into an Excel
    spreadsheet.

    Args:
        directory_path: Path to the directory containing input slice images.
        out_dir: Root output directory; an ``image_Batch`` sub-folder is
            created inside it for annotated PNGs.
        pixel_size: Physical size of one pixel in the chosen unit.
            Defaults to 0.01.
        kernel_size_mm: Morphological closing kernel diameter in mm.
        cnt_threshold: Minimum contour area (in pixels) to keep a contour.
            May be increased automatically if the LGI ratio is below 1.
            Defaults to 20.
        unit: Physical unit label written to the Excel output.
            Defaults to "mm".

    Returns:
        A tuple of (valid_slices, saved_pngs) where *valid_slices* is a
        list of integer indices of successfully processed images and
        *saved_pngs* is a list of file paths to the annotated PNG files.
    """
    
    kernel_size_px = _to_kernel_px(kernel_size_mm, pixel_size)
    sheet1 = []
    valid_slices = []
    saved_pngs = []
    total_depth: list = []
    slice_class_data: list = []
    lgi_per_image: list[float] = []
    compactness_per_image: list[float] = []
    count = 0
    out_dir_Batch = os.path.join(out_dir, "image_Batch")
    os.makedirs(out_dir_Batch, exist_ok=True)
    print(f"[Process Batch] Temp output dir: {out_dir}")
    print(
        f"[Process Batch] Perimeter method={perimeter_method}; "
        f"spacing={pixel_size:g} x {pixel_size:g} {unit}/px; "
        f"simplify={'on' if simplify_contours_for_perimeter else 'off'} "
        f"(epsilon={float(contour_simplify_epsilon):g} px)"
    )

    margin = 6

    # List image files in the directory.
    image_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    file_names = sorted(
        [n for n in os.listdir(directory_path) if n.lower().endswith(image_exts)]
    )
    for idx, file_name in enumerate(file_names):
        # Build image file path
        file_path = os.path.join(directory_path, file_name)
        if not os.path.isfile(file_path):
            continue
        # Open image and stop the whole batch on failure.
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")

        print(f"[Process Batch] {file_name} is processing")
        im_bw = _binarise_brain(image, cv2.COLOR_RGB2GRAY)
        # RETR_CCOMP + split so Contour Accounting can subtract internal holes
        # (min_area=0 here; the per-image local_threshold filter is applied in the
        # retry loop below to keep the existing LGI-parity behaviour).
        _ccnts, _chier = cv2.findContours(im_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        inner_all, internal_all = split_inner_and_internal_contours(
            _ccnts, _chier, np.zeros_like(im_bw), im_bw.shape, 0.0)

        # Per-image threshold: the auto-retry below may bump this locally when
        # the LGI ratio drops below 1, but must NOT leak into the next image —
        # always start each file from the user-supplied cnt_threshold.
        local_threshold = cnt_threshold

        annotated = image.copy()
        W, H = annotated.shape[:2]
        thickness, font_scale, radius_px = image_annotation_style(H, W, style="bold")

        # Classify slice kind (sagittal/coronal/axial vs cropped sub-slice).
        # Full MRI slices use a percent-of-slice-length window for the depth
        # filter and per-defect classification; cropped bands fall back to the
        # original fixed-millimeter rule and remain "unclassified".
        slice_kind, slice_kind_conf = classify_slice_kind(image)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        print(f"[Process Batch] {file_name}: slice_kind={slice_kind} (conf {slice_kind_conf:.2f}), "
              f"using {'percent' if use_percent_filter else 'fixed'} filter for sulci depth.")

        kernel = compute_kernel_convex(kernel_size_px)

        # Joint inner/outer filtering loop: both contour sets are filtered with
        # the SAME local_threshold every iteration, so when the retry bumps the
        # threshold to restore LGI >= 1, inner and outer stay at parity.
        # The outer source mask is rebuilt from ONLY the kept inner contours so
        # noise blobs the inner filter rejected cannot produce spurious outer
        # components after morph-close.
        max_steps = 1000
        steps = 0
        filtered_contours = []
        filtered_conv_contours = []
        perimeter_outer_envelope_sum = None
        area = 0.0
        perimeter = 0.0
        perimeter_internal = None
        perimeter_outer_envelope = None
        perimeter_rate = None
        compactness = None
        while True:
            steps += 1
            if steps > max_steps:
                break  # safety

            filtered_contours = [cnt for cnt in inner_all if cv2.contourArea(cnt) * (pixel_size ** 2) > local_threshold]
            filtered_internal = [cnt for cnt in internal_all if cv2.contourArea(cnt) * (pixel_size ** 2) > local_threshold]

            inner_mask_only = np.zeros_like(im_bw)
            cv2.drawContours(inner_mask_only, filtered_contours, -1, 255, thickness=cv2.FILLED)
            closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, kernel)
            convex_Contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_conv_contours = [
                cnt for cnt in convex_Contours
                if cv2.contourArea(cnt) * (pixel_size ** 2) > local_threshold
            ]

            # Area honours Contour Accounting; perimeter (exterior, brain
            # boundary) drives LGI/compactness and is unchanged.
            inner_area_px = sum(cv2.contourArea(cnt) for cnt in filtered_contours) if filtered_contours else 0
            internal_area_px = sum(cv2.contourArea(cnt) for cnt in filtered_internal) if filtered_internal else 0
            if contour_mode == "subtract":
                area_sum = inner_area_px - internal_area_px
            elif contour_mode == "internal_only":
                area_sum = internal_area_px
            else:  # "outer"
                area_sum = inner_area_px
            area = area_sum * pixel_size**2
            perimeter = mask_perimeter_mm(
                inner_mask_only, pixel_size, pixel_size,
                method=perimeter_method,
                simplify=simplify_contours_for_perimeter,
                epsilon=contour_simplify_epsilon,
            )
            # Interior (holes) perimeter — reported as a second value in subtract mode.
            if contour_mode == "subtract" and filtered_internal:
                internal_mask_only = np.zeros_like(im_bw)
                cv2.drawContours(internal_mask_only, filtered_internal, -1, 255, thickness=cv2.FILLED)
                perimeter_internal = mask_perimeter_mm(
                    internal_mask_only, pixel_size, pixel_size,
                    method=perimeter_method,
                    simplify=simplify_contours_for_perimeter,
                    epsilon=contour_simplify_epsilon,
                )
            else:
                perimeter_internal = None

            if not filtered_conv_contours:
                perimeter_outer_envelope_sum = None
                break

            outer_mask_only = np.zeros_like(closed_mask)
            cv2.drawContours(outer_mask_only, filtered_conv_contours, -1, 255, thickness=cv2.FILLED)
            perimeter_outer_envelope = mask_perimeter_mm(
                outer_mask_only, pixel_size, pixel_size,
                method=perimeter_method,
                simplify=simplify_contours_for_perimeter,
                epsilon=contour_simplify_epsilon,
            )
            perimeter_rate = perimeter / perimeter_outer_envelope if perimeter_outer_envelope else None
            compactness = compactness_2D(area, perimeter)
            if perimeter_rate < 1:
                local_threshold += 500 * (pixel_size ** 2)
                continue  # retry with higher threshold (both inner and outer)
            break  # condition satisfied

        # Slice length = longest side of the brain's bounding box (physical units),
        # not the raw image size. Falls back to image extent if no brain contour was found.
        if filtered_contours:
            _bx, _by, _bw_px, _bh_px = cv2.boundingRect(np.vstack(filtered_contours))
            slice_length = max(_bw_px, _bh_px) * pixel_size
        else:
            slice_length = max(W, H) * pixel_size

        if filtered_contours:
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)
        if filtered_internal and contour_mode != "outer":
            cv2.drawContours(annotated, filtered_internal, -1, (0, 255, 255), thickness)
        annotated = cv2.drawContours(annotated, filtered_conv_contours, -1, (0, 255, 0), thickness)
            
        # Sulci classification: bin each kept defect by its depth as a
        # fraction of slice_length when the image is a full MRI slice.
        depth_sets = empty_depth_sets()
        for cnt in filtered_contours:
            hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
            if hull is not None and len(hull) >= 3 and len(cnt) > 3 and np.all(np.diff(hull.ravel()) > 0):
                defects = cv2.convexityDefects(cnt, hull)

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        annotated = cv2.line(annotated, start, end, [255, 0, 0], thickness)
                        depth_value = d * pixel_size / DEFECT_FIXED_POINT
                        if use_percent_filter:
                            keep = (SULCUS_TERTIARY_MIN_FRACTION * slice_length) < depth_value < (SULCUS_PRIMARY_MAX_FRACTION * slice_length)
                        else:
                            keep = depth_value > (0.5 if unit == "mm" else 0.05 if unit == "cm" else 0)
                        if keep:
                            if use_percent_filter:
                                sulcus_class = classify_sulcus_depth(depth_value, slice_length)
                            else:
                                sulcus_class = "unclassified"
                            marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                            depth_sets[sulcus_class].append(depth_value)
                            annotated = cv2.circle(annotated, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        if use_percent_filter:
            print(f"[Process Batch] {file_name}: {format_sulcus_class_summary(depth_sets)}")

        new_name= f"{file_name}_measured.png"
        new_path = os.path.join(out_dir_Batch, new_name)
        cv2.imwrite(new_path, annotated)

        valid_slices.append(idx)
        saved_pngs.append(new_path)
        count +=1

        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)

        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns(unit))
            slice_class_data.append(None)

        if perimeter_outer_envelope is not None and perimeter_outer_envelope > 0:
            lgi_per_image.append(perimeter_rate)
        if compactness is not None:
            compactness_per_image.append(compactness)

        # Per-image row carries the values the spec-layout writer
        # reads (area / perimeters / per-class depth sets).
        sheet1.append({
            "file_name": file_name,
            "area": area,
            "perimeter": perimeter,
            "perimeter_internal": perimeter_internal,
            "perimeter_outer_envelope": perimeter_outer_envelope,
            "lgi": (perimeter_rate
                    if perimeter_outer_envelope is not None and perimeter_outer_envelope > 0
                    else None),
            "compactness": compactness,
            # Keep the depth sets in BOTH modes: the fixed-filter (cropped
            # sub-slice) path stores its kept sulci under "unclassified", so
            # passing {} here dropped the Unclassified count/mean-depth columns
            # from the workbook (drop_empty_columns saw only zeros).
            "depth_sets": depth_sets,
            "png_path": new_path,
        })

    # Write the per-image table + parameters + aggregates using the
    # shared spec layout (Results / Parameters / Mean results / Totals /
    # footer). Section cells become internal links to embedded
    # annotated-image tabs so Excel does not raise its external-link
    # security prompt.
    try:
        from helpers.results_excel_format import (
            ResultsSheet, write_results_workbook, subtype_mean,
        )

        # Per-class individual sulcus depths: one column per sulcus, deepest
        # first (depth_sets were sorted descending by flatten_depth_sets in the
        # loop). The number of columns for a class is the largest count of that
        # class across the folder's images; rows with fewer leave the trailing
        # cells blank, and drop_empty_columns removes any class with none.
        max_counts = {k: 0 for k in SULCUS_CLASSES}
        for r in sheet1:
            d = r["depth_sets"] or {}
            for k in SULCUS_CLASSES:
                max_counts[k] = max(max_counts[k], len(d.get(k, []) or []))
        # Each individual sulcus depth column is followed by its normalized
        # value (depth / max sulcus depth of that image) in the adjacent cell.
        depth_value_columns: list[str] = [
            col
            for k in SULCUS_CLASSES
            for i in range(max_counts[k])
            for col in (f"{k.capitalize()}_depth_{i + 1}",
                        f"{k.capitalize()}_depth_{i + 1}_norm")
        ]

        # Four per-image aggregate depth columns computed across ALL sulci
        # classes (primary/secondary/tertiary/unclassified) of that image:
        # the mean, max, and min sulcus depth, plus the normalized mean depth
        # = mean / max (normalized to the deepest sulcus). Empty when the image
        # has no detected sulci (drop_empty_columns then removes them); also
        # empty when max == 0 (undefined denominator).
        agg_depth_columns = [
            f"MeanSulciDepth ({unit})",
            f"MaxSulciDepth ({unit})",
            f"MinSulciDepth ({unit})",
            "NormalizedDepth",
        ]

        results_rows = []
        for r in sheet1:
            d = r["depth_sets"] or {}
            all_depths = [v for k in SULCUS_CLASSES for v in (d.get(k, []) or [])]
            if all_depths:
                _mean_d = sum(all_depths) / len(all_depths)
                _max_d = max(all_depths)
                _min_d = min(all_depths)
                _norm_d = (_mean_d / _max_d if _max_d else None)
            else:
                _mean_d = _max_d = _min_d = _norm_d = None
            row = {
                "Section": r["file_name"],
                "Area": r["area"],
                "Perimeter": r["perimeter"],
                "Perimeter_interior": r["perimeter_internal"],
                "LGI": r["lgi"],
                "Compactness": r["compactness"],
                "PrimarySulciCount": len(d.get("primary", []) or []),
                "SecondarySulciCount": len(d.get("secondary", []) or []),
                "TertiarySulciCount": len(d.get("tertiary", []) or []),
                "UnclassifiedSulciCount": len(d.get("unclassified", []) or []),
                "PrimaryMeanDepth": subtype_mean(
                    None, d.get("primary", []) or []),
                "SecondaryMeanDepth": subtype_mean(
                    None, d.get("secondary", []) or []),
                "TertiaryMeanDepth": subtype_mean(
                    None, d.get("tertiary", []) or []),
                "UnclassifiedMeanDepth": subtype_mean(
                    None, d.get("unclassified", []) or []),
                f"MeanSulciDepth ({unit})": _mean_d,
                f"MaxSulciDepth ({unit})": _max_d,
                f"MinSulciDepth ({unit})": _min_d,
                "NormalizedDepth": _norm_d,
                "_section_link": r["png_path"],
            }
            # Every individual sulcus value per class (deepest first), each
            # followed by its value normalized to the image's max sulcus depth.
            for k in SULCUS_CLASSES:
                vals = d.get(k, []) or []
                for i in range(max_counts[k]):
                    v = vals[i] if i < len(vals) else None
                    row[f"{k.capitalize()}_depth_{i + 1}"] = v
                    row[f"{k.capitalize()}_depth_{i + 1}_norm"] = (
                        v / _max_d if (v is not None and _max_d) else None)
            results_rows.append(row)

        parameters = {
            "Kernel size (mm)": float(kernel_size_mm),
            "Kernel size (px)": int(kernel_size_px),
            "Pixel spacing": f"{pixel_size} {unit}/pixel",
            "Filtered threshold (mm²)": float(cnt_threshold),
            "Length unit": unit,
            "Perimeter method": perimeter_method,
            "Isotropic spacing used": perimeter_method == "crofton",
            "Contour simplification enabled": bool(simplify_contours_for_perimeter),
            "Contour simplification epsilon": float(contour_simplify_epsilon),
            "Contour mode": contour_mode,
        }
        totals: dict = {}
        if lgi_per_image:
            totals["LGI (mean across images)"] = round(
                sum(lgi_per_image) / len(lgi_per_image), 4)
        if compactness_per_image:
            totals["Compactness (mean across images)"] = round(
                sum(compactness_per_image) / len(compactness_per_image), 4)
        overall_n = len(total_depth)
        if overall_n:
            totals["Total sulci count"] = int(overall_n)
            totals[f"Mean sulci depth ({unit})"] = round(
                sum(total_depth) / overall_n, 4)

        sheet = ResultsSheet(
            sheet_name=os.path.basename(directory_path) or "Batch",
            file_name=os.path.basename(directory_path.rstrip(os.sep))
                or "(batch)",
            folder=os.path.dirname(directory_path.rstrip(os.sep)) or None,
            parameters=parameters,
            rows=results_rows,
            totals=totals if totals else None,
            trailing_columns=tuple(agg_depth_columns + depth_value_columns),
            drop_empty_columns=True,
        )
        xlsx_path = os.path.join(out_dir, "Batch_Allmarks.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[Process Batch] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[Process Batch] WARN: could not save Excel: {ex}")

    print(f"[Process Batch] {count} images in {directory_path} have been processed")
    return valid_slices, saved_pngs
    
