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
    DEFECT_FIXED_POINT,
    SULCUS_TERTIARY_MIN_FRACTION,
    SULCUS_PRIMARY_MAX_FRACTION,
)

def process_on_images_batch(directory_path,
    out_dir,
    pixel_size: float = 0.01,
    kernel_size: int = 5,
    cnt_threshold: float = 20,
    unit: str = "mm"):
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
        kernel_size: Size of the morphological closing kernel used when
            computing the convex contour. Defaults to 5.
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
    
    sheet1 = []
    valid_slices = []
    saved_pngs = []
    total_depth: list = []
    slice_class_data: list = []
    count = 0
    out_dir_Batch = os.path.join(out_dir, "image_Batch")
    os.makedirs(out_dir_Batch, exist_ok=True)
    print(f"[Process Batch] Temp output dir: {out_dir}")

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

        print(f"{file_name} is processing")
        im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        (thresh, im_bw) = cv2.threshold(im_bw, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > cnt_threshold]

        annotated = image.copy()
        W, H = annotated.shape[:2]
        thickness, font_scale, radius_px = image_annotation_style(H, W, style="bold")

        # Classify slice kind (sagittal/coronal/axial vs cropped sub-slice).
        # Full MRI slices use a percent-of-slice-length window for the depth
        # filter and per-defect classification; cropped bands fall back to the
        # original fixed-millimeter rule and remain "unclassified".
        slice_kind, slice_kind_conf = classify_slice_kind(image)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        # Slice length = longest side of the brain's bounding box (physical units),
        # not the raw image size. Falls back to image extent if no brain contour was found.
        if filtered_contours:
            _bx, _by, _bw_px, _bh_px = cv2.boundingRect(np.vstack(filtered_contours))
            slice_length = max(_bw_px, _bh_px) * pixel_size
        else:
            slice_length = max(W, H) * pixel_size
        print(f"[Batch] {file_name}: slice_kind={slice_kind} (conf {slice_kind_conf:.2f}), "
              f"using {'percent' if use_percent_filter else 'fixed'} filter for sulci depth.")

        if filtered_contours:
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)
            area_sum = sum(cv2.contourArea(cnt) for cnt in filtered_contours)
            perimeter_sum = sum(cv2.arcLength(cnt, True) for cnt in filtered_contours)
        else:
            area_sum = perimeter_sum = 0
                
        area = area_sum * pixel_size**2
        perimeter = perimeter_sum * pixel_size
            

        closed_mask = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size))
        convex_Contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
        max_steps = 1000
        steps = 0
        while True:
            steps += 1
            if steps > max_steps:
                break  # safety

            filtered_conv_contours = [
                cnt for cnt in convex_Contours
                if cv2.contourArea(cnt) > cnt_threshold
            ]

            if not filtered_conv_contours:
                perimeter_convex_sum = 1.0  # avoid div-by-zero if needed later
                break

            perimeter_convex_sum = sum(cv2.arcLength(cnt, True) for cnt in filtered_conv_contours)
            perimeter_convex = perimeter_convex_sum * pixel_size
            perimeter_rate = perimeter / perimeter_convex if perimeter_convex else float("inf")
            compactness = compactness_2D(area, perimeter)
            if perimeter_rate < 1:
                cnt_threshold += 500
                continue  # retry with higher threshold
            break  # condition satisfied

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
            print(f"[Batch] {file_name}: {format_sulcus_class_summary(depth_sets)}")

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

        sheet1.append([
            file_name, slice_kind, area, perimeter, perimeter_convex,
            perimeter_rate, compactness,
            len(depth),
            (min(depth) if depth else None),
            (max(depth) if depth else None),
            mean_depth,
            *per_class_cells,
        ])


    base_cols = [
        'File', 'SliceKind', 'area', 'perimeter', 'perimeter_convex',
        'LGI', 'Compactness',
        'Sulci_count', f'min_depth_{unit}', f'max_depth_{unit}', f'mean_depth_{unit}',
    ]
    per_class_cols = sulcus_export_columns(unit)
    cols = base_cols + per_class_cols
    total_width = len(cols)

    overall_depth_sets = empty_depth_sets()
    for dsets in slice_class_data:
        if dsets is None:
            continue
        for k in SULCUS_CLASSES:
            overall_depth_sets[k].extend(dsets.get(k, []))


    sheet1.append(pad_row(['PixelSize:', pixel_size], total_width))
    sheet1.append(pad_row(['PixelSizeUnits:', unit], total_width))
    sheet1.append(pad_row(['KernelSize:', kernel_size], total_width))
    fd = pd.DataFrame(data=sheet1, columns=cols)
    fd = drop_empty_columns(fd)

    xlsx_path = os.path.join(out_dir, "Batch_Allmarks.xlsx")
    fd.to_excel(xlsx_path, index=False)
    print(f"[Process Batch] {count} images in {directory_path} have been processed")
    return valid_slices, saved_pngs
    
