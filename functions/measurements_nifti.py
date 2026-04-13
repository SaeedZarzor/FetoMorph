"""NIfTI segmentation volume measurements for FetoMorph.

Processes a 3-D NIfTI segmentation mask along **coronal slices** (axis 1).
For each slice the module extracts inner / outer contours, computes area,
perimeter, GI (gyrification index), and convexity-defect sulci depths,
then integrates across slices to obtain whole-brain volume (cm³), surface
area (cm²), and aggregate depth statistics.

Voxel sizes are read from the NIfTI header (``get_zooms``).  Because
voxels can be anisotropic, depth conversion uses
``defect_mm_per_px_and_fixed`` to project defect vectors onto the
physical coordinate system.
"""

from __future__ import annotations

from deps import *
from scipy.ndimage import binary_opening, binary_closing, label
from nibabel.affines import apply_affine
from helpers.helpers import (
    compute_kernel_convex,
    defect_mm_per_px_and_fixed,
    image_annotation_style,
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
from constants import DEFECT_FIXED_POINT, SULCUS_TERTIARY_MIN_FRACTION, SULCUS_PRIMARY_MAX_FRACTION

logger = logging.getLogger("fetomorph.nifti")



def compute_nifti_dims(brain_mask: np.ndarray, affine: np.ndarray) -> list[float]:
    """
    Compute voxel-space bounding box, physical bbox (mm), extents (mm), and volume (ml)
    for a 3D brain mask. Assumes the affine is in mm (standard for NIfTI).

    NOTE: The LR/PA/IS labeling assumes your image is in RAS+ orientation. If not,
    the numbers are still correct in world coords, but axis names may not map to anatomy.
    """
    mask = np.asarray(brain_mask).astype(bool)
    if mask.ndim != 3:
        raise ValueError(f"[NIfTI dimensions] mask must be 3D, got shape {mask.shape}")
    if affine.shape != (4, 4):
        raise ValueError(f"[NIfTI dimensions] affine must be 4x4, got shape {affine.shape}")
    if not np.any(mask):
        raise ValueError("[NIfTI dimensions] mask is empty (no True voxels).")

    # --- Bounding box in voxel indices (inclusive) ---
    i, j, k = np.where(mask)
    imin, jmin, kmin = int(i.min()), int(j.min()), int(k.min())
    imax, jmax, kmax = int(i.max()), int(j.max()), int(k.max())

    # --- Convert bbox to physical mm using voxel *edges* ---
    # Build the 8 voxel-edge corners: (min-0.5 ... max+0.5) in each axis
    corners_ijk = np.array(
        [[x, y, z]
         for x in (imin - 0.5, imax + 0.5)
         for y in (jmin - 0.5, jmax + 0.5)
         for z in (kmin - 0.5, kmax + 0.5)]
    )
    corners_xyz = apply_affine(affine, corners_ijk)
    mins_mm = corners_xyz.min(axis=0)
    maxs_mm = corners_xyz.max(axis=0)
    extents_mm = maxs_mm - mins_mm  # [X, Y, Z] in world (mm)

    print(f"[NIfTI dimensions] LR(X)={extents_mm[0]:.2f}  PA(Y)={extents_mm[1]:.2f}  IS(Z)={extents_mm[2]:.2f} mm")

    extents_list  = extents_mm.astype(float).tolist()
    extents_list_cm = [x/10 for x in extents_list]
    return extents_list_cm



def compute_nifti_allmarks(parent, file_path: str, out_dir: str, valid_labels: set[int],min_contour_area: float=30, kernel_size: int=5, ):
    """Compute all hallmarks (volume, area, GI, sulci depths) from a NIfTI segmentation.

    Iterates over coronal slices (axis 1), extracts inner/outer contours,
    measures area, perimeters, GI, and convexity-defect depths per slice,
    then integrates across slices.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.nii`` / ``.nii.gz`` file.
        out_dir: Output directory for slice PNGs and Excel.
        valid_labels: Set of integer segmentation labels to include.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size: Morph-close kernel diameter for outer contour.

    Returns:
        Tuple of ``(dims_cm, area_cm2, volume_cm3, GI, depths, saved_pngs, valid_slices)``
        or ``None`` if no valid slices found.
    """
    nifti_img = nib.load(file_path)
    nifti_img = nib.as_closest_canonical(nifti_img)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    header = nifti_img.header

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = header.get_zooms()[:3] # (x, y, z) in mm

    print(f"[NIfTI All hallmarks] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    # Voxel face area in the coronal plane (X × Z) — used to convert
    # pixel counts to physical mm² per slice.
    pixel_area_mm2 = pixel_size_x* pixel_size_z

    if not valid_labels:
        print(" [NIfTI All hallmarks] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI All hallmarks] Extracting regions:", valid_labels)
        brain_mask = np.isin(image_data, list(valid_labels))


    dims = compute_nifti_dims(brain_mask,affine)
    # Find coronal slice indices (axis 1) that contain brain tissue.
    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(parent, "[NIfTI All hallmarks]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI All hallmarks] Temp output dir: {out_dir}")



    sheet1=[]
    sum_area = 0
    sum_inner,sum_outer = 0, 0
    valid_slices = []
    saved_pngs = []
    total_depth = []
    slice_class_data: list = []


    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI lGI] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)
            slice_area = float(np.sum(slice_mask) * pixel_area_mm2)
            sum_area += slice_area
                
            # Inner contour
            inner_contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > min_contour_area]
            cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_contours]
            inner_perimeter = sum(cv2.arcLength(cnt, True) for cnt in cnt_mm)

            # Outer contour: rebuild a mask from ONLY the kept inner contours
            # so noise rejected by the inner filter cannot produce spurious
            # outer components after morph-close.
            inner_mask_only = np.zeros_like(slice_mask)
            cv2.drawContours(inner_mask_only, filtered_contours, -1, 1, thickness=cv2.FILLED)
            closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size))
            outer_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_outer_contours = [cnt for cnt in outer_contours if cv2.contourArea(cnt) > min_contour_area]
            outer_cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_outer_contours]
            outer_perimeter = sum(cv2.arcLength(cnt, True) for cnt in outer_cnt_mm)
                
            # Save perimeters
            sum_inner += inner_perimeter
            sum_outer += outer_perimeter
            GI_slice = inner_perimeter / outer_perimeter if outer_perimeter > 0 else 0
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            h_img, w_img = annotated.shape[:2]
            thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)  # Red contours (original)
            cv2.drawContours(annotated, filtered_outer_contours, -1, (0, 255, 0), thickness)  # green
            
            # Sulci classification: bin each kept defect by its depth as a
            # fraction of the brain's IS-extent (mm). All four sets are
            # flattened back into `depth` so the rest of the pipeline keeps
            # working unchanged.
            _is_extent_mm = dims[2] * 10
            depth_sets = empty_depth_sets()
            if filtered_contours:
                for cnt in filtered_contours:
                    hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)  # Compute convex hull
                    if hull is not None and len(hull) >= 3 and len(cnt) > 3 and np.all(np.diff(hull.ravel()) > 0):
                        defects = cv2.convexityDefects(cnt, hull)
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(cnt[s][0])
                                end = tuple(cnt[e][0])
                                far = tuple(cnt[f][0])
                                annotated = cv2.line(annotated, start, end, [255, 0, 0], thickness)
                                if d > DEFECT_FIXED_POINT:
                                    if pixel_size_x!=pixel_size_z:
                                        mm_per_px, mm_per_fixed = defect_mm_per_px_and_fixed(start, end, far, pixel_size_x, pixel_size_z)
                                        if mm_per_fixed is None:
                                            continue  # degenerate edge (start == end)
                                    else:
                                        mm_per_fixed = pixel_size_x / DEFECT_FIXED_POINT

                                    depth_mm = d *mm_per_fixed

                                    # Depth filter: keep defects within
                                    # SULCI_DEPTH_MIN_FRACTION..SULCI_DEPTH_MAX_FRACTION
                                    # of brain IS-extent. dims[2] is in cm, ×10 → mm.
                                    if (SULCUS_TERTIARY_MIN_FRACTION * _is_extent_mm) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * _is_extent_mm):
                                        sulcus_class = classify_sulcus_depth(depth_mm, _is_extent_mm)
                                        marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                        depth_sets[sulcus_class].append(depth_mm)
                                        annotated = cv2.circle(annotated, far, radius_px, marker_color, -1)

            depth = flatten_depth_sets(depth_sets)
            print(f"[NIfTI allmarks] slice {idx}: {format_sulcus_class_summary(depth_sets)}")
            mean_depth = (sum(depth)/len(depth)) if depth else None
            total_depth.extend(depth)
            slice_class_data.append(depth_sets)
            sheet1.append([idx, slice_area, inner_perimeter, outer_perimeter,
                len(depth),                         # n_defects
                (min(depth) if depth else None),    # min_depth_mm
                (max(depth) if depth else None),    # max_depth_mm
                mean_depth,                         # mean_depth_mm
                *sulcus_export_cells(depth_sets),
                ])

            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)



        # Volume integration: sum of slice areas (mm²) × slice spacing (mm) → mm³,
        # then /1000 → cm³.
        brain_volume = (sum_area * pixel_size_y)/1000
        # Surface area: sum of inner perimeters (mm) × slice spacing → mm²,
        # then /100 → cm².
        Area = sum_inner * pixel_size_y /100
        GI_total = sum_inner / sum_outer if sum_outer > 0 else 0
        # Snapshot mm-unit depths for the per-mm Excel summary row before
        # the in-place cm conversion below.
        total_depth_mm = list(total_depth)
        # Convert depth values from mm to cm for reporting.
        total_depth = [x / 10 for x in total_depth]

        # Build the integrated Sulci_overall_summary row.
        base_cols = [
            "Slice", "Inner_area_mm^2", "Inner_Perimeter_mm", "Outer_Perimeter_mm",
            "Sulci_count", "min_depth_mm", "max_depth_mm", "mean_depth_mm",
        ]
        per_class_cols = sulcus_export_columns("mm")
        cols = base_cols + per_class_cols
        total_width = len(cols)

        overall_depth_sets = empty_depth_sets()
        for dsets in slice_class_data:
            if dsets is None:
                continue
            for k in SULCUS_CLASSES:
                overall_depth_sets[k].extend(dsets.get(k, []))

        overall_n = len(total_depth_mm)
        overall_min = min(total_depth_mm) if total_depth_mm else None
        overall_max = max(total_depth_mm) if total_depth_mm else None
        overall_mean = (sum(total_depth_mm) / overall_n) if overall_n else None

        summary_base = [None] * len(base_cols)
        summary_base[0] = "Sulci_overall_summary"
        summary_base[base_cols.index("Sulci_count")]    = overall_n
        summary_base[base_cols.index("min_depth_mm")]   = overall_min
        summary_base[base_cols.index("max_depth_mm")]   = overall_max
        summary_base[base_cols.index("mean_depth_mm")]  = overall_mean
        sheet1.append(pad_row(
            [*summary_base, *sulcus_export_cells(overall_depth_sets)],
            total_width,
        ))

        sheet1.append(pad_row(["Volume cm^3", round(brain_volume, 2), "Surface Area cm^2", round(Area, 2)], total_width))
        sheet1.append(pad_row(["GI", round(GI_total, 2)], total_width))

        df = pd.DataFrame(sheet1, columns=cols)
        df = drop_empty_columns(df)
        xlsx_path = os.path.join(out_dir, "Brain_Allmarks.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI All hallmarks] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return dims, Area, brain_volume, GI_total, total_depth ,saved_pngs, valid_slices

    else:
        QMessageBox.information(parent, "[NIfTI All hallmarks]", "All slices were filtered out (too small).")
        return



def compute_nifti_volume(parent, file_path: str, out_dir: str, valid_labels: set[int]):
    """Compute brain volume (cm³) from a NIfTI segmentation by integrating slice areas.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the NIfTI file.
        out_dir: Output directory for slice PNGs and Excel.
        valid_labels: Set of integer segmentation labels to include.

    Returns:
        Tuple of ``(dims_cm, volume_cm3, saved_pngs, valid_slices)`` or ``None``.
    """
    nifti_img = nib.load(file_path)
    nifti_img = nib.as_closest_canonical(nifti_img)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI Volume] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

#    unique_labels = np.unique(image_data)
#    print("[NIfTI Volume] Unique labels in the image:", unique_labels)
#
#    # Step 1: Define the Selected Regions
#    selected_regions = {3, 4, 5, 6, 14, 15, 16, 17}  # The regions you want to include
#
#    # Check if selected regions exist in the image
#    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI Volume] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI Volume] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))

    dims = compute_nifti_dims(brain_mask,affine)

#    # --- Brain volume (ml) ---
#    voxel_vol_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
#    brain_vol_ml = brain_mask.sum() * voxel_vol_mm3 / 1000.0
#    print(f"[Brain volume] ~{brain_vol_ml:.2f} cm^3")

    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(parent, "[NIfTI Volume]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI Volume] Temp output dir: {out_dir}")


    sheet1=[]

    sum_area = 0
    valid_slices = []
    saved_pngs = []


    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI Volume] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)
                
            slice_area = float(np.sum(slice_mask) * pixel_area_mm2)
            sum_area += slice_area
                
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))

                
            sheet1.append([idx, slice_area])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)



        brain_volume = (sum_area * pixel_size_y)/1000


        sheet1.append(["Volume cm^3",round(brain_volume,2)])

        df = pd.DataFrame(sheet1, columns=["Slice", "Inner_area_mm^2", ])
        xlsx_path = os.path.join(out_dir, "Brain_Volume.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
    #    brain_extracted = image_data * filtered_mask
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI Volume] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return dims, brain_volume, saved_pngs, valid_slices

    else:
        QMessageBox.information(parent, "[NIfTI Volume]", "All slices were filtered out (too small).")
        return




def compute_nifti_area(parent, file_path: str, out_dir: str,  valid_labels: set[int], min_contour_area: float=30,):
    """Compute brain surface area (cm²) from a NIfTI segmentation.

    Sums inner-contour perimeters across coronal slices and multiplies
    by slice spacing to approximate total surface area.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the NIfTI file.
        out_dir: Output directory for slice PNGs and Excel.
        valid_labels: Set of integer segmentation labels to include.
        min_contour_area: Minimum contour area (pixels) to keep.

    Returns:
        Tuple of ``(dims_cm, area_cm2, saved_pngs, valid_slices)`` or ``None``.
    """
    nifti_img = nib.load(file_path)
    nifti_img = nib.as_closest_canonical(nifti_img)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI Area] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

#    unique_labels = np.unique(image_data)
#    print("[NIfTI Area] Unique labels in the image:", unique_labels)
#
#    # Step 1: Define the Selected Regions
#    selected_regions = {3, 4, 5, 6, 14, 15, 16, 17}  # The regions you want to include
#
#    # Check if selected regions exist in the image
#    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI Area] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI Area] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))

    dims = compute_nifti_dims(brain_mask,affine)

    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(parent, "[NIfTI Area]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI Area] Temp output dir: {out_dir}")
    
    sheet1=[]

    sum_inner = 0
    valid_slices = []
    saved_pngs = []


    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI lGI] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)

                
            # Inner contour
            inner_contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > min_contour_area]
            cont_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_contours]
            inner_perimeter = sum(cv2.arcLength(cnt, True) for cnt in cont_mm)
                

                
            # Save perimeters
            sum_inner += inner_perimeter
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            h_img, w_img = annotated.shape[:2]
            thickness, _, _ = image_annotation_style(h_img, w_img, style="regular")
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)  # Red contours (original)
                            
            sheet1.append([idx, inner_perimeter])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)


        Area = sum_inner * pixel_size_y/ 100


        sheet1.append(["Surface Area cm^2",round(Area,2)])

        df = pd.DataFrame(sheet1, columns=["Slice", "Inner_Perimeter_mm"])
        xlsx_path = os.path.join(out_dir, "Brain_Surface_Area.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
    #    brain_extracted = image_data * filtered_mask
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI Area] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return dims,Area, saved_pngs, valid_slices

    else:
        QMessageBox.information(parent, "[NIfTI Area]", "All slices were filtered out (too small).")
        return
    
    

def compute_nifti_lGI(parent, file_path: str, out_dir: str,  valid_labels: set[int], min_contour_area: float=30, kernel_size: int=5):
    """Compute the gyrification index (GI) from a NIfTI segmentation.

    GI = total inner perimeter / total outer perimeter across all coronal
    slices.  The "outer" contour is obtained by morphologically closing
    each slice mask to fill sulci.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the NIfTI file.
        out_dir: Output directory for slice PNGs and Excel.
        valid_labels: Set of integer segmentation labels to include.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size: Morph-close kernel diameter.

    Returns:
        Tuple of ``(GI_total, saved_pngs, valid_slices)`` or ``None``.
    """
    nifti_img = nib.load(file_path)
    nifti_img = nib.as_closest_canonical(nifti_img)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI lGI] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

#    unique_labels = np.unique(image_data)
#    print("[NIfTI lGI] Unique labels in the image:", unique_labels)
#
#    # Step 1: Define the Selected Regions
#    selected_regions = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17}  # The regions you want to include
#
#    # Check if selected regions exist in the image
#    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI lGI] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI lGI] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))


    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(parent, "[NIfTI lGI]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI lGI] Temp output dir: {out_dir}")



    sheet1=[]

    sum_inner,sum_outer = 0, 0
    valid_slices = []
    saved_pngs = []
    
    
    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI lGI] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)
                
#                slice_area = float(np.sum(slice_mask) * pixel_area_mm2)
#                sum_area += slice_area
                
            # Inner contour
            inner_contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > min_contour_area]
            cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_contours]
            inner_perimeter = sum(cv2.arcLength(cnt, True) for cnt in cnt_mm)

            # Outer contour: rebuild a mask from ONLY the kept inner contours
            # so noise rejected by the inner filter cannot produce spurious
            # outer components after morph-close.
            inner_mask_only = np.zeros_like(slice_mask)
            cv2.drawContours(inner_mask_only, filtered_contours, -1, 1, thickness=cv2.FILLED)
            closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size))
            outer_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_outer_contours = [cnt for cnt in outer_contours if cv2.contourArea(cnt) > min_contour_area]
            outer_cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_outer_contours]
            outer_perimeter = sum(cv2.arcLength(cnt, True) for cnt in outer_cnt_mm)
                
            # Save perimeters
            sum_inner += inner_perimeter
                
            sum_outer += outer_perimeter
            GI_slice = inner_perimeter / outer_perimeter if outer_perimeter > 0 else 0
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            h_img, w_img = annotated.shape[:2]
            thickness, _, _ = image_annotation_style(h_img, w_img, style="regular")
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)  # Red contours (original)
            cv2.drawContours(annotated, filtered_outer_contours, -1, (0, 255, 0), thickness)  # green


                
    
#                    all_points = np.vstack(contours_mm)  # Stack all contour points
#                    hull = cv2.convexHull(all_points)  # Compute convex hull
#                    perimeter_convex = cv2.arcLength(hull, True)  # Perimeter in mm
#                    sum_perimeter_conv += perimeter_convex  # Sum total convex perimeters
                    
                    
#                    # Compute Perimeter Rate
#                    perimeter_rate = slice_perimeter / perimeter_convex if perimeter_convex > 0 else 0
#                    perimeters.append(slice_perimeter)  # Save to array
#                    convex_hull_perimeters.append(perimeter_convex)  # Save to array
#    
#                    hull_draw= cv2.convexHull(np.vstack(filtered_contours))
#                    # Draw contours (original in red, convex hull in green)
#                    cv2.drawContours(contour_image, [hull_draw], -1, (0, 255, 0), 1)  # Green convex hull

                
            sheet1.append([idx, inner_perimeter, outer_perimeter])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)



#        brain_length_cm = ((valid_slices[-1] - valid_slices[0] + 1) * pixel_size_y) /10
#        brain_volume = (sum_area * pixel_size_y)/1000
        GI_total = sum_inner / sum_outer if sum_outer > 0 else 0


        sheet1.append(["GI",round(GI_total,2)])

        df = pd.DataFrame(sheet1, columns=["Slice", "Inner_Perimeter_mm", "Outer_Perimeter_mm"])
        xlsx_path = os.path.join(out_dir, "Brain_lGI.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
    #    brain_extracted = image_data * filtered_mask
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI lGI] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return GI_total, saved_pngs, valid_slices

    else:
        QMessageBox.information(parent, "[NIfTI lGI]", "All slices were filtered out (too small).")
        return


        
def compute_nifti_sulci_depth(parent, file_path: str, out_dir: str,  valid_labels: set[int], min_contour_area: float=30,):
    """Compute sulci depths from convexity defects across NIfTI coronal slices.

    For each slice, extracts contours, computes convex hulls, and measures
    defect depths.  Anisotropic voxels are handled via
    ``defect_mm_per_px_and_fixed``.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the NIfTI file.
        out_dir: Output directory for slice PNGs and Excel.
        valid_labels: Set of integer segmentation labels to include.
        min_contour_area: Minimum contour area (pixels) to keep.

    Returns:
        Tuple of ``(dims_cm, depths_cm, saved_pngs, valid_slices)`` or ``None``.
    """
    nifti_img = nib.load(file_path)
    nifti_img = nib.as_closest_canonical(nifti_img)
    image_data = nifti_img.get_fdata()  # Get voxel data (3D NumPy array)
    affine = nifti_img.affine        # ✅ Needed for spatial info
    header = nifti_img.header        # ✅ Optional but keeps meta info
    # Get voxel size (in mm)
    voxel_size = nifti_img.header.get_zooms()  # (x, y, z) in mm

    # Extract pixel size
    pixel_size_x, pixel_size_y, pixel_size_z = voxel_size[:3]

    print(f"[NIfTI Sulci depth] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")


    pixel_area_mm2 = pixel_size_x* pixel_size_z

#    unique_labels = np.unique(image_data)
#    print("[NIfTI Sulci depth] Unique labels in the image:", unique_labels)
#
#    # Step 1: Define the Selected Regions
#    selected_regions = {2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17}  # The regions you want to include
#
#    # Check if selected regions exist in the image
#    valid_labels = selected_regions.intersection(set(unique_labels))
    if not valid_labels:
        print(" [NIfTI Sulci depth] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI Sulci depth] Extracting regions:", valid_labels)
        # Step 2: Create a Mask Including Only These Regions
        brain_mask = np.isin(image_data, list(valid_labels))

    dims = compute_nifti_dims(brain_mask,affine)

    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(parent, "[NIfTI Sulci depth]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI Sulci depth] Temp output dir: {out_dir}")



    sheet1=[]
    valid_slices = []
    saved_pngs = []
    total_depth = []
    slice_class_data: list = []


    # Loop through all slices and save each as an image
    if len(brain_slices) > 0:
        for idx in brain_slices:  # Iterate over slices (z-dimension)
            slice_mask = brain_mask[:, idx, :].astype(np.uint8)  # Convert to uint8

            nonzero_count = np.count_nonzero(slice_mask)
#            if nonzero_count < 100:
#                print(f"[NIfTI lGI] Slice {idx} ignored: too few non-zero pixels ({nonzero_count})")
#                continue  # Skip this slice
#            else:
            valid_slices.append(idx)


            # Inner contour
            inner_contours, _ = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) > min_contour_area]
            cnt_mm = [cnt.astype(np.float32) * [pixel_size_x, pixel_size_z] for cnt in filtered_contours]
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            h_img, w_img = annotated.shape[:2]
            thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)  # Red contours (original)
                
            # Sulci classification: bin each kept defect by its depth as a
            # fraction of the brain's IS-extent (mm).
            _is_extent_mm = dims[2] * 10
            depth_sets = empty_depth_sets()
            if filtered_contours:
                for cnt in filtered_contours:
                    hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)  # Compute convex hull
                    if hull is not None and len(hull) >= 3 and len(cnt) > 3 and np.all(np.diff(hull.ravel()) > 0):
                        defects = cv2.convexityDefects(cnt, hull)
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(cnt[s][0])
                                end = tuple(cnt[e][0])
                                far = tuple(cnt[f][0])
                                annotated = cv2.line(annotated, start, end, [255, 0, 0], thickness)
                                if d > DEFECT_FIXED_POINT:
                                    if pixel_size_x!=pixel_size_z:
                                        mm_per_px, mm_per_fixed = defect_mm_per_px_and_fixed(start, end, far, pixel_size_x, pixel_size_z)
                                        if mm_per_fixed is None:
                                            continue  # degenerate edge (start == end)
                                    else:
                                        mm_per_fixed = pixel_size_x / DEFECT_FIXED_POINT

                                    depth_mm = d *mm_per_fixed

                                    # Depth filter: keep defects within
                                    # SULCI_DEPTH_MIN_FRACTION..SULCI_DEPTH_MAX_FRACTION
                                    # of brain IS-extent. dims[2] is in cm, ×10 → mm.
                                    if (SULCI_DEPTH_MIN_FRACTION * _is_extent_mm) < depth_mm < (SULCI_DEPTH_MAX_FRACTION * _is_extent_mm):
                                        sulcus_class = classify_sulcus_depth(depth_mm, _is_extent_mm)
                                        marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                        depth_sets[sulcus_class].append(depth_mm)
                                        annotated = cv2.circle(annotated, far, radius_px, marker_color, -1)

            depth = flatten_depth_sets(depth_sets)
            print(f"[NIfTI sulci] slice {idx}: {format_sulcus_class_summary(depth_sets)}")
            total_depth.extend(depth)
            slice_class_data.append(depth_sets)
            mean_depth = (sum(depth)/len(depth)) if depth else None

            sheet1.append([idx,
                len(depth),                         # n_defects
                (min(depth) if depth else None),    # min_depth_mm
                (max(depth) if depth else None),    # max_depth_mm
                mean_depth,                         # mean_depth_mm
                *sulcus_export_cells(depth_sets),
                ])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)
        
        # Snapshot mm-unit depths for the per-mm Excel summary row before
        # the in-place cm conversion below (which only affects the value
        # returned to the dispatcher).
        total_depth_mm = list(total_depth)
        total_depth = [x / 10 for x in total_depth]

        base_cols = ["Slice", "Sulci_count", "min_depth_mm", "max_depth_mm", "mean_depth_mm"]
        per_class_cols = sulcus_export_columns("mm")
        cols = base_cols + per_class_cols
        total_width = len(cols)

        overall_depth_sets = empty_depth_sets()
        for dsets in slice_class_data:
            if dsets is None:
                continue
            for k in SULCUS_CLASSES:
                overall_depth_sets[k].extend(dsets.get(k, []))

        overall_n = len(total_depth_mm)
        overall_min = min(total_depth_mm) if total_depth_mm else None
        overall_max = max(total_depth_mm) if total_depth_mm else None
        overall_mean = (sum(total_depth_mm) / overall_n) if overall_n else None

        summary_base = [None] * len(base_cols)
        summary_base[0] = "Sulci_overall_summary"
        summary_base[base_cols.index("Sulci_count")]   = overall_n
        summary_base[base_cols.index("min_depth_mm")]  = overall_min
        summary_base[base_cols.index("max_depth_mm")]  = overall_max
        summary_base[base_cols.index("mean_depth_mm")] = overall_mean
        sheet1.append(pad_row(
            [*summary_base, *sulcus_export_cells(overall_depth_sets)],
            total_width,
        ))

        df = pd.DataFrame(sheet1, columns=cols)
        df = drop_empty_columns(df)
        xlsx_path = os.path.join(out_dir, "Brain_Sulci.xlsx")
        df.to_excel(xlsx_path, index=False)
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI Slice] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return  dims, total_depth, saved_pngs, valid_slices

    else:
        QMessageBox.information(parent, "[NIfTI Sulci depth]", "All slices were filtered out (too small).")
        return


