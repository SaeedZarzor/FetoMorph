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
    compactness_2D,
    mask_perimeter_mm,
    compute_kernel_convex,
    lateral_area_simpson,
    volume_simpson,
    total_surface_area_simpson,
    defect_mm_per_px_and_fixed,
    image_annotation_style,
    SULCUS_CLASS_COLORS,
    SULCUS_CLASSES,
    classify_sulcus_depth,
    sulcus_depth_min,
    empty_depth_sets,
    flatten_depth_sets,
    format_sulcus_class_summary,
    sulcus_export_columns,
    sulcus_export_cells,
    pad_row,
    drop_empty_columns,
)
from helpers.cavities import cavity_correction_nifti, CavityCorrection
from constants import (
    DEFAULT_KERNEL_SIZE_MM, DEFECT_FIXED_POINT,
    SULCUS_TERTIARY_MIN_FRACTION, SULCUS_PRIMARY_MAX_FRACTION,
    DEFAULT_CAVITY_CORRECTION_ENABLED, DEFAULT_CAVITY_AREA_THRESHOLD_MM2,
    DEFAULT_PERIMETER_METHOD, DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
)

logger = logging.getLogger("fetomorph.nifti")


def _to_kernel_px(kernel_size_mm: float, pixel_size_mm: float) -> int:
    px = max(3, int(round(float(kernel_size_mm) / max(float(pixel_size_mm), 1e-9))))
    return px


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



def compute_nifti_allmarks(parent, file_path: str, out_dir: str, valid_labels: set[int], min_contour_area: float = 30, kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON):
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
        kernel_size_mm: Morph-close kernel diameter for outer contour, in mm.

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

    print(f"[NIfTI All Hallmarks] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")
    print(
        f"[NIfTI All Hallmarks] Perimeter method={perimeter_method}; "
        f"in-plane spacing={pixel_size_x:.4f} x {pixel_size_z:.4f} mm/px; "
        f"simplify={'on' if simplify_contours_for_perimeter else 'off'} "
        f"(epsilon={float(contour_simplify_epsilon):g} px)"
    )


    # Voxel face area in the coronal plane (X × Z) — used to convert
    # pixel counts to physical mm² per slice.
    pixel_area_mm2 = pixel_size_x* pixel_size_z
    kernel_size_px = _to_kernel_px(kernel_size_mm, max(pixel_size_x, pixel_size_z))

    if not valid_labels:
        print("[NIfTI All Hallmarks] Warning: None of the selected regions are present in this NIfTI file!")
        threshold = np.percentile(image_data, 50)
        brain_mask = image_data #> threshold
    else:
        print("[NIfTI All Hallmarks] Extracting regions:", valid_labels)
        brain_mask = np.isin(image_data, list(valid_labels))


    dims = compute_nifti_dims(brain_mask,affine)
    # Find coronal slice indices (axis 1) that contain brain tissue.
    brain_slices = np.where(np.any(brain_mask, axis=(0, 2)))[0]
    if brain_slices.size == 0:
        QMessageBox.information(parent, "[NIfTI All Hallmarks]", "No slices contain the selected mask.")
        return
        
   
    out_dir_slices = os.path.join(out_dir, "brain_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    print(f"[NIfTI All Hallmarks] Temp output dir: {out_dir}")



    # Surface-connected cavity classification (exact 3-D via binary_fill_holes).
    # `filled` = tissue + enclosed voids (open cavities stay background); it is the
    # corrected cross-section for the volume integral. Open-cavity wall perimeters
    # are added to the surface lateral only (GI is untouched).
    if cavity_correction_enabled:
        cavity_corr, filled_mask = cavity_correction_nifti(
            brain_mask > 0, list(brain_slices), axis=1,
            pixel_size_x=float(pixel_size_x), pixel_size_z=float(pixel_size_z),
            area_threshold_mm2=float(cavity_area_threshold_mm2))
    else:
        cavity_corr, filled_mask = CavityCorrection.empty(), None

    sheet1=[]
    sum_area = 0
    sum_inner,sum_outer = 0, 0
    # (slice position mm, exterior perimeter mm) for the Simpson lateral integral,
    # and (position mm, cross-section area mm²) for the Simpson volume integral.
    lateral_samples: list[tuple[float, float]] = []
    volume_samples: list[tuple[float, float]] = []
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
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) * pixel_area_mm2 > min_contour_area]
            # Outer contour: rebuild a mask from ONLY the kept inner contours
            # so noise rejected by the inner filter cannot produce spurious
            # outer components after morph-close.
            inner_mask_only = np.zeros_like(slice_mask)
            cv2.drawContours(inner_mask_only, filtered_contours, -1, 1, thickness=cv2.FILLED)
            inner_perimeter = mask_perimeter_mm(
                inner_mask_only, pixel_size_x, pixel_size_z,
                method=perimeter_method,
                simplify=simplify_contours_for_perimeter,
                epsilon=contour_simplify_epsilon,
            )
            closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size_px))
            outer_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_outer_contours = [cnt for cnt in outer_contours if cv2.contourArea(cnt) * pixel_area_mm2 > min_contour_area]
            outer_mask_only = np.zeros_like(closed_mask)
            cv2.drawContours(outer_mask_only, filtered_outer_contours, -1, 1, thickness=cv2.FILLED)
            outer_perimeter = mask_perimeter_mm(
                outer_mask_only, pixel_size_x, pixel_size_z,
                method=perimeter_method,
                simplify=simplify_contours_for_perimeter,
                epsilon=contour_simplify_epsilon,
            )
                
            # Save perimeters
            sum_inner += inner_perimeter
            sum_outer += outer_perimeter
            # Cavity correction: volume cross-section uses the filled mask (tissue
            # + enclosed; open cavities removed); surface lateral gets the open
            # cavity wall perimeter added. GI (exterior/closed-envelope) is unchanged.
            vol_area = (float(np.sum(filled_mask[:, idx, :]) * pixel_area_mm2)
                        if filled_mask is not None else slice_area)
            lateral_samples.append((float(idx) * pixel_size_y,
                                    inner_perimeter + cavity_corr.perim_add(idx)))
            volume_samples.append((float(idx) * pixel_size_y, vol_area))
            GI_slice = inner_perimeter / outer_perimeter if outer_perimeter > 0 else 0
                
            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            h_img, w_img = annotated.shape[:2]
            thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)  # Red contours (original)
            cv2.drawContours(annotated, filtered_outer_contours, -1, (0, 255, 0), thickness)  # green
            # Yellow outline around surface-connected cavities (classified before
            # the loop via binary_fill_holes); enclosed voids are left unmarked.
            _cav_cnts = cavity_corr.surface_connected_by_idx.get(idx, [])
            if _cav_cnts:
                cv2.drawContours(annotated, _cav_cnts, -1, (0, 255, 255), thickness)  # yellow

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
                                    if (SULCUS_TERTIARY_MIN_FRACTION * _is_extent_mm) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * _is_extent_mm) and depth_mm > sulcus_depth_min("mm"):
                                        sulcus_class = classify_sulcus_depth(depth_mm, _is_extent_mm)
                                        marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                        depth_sets[sulcus_class].append(depth_mm)
                                        annotated = cv2.circle(annotated, far, radius_px, marker_color, -1)

            depth = flatten_depth_sets(depth_sets)
            print(f"[NIfTI All Hallmarks] slice {idx}: {format_sulcus_class_summary(depth_sets)}")
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



        # Volume = ∫ slice area dh via Simpson's rule (mm³ → /1000 cm³).
        # The voxel-sum volume (Σ area × spacing) is kept only for verification.
        brain_volume = volume_simpson(volume_samples) / 1000
        _voxel_volume_cm3 = (sum_area * pixel_size_y) / 1000
        print(f"[NIfTI All Hallmarks] Volume={brain_volume:.3f} cm³")
        # Surface area = Simpson lateral (∫ exterior perimeter dh) + top & bottom
        # caps (mm² → /100 cm²).
        _total_area_mm2, _lat_mm2, _caps_mm2 = total_surface_area_simpson(
            lateral_samples, volume_samples)
        Area = _total_area_mm2 / 100
        print(f"[NIfTI All Hallmarks] Surface area: lateral+caps={Area:.3f} cm²")
        if cavity_correction_enabled:
            print(f"[NIfTI All Hallmarks] Cavity correction: "
                  f"{cavity_corr.n_surface_connected} surface-connected, "
                  f"{cavity_corr.n_enclosed} enclosed; "
                  f"area removed={cavity_corr.total_cavity_area_mm2 / 100:.3f} cm², "
                  f"wall added={cavity_corr.total_wall_perim_mm / 10:.3f} cm")
        GI_total = sum_inner / sum_outer if sum_outer > 0 else 0
        # Snapshot mm-unit depths for the per-mm Excel summary row before
        # the in-place cm conversion below.
        total_depth_mm = list(total_depth)
        # Convert depth values from mm to cm for reporting.
        total_depth = [x / 10 for x in total_depth]

        # Save per-slice + totals to Excel using the shared spec layout.
        try:
            from helpers.results_excel_format import (
                ResultsSheet, write_results_workbook, subtype_mean,
            )

            overall_n = len(total_depth_mm)
            overall_mean = ((sum(total_depth_mm) / overall_n)
                            if overall_n else None)

            results_rows = []
            for r, dsets, png_path in zip(
                    sheet1, slice_class_data, saved_pngs):
                idx, slice_area_mm, inner_perim_mm, outer_perim_mm = r[:4]
                lgi = ((inner_perim_mm / outer_perim_mm)
                       if outer_perim_mm else None)
                compact = (compactness_2D(slice_area_mm, inner_perim_mm)
                           if inner_perim_mm else None)
                d = dsets if isinstance(dsets, dict) else {}
                results_rows.append({
                    "Section": int(idx),
                    "Area": slice_area_mm,
                    "Perimeter": inner_perim_mm,
                    "LGI": lgi,
                    "Compactness": compact,
                    "PrimarySulciCount": len(d.get("primary", []) or []),
                    "SecondarySulciCount": len(d.get("secondary", []) or []),
                    "TertiarySulciCount": len(d.get("tertiary", []) or []),
                    "UnclassifiedSulciCount":
                        len(d.get("unclassified", []) or []),
                    "PrimaryMeanDepth": subtype_mean(
                        None, d.get("primary", []) or []),
                    "SecondaryMeanDepth": subtype_mean(
                        None, d.get("secondary", []) or []),
                    "TertiaryMeanDepth": subtype_mean(
                        None, d.get("tertiary", []) or []),
                    "UnclassifiedMeanDepth": subtype_mean(
                        None, d.get("unclassified", []) or []),
                    "_section_link": png_path,
                })

            parameters = {
                "Kernel size (mm)": float(kernel_size_mm),
                "Kernel size (px)": int(kernel_size_px),
                "Pixel spacing": (
                    f"{float(pixel_size_x):.4f} × {float(pixel_size_z):.4f} mm "
                    "(in-plane)"),
                "Slice thickness": float(pixel_size_y),
                "Filtered threshold (mm²)": float(min_contour_area),
                "Perimeter method": perimeter_method,
                "Isotropic spacing used": perimeter_method == "crofton",
                "Contour simplification enabled": bool(simplify_contours_for_perimeter),
                "Contour simplification epsilon": float(contour_simplify_epsilon),
                "Cavity correction": "on" if cavity_correction_enabled else "off",
            }
            if cavity_correction_enabled:
                parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
            totals = {
                "Volume (cm^3)": round(float(brain_volume), 4),
                "Surface Area (cm^2)": round(float(Area), 4),
                "GI": round(float(GI_total), 4),
                "Total sulci count": int(overall_n),
                "Mean sulci depth (mm)": (round(float(overall_mean), 4)
                                           if overall_mean is not None
                                           else None),
            }
            if cavity_correction_enabled:
                totals.update({
                    "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                    "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                    "Surface connected cavity area (cm^2)": round(cavity_corr.total_cavity_area_mm2 / 100, 4),
                    "Cavity wall surface (cm^2)": round(cavity_corr.total_wall_perim_mm * float(pixel_size_y) / 100, 4),
                })
            sheet = ResultsSheet(
                sheet_name=os.path.basename(file_path) or "Results",
                file_name=os.path.basename(file_path),
                folder=os.path.dirname(file_path) or None,
                parameters=parameters,
                rows=results_rows,
                totals=totals,
                drop_empty_columns=True,
            )
            xlsx_path = os.path.join(out_dir, "Brain_Allmarks.xlsx")
            write_results_workbook(xlsx_path, [sheet])
            print(f"[NIfTI All Hallmarks] Saved Excel → {xlsx_path}")
        except Exception as ex:
            print(f"[NIfTI All Hallmarks] WARN: could not save Excel: {ex}")
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI All Hallmarks] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return dims, Area, brain_volume, GI_total, total_depth ,saved_pngs, valid_slices

    else:
        QMessageBox.information(parent, "[NIfTI All Hallmarks]", "All slices were filtered out (too small).")
        return



def compute_nifti_volume(parent, file_path: str, out_dir: str, valid_labels: set[int],
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2):
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
        print("[NIfTI Volume] Warning: None of the selected regions are present in this NIfTI file!")
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

    # Surface-connected cavity classification (exact 3-D via binary_fill_holes).
    # `filled_mask` = tissue + enclosed voids (open cavities stay background); it
    # is the corrected cross-section for the volume integral.
    if cavity_correction_enabled:
        cavity_corr, filled_mask = cavity_correction_nifti(
            brain_mask > 0, list(brain_slices), axis=1,
            pixel_size_x=float(pixel_size_x), pixel_size_z=float(pixel_size_z),
            area_threshold_mm2=float(cavity_area_threshold_mm2))
    else:
        cavity_corr, filled_mask = CavityCorrection.empty(), None

    sheet1=[]

    sum_area = 0
    volume_samples: list[tuple[float, float]] = []  # (position mm, cross-section area mm²)
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

            # Holes-filled cross-section (enclosed voids solid, open cavities
            # excluded) for the volume integral.
            slice_area = (float(np.sum(filled_mask[:, idx, :]) * pixel_area_mm2)
                          if filled_mask is not None
                          else float(np.sum(slice_mask) * pixel_area_mm2))
            sum_area += slice_area
            volume_samples.append((float(idx) * pixel_size_y, slice_area))

            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            # Yellow-outline surface-connected cavities on the saved slice.
            _cav_cnts = cavity_corr.surface_connected_by_idx.get(idx, [])
            if _cav_cnts:
                h_img, w_img = annotated.shape[:2]
                _ct, _, _ = image_annotation_style(h_img, w_img, style="regular")
                cv2.drawContours(annotated, _cav_cnts, -1, (0, 255, 255), _ct)

            sheet1.append([idx, slice_area])

            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)



        # Volume = ∫ slice area dh via Simpson's rule (mm³ → cm³); voxel-sum kept
        # only for verification.
        brain_volume = volume_simpson(volume_samples) / 1000
        _voxel_volume_cm3 = (sum_area * pixel_size_y) / 1000
        print(f"[NIfTI Volume] Volume={brain_volume:.3f} cm³")
        if cavity_correction_enabled:
            print(f"[NIfTI Volume] Cavity correction: "
                  f"{cavity_corr.n_surface_connected} surface-connected, "
                  f"{cavity_corr.n_enclosed} enclosed; "
                  f"area removed={cavity_corr.total_cavity_area_mm2 / 100:.3f} cm²")


        from helpers.results_excel_format import build_measurement_sheet, write_results_workbook
        results_rows = [{"Section": r[0], "Area": r[1]} for r in sheet1]
        parameters = {
            "Pixel spacing": (
                f"{float(pixel_size_x):.4f} × {float(pixel_size_z):.4f} mm "
                "(in-plane)"),
            "Slice thickness": float(pixel_size_y),
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {"Volume (cm^3)": round(float(brain_volume), 4)}
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                "Cavity area removed (cm²)": round(float(cavity_corr.total_cavity_area_mm2 / 100), 4),
            })
        sheet = build_measurement_sheet(file_path, "Volume", results_rows, parameters, totals)
        xlsx_path = os.path.join(out_dir, "Brain_Volume.xlsx")
        write_results_workbook(xlsx_path, [sheet])
            

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




def compute_nifti_area(parent, file_path: str, out_dir: str,  valid_labels: set[int], min_contour_area: float=30,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
    cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED,
    cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2):
    """Compute brain surface area (cm²) from a NIfTI segmentation.

    Sums exterior-contour perimeters across coronal slices and multiplies
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
    print(
        f"[NIfTI Area] Perimeter method={perimeter_method}; "
        f"in-plane spacing={pixel_size_x:.4f} x {pixel_size_z:.4f} mm/px; "
        f"simplify={'on' if simplify_contours_for_perimeter else 'off'} "
        f"(epsilon={float(contour_simplify_epsilon):g} px)"
    )


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
        print("[NIfTI Area] Warning: None of the selected regions are present in this NIfTI file!")
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

    # Surface-connected cavity classification (exact 3-D). Open-cavity wall
    # perimeters are added to the lateral surface (enclosed voids stay solid).
    if cavity_correction_enabled:
        cavity_corr, _filled_mask = cavity_correction_nifti(
            brain_mask > 0, list(brain_slices), axis=1,
            pixel_size_x=float(pixel_size_x), pixel_size_z=float(pixel_size_z),
            area_threshold_mm2=float(cavity_area_threshold_mm2))
    else:
        cavity_corr, _filled_mask = CavityCorrection.empty(), None

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
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) * pixel_area_mm2 > min_contour_area]
            inner_mask_only = np.zeros_like(slice_mask)
            cv2.drawContours(inner_mask_only, filtered_contours, -1, 1, thickness=cv2.FILLED)
            inner_perimeter = mask_perimeter_mm(
                inner_mask_only, pixel_size_x, pixel_size_z,
                method=perimeter_method,
                simplify=simplify_contours_for_perimeter,
                epsilon=contour_simplify_epsilon,
            )
            # Add the open-cavity wall perimeter to the surface lateral.
            inner_perimeter += cavity_corr.perim_add(idx)


            # Save perimeters
            sum_inner += inner_perimeter

            # Create a grayscale image for visualization
            annotated = np.stack([slice_mask * 255] * 3, axis=-1)  # Convert binary mask to RGB
            annotated = annotated.reshape((annotated.shape[0], annotated.shape[1], 3))
            h_img, w_img = annotated.shape[:2]
            thickness, _, _ = image_annotation_style(h_img, w_img, style="regular")
            cv2.drawContours(annotated, filtered_contours, -1, (0, 0, 255), thickness)  # Red contours (original)
            # Yellow-outline surface-connected cavities.
            _cav_cnts = cavity_corr.surface_connected_by_idx.get(idx, [])
            if _cav_cnts:
                cv2.drawContours(annotated, _cav_cnts, -1, (0, 255, 255), thickness)

            sheet1.append([idx, inner_perimeter])

            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)


        Area = sum_inner * pixel_size_y/ 100
        print(f"[NIfTI Area] Surface area: lateral+caps={Area:.3f} cm²")
        if cavity_correction_enabled:
            print(f"[NIfTI Area] Cavity correction: "
                  f"{cavity_corr.n_surface_connected} surface-connected, "
                  f"{cavity_corr.n_enclosed} enclosed; "
                  f"wall perim added={cavity_corr.total_wall_perim_mm / 10:.3f} cm")


        from helpers.results_excel_format import build_measurement_sheet, write_results_workbook
        results_rows = [{"Section": r[0], "Perimeter": r[1]} for r in sheet1]
        parameters = {
            "Perimeter method": perimeter_method,
            "Pixel spacing": (
                f"{float(pixel_size_x):.4f} × {float(pixel_size_z):.4f} mm "
                "(in-plane)"),
            "Slice thickness": float(pixel_size_y),
            "Isotropic spacing used": perimeter_method == "crofton",
            "Contour simplification enabled": bool(simplify_contours_for_perimeter),
            "Contour simplification epsilon": float(contour_simplify_epsilon),
            "Cavity correction": "on" if cavity_correction_enabled else "off",
        }
        if cavity_correction_enabled:
            parameters["Cavity area threshold (mm²)"] = float(cavity_area_threshold_mm2)
        totals = {"Surface Area (cm^2)": round(float(Area), 4)}
        if cavity_correction_enabled:
            totals.update({
                "Number of surface-connected cavities": int(cavity_corr.n_surface_connected),
                "Number of enclosed cavities": int(cavity_corr.n_enclosed),
                "Cavity wall perimeter added (cm)": round(float(cavity_corr.total_wall_perim_mm / 10), 4),
            })
        sheet = build_measurement_sheet(file_path, "Area", results_rows, parameters, totals)
        xlsx_path = os.path.join(out_dir, "Brain_Surface_Area.xlsx")
        write_results_workbook(xlsx_path, [sheet])
            

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
    
    

def compute_nifti_lGI(parent, file_path: str, out_dir: str,  valid_labels: set[int], min_contour_area: float=30, kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    perimeter_method: str = DEFAULT_PERIMETER_METHOD,
    simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON):
    """Compute the gyrification index (GI) from a NIfTI segmentation.

    GI = total exterior perimeter / total closed-envelope perimeter across all coronal
    slices.  The "outer" contour is obtained by morphologically closing
    each slice mask to fill sulci.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the NIfTI file.
        out_dir: Output directory for slice PNGs and Excel.
        valid_labels: Set of integer segmentation labels to include.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size_mm: Morph-close kernel diameter, in mm.

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
    kernel_size_px = _to_kernel_px(kernel_size_mm, max(pixel_size_x, pixel_size_z))

    print(f"[NIfTI lGI] voxel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} x {pixel_size_z:.4f} mm")
    print(
        f"[NIfTI lGI] Perimeter method={perimeter_method}; "
        f"in-plane spacing={pixel_size_x:.4f} x {pixel_size_z:.4f} mm/px; "
        f"simplify={'on' if simplify_contours_for_perimeter else 'off'} "
        f"(epsilon={float(contour_simplify_epsilon):g} px)"
    )


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
        print("[NIfTI lGI] Warning: None of the selected regions are present in this NIfTI file!")
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
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) * pixel_area_mm2 > min_contour_area]
            # Outer contour: rebuild a mask from ONLY the kept inner contours
            # so noise rejected by the inner filter cannot produce spurious
            # outer components after morph-close.
            inner_mask_only = np.zeros_like(slice_mask)
            cv2.drawContours(inner_mask_only, filtered_contours, -1, 1, thickness=cv2.FILLED)
            inner_perimeter = mask_perimeter_mm(
                inner_mask_only, pixel_size_x, pixel_size_z,
                method=perimeter_method,
                simplify=simplify_contours_for_perimeter,
                epsilon=contour_simplify_epsilon,
            )
            closed_mask = cv2.morphologyEx(inner_mask_only, cv2.MORPH_CLOSE, compute_kernel_convex(kernel_size_px))
            outer_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_outer_contours = [cnt for cnt in outer_contours if cv2.contourArea(cnt) * pixel_area_mm2 > min_contour_area]
            outer_mask_only = np.zeros_like(closed_mask)
            cv2.drawContours(outer_mask_only, filtered_outer_contours, -1, 1, thickness=cv2.FILLED)
            outer_perimeter = mask_perimeter_mm(
                outer_mask_only, pixel_size_x, pixel_size_z,
                method=perimeter_method,
                simplify=simplify_contours_for_perimeter,
                epsilon=contour_simplify_epsilon,
            )
                
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

                
            sheet1.append([idx, kernel_size_px, inner_perimeter, outer_perimeter, GI_slice])
                    
            slice_path = os.path.join(out_dir_slices, f"brain_slice_{idx:03d}.png")
            cv2.imwrite(slice_path, annotated)
            saved_pngs.append(slice_path)



#        brain_length_cm = ((valid_slices[-1] - valid_slices[0] + 1) * pixel_size_y) /10
#        brain_volume = (sum_area * pixel_size_y)/1000
        GI_total = sum_inner / sum_outer if sum_outer > 0 else 0


        from helpers.results_excel_format import build_measurement_sheet, write_results_workbook
        results_rows = [{
            "Section": r[0], "Kernel px": r[1],
            "Perimeter": r[2], "Closed-envelope perimeter": r[3], "LGI": r[4],
        } for r in sheet1]
        parameters = {
            "Kernel size (mm)": float(kernel_size_mm),
            "Perimeter method": perimeter_method,
            "Contour simplification enabled": bool(simplify_contours_for_perimeter),
            "Contour simplification epsilon": float(contour_simplify_epsilon),
        }
        totals = {"GI": round(float(GI_total), 4)}
        sheet = build_measurement_sheet(
            file_path, "LGI", results_rows, parameters, totals,
            extra_columns=("Kernel px", "Closed-envelope perimeter"))
        xlsx_path = os.path.join(out_dir, "Brain_lGI.xlsx")
        write_results_workbook(xlsx_path, [sheet])
            

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
        print("[NIfTI Sulci depth] Warning: None of the selected regions are present in this NIfTI file!")
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
            filtered_contours = [cnt for cnt in inner_contours if cv2.contourArea(cnt) * pixel_area_mm2 > min_contour_area]
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
            print(f"[NIfTI Sulci depth] slice {idx}: {format_sulcus_class_summary(depth_sets)}")
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

        from helpers.results_excel_format import (
            build_measurement_sheet, write_results_workbook, subtype_mean)
        results_rows = []
        for r, dsets in zip(sheet1, slice_class_data):
            d = dsets if isinstance(dsets, dict) else {}
            results_rows.append({
                "Section": r[0],
                "PrimarySulciCount": len(d.get("primary", []) or []),
                "SecondarySulciCount": len(d.get("secondary", []) or []),
                "TertiarySulciCount": len(d.get("tertiary", []) or []),
                "UnclassifiedSulciCount": len(d.get("unclassified", []) or []),
                "PrimaryMeanDepth": subtype_mean(None, d.get("primary", []) or []),
                "SecondaryMeanDepth": subtype_mean(None, d.get("secondary", []) or []),
                "TertiaryMeanDepth": subtype_mean(None, d.get("tertiary", []) or []),
                "UnclassifiedMeanDepth": subtype_mean(None, d.get("unclassified", []) or []),
            })
        overall_n = len(total_depth_mm)
        overall_mean = (sum(total_depth_mm) / overall_n) if overall_n else None
        totals = {
            "Total sulci count": int(overall_n),
            "Min sulci depth (mm)": (round(float(min(total_depth_mm)), 4) if total_depth_mm else None),
            "Max sulci depth (mm)": (round(float(max(total_depth_mm)), 4) if total_depth_mm else None),
            "Mean sulci depth (mm)": (round(float(overall_mean), 4) if overall_mean is not None else None),
        }
        sheet = build_measurement_sheet(file_path, "Sulci depth", results_rows, {}, totals)
        xlsx_path = os.path.join(out_dir, "Brain_Sulci.xlsx")
        write_results_workbook(xlsx_path, [sheet])
            

        # Step 3: Apply Mask & Save Extracted Brain NIfTI
        filtered_mask = brain_mask[:, valid_slices, :]
        brain_nii = nib.Nifti1Image(filtered_mask, affine=affine, header=header)
        
        brain_extracted = os.path.join(out_dir, "brain_extracted.nii.gz")
        nib.save(brain_nii, brain_extracted)
        print("[NIfTI Sulci depth] Brain-extracted NIfTI file saved as 'brain_extracted.nii.gz'")
        
        return  dims, total_depth, saved_pngs, valid_slices

    else:
        QMessageBox.information(parent, "[NIfTI Sulci depth]", "All slices were filtered out (too small).")
        return
