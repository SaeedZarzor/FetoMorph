"""STL mesh measurement functions for FetoMorph.

Pipeline: STL mesh → PyVista off-screen slice renders → OpenCV contour
analysis.  A red reference cube (10 % of Y extent) is rendered alongside
each cross-section and detected in the screenshot to calibrate mm/px.

Red-cube contours are excluded from brain measurements via
``contours_exclude`` so they don't pollute area / perimeter values.

Unit conversions:
    * ``/1000``: mm³ → cm³  (volume)
    * ``/100``:  mm² → cm²  (surface area)
    * ``/10``:   mm  → cm   (depth)
"""

from __future__ import annotations

from deps import *
from helpers.helpers import (
    compute_kernel_convex,
    contours_exclude,
    calc_scale_with_metadata,
    get_red_rect_offset,
    make_scale_cube,
    prepare_orthographic_slice_render,
    slice_at,
    validate_scale_cube_sanity,
    compactness_2D,
    compactness_3D,
    frustum_surface_area,
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
from helpers.slice_kind_classifier import classify_slice_kind
from helpers.check_mesh import check_brain
from managers.visualization_settings import get_active as _get_viz
from constants import (
    BINARY_THRESHOLD_DEFAULT,
    DEFAULT_KERNEL_SIZE_MM,
    DEFECT_FIXED_POINT,
    SULCUS_TERTIARY_MIN_FRACTION,
    SULCUS_PRIMARY_MAX_FRACTION,
)

logger = logging.getLogger("fetomorph.stl")


def _to_kernel_px(kernel_size_mm: float, pixel_size_mm: float) -> int:
    px = max(3, int(round(float(kernel_size_mm) / max(float(pixel_size_mm), 1e-9))))
    return px


def _stl_slice_setup(mesh, brain_dim, Slice_direction: str, pixel_spacing: float = 0.1):
    """Compute direction-dependent slicing parameters for STL rendering.

    Returns a dict with keys: axis_index, pre_axis, next_axis, low, high,
    image_width, image_height, view_fn_name, cube_len, max_dim.
    """
    axis_index = {"X": 0, "Y": 1, "Z": 2}[Slice_direction]
    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3

    bounds_pairs = list(zip(mesh.bounds[::2], mesh.bounds[1::2]))
    low, high = bounds_pairs[axis_index]

    image_width = int(np.clip(np.ceil(brain_dim[pre_axis] / pixel_spacing), 64, 4096))
    image_height = int(np.clip(np.ceil(brain_dim[next_axis] / pixel_spacing), 64, 4096))
    view_fn_name = {"X": "view_yz", "Y": "view_xz", "Z": "view_xy"}[Slice_direction]
    cube_len = max(1e-6, brain_dim[axis_index] / 10.0)
    max_dim = max(brain_dim[pre_axis], brain_dim[next_axis])

    return {
        "axis_index": axis_index,
        "pre_axis": pre_axis,
        "next_axis": next_axis,
        "low": low,
        "high": high,
        "image_width": image_width,
        "image_height": image_height,
        "view_fn_name": view_fn_name,
        "cube_len": cube_len,
        "max_dim": max_dim,
    }


def _require_scale_calibration(
    p,
    Slice_direction: str,
    cube_len: float,
    mesh_center,
    sample_slice: float,
    offset: float,
    view_fn_name: str,
    *,
    physical_cube_len: float | None = None,
) -> dict:
    scale_cube = make_scale_cube(Slice_direction, cube_len, mesh_center, sample_slice, offset)
    ok, metadata = validate_scale_cube_sanity(
        p,
        scale_cube,
        float(physical_cube_len if physical_cube_len is not None else cube_len),
        getattr(p, view_fn_name),
        background="white",
    )
    if not ok:
        raise ValueError(
            "Scale cube calibration failed before processing slices: "
            f"{metadata.get('calibration_status')}"
        )
    return metadata


# ----------------- main API -----------------
def compute_stl_allmarks(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y"):
    """Compute all hallmarks (volume, area, GI, sulci depths) from an STL mesh.

    Slices the mesh along Y, renders each cross-section with a red scale
    cube, then runs contour analysis on the screenshots.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory for slice PNGs and Excel.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size_mm: Morph-close kernel diameter for outer contour, in mm.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, area_cm2, volume_cm3, GI, depths_cm,
        saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    dic = check_brain(file_path)
    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],0,0,0,[],[],[]
        
    mesh = pv.read(str(file_path))
    print(f"[STL All Hallmarks] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL All Hallmarks] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL All Hallmarks] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL All Hallmarks] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL All Hallmarks] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    kernel_px_values: list[int] = []
    slice_class_data: list = []
    sections_list: list[pv.PolyData] = []
    total_depth = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0
    sum_area = 0.0

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    max_dim = sd["max_dim"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        max_dim, sd["view_fn_name"],
    )
    calibration_metadata: list[dict] = []
    invalid_calibration_rows: list[dict] = []
    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, max_dim)

        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(section, color="black", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            invalid_calibration_rows.append({
                "Section": idx,
                "cube_len_mm": scale_meta.get("cube_len_mm"),
                "detected_cube_size_px": scale_meta.get("detected_cube_size_px"),
                "computed_mm_per_px": scale_meta.get("computed_mm_per_px"),
                "projection_mode": scale_meta.get("projection_mode"),
                "cube_detection_method": scale_meta.get("cube_detection_method"),
                "calibration_status": scale_meta.get("calibration_status"),
                "calibration_error_percent": scale_meta.get("calibration_error_percent"),
            })
            continue
        calibration_metadata.append(scale_meta)
        kernel_size_px = _to_kernel_px(kernel_size_mm, mm_per_px)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Outer contours: rebuild a mask from ONLY the kept inner contours so
        # noise blobs rejected by the inner filter can't produce spurious outer
        # components after morph-close.
        inner_mask = np.zeros_like(bw)
        cv2.drawContours(inner_mask, inner_filtered, -1, 255, thickness=cv2.FILLED)
        kernel = compute_kernel_convex(kernel_size_px)
        closed = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel)
        outer_candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_filtered = [c for c in outer_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, outer_filtered, -1, tuple(_get_viz().contour_outer_color_bgr), thickness)

        # Perimeters (mm)
        area_px        = sum(cv2.contourArea(c)     for c in inner_filtered)
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        outer_perim_px = sum(cv2.arcLength(c, True) for c in outer_filtered)
        inner_perim_mm = inner_perim_px * mm_per_px
        outer_perim_mm = outer_perim_px * mm_per_px
        area_mm  = area_px * (mm_per_px ** 2)

        # Classify the rendered slice and gate the percent filter on it.
        # Slices that don't look like a full MRI fall back to the original
        # fixed-millimeter rule and stay "unclassified".
        slice_kind, slice_kind_conf = classify_slice_kind(bgr)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        # Sulci classification: bin each kept defect by its depth as a
        # fraction of `max_dim` (longest brain extent in mm).
        depth_sets = empty_depth_sets()
        if inner_filtered:
            for cnt in inner_filtered:
                hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)  # Compute convex hull
                if hull is not None and np.all(np.diff(hull.ravel()) > 0) and len(hull) >= 3 and len(cnt) > 3:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            bgr = cv2.line(bgr, start, end, list(_get_viz().measurement_line_color_bgr), thickness)
                            if d > DEFECT_FIXED_POINT:
                                depth_mm = d * mm_per_px / DEFECT_FIXED_POINT
                                if use_percent_filter:
                                    keep = (SULCUS_TERTIARY_MIN_FRACTION * max_dim) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * max_dim)
                                else:
                                    keep = depth_mm > 0.5
                                if keep:
                                    if use_percent_filter:
                                        sulcus_class = classify_sulcus_depth(depth_mm, max_dim)
                                    else:
                                        sulcus_class = "unclassified"
                                    marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                    depth_sets[sulcus_class].append(depth_mm)
                                    bgr = cv2.circle(bgr, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        print(f"[STL allmarks] slice {idx}: kind={slice_kind} ({slice_kind_conf:.2f}), {format_sulcus_class_summary(depth_sets)}")
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns("mm"))
            slice_class_data.append(None)
        rows.append([
            idx, slice_kind, len(inner_filtered), area_mm, inner_perim_mm, outer_perim_mm,
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_mm
            (max(depth) if depth else None),    # max_depth_mm
            mean_depth,                         # mean_depth_mm
            *per_class_cells,
        ])
        kernel_px_values.append(kernel_size_px)
        # Accumulate
        sum_inner_mm += inner_perim_mm
        sum_outer_mm += outer_perim_mm
        sum_area     += area_mm
#        GI_slice = (inner_perim_mm / outer_perim_mm) if outer_perim_mm > 0 else 0.0

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(plane)
        sections_list.append(section)

    # ---- end with: plotter is fully and safely closed here ----

    # Totals
    brain_volume = (sum_area * slice_thickness_eff)/1000
    Area = sum_inner_mm * slice_thickness_eff /100
    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
    comp_3D  = compactness_3D(brain_volume, Area) 
    
    # Snapshot mm-unit depths for the per-mm Excel summary row before
    # the in-place cm conversion below.
    total_depth_mm = list(total_depth)
    total_depth = [x/10 for x in total_depth]

    mean_total = (sum(total_depth)/ len(total_depth))  if len(total_depth)>0 else None
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()

    # Save per-slice + totals to Excel using the shared spec layout.
    try:
        from helpers.results_excel_format import (
            ResultsSheet, write_results_workbook, subtype_mean,
        )

        overall_n = len(total_depth_mm)
        overall_mean = (sum(total_depth_mm) / overall_n) if overall_n else None

        results_rows = []
        for r, dsets, png_path, kernel_px, cal_meta in zip(rows, slice_class_data, saved_pngs, kernel_px_values, calibration_metadata):
            idx, _kind, _ncont, area_mm, inner_perim_mm, outer_perim_mm = r[:6]
            lgi = ((inner_perim_mm / outer_perim_mm)
                   if outer_perim_mm else None)
            compact = (compactness_2D(area_mm, inner_perim_mm)
                       if inner_perim_mm else None)
            d = dsets if isinstance(dsets, dict) else {}
            results_rows.append({
                "Section": idx,
                "Kernel_px": int(kernel_px),
                "cube_len_mm": cal_meta.get("cube_len_mm"),
                "detected_cube_size_px": cal_meta.get("detected_cube_size_px"),
                "computed_mm_per_px": cal_meta.get("computed_mm_per_px"),
                "projection_mode": cal_meta.get("projection_mode"),
                "cube_detection_method": cal_meta.get("cube_detection_method"),
                "calibration_status": cal_meta.get("calibration_status"),
                "calibration_error_percent": cal_meta.get("calibration_error_percent"),
                "Area": area_mm,
                "Perimeter": inner_perim_mm,
                "LGI": lgi,
                "Compactness": compact,
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
                "_section_link": png_path,
            })
        results_rows.extend(invalid_calibration_rows)

        parameters = {
            "Kernel size (mm)": float(kernel_size_mm),
            "Kernel size (px)": (
                f"per-slice, range {min(kernel_px_values)}-{max(kernel_px_values)}"
                if kernel_px_values else None),
            "Pixel spacing": "0.1 mm/pixel (rendering); per-slice from cube",
            "Slice thickness": float(slice_thickness),
            "Filtered threshold": float(min_contour_area),
            "Slice direction": Slice_direction,
            "projection_mode": "parallel",
            "cube_detection_method": "HSV red mask + morphology + minAreaRect",
            "calibration_status": "valid" if calibration_metadata else "no valid calibrated slices",
        }
        totals = {
            "Volume (cm^3)": round(float(brain_volume), 2),
            "Surface Area (cm^2)": round(float(Area), 2),
            "GI": round(float(GI_total), 4),
            "Compactness": round(float(comp_3D), 4),
            "Total sulci count": int(overall_n),
            "Mean sulci depth (mm)": (round(float(overall_mean), 3)
                                       if overall_mean is not None else None),
        }
        sheet = ResultsSheet(
            sheet_name=os.path.basename(file_path) or "Results",
            file_name=os.path.basename(file_path),
            folder=os.path.dirname(file_path) or None,
            parameters=parameters,
            rows=results_rows,
            extra_columns=(
                "Kernel_px",
                "cube_len_mm",
                "detected_cube_size_px",
                "computed_mm_per_px",
                "projection_mode",
                "cube_detection_method",
                "calibration_status",
                "calibration_error_percent",
            ),
            totals=totals,
            drop_empty_columns=True,
        )
        xlsx_path = os.path.join(out_dir, "Mesh_Allmarks.xlsx")
        write_results_workbook(xlsx_path, [sheet])
        print(f"[STL All Hallmarks] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL All Hallmarks] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, Area, brain_volume, GI_total, comp_3D ,total_depth ,saved_pngs, valid_slices 


def compute_stl_lGI(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM,
    slice_thickness: float = 0.5,
    build_solid: bool = False,
    Slice_direction: str = "Y",
):
    """Compute the gyrification index (GI) from an STL mesh via slice rendering.

    GI = total inner perimeter / total outer perimeter across all Y slices.
    Optionally builds an extruded 3-D solid from the outer contours.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size_mm: Morph-close kernel diameter, in mm.
        slice_thickness: Distance between slices (mm).
        build_solid: If True, extrude outer contours into a solid mesh.

    Returns:
        Tuple of ``(label, dims_cm, GI_total, saved_pngs, valid_slices)``.
    """
    
    dic = check_brain(file_path)
    

    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],0,[],[]
    # --- Load mesh
    mesh = pv.read(str(file_path))
    print(f"[STL lGI] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL lGI] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL lGI] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"],[],0,[],[]

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL lGI] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL lGI] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    all_3d_contours = []
    sections_list: list[pv.PolyData] = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        sd["max_dim"], sd["view_fn_name"],
    )

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, sd["max_dim"])

        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(section, color="black", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue
        kernel_size_px = _to_kernel_px(kernel_size_mm, mm_per_px)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Outer contours: rebuild a mask from ONLY the kept inner contours so
        # noise blobs rejected by the inner filter can't produce spurious outer
        # components after morph-close.
        inner_mask = np.zeros_like(bw)
        cv2.drawContours(inner_mask, inner_filtered, -1, 255, thickness=cv2.FILLED)
        kernel = compute_kernel_convex(kernel_size_px)
        closed = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel)
        outer_candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_filtered = [c for c in outer_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, outer_filtered, -1, tuple(_get_viz().contour_outer_color_bgr), thickness)

        # Perimeters (mm)
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        outer_perim_px = sum(cv2.arcLength(c, True) for c in outer_filtered)
        inner_perim_mm = inner_perim_px * mm_per_px
        outer_perim_mm = outer_perim_px * mm_per_px

        # Accumulate
        sum_inner_mm += inner_perim_mm
        sum_outer_mm += outer_perim_mm
        GI_slice = (inner_perim_mm / outer_perim_mm) if outer_perim_mm > 0 else 0.0
        rows.append([idx, len(inner_filtered), kernel_size_px, inner_perim_mm, outer_perim_mm, GI_slice])

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)

            # Store 3D outer contours in mm coords for optional solid build
        if build_solid and outer_filtered:
            offset_2d = get_red_rect_offset(img_rgb)  # center of red cube in px
            for cnt in outer_filtered:
                pts = cnt.squeeze()
                if pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                # XZ from pixels -> mm; Y from slice index
                aligned = pts - offset_2d
                cnt_3d = np.column_stack([
                    aligned[:, 0] * mm_per_px,                       # X
                    np.full((pts.shape[0],), idx * slice_thickness_eff),  # Y
                    aligned[:, 1] * mm_per_px,                       # Z
                ])
                all_3d_contours.append(cnt_3d)

    # ---- end with: plotter is fully and safely closed here ----

    # Totals
    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
    print(f"[STL lGI] GI_total = {GI_total:.6f}")
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
        
    # Save per-slice + total to Excel
    try:
        df = pd.DataFrame(rows, columns=["Slice", "Count_of_cont.", "Kernel_px", "Inner_Perimeter_mm", "Outer_Perimeter_mm", "GI"])
        df.insert(0, "Kernel_size_mm", float(kernel_size_mm))
        df.loc[len(df)] = [float(kernel_size_mm), "GI", None, None, None, None, round(GI_total, 3)]
        xlsx_path = os.path.join(out_dir, "STL_lGI.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[STL lGI] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL lGI] WARN: could not save Excel: {ex}")

    # Optional: build extruded solid (can be fragile on some macOS OpenGL stacks)
    if build_solid and all_3d_contours:
        try:
            solids = []
            for cnt in all_3d_contours:
                if cnt.shape[0] < 3:
                    continue
                n = cnt.shape[0]
                faces = np.hstack(([n], np.arange(n, dtype=np.int64))).astype(np.int64)
                surf = pv.PolyData(cnt, faces=faces).triangulate()
                extruded = surf.extrude([0, slice_thickness_eff, 0], capping=True)
                solids.append(extruded)
            if solids:
                merged = solids[0].copy()
                for part in solids[1:]:
                    merged = merged.merge(part)
                solid_path = os.path.join(out_dir, "solid_volume_convex.stl")
                merged.save(solid_path)
                print(f"[STL lGI] Saved extruded solid → {solid_path}")
        except Exception as ex:
            print(f"[STL lGI] NOTE: solid build skipped: {ex}")

    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, float(GI_total), saved_pngs, valid_slices


def compute_stl_volume(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y"):
    """Compute brain volume (cm³) from an STL mesh by integrating slice cross-section areas.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, volume_cm3, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    dic = check_brain(file_path)
    
    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],0,[],[]
        
    mesh = pv.read(str(file_path))
    print(f"[STL Volume] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL Volume] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL Volume] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"],[],0,[],[]

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL Volume] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Volume] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sections_list: list[pv.PolyData] = []
    sum_area = 0.0

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        sd["max_dim"], sd["view_fn_name"],
    )

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, sd["max_dim"])
        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(section, color="black", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Perimeters (mm)
        area_px  = sum(cv2.contourArea(c)     for c in inner_filtered)
        area_mm  = area_px * (mm_per_px ** 2)


        rows.append([idx, len(inner_filtered), area_mm])
        # Accumulate
        sum_area     += area_mm

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)

      
    # ---- end with: plotter is fully and safely closed here ----

    # Totals
    brain_volume = (sum_area * slice_thickness_eff)/1000
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
        
    # Save per-slice + total to Excel
    try:
        rows.append(["Volume cm^3",round(brain_volume,2)])

        df = pd.DataFrame(rows, columns=["Slice", "Count_of_cont.", "Inner_area_mm^2"])

        
        xlsx_path = os.path.join(out_dir, "Mesh_Volume.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[STL Volume] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL Volume] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, brain_volume,saved_pngs, valid_slices


def compute_stl_area(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y"):
    """Compute brain surface area (cm²) from an STL mesh.

    Sums inner-contour perimeters across Y slices and multiplies by
    effective slice thickness.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, area_cm2, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    dic = check_brain(file_path)

    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],0,[],[]
        
    mesh = pv.read(str(file_path))
    print(f"[STL Area] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL Area] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL Area] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"],[],0,[],[]

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL Area] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Area] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sum_inner_mm = 0.0
    frustum_slices: list[tuple[float, float, float]] = []  # (h_mm, perimeter_mm, area_mm2)
    sections_list: list[pv.PolyData] = []

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        sd["max_dim"], sd["view_fn_name"],
    )

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, sd["max_dim"])
        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(section, color="black", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)


        # Perimeters (mm)
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        inner_perim_mm = inner_perim_px * mm_per_px

        # Cross-sectional area (mm²) for the frustum surface approximation
        inner_area_px = sum(cv2.contourArea(c) for c in inner_filtered)
        inner_area_mm2 = inner_area_px * (mm_per_px ** 2)
        frustum_slices.append((float(k), inner_perim_mm, inner_area_mm2))

        rows.append([idx, len(inner_filtered), inner_perim_mm])
        # Accumulate
        sum_inner_mm += inner_perim_mm


        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)

      
    # ---- end with: plotter is fully and safely closed here ----

    # Totals — frustum-with-caps surface area (mm²), then /100 → cm².
    # The legacy stack-of-slabs lateral area (Σ perimeter × thickness) is kept
    # for comparison in `sa_meta["surface_area_lateral_old"]`.
    sa_total_mm2, sa_meta = frustum_surface_area(
        frustum_slices, legacy_slice_thickness=slice_thickness_eff)
    for w in sa_meta["warnings"]:
        print(f"[STL Area] WARN: {w}")
    if sa_total_mm2 is None:
        print("[STL Area] WARN: no valid slices for frustum surface; reporting 0.")
        Area = 0.0
    else:
        Area = sa_total_mm2 / 100
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
        
    # Save per-slice + total to Excel
    try:
        rows.append(["Surface Area cm^2",round(Area,2)])
        rows.append(["surface_area_method", sa_meta["surface_area_method"]])
        rows.append(["surface_area_frustum_total_mm2", sa_meta["surface_area_frustum_total"]])
        rows.append(["surface_area_frustum_lateral_mm2", sa_meta["surface_area_frustum_lateral"]])
        rows.append(["surface_area_caps_mm2", sa_meta["surface_area_caps"]])
        rows.append(["surface_area_lateral_old_mm2", sa_meta["surface_area_lateral_old"]])
        rows.append(["top_cap_area_mm2", sa_meta["top_cap_area"]])
        rows.append(["bottom_cap_area_mm2", sa_meta["bottom_cap_area"]])
        rows.append(["number_of_valid_slices", sa_meta["number_of_valid_slices"]])
        rows.append(["slice_spacing_mode", sa_meta["slice_spacing_mode"]])
        df = pd.DataFrame(rows, columns=["Slice","Count_of_cont.", "Inner_Perimeter_mm"])

        xlsx_path = os.path.join(out_dir, "Mesh_Area.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[STL Area] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL Area] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, Area, saved_pngs, valid_slices



def compute_stl_sulci_depth(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y"):
    """Compute sulci depths from convexity defects across STL mesh slices.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, depths_mm, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    dic = check_brain(file_path)
    
    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent,"Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"],[],[],[],[]
        
    mesh = pv.read(str(file_path))
    print(f"[STL Sulci depth] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL Sulci depth] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL Sulci depth] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"],[],[],[],[]

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL Sulci depth] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Sulci depth] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    slice_class_data: list = []
    total_depth = []
    sections_list: list[pv.PolyData] = []

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    max_dim = sd["max_dim"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        max_dim, sd["view_fn_name"],
    )
    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, max_dim)
        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]]*1.5, j_size=brain_dim[sd["next_axis"]]*1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(section, color="black", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Classify the rendered slice and gate the percent filter on it.
        slice_kind, slice_kind_conf = classify_slice_kind(bgr)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        # Sulci classification: bin each kept defect by its depth as a
        # fraction of `max_dim` (longest brain extent in mm).
        depth_sets = empty_depth_sets()
        if inner_filtered:
            for cnt in inner_filtered:
                hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)  # Compute convex hull
                if hull is not None and np.all(np.diff(hull.ravel()) > 0) and len(hull) >= 3 and len(cnt) > 3:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            bgr = cv2.line(bgr, start, end, list(_get_viz().measurement_line_color_bgr), thickness)
                            if d > DEFECT_FIXED_POINT:
                                mm_per_fixed = mm_per_px / DEFECT_FIXED_POINT
                                depth_mm = d *mm_per_fixed
                                if use_percent_filter:
                                    keep = (SULCUS_TERTIARY_MIN_FRACTION * max_dim) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * max_dim)
                                else:
                                    keep = depth_mm > 0.5
                                if keep:
                                    if use_percent_filter:
                                        sulcus_class = classify_sulcus_depth(depth_mm, max_dim)
                                    else:
                                        sulcus_class = "unclassified"
                                    marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                    depth_sets[sulcus_class].append(depth_mm)
                                    bgr = cv2.circle(bgr, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        print(f"[STL sulci] slice {idx}: kind={slice_kind} ({slice_kind_conf:.2f}), {format_sulcus_class_summary(depth_sets)}")
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns("mm"))
            slice_class_data.append(None)
        rows.append([
            idx, slice_kind, len(inner_filtered),
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_mm
            (max(depth) if depth else None),    # max_depth_mm
            mean_depth,                         # mean_depth_mm
            *per_class_cells,
        ])


        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)
      
    # ---- end with: plotter is fully and safely closed here ----

    mean_total = (sum(total_depth)/ len(total_depth))  if len(total_depth)>0 else None
    
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
        
    # Save per-slice + total to Excel
    try:
        per_class_cols = sulcus_export_columns("mm")
        base_cols = [
            "Slice", "SliceKind", "Count_of_cont.", "Sulci_count",
            "min_depth_mm", "max_depth_mm", "mean_depth_mm",
        ]
        total_width = len(base_cols) + len(per_class_cols)

        overall_depth_sets = empty_depth_sets()
        for dsets in slice_class_data:
            if dsets is None:
                continue
            for k in SULCUS_CLASSES:
                overall_depth_sets[k].extend(dsets.get(k, []))

        overall_n = len(total_depth)
        overall_min = min(total_depth) if total_depth else None
        overall_max = max(total_depth) if total_depth else None
        overall_mean = (sum(total_depth) / overall_n) if overall_n else None

        summary_base = [None] * len(base_cols)
        summary_base[0] = "Sulci_overall_summary"
        summary_base[base_cols.index("Sulci_count")]   = overall_n
        summary_base[base_cols.index("min_depth_mm")]  = overall_min
        summary_base[base_cols.index("max_depth_mm")]  = overall_max
        summary_base[base_cols.index("mean_depth_mm")] = overall_mean
        rows.append(pad_row(
            [*summary_base, *sulcus_export_cells(overall_depth_sets)],
            total_width,
        ))

        df = pd.DataFrame(rows, columns=[*base_cols, *per_class_cols])
        df = drop_empty_columns(df)

        xlsx_path = os.path.join(out_dir, "Mesh_Sulci_depth.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[STL Sulci depth] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL Sulci depth] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, total_depth ,saved_pngs, valid_slices


def compute_compactness_stl(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    slice_thickness: float = 0.5,
    Slice_direction: str = "Y"):
    """Compute 3D compactness (sphericity) from an STL mesh.

    Combines per-slice area and perimeter accumulation to derive volume (cm³)
    and surface area (cm²), then applies ``compactness_3D(V, SA)``.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        slice_thickness: Distance between slices (mm).

    Returns:
        Tuple of ``(label, dims_cm, compactness, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    dic = check_brain(file_path)

    if dic["label"] == "not_brain":
        reply = QMessageBox.question(parent, "Check measurement",
            "The imported mesh does not represent a human brain. Do you want to continue processing it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return dic["label"], [], 0, [], []

    mesh = pv.read(str(file_path))
    print(f"[STL Compactness] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL Compactness] mesh dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim / 10 for dim in brain_dim]
    brain_dim_cm = sorted(brain_dim_cm, reverse=True)

    # --- Direction-dependent slicing setup
    sd = _stl_slice_setup(mesh, brain_dim, Slice_direction)

    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(sd["low"], sd["high"], slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print(f"[STL Compactness] No slices to process (thickness too large vs. {Slice_direction} range).")
        return dic["label"], [], 0, [], []

    slice_thickness_eff = brain_dim[sd["axis_index"]] / N
    print(f"[STL Compactness] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Compactness] Temp output dir: {out_dir}")

    window_size = (sd["image_width"], sd["image_height"])

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sections_list: list[pv.PolyData] = []
    sum_area_mm2 = 0.0
    sum_inner_mm = 0.0

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.parallel_projection = True
    cube_len = sd["cube_len"]
    _require_scale_calibration(
        p, Slice_direction, cube_len, mesh.center, float(slice_positions[0]),
        sd["max_dim"], sd["view_fn_name"],
    )

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal, origin=origin)
        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, sd["max_dim"])
        plane = pv.Plane(center=origin,
                         direction=normal,
                         i_size=brain_dim[sd["pre_axis"]] * 1.5, j_size=brain_dim[sd["next_axis"]] * 1.5,
                         i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(scale_cube, color="red", lighting=False)
        p.add_mesh(section, color="black", lighting=False)
        prepare_orthographic_slice_render(p, getattr(p, sd["view_fn_name"]))

        # Screenshot
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px, scale_meta, red_rect = calc_scale_with_metadata(img_rgb, cube_len)
        if mm_per_px is None:
            print(f"[STL Scale] Scale cube was not detected or failed validation. {scale_meta.get('calibration_status')}. Measurements for this slice were skipped.")
            continue

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)

        # Convert pixel measurements to physical units
        area_px = sum(cv2.contourArea(c) for c in inner_filtered)
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        area_mm2 = area_px * (mm_per_px ** 2)
        inner_perim_mm = inner_perim_px * mm_per_px
        comp_2D = compactness_2D(area_mm2, inner_perim_mm)

        # Accumulate
        sum_area_mm2 += area_mm2
        sum_inner_mm += inner_perim_mm

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        rows.append([idx, len(inner_filtered), comp_2D])
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)

    # ---- end of slice loop ----

    # Totals (mm → cm conversions)
    Volume = sum_area_mm2 * slice_thickness_eff / 1000   # cm³
    Area = sum_inner_mm * slice_thickness_eff / 100       # cm²
    comp_3D = compactness_3D(Volume, Area)

    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()

    # Save per-slice + total to Excel
    try:
        rows.append(["Compactness", round(comp_3D, 2)])
        rows.append([f"Volume cm^3", round(Volume, 2)])
        rows.append([f"Area cm^2", round(Area, 2)])
        df = pd.DataFrame(rows, columns=["Slice", "Count_of_cont.", "Slice_compactness"])

        xlsx_path = os.path.join(out_dir, "Mesh_Compactness.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[STL Compactness] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL Compactness] WARN: could not save Excel: {ex}")

    # Always return a 5-tuple
    return dic["label"], brain_dim_cm, comp_3D, saved_pngs, valid_slices
