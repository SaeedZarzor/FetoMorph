"""VTK mesh measurement functions for FetoMorph.

Same render-and-measure pipeline as the STL module, but adapted for VTK
meshes that may use arbitrary model units.  A ``Physical_dim`` parameter
supplies the real-world dimensions so that a ``mesh_dim_scaled`` factor
converts model units → physical mm.

Key differences from the STL pipeline:
    * ``BINARY_THRESHOLD_VTK = 150`` (vs 200 for STL) because VTK renders
      use a black background instead of white.
    * Slice direction is configurable (X / Y / Z) via cyclic axis indexing:
      ``pre_axis = (axis_index - 1) % 3``, ``next_axis = (axis_index + 1) % 3``.
"""

from __future__ import annotations

from deps import *
from helpers.helpers import (
    compute_kernel_convex,
    contours_exclude,
    split_inner_and_internal_contours,
    calc_scale,
    get_red_rect_offset,
    slice_at,
    make_scale_cube,
    compactness_3D,
    compactness_2D,
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
from helpers.check_mesh import check_brain
from helpers.slice_kind_classifier import classify_slice_kind
from managers.visualization_settings import get_active as _get_viz
from constants import (
    BINARY_THRESHOLD_VTK,
    RED_CHANNEL_MIN,
    GREEN_CHANNEL_MAX,
    DEFECT_FIXED_POINT,
    SULCUS_TERTIARY_MIN_FRACTION,
    SULCUS_PRIMARY_MAX_FRACTION,
)

logger = logging.getLogger("fetomorph.vtk")

    
# ----------------- main API -----------------
def compute_vtk_allmarks(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size: int = 5,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5,
    contour_mode: str = "outer"):
    """Compute all hallmarks (area, volume, GI, sulci depths) from a VTK mesh.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size: Morph-close kernel diameter.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(area, volume, GI, depths, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    mesh = pv.read(str(file_path))
    print(f"[VTK All Hallmarks] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    # Scale factor: converts model units to physical mm per axis.
    # E.g. if the mesh is 50 units wide but physically 100 mm, scale = 2.0.
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along the chosen axis
    if slice_thickness <= 0:
        slice_thickness = 0.05

    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)

    N = len(slice_positions)
    if N == 0:
        print(f"[VTK All Hallmarks] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective slice thickness in physical units (model units × scale).
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK All Hallmarks] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK All Hallmarks] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sections_list: list[pv.PolyData] = []
    total_depth = []
    slice_class_data: list = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0
    sum_area = 0.0

    # VTK renders use a black background → threshold is BINARY_THRESHOLD_VTK (150)
    # instead of the 200 used for white-background STL renders.
    p = pv.Plotter(off_screen=True)
    p.set_background("black")

    # Cyclic axis indexing: for a given slice axis, pre_axis and next_axis
    # give the two in-plane axes used for plane sizing and cube placement.
    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)

    for idx, k in enumerate(slice_positions):
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)

        if section.n_points == 0:
            continue

        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, mesh_dim[pre_axis]/2)

        plane = pv.Plane(center=origin,
                 direction=normal,
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,
                 i_resolution=1, j_resolution=1)
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1)
        p.add_mesh(scale_cube, color="red")
        # Select camera view matching the slice plane orientation.
        {"X": p.view_yz, "Y": p.view_xz, "Z": p.view_xy}[Slice_direction]()

        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Scale cube is rendered in physical mm (cube_len × mesh_dim_scaled[0]).
        mm_per_px = calc_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Use BINARY_THRESHOLD_VTK (150) for black-background VTK renders.
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_VTK, 255, 0)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area),
        )
        # Only highlight contours that actually contribute to the measured area.
        if contour_mode != "internal_only":
            cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)
        if contour_mode != "outer" and internal_filtered:
            cv2.drawContours(bgr, internal_filtered, -1, tuple(_get_viz().contour_internal_color_bgr), thickness)

        # Outer contours: rebuild a mask from ONLY the kept inner contours so
        # noise blobs rejected by the inner filter can't produce spurious outer
        # components after morph-close.
        inner_mask = np.zeros_like(bw)
        cv2.drawContours(inner_mask, inner_filtered, -1, 255, thickness=cv2.FILLED)
        kernel = compute_kernel_convex(max(1, int(kernel_size)))
        closed = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel)
        outer_candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_filtered = [c for c in outer_candidates if cv2.contourArea(c) > float(min_contour_area)]
        if contour_mode != "internal_only":
            cv2.drawContours(bgr, outer_filtered, -1, tuple(_get_viz().contour_outer_color_bgr), thickness)

        inner_area_px = sum(cv2.contourArea(c) for c in inner_filtered)
        internal_area_px = sum(cv2.contourArea(c) for c in internal_filtered)
        if contour_mode == "subtract":
            area_perim_px = inner_area_px - internal_area_px
        elif contour_mode == "internal_only":
            area_perim_px = internal_area_px
        else:  # "outer"
            area_perim_px = inner_area_px
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        outer_perim_px = sum(cv2.arcLength(c, True) for c in outer_filtered)
        inner_perim_mm = inner_perim_px * mm_per_px
        outer_perim_mm = outer_perim_px * mm_per_px
        area_perim_mm  = area_perim_px * (mm_per_px ** 2)


        # Classify the rendered slice and gate the percent filter on it.
        slice_kind, slice_kind_conf = classify_slice_kind(bgr)
        use_percent_filter = slice_kind != "not_full_slice" and slice_kind_conf >= 0.7
        # Sulci classification: bin each kept defect by its depth as a
        # fraction of the brain's longest physical extent (mm).
        max_dim_mm = float(max(Physical_dim)) if Physical_dim else float(max(mesh_dim))
        depth_sets = empty_depth_sets()
        if inner_filtered:
            for cnt in inner_filtered:
                hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
                if hull is not None and np.all(np.diff(hull.ravel()) > 0) and len(hull) >= 3 and len(cnt) > 3:
                    defects = cv2.convexityDefects(cnt, hull)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            # s=start, e=end, f=farthest, d=depth (8.8 fixed-point)
                            s, e, f, d = defects[i, 0]
                            start = tuple(cnt[s][0])
                            end = tuple(cnt[e][0])
                            far = tuple(cnt[f][0])
                            bgr = cv2.line(bgr, start, end, list(_get_viz().measurement_line_color_bgr), thickness)
                            if d > DEFECT_FIXED_POINT:
                                depth_mm = d * mm_per_px / DEFECT_FIXED_POINT
                                if use_percent_filter:
                                    keep = (SULCUS_TERTIARY_MIN_FRACTION * max_dim_mm) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * max_dim_mm)
                                else:
                                    keep = depth_mm > 0.5
                                if keep:
                                    if use_percent_filter:
                                        sulcus_class = classify_sulcus_depth(depth_mm, max_dim_mm)
                                    else:
                                        sulcus_class = "unclassified"
                                    marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                    depth_sets[sulcus_class].append(depth_mm)
                                    bgr = cv2.circle(bgr, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        print(f"[VTK allmarks] slice {idx}: kind={slice_kind} ({slice_kind_conf:.2f}), {format_sulcus_class_summary(depth_sets)}")
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns(unit))
            slice_class_data.append(None)
        rows.append([idx, len(inner_filtered), area_perim_mm, inner_perim_mm, outer_perim_mm,
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_{unit}
            (max(depth) if depth else None),    # max_depth_{unit}
            mean_depth,                         # mean_depth_{unit}
            *per_class_cells,
            ])
        # Accumulate
        sum_inner_mm += inner_perim_mm
        sum_outer_mm += outer_perim_mm
        sum_area     += area_perim_mm
#        GI_slice = (inner_perim_mm / outer_perim_mm) if outer_perim_mm > 0 else 0.0

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

    # Totals
    Volume = (sum_area * slice_thickness_eff)
    Area = sum_inner_mm * slice_thickness_eff
    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
    comp = compactness_3D(Volume, Area)
    
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
        base_cols = [
            "Slice", "Count_of_cont.", f"Inner_area_{unit}^2",
            f"Inner_Perimeter_{unit}", f"Outer_Perimeter_{unit}",
            "Sulci_count", f"min_depth_{unit}", f"max_depth_{unit}", f"mean_depth_{unit}",
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

        overall_n = len(total_depth)
        overall_min = min(total_depth) if total_depth else None
        overall_max = max(total_depth) if total_depth else None
        overall_mean = (sum(total_depth) / overall_n) if overall_n else None

        summary_base = [None] * len(base_cols)
        summary_base[0] = "Sulci_overall_summary"
        summary_base[base_cols.index("Sulci_count")]        = overall_n
        summary_base[base_cols.index(f"min_depth_{unit}")]  = overall_min
        summary_base[base_cols.index(f"max_depth_{unit}")]  = overall_max
        summary_base[base_cols.index(f"mean_depth_{unit}")] = overall_mean
        rows.append(pad_row(
            [*summary_base, *sulcus_export_cells(overall_depth_sets)],
            total_width,
        ))

        rows.append(pad_row([f"Volume {unit}^3", Volume, f"Surface Area {unit}^2", Area], total_width))
        rows.append(pad_row(["GI", round(GI_total, 2)], total_width))
        rows.append(pad_row(["Compactness", round(comp, 2)], total_width))

        df = pd.DataFrame(rows, columns=cols)
        df = drop_empty_columns(df)

        xlsx_path = os.path.join(out_dir, f"Mesh_Allmarks_{Slice_direction}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[VTK All Hallmarks] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK All Hallmarks] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return Area, Volume, GI_total, comp ,total_depth ,saved_pngs, valid_slices


def compute_vtk_lGI(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size: int = 5,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5):
    """Compute the gyrification index (GI) from a VTK mesh.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size: Morph-close kernel diameter.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(GI_total, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
    mesh = pv.read(str(file_path))
    print(f"[VTK lGI] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.05
        
    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)
        
    N = len(slice_positions)
    if N == 0:
        print(f"[VTK lGI] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK lGI] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK lGI] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0
    
    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True)
    p.set_background("black")
    
    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)
        
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, mesh_dim[pre_axis]/2)
        
        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)        # Render: section + scale cube
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1)
        p.add_mesh(scale_cube, color="red")
        {"X": p.view_yz, "Y": p.view_xz, "Z": p.view_xy}[Slice_direction]()

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = calc_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_VTK, 255, 0)
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
        kernel = compute_kernel_convex(max(1, int(kernel_size)))
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

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        rows.append([idx, len(inner_filtered), inner_perim_mm, outer_perim_mm, GI_slice])
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)
      
    # ---- end with: plotter is fully and safely closed here ----

    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
    # Save per-slice + total to Excel
    try:
        rows.append(["GI",round(GI_total,2)])
        df = pd.DataFrame(rows, columns=["Slice", "Count_of_cont.", f"Inner_Perimeter_{unit}", f"Outer_Perimeter_{unit}" ,"Sulci_lGI"])
        
       
        
        xlsx_path = os.path.join(out_dir, f"Mesh_lGI_{Slice_direction}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[VTK lGI] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK LGI] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return  GI_total, saved_pngs, valid_slices


def compute_vtk_volume(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 0.0,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5,
    contour_mode: str = "outer"):
    """Compute volume from a VTK mesh by integrating slice cross-section areas.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(volume, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
        
    mesh = pv.read(str(file_path))
    print(f"[VTK Volume] Loaded mesh: {mesh}")
    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)
    mesh_center = mesh.center

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.05
        
    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)
        
    N = len(slice_positions)
    if N == 0:
        print(f"[VTK Volume] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK Volume] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK Volume] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    sum_area = 0.0
    
    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True)
    p.set_background("black")
    
    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)
        
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, mesh_dim[pre_axis]/2)
        
        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)
        # Render: section + scale cube
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1)
        p.add_mesh(scale_cube, color="red")
        {"X": p.view_yz, "Y": p.view_xz, "Z": p.view_xy}[Slice_direction]()

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = calc_scale(img_rgb, cube_len*mesh_dim_scaled[0])


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_VTK, 255, 0)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area),
        )
        # Only highlight contours that actually contribute to the measured area.
        if contour_mode != "internal_only":
            cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)
        if contour_mode != "outer" and internal_filtered:
            cv2.drawContours(bgr, internal_filtered, -1, tuple(_get_viz().contour_internal_color_bgr), thickness)

        

        # Perimeters (mm)
        inner_area_px = sum(cv2.contourArea(c) for c in inner_filtered)
        internal_area_px = sum(cv2.contourArea(c) for c in internal_filtered)
        if contour_mode == "subtract":
            area_perim_px = inner_area_px - internal_area_px
        elif contour_mode == "internal_only":
            area_perim_px = internal_area_px
        else:  # "outer"
            area_perim_px = inner_area_px
        area_perim_mm  = area_perim_px * (mm_per_px ** 2)
        sum_area      += area_perim_mm

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        rows.append([idx, len(inner_filtered) ,area_perim_mm])
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)
      
    # ---- end with: plotter is fully and safely closed here ----

    # Totals
    Volume = sum_area * slice_thickness_eff
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()

    # Save per-slice + total to Excel
    try:
        
        rows.append([f"Volume {unit}^3",Volume])
        df = pd.DataFrame(rows, columns=["Slice_indx", "Count_of_cont." ,f"Inner_area_{unit}^2"])
        
        xlsx_path = os.path.join(out_dir, f"Mesh_Volume_{Slice_direction}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[VTK Volume] Saved Excel → {xlsx_path}")
        
    except Exception as ex:
        print(f"[VTK Volume] WARN: could not save Excel: {ex}")

    return Volume, saved_pngs, valid_slices

def compute_vtk_area(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5):
    """Compute surface area from a VTK mesh by summing slice perimeters.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(area, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
        
    mesh = pv.read(str(file_path))
    print(f"[VTK Area] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.05
        
    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)
        
    N = len(slice_positions)
    if N == 0:
        print(f"[VTK Area] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK Area] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK Area] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    sum_inner_mm = 0.0
    
    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True)
    p.set_background("black")
    
    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)
        
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, mesh_dim[pre_axis]/2)
        
        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)        # Render: section + scale cube
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1)
        p.add_mesh(scale_cube, color="red")
        {"X": p.view_yz, "Y": p.view_xz, "Z": p.view_xy}[Slice_direction]()

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = calc_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_VTK, 255, 0)
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
        sum_inner_mm += inner_perim_mm

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        rows.append([idx, len(inner_filtered), inner_perim_mm])
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)
    # ---- end with: plotter is fully and safely closed here ----

    # Totals
    Area = sum_inner_mm * slice_thickness_eff
    if sections_list:
        all_slices_mesh = pv.merge(sections_list)
#        all_slices_mesh.active_scalars_name = "RGB"
        slice_mesh_path = os.path.join(out_dir, "all_slices_mesh.vtk")
        all_slices_mesh.save(slice_mesh_path)
    else:
        all_slices_mesh = pv.PolyData()
    # Save per-slice + total to Excel
    try:
        rows.append([f"Surface Area {unit}^2", Area])

        df = pd.DataFrame(rows, columns=["Slice", "Count_of_cont.", f"Inner_Perimeter_{unit}"])
            
        xlsx_path = os.path.join(out_dir, f"Mesh_Area_{Slice_direction}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[VTK Area] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK Area] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return Area, saved_pngs, valid_slices



def compute_vtk_sulci_depth(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5):
    """Compute sulci depths from convexity defects across VTK mesh slices.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.vtk`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        Slice_direction: Axis to slice along.
        Physical_dim: Real-world ``[X, Y, Z]`` dimensions for unit scaling.
        unit: Label for output units.
        slice_thickness: Distance between slices in model units.

    Returns:
        Tuple of ``(depths, saved_pngs, valid_slices)``.
    """
    # --- Load mesh
    
        
    mesh = pv.read(str(file_path))
    print(f"[VTK Sulci depth] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.05
        
    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)
        
    N = len(slice_positions)
    if N == 0:
        print(f"[VTK Sulci depth] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK Sulci depth] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[VTK Sulci depth] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    total_depth = []
    slice_class_data: list = []

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True)
    p.set_background("black")

    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)

        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, mesh_dim[pre_axis]/2)

        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)        # Render: section + scale cube
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1)
        p.add_mesh(scale_cube, color="red")
        {"X": p.view_yz, "Y": p.view_xz, "Z": p.view_xy}[Slice_direction]()

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = calc_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, radius_px = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_VTK, 255, 0)
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
        # fraction of the brain's longest physical extent (mm).
        max_dim_mm = float(max(Physical_dim)) if Physical_dim else float(max(mesh_dim))
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
                                    keep = (SULCUS_TERTIARY_MIN_FRACTION * max_dim_mm) < depth_mm < (SULCUS_PRIMARY_MAX_FRACTION * max_dim_mm)
                                else:
                                    keep = depth_mm > 0.5
                                if keep:
                                    if use_percent_filter:
                                        sulcus_class = classify_sulcus_depth(depth_mm, max_dim_mm)
                                    else:
                                        sulcus_class = "unclassified"
                                    marker_color = SULCUS_CLASS_COLORS[sulcus_class]
                                    depth_sets[sulcus_class].append(depth_mm)
                                    bgr = cv2.circle(bgr, far, radius_px, marker_color, -1)

        depth = flatten_depth_sets(depth_sets)
        print(f"[VTK sulci] slice {idx}: kind={slice_kind} ({slice_kind_conf:.2f}), {format_sulcus_class_summary(depth_sets)}")
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        if use_percent_filter:
            per_class_cells = sulcus_export_cells(depth_sets)
            slice_class_data.append(depth_sets)
        else:
            per_class_cells = [None] * len(sulcus_export_columns(unit))
            slice_class_data.append(None)
        rows.append([idx, len(inner_filtered),
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_{unit}
            (max(depth) if depth else None),    # max_depth_{unit}
            mean_depth,                         # mean_depth_{unit}
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
        base_cols = [
            "Slice", "Count_of_cont.", "Sulci_count",
            f"min_depth_{unit}", f"max_depth_{unit}", f"mean_depth_{unit}",
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

        overall_n = len(total_depth)
        overall_min = min(total_depth) if total_depth else None
        overall_max = max(total_depth) if total_depth else None
        overall_mean = (sum(total_depth) / overall_n) if overall_n else None

        summary_base = [None] * len(base_cols)
        summary_base[0] = "Sulci_overall_summary"
        summary_base[base_cols.index("Sulci_count")]        = overall_n
        summary_base[base_cols.index(f"min_depth_{unit}")]  = overall_min
        summary_base[base_cols.index(f"max_depth_{unit}")]  = overall_max
        summary_base[base_cols.index(f"mean_depth_{unit}")] = overall_mean
        rows.append(pad_row(
            [*summary_base, *sulcus_export_cells(overall_depth_sets)],
            total_width,
        ))

        df = pd.DataFrame(rows, columns=cols)
        df = drop_empty_columns(df)

        xlsx_path = os.path.join(out_dir, f"Mesh_Sulci_depth_{Slice_direction}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[VTK Sulci depth] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK Sulci depth] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return total_depth ,saved_pngs, valid_slices

def compute_compactness_vtk(parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    Slice_direction: Literal["X", "Y", "Z"] = "Y",
    Physical_dim: Sequence[int] | None = None,
    unit: str = "mm",
    slice_thickness: float = 0.5,
    contour_mode: str = "outer"):

    # --- Load mesh

    mesh = pv.read(str(file_path))
    print(f"[VTK compactness] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    mesh_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    mesh_dim_scaled = np.array(Physical_dim) / np.array(mesh_dim)

    axis_bounds = {
        "X": (x_min, x_max),
        "Y": (y_min, y_max),
        "Z": (z_min, z_max),
    }

    # --- Slice positions along the chosen axis
    if slice_thickness <= 0:
        slice_thickness = 0.05

    low, high = axis_bounds[Slice_direction]
    slice_positions = np.arange(low, high, slice_thickness)

    N = len(slice_positions)
    if N == 0:
        print(f"[VTK compactness] No slices to process (thickness too large vs. {Slice_direction} range).")
        return 0.0, [], []

    # Effective thickness to exactly span the axis extent, scaled to physical units
    axis_index = {"X": 0, "Y": 1, "Z": 2}.get(Slice_direction)
    slice_thickness_eff = mesh_dim[axis_index] / N
    slice_thickness_eff *= mesh_dim_scaled[axis_index]
    print(f"[VTK compactness] Effective slice thickness: {slice_thickness_eff} {unit}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "vtk_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "vtk_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[VTK compactness] Temp output dir: {out_dir}")

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    sections_list: list[pv.PolyData] = []
    rows = []
    sum_inner_mm = 0.0
    sum_area_mm2 = 0.0

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True)
    p.set_background("black")

    pre_axis = (axis_index - 1) % 3
    next_axis = (axis_index + 1) % 3
    cube_len = max(1e-6, mesh_dim[0] / 10.0)

    for idx, k in enumerate(slice_positions):
        # Cross-section slice
        normal, origin = slice_at(mesh, Slice_direction, k)
        section = mesh.slice(normal=normal,origin=origin)

        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube(Slice_direction, cube_len, mesh.center, k, mesh_dim[pre_axis]/2)

        plane = pv.Plane(center=origin,
                 direction=normal,  # plane normal
                 i_size=mesh_dim[pre_axis]*1.5, j_size=mesh_dim[next_axis]*1.5,  # side lengths
                 i_resolution=1, j_resolution=1)        # Render: section + scale cube
        p.clear()
        p.add_mesh(section, color="#ffffff", opacity=1)
        p.add_mesh(scale_cube, color="red")
        {"X": p.view_yz, "Y": p.view_xz, "Z": p.view_xy}[Slice_direction]()

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = calc_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h_img, w_img = bgr.shape[:2]
        thickness, _, _ = image_annotation_style(h_img, w_img, style="thin")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_VTK, 255, 0)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        inner_filtered, internal_filtered = split_inner_and_internal_contours(
            contours, hierarchy, red_rect, bw.shape, float(min_contour_area),
        )
        # Only highlight contours that actually contribute to the measured area.
        if contour_mode != "internal_only":
            cv2.drawContours(bgr, inner_filtered, -1, tuple(_get_viz().contour_inner_color_bgr), thickness)
        if contour_mode != "outer" and internal_filtered:
            cv2.drawContours(bgr, internal_filtered, -1, tuple(_get_viz().contour_internal_color_bgr), thickness)

        # Convert pixel measurements to physical units
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        inner_area_px = sum(cv2.contourArea(c) for c in inner_filtered)
        internal_area_px = sum(cv2.contourArea(c) for c in internal_filtered)
        if contour_mode == "subtract":
            area_px = inner_area_px - internal_area_px
        elif contour_mode == "internal_only":
            area_px = internal_area_px
        else:  # "outer"
            area_px = inner_area_px
        inner_perim_mm = inner_perim_px * mm_per_px
        area_mm2 = area_px * (mm_per_px ** 2)
        comp_2D = compactness_2D(area_mm2, inner_perim_mm)

        # Accumulate in physical units
        sum_inner_mm += inner_perim_mm
        sum_area_mm2 += area_mm2

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)
        rows.append([idx, len(inner_filtered), comp_2D])
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        sections_list.append(section)
        sections_list.append(plane)

    # ---- end with: plotter is fully and safely closed here ----

    Volume = sum_area_mm2 * slice_thickness_eff
    Area = sum_inner_mm * slice_thickness_eff
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
        rows.append([f"Volume {unit}^3", round(Volume, 2)])
        rows.append([f"Area {unit}^2", round(Area, 2)])
        df = pd.DataFrame(rows, columns=["Slice", "Count_of_cont.", "Slice_compactness"])

        xlsx_path = os.path.join(out_dir, f"Mesh_compactness_{Slice_direction}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[VTK compactness] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK compactness] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return  comp_3D, saved_pngs, valid_slices
