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

from deps import *
import pyvista as pv
from helpers.Helpers import compute_kernel_convex, contours_exclude, clac_scale, get_red_rect_offset, slice_at, make_scale_cube, compactness_3D, compactness_2D
from helpers.check_mesh import check_brain
from typing import Any, Literal, Sequence
from constants import BINARY_THRESHOLD_VTK, RED_CHANNEL_MIN, GREEN_CHANNEL_MAX, DEFECT_FIXED_POINT

    
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
    slice_thickness: float = 0.5):
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
        mm_per_px = clac_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Use BINARY_THRESHOLD_VTK (150) for black-background VTK renders.
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_VTK, 255, 0)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Exclude red reference-cube contours from brain measurements.
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, (0, 0, 255), 1)

        kernel = compute_kernel_convex(max(1, int(kernel_size)))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        outer_candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_candidates = contours_exclude(outer_candidates, red_rect, bw.shape)
        outer_filtered = [c for c in outer_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, outer_filtered, -1, (0, 255, 0), 1)

        area_perim_px  = sum(cv2.contourArea(c)     for c in inner_filtered)
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        outer_perim_px = sum(cv2.arcLength(c, True) for c in outer_filtered)
        inner_perim_mm = inner_perim_px * mm_per_px
        outer_perim_mm = outer_perim_px * mm_per_px
        area_perim_mm  = area_perim_px * (mm_per_px ** 2)


        depth = []
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
                            bgr = cv2.line(bgr, start, end, [255, 0, 0], 1)
                            if d > DEFECT_FIXED_POINT:
                                depth_mm = d * mm_per_px / DEFECT_FIXED_POINT
                                bgr = cv2.circle(bgr, far, 2, [255, 255, 0], -1)
                                depth.append(depth_mm)
                
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        rows.append([idx, len(inner_filtered), area_perim_mm, inner_perim_mm, outer_perim_mm,
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_mm
            (max(depth) if depth else None),    # max_depth_mm
            mean_depth                          # mean_depth_mm
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
        rows.append([f"Volume {unit}^3", Volume, f"Surface Area {unit}^2", Area])
        rows.append(["GI",round(GI_total,2)])
        rows.append(["Total_Number_of_Sluci",len(total_depth), f"Mean_value_across_slices_{unit}",(round(mean_total, 2) if mean_total is not None else None)])
        rows.append([f"Max_sulci_across_slices_{unit}",(round(max(total_depth),2) if total_depth else None),
        f"Min_sulci_across_slices_{unit}",(round(min(total_depth),2) if total_depth else None)])
        
        df = pd.DataFrame(rows, columns=["Slice", "Count_of_cont.",f"Inner_area_{unit}^2", f"Inner_Perimeter_{unit}", f"Outer_Perimeter_{unit}" ,"Sulci_count",
            f"min_depth_{unit}", f"max_depth_{unit}", f"mean_depth_{unit}"])
        
        xlsx_path = os.path.join(out_dir, f"Mesh_Allmarks_{Slice_direction}.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[VTK All Hallmarks] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[VTK All Hallmarks] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return Area, Volume, GI_total, total_depth ,saved_pngs, valid_slices


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
        mm_per_px = clac_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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
        cv2.drawContours(bgr, inner_filtered, -1, (0, 0, 255), 1)

        # Outer contours via morph close, exclude red + area filter
        kernel = compute_kernel_convex(max(1, int(kernel_size)))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        outer_candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_candidates = contours_exclude(outer_candidates, red_rect, bw.shape)
        outer_filtered = [c for c in outer_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, outer_filtered, -1, (0, 255, 0), 1)

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
    slice_thickness: float = 0.5):
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
        mm_per_px = clac_scale(img_rgb, cube_len*mesh_dim_scaled[0])


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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
        cv2.drawContours(bgr, inner_filtered, -1, (0, 255, 255), 2)

        

        # Perimeters (mm)
        area_perim_px  = sum(cv2.contourArea(c)   for c in inner_filtered)
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
        mm_per_px = clac_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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
        cv2.drawContours(bgr, inner_filtered, -1, (0, 0, 255), 1)

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
        mm_per_px = clac_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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
        cv2.drawContours(bgr, inner_filtered, -1, (0, 0, 255), 1)

        depth = []
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
                            bgr = cv2.line(bgr, start, end, [255, 0, 0], 1)
                            if d > DEFECT_FIXED_POINT:
                                depth_mm = d * mm_per_px / DEFECT_FIXED_POINT
                                bgr = cv2.circle(bgr, far, 2, [255, 255, 0], -1)
                                depth.append(depth_mm)
                
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        rows.append([idx, len(inner_filtered),
            len(depth),                         # n_defects
            (min(depth) if depth else None),    # min_depth_mm
            (max(depth) if depth else None),    # max_depth_mm
            mean_depth                          # mean_depth_mm
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
        rows.append(["Total_Number_of_Sluci",len(total_depth), f"Mean_value_across_slices_{unit}",(round(mean_total, 2) if mean_total is not None else None)])
        rows.append([f"Max_sulci_across_slices_{unit}",(round(max(total_depth),2) if total_depth else None),
        f"Min_sulci_across_slices_{unit}",(round(min(total_depth),2) if total_depth else None)])
        
        df = pd.DataFrame(rows, columns=["Slice", "Sulci_count", f"min_depth_{unit}", f"max_depth_{unit}", f"mean_depth_{unit}"])
        
        
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
    slice_thickness: float = 0.5):

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
        mm_per_px = clac_scale(img_rgb, cube_len*mesh_dim_scaled[0])

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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
        cv2.drawContours(bgr, inner_filtered, -1, (0, 0, 255), 1)

        # Convert pixel measurements to physical units
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        area_px  = sum(cv2.contourArea(c)     for c in inner_filtered)
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