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

from deps import *
import pyvista as pv
from helpers.Helpers import compute_kernel_convex, contours_exclude, clac_scale, get_red_rect_offset, make_scale_cube
from helpers.check_mesh import check_brain
from constants import BINARY_THRESHOLD_DEFAULT, RED_CHANNEL_MIN, GREEN_CHANNEL_MAX, DEFECT_FIXED_POINT
from helpers.Helpers import compactness_2D, compactness_3D

# ----------------- main API -----------------
def compute_stl_allmarks(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size: int = 5,
    slice_thickness: float = 0.5):
    """Compute all hallmarks (volume, area, GI, sulci depths) from an STL mesh.

    Slices the mesh along Y, renders each cross-section with a red scale
    cube, then runs contour analysis on the screenshots.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory for slice PNGs and Excel.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size: Morph-close kernel diameter for outer contour.
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
    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(y_min, y_max, slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print("[STL All Hallmarks] No slices to process (thickness too large vs. Y range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    slice_thickness_eff = brain_dim[1] / N
    print(f"[STL All Hallmarks] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[STL All Hallmarks] Temp output dir: {out_dir}")

    # --- Screenshot resolution via target mm/px spacing ---
    # pixel_spacing controls render resolution: 0.1 mm/px gives ~10 px per mm.
    pixel_spacing = 0.1  # mm per pixel
    image_width = int(np.clip(np.ceil(brain_dim[0] / pixel_spacing), 64, 4096))
    image_height = int(np.clip(np.ceil(brain_dim[2] / pixel_spacing), 64, 4096))
    window_size = (image_width, image_height)

    # --- Camera: positioned along +Y, looking at the XZ plane ---
    center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0]
    cam_position = [
        (center[0], y_max + 100.0, center[2]),  # camera sits 100 mm beyond Y-max
        (center[0], center[1], center[2]),       # focal point = mesh centre
        (0.0, 0.0, 1.0),                         # Z-up
    ]

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sections_list: list[pv.PolyData] = []
    total_depth = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0
    sum_area = 0.0

    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.camera_position = cam_position
    # Red reference cube side length = 10% of Y extent, used for scale calibration.
    cube_len = max(1e-6, brain_dim[1] / 10.0)
    max_dim =  max(brain_dim[0],brain_dim[2])
    for idx, y in enumerate(slice_positions):
        # Cross-section slice
        origin =  mesh.center
        section = mesh.slice(normal=[0, 1, 0], origin=[origin[0], float(y), origin[2]])
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube("Y", cube_len, mesh.center, y, max_dim)
#        scale_cube = pv.Cube(x_length=cube_len, y_length=0.01, z_length=cube_len)
#        scale_cube.translate((50, 0, 50), inplace=True)

        plane = pv.Plane(center=(0, float(y), 0),
                         direction=(0, 1, 0),  # plane normal
                         i_size=brain_dim[0]*1.5, j_size=brain_dim[2]*1.5,  # side lengths
                         i_resolution=1, j_resolution=1)
        # Render: section + scale cube
        p.clear()
        p.add_mesh(scale_cube, color="red")
        p.add_mesh(section, color="black")
        p.view_xz()  # no .show()!

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = clac_scale(img_rgb, cube_len)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
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
        area_perim_px  = sum(cv2.contourArea(c)     for c in inner_filtered)
        inner_perim_px = sum(cv2.arcLength(c, True) for c in inner_filtered)
        outer_perim_px = sum(cv2.arcLength(c, True) for c in outer_filtered)
        inner_perim_mm = inner_perim_px * mm_per_px
        outer_perim_mm = outer_perim_px * mm_per_px
        area_perim_mm  = area_perim_px * (mm_per_px ** 2)


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
                                if depth_mm < (0.25 * max_dim) and depth_mm > (0.005* max_dim):
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
        plane["slice_idx"] = np.full(plane.n_points, idx, dtype=np.int32)
        section["slice_idx"] = np.full(section.n_points, idx, dtype=np.int32)
        sections_list.append(plane)
        sections_list.append(section)

    # ---- end with: plotter is fully and safely closed here ----

    # Totals
    brain_volume = (sum_area * slice_thickness_eff)/1000
    Area = sum_inner_mm * slice_thickness_eff /100
    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
    
    total_depth = [x/10 for x in total_depth]

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
    
        rows.append(["Volume cm^3",round(brain_volume,2), "Surface Area cm^2",round(Area,2)])
        rows.append(["GI",round(GI_total,2)])
        rows.append(["Total_Number_of_Sluci",len(total_depth), "Mean_value_across_slices_cm",(round(mean_total, 2) if mean_total is not None else None)])
        rows.append(["Max_sulci_across_slices_cm",(round(max(total_depth),2) if total_depth else None),
        "Min_sulci_across_slices_cm",(round(min(total_depth),2) if total_depth else None)])
        df = pd.DataFrame(rows, columns=["Slice", "Count_of_cont.","Inner_area_mm^2", "Inner_Perimeter_mm", "Outer_Perimeter_mm" ,"Sulci_count",
            "min_depth_mm", "max_depth_mm", "mean_depth_mm"])
    
        
        xlsx_path = os.path.join(out_dir, "Mesh_Allmarks.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[STL All Hallmarks] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL All Hallmarks] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return dic["label"], brain_dim_cm, Area, brain_volume, GI_total, total_depth ,saved_pngs, valid_slices 


def compute_stl_lGI(
    parent,
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size: int = 5,
    slice_thickness: float = 0.5,
    build_solid: bool = False,
):
    """Compute the gyrification index (GI) from an STL mesh via slice rendering.

    GI = total inner perimeter / total outer perimeter across all Y slices.
    Optionally builds an extruded 3-D solid from the outer contours.

    Args:
        parent: Qt parent widget for message boxes.
        file_path: Path to the ``.stl`` file.
        out_dir: Output directory.
        min_contour_area: Minimum contour area (pixels) to keep.
        kernel_size: Morph-close kernel diameter.
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


    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(y_min, y_max, slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print("[STL lGI] No slices to process (thickness too large vs. Y range).")
        return dic["label"],[],0,[],[]

    # Effective thickness to exactly span Y-extent
    slice_thickness_eff = brain_dim[1] / N
    print(f"[STL lGI] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL lGI] Temp output dir: {out_dir}")

    # --- Screenshot resolution via target mm/px spacing
    pixel_spacing = 0.1  # mm per pixel
    image_width = int(np.clip(np.ceil(brain_dim[0] / pixel_spacing), 64, 4096))
    image_height = int(np.clip(np.ceil(brain_dim[2] / pixel_spacing), 64, 4096))
    window_size = (image_width, image_height)

    # --- Camera (look along +Y onto XZ)
    center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0]
    cam_position = [
        (center[0], y_max + 100.0, center[2]),  # camera position
        (center[0], center[1], center[2]),      # focal point
        (0.0, 0.0, 1.0),                        # view up
    ]

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    all_3d_contours = []
    sections_list: list[pv.PolyData] = []
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.camera_position = cam_position

    for idx, y in enumerate(slice_positions):
        # Cross-section slice
        origin =  mesh.center
        section = mesh.slice(normal=[0, 1, 0], origin=[origin[0], float(y), origin[2]])
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        cube_len = max(1e-6, brain_dim[1] / 10.0)
        scale_cube = make_scale_cube("Y", cube_len, mesh.center, y, max(brain_dim[0],brain_dim[2]))

        plane = pv.Plane(center=(0, float(y), 0),
                         direction=(0, 1, 0),  # plane normal
                         i_size=brain_dim[0]*1.5, j_size=brain_dim[2]*1.5,  # side lengths
                         i_resolution=1, j_resolution=1)
        # Render: section + scale cube
        p.clear()
        p.add_mesh(scale_cube, color="red")
        p.add_mesh(section, color="black")
        p.view_xz()  # no .show()!

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = clac_scale(img_rgb, cube_len)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
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
        rows.append([idx, len(inner_filtered), inner_perim_mm, outer_perim_mm, GI_slice])

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
        df = pd.DataFrame(rows, columns=["Slice", "Inner_Perimeter_mm", "Outer_Perimeter_mm", "GI"])
        df.loc[len(df)] = ["GI", round(GI_total, 3), None, None]
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
    slice_thickness: float = 0.5):
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
    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(y_min, y_max, slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print("[STL Volume] No slices to process (thickness too large vs. Y range).")
        return dic["label"],[],0,[],[]

    # Effective thickness to exactly span Y-extent
    slice_thickness_eff = brain_dim[1] / N
    print(f"[STL Volume] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[STL Volume] Temp output dir: {out_dir}")

    # --- Screenshot resolution via target mm/px spacing
    pixel_spacing = 0.1  # mm per pixel
    image_width = int(np.clip(np.ceil(brain_dim[0] / pixel_spacing), 64, 4096))
    image_height = int(np.clip(np.ceil(brain_dim[2] / pixel_spacing), 64, 4096))
    window_size = (image_width, image_height)

    # --- Camera (look along +Y onto XZ)
    center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0]
    cam_position = [
        (center[0], y_max + 100.0, center[2]),  # camera position
        (center[0], center[1], center[2]),      # focal point
        (0.0, 0.0, 1.0),                        # view up
    ]

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sections_list: list[pv.PolyData] = []
    sum_area = 0.0

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.camera_position = cam_position

    for idx, y in enumerate(slice_positions):
        # Cross-section slice
        origin =  mesh.center
        section = mesh.slice(normal=[0, 1, 0], origin=[origin[0], float(y), origin[2]])
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        cube_len = max(1e-6, brain_dim[1] / 10.0)
        scale_cube = make_scale_cube("Y", cube_len, mesh.center, y, max(brain_dim[0],brain_dim[2]))
        plane = pv.Plane(center=(0, float(y), 0),
                         direction=(0, 1, 0),  # plane normal
                         i_size=brain_dim[0]*1.5, j_size=brain_dim[2]*1.5,  # side lengths
                         i_resolution=1, j_resolution=1)
        # Render: section + scale cube
        p.clear()
        p.add_mesh(scale_cube, color="red")
        p.add_mesh(section, color="black")
        p.view_xz()  # no .show()!

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = clac_scale(img_rgb, cube_len)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, (0, 0, 255), 1)

        # Perimeters (mm)
        area_perim_px  = sum(cv2.contourArea(c)     for c in inner_filtered)
        area_perim_mm  = area_perim_px * (mm_per_px ** 2)


        rows.append([idx, len(inner_filtered), area_perim_mm])
        # Accumulate
        sum_area     += area_perim_mm

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
    slice_thickness: float = 0.5):
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
    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(y_min, y_max, slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print("[STL Area] No slices to process (thickness too large vs. Y range).")
        return dic["label"],[],0,[],[]

    # Effective thickness to exactly span Y-extent
    slice_thickness_eff = brain_dim[1] / N
    print(f"[STL Area] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[STL Area] Temp output dir: {out_dir}")

    # --- Screenshot resolution via target mm/px spacing
    pixel_spacing = 0.1  # mm per pixel
    image_width = int(np.clip(np.ceil(brain_dim[0] / pixel_spacing), 64, 4096))
    image_height = int(np.clip(np.ceil(brain_dim[2] / pixel_spacing), 64, 4096))
    window_size = (image_width, image_height)

    # --- Camera (look along +Y onto XZ)
    center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0]
    cam_position = [
        (center[0], y_max + 100.0, center[2]),  # camera position
        (center[0], center[1], center[2]),      # focal point
        (0.0, 0.0, 1.0),                        # view up
    ]

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sum_inner_mm = 0.0
    sections_list: list[pv.PolyData] = []


    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.camera_position = cam_position
    cube_len = max(1e-6, brain_dim[1] / 10.0)

    for idx, y in enumerate(slice_positions):
        # Cross-section slice
        origin =  mesh.center

        section = mesh.slice(normal=[0, 1, 0], origin=[origin[0], float(y), origin[2]])
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube("Y", cube_len, mesh.center, y, max(brain_dim[0],brain_dim[2]))
        plane = pv.Plane(center=(0, float(y), 0),
                         direction=(0, 1, 0),  # plane normal
                         i_size=brain_dim[0]*1.5, j_size=brain_dim[2]*1.5,  # side lengths
                         i_resolution=1, j_resolution=1)
        # Render: section + scale cube
        p.clear()
        p.add_mesh(scale_cube, color="red")
        p.add_mesh(section, color="black")
        p.view_xz()  # no .show()!

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = clac_scale(img_rgb, cube_len)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
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

    # Totals
    Area = sum_inner_mm * slice_thickness_eff /100
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
    slice_thickness: float = 0.5):
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
    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(y_min, y_max, slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print("[STL Sulci depth] No slices to process (thickness too large vs. Y range).")
        return dic["label"],[],[],[],[]

    # Effective thickness to exactly span Y-extent
    slice_thickness_eff = brain_dim[1] / N
    print(f"[STL Sulci depth] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
    
    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)
    
    print(f"[STL Sulci depth] Temp output dir: {out_dir}")

    # --- Screenshot resolution via target mm/px spacing
    pixel_spacing = 0.1  # mm per pixel
    image_width = int(np.clip(np.ceil(brain_dim[0] / pixel_spacing), 64, 4096))
    image_height = int(np.clip(np.ceil(brain_dim[2] / pixel_spacing), 64, 4096))
    window_size = (image_width, image_height)

    # --- Camera (look along +Y onto XZ)
    center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0]
    cam_position = [
        (center[0], y_max + 100.0, center[2]),  # camera position
        (center[0], center[1], center[2]),      # focal point
        (0.0, 0.0, 1.0),                        # view up
    ]

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    total_depth = []
    sections_list: list[pv.PolyData] = []


    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.camera_position = cam_position
    cube_len = max(1e-6, brain_dim[1] / 10.0)
    max_dim =  max(brain_dim[0],brain_dim[2])
    for idx, y in enumerate(slice_positions):
        # Cross-section slice
        origin =  mesh.center
        section = mesh.slice(normal=[0, 1, 0], origin=[origin[0], float(y), origin[2]])
        if section.n_points == 0:
            continue
            
        # Red cube reference (10% of X extent)
        scale_cube = make_scale_cube("Y", cube_len, mesh.center, y, max_dim)
        plane = pv.Plane(center=(0, float(y), 0),
                         direction=(0, 1, 0),  # plane normal
                         i_size=brain_dim[0]*1.5, j_size=brain_dim[2]*1.5,  # side lengths
                         i_resolution=1, j_resolution=1)
        # Render: section + scale cube
        p.clear()
        p.add_mesh(scale_cube, color="red")
        p.add_mesh(section, color="black")
        p.view_xz()  # no .show()!

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = clac_scale(img_rgb, cube_len)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
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
                                mm_per_fixed = mm_per_px / DEFECT_FIXED_POINT
                                depth_mm = d *mm_per_fixed
                                if depth_mm < (0.25* max_dim) and depth_mm > (0.005* max_dim):
                                    bgr = cv2.circle(bgr, far, 2, [255, 255, 0], -1)
                                    depth.append(depth_mm)
                
        mean_depth = (sum(depth)/len(depth)) if depth else None
        total_depth.extend(depth)
        rows.append([idx,len(inner_filtered),
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
        rows.append(["Total_Number_of_Sluci",len(total_depth), "Mean_value_across_slices_mm",(round(mean_total, 2) if mean_total is not None else None)])
        rows.append(["Max_sulci_across_slices_mm",(round(max(total_depth),2) if total_depth else None),
        "Min_sulci_across_slices_mm",(round(min(total_depth),2) if total_depth else None)])
        df = pd.DataFrame(rows, columns=["Slice","Sulci_count", "min_depth_mm", "max_depth_mm", "mean_depth_mm"])
    
        
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
    slice_thickness: float = 0.5):
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

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(y_min, y_max, slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print("[STL Compactness] No slices to process (thickness too large vs. Y range).")
        return dic["label"], [], 0, [], []

    # Effective thickness to exactly span Y-extent
    slice_thickness_eff = brain_dim[1] / N
    print(f"[STL Compactness] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)

    out_dir_origin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_origin, exist_ok=True)

    print(f"[STL Compactness] Temp output dir: {out_dir}")

    # --- Screenshot resolution via target mm/px spacing
    pixel_spacing = 0.1  # mm per pixel
    image_width = int(np.clip(np.ceil(brain_dim[0] / pixel_spacing), 64, 4096))
    image_height = int(np.clip(np.ceil(brain_dim[2] / pixel_spacing), 64, 4096))
    window_size = (image_width, image_height)

    # --- Camera (look along +Y onto XZ)
    center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, (z_min + z_max) / 2.0]
    cam_position = [
        (center[0], y_max + 100.0, center[2]),  # camera position
        (center[0], center[1], center[2]),       # focal point
        (0.0, 0.0, 1.0),                         # view up
    ]

    saved_pngs: list[str] = []
    valid_slices: list[int] = []
    rows = []
    sections_list: list[pv.PolyData] = []
    sum_area_mm2 = 0.0
    sum_inner_mm = 0.0

    # --- Plotter
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.camera_position = cam_position
    cube_len = max(1e-6, brain_dim[1] / 10.0)

    for idx, y in enumerate(slice_positions):
        # Cross-section slice
        origin = mesh.center
        section = mesh.slice(normal=[0, 1, 0], origin=[origin[0], float(y), origin[2]])
        if section.n_points == 0:
            continue

        # Red cube reference (10% of Y extent)
        scale_cube = make_scale_cube("Y", cube_len, mesh.center, y, max(brain_dim[0], brain_dim[2]))
        plane = pv.Plane(center=(0, float(y), 0),
                         direction=(0, 1, 0),
                         i_size=brain_dim[0] * 1.5, j_size=brain_dim[2] * 1.5,
                         i_resolution=1, j_resolution=1)
        # Render: section + scale cube
        p.clear()
        p.add_mesh(scale_cube, color="red")
        p.add_mesh(section, color="black")
        p.view_xz()

        # Screenshot
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_origin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = clac_scale(img_rgb, cube_len)

        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > RED_CHANNEL_MIN) & (img_rgb[:, :, 1] < GREEN_CHANNEL_MAX), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Inner contours: exclude red ref + area filter
        inner_candidates = contours_exclude(contours, red_rect, bw.shape)
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(min_contour_area)]
        cv2.drawContours(bgr, inner_filtered, -1, (0, 0, 255), 1)

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
