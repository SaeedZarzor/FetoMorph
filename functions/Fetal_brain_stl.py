# Fetal_brain_stl.py
from deps import *
import pyvista as pv

from functions.Helpers import compute_kernel_convex, contours_exclude, clac_scale, get_red_rect_offset

# ----------------- main API -----------------
def compute_stl_allmarks(
    file_path: str,
    out_dir: str,
    min_contour_area: float = 20.0,
    kernel_size: int = 5,
    slice_thickness: float = 0.5):
    # --- Load mesh
    mesh = pv.read(str(file_path))
    print(f"[STL All Hallmarks] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL All Hallmarks] Brain dimensions (mm): {brain_dim}")

    brain_dim_cm = [dim/10 for dim in brain_dim]
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
    
    out_dir_orgin = os.path.join(out_dir, "stl_orgin")
    os.makedirs(out_dir_orgin, exist_ok=True)
    
    print(f"[STL All Hallmarks] Temp output dir: {out_dir}")

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
    sum_outer_mm = 0.0
    sum_area = 0.0

    # --- Use a context manager so the plotter is *guaranteed* to be closed safely
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")
    p.camera_position = cam_position

    for idx, y in enumerate(slice_positions):
        # Cross-section slice
        section = mesh.slice(normal=[0, 1, 0], origin=[0.0, float(y), 0.0])
        if section.n_points == 0:
            continue

        # Red cube reference (10% of X extent)
        cube_len = max(1e-6, brain_dim[1] / 10.0)
        scale_cube = pv.Cube(x_length=cube_len, y_length=0.01, z_length=cube_len)
        scale_cube.translate((50, 0, 50), inplace=True)

        # Render: section + scale cube
        p.clear()
        p.add_mesh(scale_cube, color="red")
        p.add_mesh(section, color="black")
        p.view_xz()  # no .show()!

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir_orgin, f"image_{idx:03d}.png"))

        # Compute mm/px scale from the red cube
        mm_per_px = clac_scale(img_rgb, cube_len)


        # Prepare masks / contours (pixel space)
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        red_rect = np.where((img_rgb[:, :, 0] > 150) & (img_rgb[:, :, 1] < 50), 255, 0).astype("uint8")

        # Binary for contours
        _, bw = cv2.threshold(gray, 200, 255, 1)
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

        # Accumulate
        sum_inner_mm += inner_perim_mm
        sum_outer_mm += outer_perim_mm
        sum_area     += area_perim_mm
        GI_slice = (inner_perim_mm / outer_perim_mm) if outer_perim_mm > 0 else 0.0
        rows.append([idx, area_perim_mm,  inner_perim_mm, outer_perim_mm , GI_slice])

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)

      
    # ---- end with: plotter is fully and safely closed here ----

    # Totals
    brain_volume = (sum_area * slice_thickness_eff)/1000
    Area = sum_inner_mm * slice_thickness_eff /100
    GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0

    # Save per-slice + total to Excel
    try:
        df = pd.DataFrame(rows, columns=["Slice", "Inner_area_mm^2", "Inner_Perimeter_mm", "Outer_Perimeter_mm" ,"GI"])
        
        rows.append(["Volume cm^3",round(brain_volume,2), "Surface Area cm^2",round(Area,2)])
        rows.append(["GI",round(GI_total,2)])
        xlsx_path = os.path.join(out_dir, "Brain_Allmarks.xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[STL All Hallmarks] Saved Excel → {xlsx_path}")
    except Exception as ex:
        print(f"[STL All Hallmarks] WARN: could not save Excel: {ex}")


    # Always return a 3-tuple
    return brain_dim_cm, Area, brain_volume, GI_total, saved_pngs, valid_slices
