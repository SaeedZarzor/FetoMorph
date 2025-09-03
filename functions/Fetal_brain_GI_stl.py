# Fetal_brain_GI_stl.py — STL-based LGI computation with robust off-screen screenshots (macOS-safe)
import os
import numpy as np
import pandas as pd
import cv2
import pyvista as pv

from functions.measurements_image import compute_kernel_convex  # must return a cv2 kernel (k x k)

# ----------------- helpers -----------------
def contours_exclude(contours, excluded_space, image_shape):
    """
    Filter contours by excluding any that overlap with 'excluded_space' (uint8 mask).
    """
    filtered = []
    for cnt in contours:
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if np.count_nonzero(cv2.bitwise_and(mask, excluded_space)) == 0:
            filtered.append(cnt)
    return filtered


def clac_scale(image_rgb, cube_length_mm):
    """
    Compute mm-per-pixel from a red reference cube drawn in the render.
    cube_length_mm: the real cube side length (x_length) in mm.
    """
    red_rect = np.where((image_rgb[:, :, 0] > 150) & (image_rgb[:, :, 1] < 50), 255, 0).astype("uint8")
    _, thresh_red = cv2.threshold(red_rect, 150, 255, 0)
    contours, _ = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[STL lGI] No red reference contour found; default scale 1.0 mm/px")
        return 1.0
    # Use the largest red blob as reference
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (cube_length_mm / float(w)) if w > 0 else 1.0


def get_red_rect_offset(image_rgb):
    """
    Detect red rectangle and return its center (x,y) in pixels — used to zero-align contours.
    """
    red_mask = (image_rgb[:, :, 0] > 150) & (image_rgb[:, :, 1] < 50)
    coords = np.argwhere(red_mask)
    if coords.size == 0:
        return np.array([0, 0])
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return np.array([(x_min + x_max) // 2, (y_min + y_max) // 2])


# ----------------- main API -----------------
def compute_stl_lGI(
    file_path: str,
    out_dir: str,
    cnt_threshold: float = 20.0,
    kernel_size: int = 5,
    slice_thickness: float = 0.5,
    build_solid: bool = False,  # set True if you want the extruded solid (may be crashy on some macOS stacks)
):
    """
    Compute a slice-based LGI proxy from an STL surface using off-screen PyVista rendering.

    Returns:
        GI_total (float), saved_pngs (list[str]), valid_slices (list[int])
    """
    # --- Load mesh
    mesh = pv.read(str(file_path))
    print(f"[STL lGI] Loaded mesh: {mesh}")

    # --- Bounds / dims (mm)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    brain_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
    print(f"[STL lGI] Brain dimensions (mm): {brain_dim}")

    # --- Slice positions along Y
    if slice_thickness <= 0:
        slice_thickness = 0.5
    slice_positions = np.arange(y_min, y_max, slice_thickness)
    N = len(slice_positions)
    if N == 0:
        print("[STL lGI] No slices to process (thickness too large vs. Y range).")
        return 0.0, [], []

    # Effective thickness to exactly span Y-extent
    slice_thickness_eff = brain_dim[1] / N
    print(f"[STL lGI] Effective slice thickness (mm): {slice_thickness_eff}")

    # --- Outputs
    os.makedirs(out_dir, exist_ok=True)
    out_dir_slices = os.path.join(out_dir, "stl_slices")
    os.makedirs(out_dir_slices, exist_ok=True)
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
    sum_inner_mm = 0.0
    sum_outer_mm = 0.0

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
        cube_len = max(1e-6, brain_dim[0] / 10.0)
        scale_cube = pv.Cube(x_length=cube_len, y_length=1.0, z_length=cube_len)
        scale_cube.translate((50, -1, 50), inplace=True)

        # Render: section + scale cube
        p.clear()
        p.add_mesh(scale_cube, color="red")
        p.add_mesh(section, color="black")
        p.view_xz()  # no .show()!

        # Screenshot (array for processing, file for debugging)
        img_rgb = p.screenshot(return_img=True, filename=os.path.join(out_dir, f"image_{idx:03d}.png"))

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
        inner_filtered = [c for c in inner_candidates if cv2.contourArea(c) > float(cnt_threshold)]
        cv2.drawContours(bgr, inner_filtered, -1, (0, 0, 255), 1)

        # Outer contours via morph close, exclude red + area filter
        kernel = compute_kernel_convex(max(1, int(kernel_size)))
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        outer_candidates, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outer_candidates = contours_exclude(outer_candidates, red_rect, bw.shape)
        outer_filtered = [c for c in outer_candidates if cv2.contourArea(c) > float(cnt_threshold)]
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
        rows.append([idx, inner_perim_mm, outer_perim_mm, GI_slice])

        # Save annotated slice
        slice_path = os.path.join(out_dir_slices, f"slice_{idx:03d}.png")
        cv2.imwrite(slice_path, bgr)
        saved_pngs.append(slice_path)
        valid_slices.append(idx)

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
    return float(GI_total), saved_pngs, valid_slices
