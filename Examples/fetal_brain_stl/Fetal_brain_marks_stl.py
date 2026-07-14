import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import trimesh
import math
import pyvista as pv
from skimage.draw import polygon2mask


def contours_exclude(contours, excluded_space, image_shape):
    """Filters contours by excluding those that overlap with excluded_space."""
    filtered_contours = []
    for cnt in contours:
        mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        overlap = cv2.bitwise_and(mask, excluded_space)
        if np.count_nonzero(overlap) == 0:
            filtered_contours.append(cnt)
    return filtered_contours
    
    
def clac_scale(image, cube_length_mm):
    """
    Computes mm-per-pixel scale using the detected red reference square in the render.
    cube_length_mm is the real-world length of the red cube in mm (x_length).
    """
    red_rect = np.where((image[:, :, 0] > 150) & (image[:, :, 1] < 50), 255, 0).astype('uint8')
    _, thresh_red = cv2.threshold(red_rect, 150, 255, 0)
    contours, _ = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No red reference contour found")
        return 1.0
    # Take largest red contour
    contour_red = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour_red)
    # mm per pixel based on cube side length in mm
    mm_per_pixel = (cube_length_mm / float(w))
    return mm_per_pixel


def get_red_rect_offset(image):
    """Detect red rectangle and return its center (x, y) in pixels."""
    red_mask = (image[:, :, 0] > 150) & (image[:, :, 1] < 50)  # red dominant
    coords = np.argwhere(red_mask)
    if coords.size == 0:
        return np.array([0, 0])  # default to no offset
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return np.array([center_x, center_y])


# --- Tkinter file picker ---
root = tk.Tk()
root.withdraw()
file_path_str = filedialog.askopenfilename(
    title='Select file',
    filetypes=[('STL files', '*.stl'), ('VTC files', '*.vtk'), ('All files', '*.*')]
)
if not file_path_str:
    print("No file selected, exiting.")
    exit()

# Load mesh
file_path = Path(file_path_str)
directory = file_path.parent
mesh = pv.read(str(file_path))

print(f"mesh data is {mesh}")

# Brain real dimensions in mm
x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
brian_dim = [x_max - x_min, y_max - y_min, z_max - z_min]
print(f"Brain dimensions (mm): {brian_dim}")

# Render window size in pixels
pixel_spacing = 0.1  # mm per pixel for screenshot resolution
image_width = int(np.ceil(brian_dim[0] / pixel_spacing))
image_height = int(np.ceil(brian_dim[2] / pixel_spacing))
window_size = (image_width, image_height)


# Slice positions (Y direction)
slice_thickness = 0.5  # mm
slice_positions = np.arange(y_min, y_max, slice_thickness)
N = len(slice_positions)
if N == 0:
    print("No slices to process.")
    exit()

# Effective thickness so total height matches By
slice_thickness_eff = brian_dim[1] / N
print(f"Effective slice thickness (mm): {slice_thickness_eff}")

# Output folders
output_folder = directory / "stl_slices"
os.makedirs(output_folder, exist_ok=True)
output_temp = directory / "out"
os.makedirs(output_temp, exist_ok=True)

cnt_threshold = 20
font_scale = 0.1 / pixel_spacing

# Morphological kernel
kernel_size = int(2 / pixel_spacing)
if kernel_size % 2 == 0:
    kernel_size += 1
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

sheet = []
sum_inner , sum_outer = 0.0, 0.0
inner_perimeter_all = []
outer_perimeter_all = []

pv.global_theme.allow_empty_mesh = True

p = pv.Plotter(off_screen=True, window_size=window_size)
p.set_background('white')
all_3d_contours = []

# Camera centered on XZ plane
center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
p.camera_position = [
    (center[0], y_max + 100, center[2]),
    (center[0], center[1], center[2]),
    (0, 0, 1)
]

for idx, y in enumerate(slice_positions):
    origin = [0, y, 0]
    slice_section = mesh.slice(normal=[0, 1, 0], origin=origin)
    if slice_section.n_points == 0:
        continue

    # Red cube reference: set real cube length (unite)
    cube_length = brian_dim[0]/10
    scale_cube = pv.Cube(x_length=cube_length, y_length=1, z_length=cube_length).translate(50, -1, 50)

    p.clear()
    p.add_mesh(scale_cube, color='red')
    p.add_mesh(slice_section, color="black")
    p.show(auto_close=False, cpos="xz")

    image = p.screenshot(return_img=True, filename=f"{output_temp}/image_{idx:03d}.png")
    image_scale = clac_scale(image, cube_length)  # unite per pixel

    imag = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im_gray = cv2.cvtColor(imag, cv2.COLOR_RGB2GRAY)
    red_rect = np.where((image[:, :, 0] > 150) & (image[:, :, 1] < 50), 255, 0).astype('uint8')

    _, im_bw = cv2.threshold(im_gray, 200, 255, 1)
    contours, _ = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    filtered_contours = contours_exclude(contours, red_rect, im_bw.shape)
    cv2.drawContours(imag, filtered_contours, -1, (0, 0, 255), 1)

    # Real area in mm²
    area_sum_px2 = sum(cv2.contourArea(cnt) for cnt in filtered_contours)
    area = area_sum_px2 * (image_scale ** 2)

    # Real inner perimeter in mm
    perimeter_sum_px = sum(cv2.arcLength(cnt, True) for cnt in filtered_contours)
    inner_perimeter = perimeter_sum_px * image_scale

    x1, y1 = contours[0][0][0]
    cv2.putText(imag, f"Area:{area:.2f}", (x1, y1 - 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 100), 2)
    cv2.putText(imag, f"Perimeter:{inner_perimeter:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 200), 2)

    # Outer perimeter (convex mask)
    closed_mask = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
    convex_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_conv_contours = contours_exclude(convex_contours, red_rect, im_bw.shape)
    cv2.drawContours(imag, filtered_conv_contours, -1, (0, 255, 0), 1)
    outer_perimeter_px = sum(cv2.arcLength(cnt, True) for cnt in filtered_conv_contours)
    outer_perimeter = outer_perimeter_px * image_scale

    offset_2d = get_red_rect_offset(image)

    # Store 3D contours in real mm coordinates
    for cnt in convex_contours:
        cnt = cnt.squeeze()
        if len(cnt.shape) != 2:
            continue
        aligned_cnt = cnt - offset_2d
        cnt_3d = np.hstack([
            aligned_cnt * np.array([image_scale, image_scale]),  # mm
                        np.full((cnt.shape[0], 1), idx * slice_thickness_eff)  # mm
        ])
        all_3d_contours.append(cnt_3d)

    # Accumulate totals
    sum_inner += inner_perimeter
    sum_outer += outer_perimeter
    GI_slice = inner_perimeter / outer_perimeter if outer_perimeter > 0 else 0

    out_path = os.path.join(output_folder, f"slice_{idx:03d}.png")
    cv2.imwrite(out_path, imag)
    sheet.append([idx, area, inner_perimeter, outer_perimeter , GI_slice])

p.close()

# Totals in real units
GI_total = sum_inner / sum_outer if sum_outer > 0 else 0

print(f"GI_total equls to {GI_total}")

sheet.append([
    "GI", round(GI_total, 3)
])

# Save to Excel
# Save to Excel
df = pd.DataFrame(sheet, columns=["Slice", "Area ", "Inner_Perimeter ", "Outer_Perimeter ", "GI"])
outfile_name = "Gyrification_Index_Results.xlsx"
output_path = directory / outfile_name
df.to_excel(output_path, index=False)

print("✅ STL-based Gyrification index computed and saved!")

# Build solid from contours
solids = []
for cnt in all_3d_contours:
    if cnt.shape[0] < 3:
        continue
    if not np.allclose(cnt[0], cnt[-1]):
        cnt = np.vstack([cnt, cnt[0]])
    faces = np.hstack([[cnt.shape[0]], np.arange(cnt.shape[0])])
    surf = pv.PolyData(cnt, faces=faces)
    try:
        surf = surf.triangulate()
        extruded = surf.extrude([0, 0, slice_thickness_eff], capping=True)
    except:
        continue
    solids.append(extruded)

if solids:
    solid_volume = solids[0].copy()
    for part in solids[1:]:
        solid_volume = solid_volume.merge(part)
    solid_volume.save("solid_volume_convex.stl")
