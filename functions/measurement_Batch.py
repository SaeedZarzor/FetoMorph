"""Batch morphometric measurement of brain slice images.

Processes a directory of 2-D slice images, extracts contours,
computes area, perimeter, local Gyrification Index (LGI), and
convexity-defect depths, then writes annotated PNGs and an Excel
summary.
"""

import cv2
import os
import numpy as np
from typing import Tuple, Union
from helpers.Helpers import text_thickness, compute_kernel_convex, compactness_2D
from constants import BINARY_THRESHOLD_DEFAULT, DEFECT_FIXED_POINT
import pandas as pd

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
    count = 0
    out_dir_Batch = os.path.join(out_dir, "image_Batch")
    os.makedirs(out_dir_Batch, exist_ok=True)
    print(f"[Process Batch] Temp output dir: {out_dir}")

    font_scale = 0.01/pixel_size
    margin = 6
    radius_px = int(round(0.1 / pixel_size))

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
        thickness = text_thickness(H, style="thin")

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
            
        depth = []
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
                        if (d * pixel_size / DEFECT_FIXED_POINT) > 0.5:
                            annotated = cv2.circle(annotated, tuple(map(int, far)), radius_px, [255, 255, 0], -1)
                            depth.append(d * pixel_size / DEFECT_FIXED_POINT )

                depth.sort(reverse=True)

        new_name= f"{file_name}_measured.png"
        new_path = os.path.join(out_dir_Batch, new_name)
        cv2.imwrite(new_path, annotated)
        
        valid_slices.append(idx)
        saved_pngs.append(new_path)
        count +=1
        
        mean_depth = (sum(depth)/len(depth)) if depth else None

        sheet1.append([file_name,area, perimeter, perimeter_convex, perimeter_rate, compactness, len(depth), (max(depth) if depth else None), (min(depth) if depth else None), mean_depth])


    sheet1.append(['PixelSize:', pixel_size])
    sheet1.append(['PixelSizeUnits:', unit])
    sheet1.append(['KernelSize:', kernel_size])
    fd = pd.DataFrame(data=sheet1, columns=['File ', 'area', 'perimeter', 'perimeter_convex', 'LGI', 'Compactness', 'SulciCount', 'MaxDepth', 'MinDepth', 'MeanDepth'])

    xlsx_path = os.path.join(out_dir, "Batch_Allmarks.xlsx")
    fd.to_excel(xlsx_path, index=False)
    print(f"[Process Batch] {count} images in {directory_path} have been processed")
    return valid_slices, saved_pngs
    
