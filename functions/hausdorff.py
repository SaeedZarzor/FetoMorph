import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
from scipy.spatial.distance import directed_hausdorff, cdist
import matplotlib.pyplot as plt
from helpers.Helpers import text_thickness
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def _to_xy2(a):
    a = np.squeeze(np.asarray(a)).astype(float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Expected (N,2) array")
    return a

def _align_points(c1, c2, mode="right_bottom"):
    if mode == "right_bottom":
        dx = c2[:,0].max() - c1[:,0].max()
        dy = c2[:,1].max() - c1[:,1].max()
    elif mode == "left_top":
        dx = c2[:,0].min() - c1[:,0].min()
        dy = c2[:,1].min() - c1[:,1].min()
    elif mode == "centroid":
        dx = c2[:,0].mean() - c1[:,0].mean()
        dy = c2[:,1].mean() - c1[:,1].mean()
    else:
        raise ValueError("mode must be 'right_bottom', 'left_top', or 'centroid'")
    return c1 + np.array([dx, dy]), (dx, dy)


def convert_image(
        image_path, out_dir,
        pixel_spacing: float = 0.01,
        min_contour_area: float =200):

    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"[Hausdorff] Error: Could not load image from {image_path}")
        return
    
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to grayscale
    im_bw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold
    (thresh, im_bw) = cv2.threshold(im_bw, 200, 255, 1)
    
    # Display binary/threshold image
#    cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
#    cv2.imshow('Binary Image', im_bw)
    
    # Find contours
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    annotated = image.copy()
    W, H = annotated.shape[:2]
    thickness = text_thickness(H, style="thin")
    
    if len(filtered_contours) > 0:

        # Draw contours on original image
        annotated = cv2.drawContours(annotated, filtered_contours, -1, (0, 255, 255), thickness)
        
        # Extract contour coordinates
        contour_coordinates = []
        for contour in filtered_contours:
            # Each contour is a NumPy array of shape (n, 1, 2)
            for point in contour:
                x, y = point[0]  # Extract (x, y) coordinates
                contour_coordinates.append([x, y])  # Append coordinates to list
        
        # Convert the list to a NumPy array
        contour_coordinates_array = np.array(contour_coordinates)
        
        print(f"[Hausdorff] Total contour points: {len(contour_coordinates_array)}")
        
          # Force conversion to float, removing any string data
        contour_coordinates_numeric = contour_coordinates_array.astype(float)
         # Now multiply with pixel spacing

        corrected_pixel_array = contour_coordinates_numeric * pixel_spacing


        os.makedirs(out_dir, exist_ok=True)
        basename = Path(image_path).stem
        path = os.path.join(out_dir, f"{basename}_annotated.png")
        cv2.imwrite(path, annotated)

        return annotated, basename, corrected_pixel_array
        
    else:
        print("[Hausdorff] No contours found!")
        print("Try adjusting the threshold value.")
        return None, None
        
        
def calculate_hausdorff_distance(
    contours_coords_first, contours_coords_second,
    First_label="First", Second_label="Second",
    invert_y=True, align_mode="right_bottom",
    out_dir=None, filename="Hausdorff_distance_plot.png"):
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    
    if contours_coords_first is None or contours_coords_second is None:
        print("[Hausdorff] Error: inputs are None"); return None, None, None

    c1 = _to_xy2(contours_coords_first)
    c2 = _to_xy2(contours_coords_second)

    # align SIM to MRI (to the right and down by default)
    c1_aligned, shift = _align_points(c1, c2, mode=align_mode)

    # Hausdorff
    d12 = directed_hausdorff(c1_aligned, c2)[0]
    d21 = directed_hausdorff(c2, c1_aligned)[0]
    hd  = max(d12, d21)
    
    # ----- find the actual extremal pairs to draw -----
    # First->Second (d12)
    D12 = cdist(c1_aligned, c2)                 # shape (N,M)
    nn12_idx = D12.argmin(axis=1)               # nearest B index for each A point
    min12     = D12[np.arange(D12.shape[0]), nn12_idx]
    i12 = int(min12.argmax())                   # A index giving the max of mins
    j12 = int(nn12_idx[i12])                    # its nearest B index
    p12 = c1_aligned[i12]                       # point in First (aligned)
    q12 = c2[j12]                               # nearest in Second

    # Second->First (d21)
    D21 = D12.T                                 # reuse by transposing
    nn21_idx = D21.argmin(axis=1)               # nearest A index for each B point
    min21     = D21[np.arange(D21.shape[0]), nn21_idx]
    j21 = int(min21.argmax())                   # B index giving the max of mins
    i21 = int(nn21_idx[j21])                    # its nearest A index
    p21 = c2[j21]                               # point in Second
    q21 = c1_aligned[i21]                       # nearest in First (aligned)


    # plot
    fig, ax = plt.subplots()

    ax.plot(c1_aligned[:,0], c1_aligned[:,1], linewidth=1.5,
            label=f"{First_label} aligned ({shift[0]:.2f},{shift[1]:.2f})")
    ax.plot(c2[:,0], c2[:,1], linewidth=1.5, label=f"{Second_label}")
    ax.scatter(c1_aligned[:,0], c1_aligned[:,1], s=4)
    ax.scatter(c2[:,0], c2[:,1], s=4)

    #draw Hausdorff segments
    ax.plot([p12[0], q12[0]], [p12[1], q12[1]], linestyle='--', linewidth=2, label=f"d12={d12:.3f}")
    ax.plot([p21[0], q21[0]], [p21[1], q21[1]], linestyle='--', linewidth=2, label=f"d21={d21:.3f}")
    
    # emphasize endpoints
    ax.scatter([p12[0], q12[0]], [p12[1], q12[1]], s=30, marker='o')
    ax.scatter([p21[0], q21[0]], [p21[1], q21[1]], s=30, marker='s')
    
    # annotate near midpoints
    m12 = (p12 + q12) / 2.0
    m21 = (p21 + q21) / 2.0
    ax.text(m12[0], m12[1], f"d12={d12:.3f}", fontsize=9, va='bottom', ha='center')
    ax.text(m21[0], m21[1], f"d21={d21:.3f}", fontsize=9, va='bottom', ha='center')
    
    ax.set_aspect('equal', adjustable='box')
    if invert_y:
        ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Hausdorff: {hd:.3f}  (d12={d12:.3f}, d21={d21:.3f})")
    fig.tight_layout()

    if out_dir:
        fig.savefig(path, dpi=300, bbox_inches='tight')
    
    print(f"[Hausdorff] the image {filename} has been saved")

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    
    return img, hd, d12, d21
   


