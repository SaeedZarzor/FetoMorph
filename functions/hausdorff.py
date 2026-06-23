"""Hausdorff distance computation between two brain-slice contours.

Workflow:
    1. ``convert_image`` extracts contour coordinates from a brain-slice
       image and scales them to physical units.
    2. ``calculate_hausdorff_distance`` aligns the two point sets and
       computes the directed and symmetric Hausdorff distances.

**Why alignment is needed:** contours from different imaging modalities
(e.g. MRI vs simulation) may be offset in pixel space even though they
represent the same anatomy.  Alignment modes (``right_bottom``,
``left_top``, ``centroid``) translate the first contour so that a
reference corner or centroid matches the second.
"""

from __future__ import annotations

from deps import *
from PIL import Image
from scipy.spatial.distance import directed_hausdorff, cdist
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from helpers.helpers import image_annotation_style, threshold_binary


def _to_xy2(a: np.ndarray) -> np.ndarray:
    """Ensure *a* is an ``(N, 2)`` float array."""
    a = np.squeeze(np.asarray(a)).astype(float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Expected (N,2) array")
    return a

def _align_points(c1: np.ndarray, c2: np.ndarray, mode: str = "right_bottom") -> tuple[np.ndarray, tuple[float, float]]:
    """Translate *c1* so that a reference anchor matches *c2*.

    Alignment modes:
        * ``"right_bottom"``: align maximum X and Y coordinates.
        * ``"left_top"``:     align minimum X and Y coordinates.
        * ``"centroid"``:     align centroids.

    Returns:
        Tuple of ``(aligned_c1, (dx, dy))``.
    """
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
        image_path: str, out_dir: str,
        pixel_spacing: float = 0.01,
        min_contour_area: float = 200) -> tuple[np.ndarray | None, str | None, np.ndarray | None]:
    """Extract contour coordinates from a brain-slice image.

    Thresholds the image, filters contours by area, scales to physical
    units, and saves an annotated copy.

    Args:
        image_path: Path to the image file.
        out_dir: Directory for the annotated output image.
        pixel_spacing: mm per pixel for coordinate scaling.
        min_contour_area: Minimum contour area (pixels) to keep.

    Returns:
        Tuple of ``(annotated_bgr, basename, contour_coords_mm)``
        or ``(None, None, None)`` if the image can't be loaded or
        no contours are found.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"[Hausdorff] Error: Could not load image from {image_path}")
        return None, None, None
    
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to grayscale
    im_bw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold
    im_bw = threshold_binary(im_bw, 200, invert=True)
    
    # Display binary/threshold image
#    cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
#    cv2.imshow('Binary Image', im_bw)
    
    # Find contours
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) * (pixel_spacing ** 2) > min_contour_area]
    
    annotated = image.copy()
    W, H = annotated.shape[:2]
    thickness, _, _ = image_annotation_style(H, W, style="bold")
    
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
        return None, None, None
        
        
def calculate_hausdorff_distance(
    contours_coords_first: np.ndarray | None, contours_coords_second: np.ndarray | None,
    First_label: str = "First", Second_label: str = "Second",
    invert_y: bool = True, align_mode: str = "right_bottom",
    out_dir: str | None = None, filename: str = "Hausdorff_distance_plot.png") -> tuple[np.ndarray | None, float | None, float | None, float | None]:
    """Compute symmetric Hausdorff distance between two 2-D contour sets.

    Optionally aligns the first contour to the second before measuring
    (see ``_align_points`` for alignment modes).

    Args:
        contours_coords_first: ``(N, 2)`` array of the first contour.
        contours_coords_second: ``(M, 2)`` array of the second contour.
        First_label: Legend label for the first contour.
        Second_label: Legend label for the second contour.
        invert_y: If True, invert Y axis on the plot (image convention).
        align_mode: ``"right_bottom"``, ``"left_top"``, ``"centroid"``, or
            ``"none"`` to skip alignment.
        out_dir: Directory to save the plot.
        filename: Output file name.

    Returns:
        Tuple of ``(plot_rgba, hausdorff_dist, d12, d21)`` or ``(None, …)``.
    """
    if contours_coords_first is None or contours_coords_second is None:
        print("[Hausdorff] Error: inputs are None")
        return None, None, None, None

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)

    c1 = _to_xy2(contours_coords_first)
    c2 = _to_xy2(contours_coords_second)

    if align_mode != "none" :
        # align SIM to MRI (to the right and down by default)
        c1, shift = _align_points(c1, c2, mode=align_mode)

    # Hausdorff
    d12, i12_c1, j12_c2 = directed_hausdorff(c1, c2)
    d21, j21_c2, i21_c1 = directed_hausdorff(c2, c1)
    hd  = max(d12, d21)

    p12 = c1[i12_c1]   # [x,y]
    q12 = c2[j12_c2]           # [x,y]
    p21 = c2[j21_c2]
    q21 = c1[i21_c1]
    

    # plot
    fig, ax = plt.subplots()

    ax.plot(c1[:,0], c1[:,1], linewidth=1.5,
            label=f"{First_label}")
    ax.plot(c2[:,0], c2[:,1], linewidth=1.5, label=f"{Second_label}")
    ax.scatter(c1[:,0], c1[:,1], s=4)
    ax.scatter(c2[:,0], c2[:,1], s=4)

    # draw Hausdorff segments (note the lists)
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
   


