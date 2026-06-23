#!/usr/bin/python
"""Curvature analysis on binary brain-slice masks.

For each contour point, the local curvature is estimated by:
    1. Extracting a neighbourhood window along the contour.
    2. Rotating the neighbourhood into a local coordinate frame aligned
       with the tangent direction — this makes the curvature computation
       independent of the curve's orientation.
    3. Fitting a degree-2 polynomial to the rotated points.
    4. Taking the **2nd derivative** of the polynomial.  Its sign indicates
       whether the surface is **convex** (positive) or **concave** (negative)
       at that point.

Author: Stefan Herdy (14.11.2023), adapted for FetoMorph.
"""

from __future__ import annotations

from deps import *
from skimage import measure
from shapely.geometry import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def rgb_to_bw_mask(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Convert a BGR image to a binary mask using Otsu thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold to create mask
    _, mask = cv2.threshold(gray, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

def compute_curvature(point: np.ndarray, i: int, contour: np.ndarray, window_size: int) -> float:
    """Compute local curvature at a contour point via rotated polynomial fitting.

    The neighbourhood is rotated so that the local tangent aligns with the
    X axis.  A degree-2 polynomial is fitted; its 2nd derivative gives the
    curvature sign (positive = convex, negative = concave).

    Args:
        point: The ``(row, col)`` contour point.
        i: Index of *point* within *contour*.
        contour: Full contour array ``(N, 2)``.
        window_size: Number of neighbouring points to include.

    Returns:
        Mean curvature (2nd derivative) over the neighbourhood window.
    """
    start = max(0, i - window_size // 2)
    end = min(len(contour), i + window_size // 2 + 1)
    neighborhood = contour[start:end]

    # Extract x and y coordinates from the neighborhood
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]

    # Compute the tangent direction over the entire neighborhood and rotate the points
    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood), np.gradient(x_neighborhood))
    tangent_direction_original.fill(tangent_direction_original[len(tangent_direction_original)//2])

    # Translate the neighborhood points to the central point
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]


    # Apply rotation to the translated neighborhood points
    # We have to rotate the oints to be able to compute the curvature independend of the local orientation of the curve
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    # Fit a polynomial of degree 2 to the rotated coordinates
    coeffs = np.polyfit(rotated_x, rotated_y, 2)


    # You can compute the curvature using the formula: curvature = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
    # dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
    # d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)
    # curvature = np.abs(d2y_dx2) / np.power(1 + np.power(dy_dx, 2), 1.5)

    # We compute the 2nd derivative in order to determine wether the curve at the certain point is convex or concave
    curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)

    # Return the mean curvature for the central point
    return np.mean(curvature)

def filter_contours_by_area(mask: np.ndarray, min_area: float = 500, pixel_size: float = 1.0) -> list[np.ndarray]:
    """Return contours whose enclosed polygon area exceeds *min_area* (mm²).

    ``pixel_size`` (mm/px) converts the polygon's pixel area to mm² so the
    threshold is physical, matching the rest of the filtered-threshold setting.
    """
    contours = measure.find_contours(mask, 0.5)
    filtered = []
    for c in contours:
        if len(c) >= 3:  # must form a polygon
            poly = Polygon(c[:, ::-1])  # swap to (x, y)
            if poly.area * (pixel_size ** 2) >= min_area:
                filtered.append(c)
    return filtered

def compute_curvature_profile(path: str, window_size_ratio: int = 5, second_derivative: bool = True, min_area: float = 20, pixel_size: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Compute curvature at every contour point in a brain-slice image.

    Args:
        path: Path to the image file.
        window_size_ratio: Contour length is divided by this to set the
            neighbourhood window size.
        second_derivative: Unused (kept for API compatibility).
        min_area: Minimum polygon area (mm²) for contour filtering.
        pixel_size: Physical size of one pixel (mm/px) for the area conversion.

    Returns:
        Tuple of ``(mask, edge_pixels, curvature_values, curvature_signs)``.
    """

    img = cv2.imread(str(path))
    mask = rgb_to_bw_mask(img)

    contours = filter_contours_by_area(mask, min_area, pixel_size)

    # Initialize arrays to store the curvature information for each edge pixel
    curvature_values = []
    curvature_values_s = []
    edge_pixels = []

    # Iterate over each contour
    for contour in contours:
        # Iterate over each point in the contour
        for i, point in enumerate(contour):
#            if contour.shape[0] > min_contour_length:
                # Compute the curvature for the point
                # We set the window size to 1/5 of the whole contour edge. Adjust this value according to your specific task
            window_size = int(contour.shape[0]/window_size_ratio)
            curvature = compute_curvature(point, i, contour, window_size)
                # We compute, whether a point is convex or concave.
                # Store curvature information and corresponding edge pixel
            curvature_values.append(curvature)
            curvature_values_s.append(1 if curvature > 0 else -1)
            edge_pixels.append(point)

    # Convert lists to numpy arrays for further processing
    curvature_values = np.array(curvature_values)
    edge_pixels = np.array(edge_pixels)

    return mask, edge_pixels, curvature_values, curvature_values_s

def save_curvature_plot(out_dir: str, mask: np.ndarray, edge_pixels: np.ndarray, curvature_values: np.ndarray, filename: str = "curvature_plot.png") -> np.ndarray:
    """Render a jet-coloured curvature overlay on the mask and save to disk.

    Args:
        out_dir: Output directory.
        mask: Binary mask image.
        edge_pixels: ``(N, 2)`` array of contour point coordinates.
        curvature_values: ``(N,)`` array of curvature values.
        filename: Output file name.

    Returns:
        RGBA image as a NumPy array (for display in the viewer).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    fig, ax = plt.subplots()
    ax.imshow(mask, cmap='gray')
    threshold = np.percentile(np.abs(curvature_values), 98)
    sc = ax.scatter(edge_pixels[:, 1], edge_pixels[:, 0],
                c=curvature_values, cmap='jet', s=5,
                vmin=-threshold, vmax=threshold)
    fig.colorbar(sc, ax=ax, label='Curvature')
    ax.set_title("Curvature of Binary Mask")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"[Curvature] the image {filename} has been saved")
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return img
