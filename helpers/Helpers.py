"""Shared helper functions for FetoMorph measurement pipelines.

Provides utilities for OpenCV text sizing, morphological kernels, convexity-
defect depth conversion (pixel ↔ mm), red-cube scale calibration, scalebar
drawing, and PyVista slice geometry.
"""

from __future__ import annotations

import os
import math
import logging
from pathlib import Path

import numpy as np
import cv2
import pyvista as pv
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from functions.Nifti2image import draw_new_scale_bar

logger = logging.getLogger(__name__)

def text_thickness(H: int, style: str = "regular", cap: int = 10) -> int:
    """Compute an OpenCV line thickness that scales with image height.

    Args:
        H: Image height in pixels.
        style: One of ``"thin"``, ``"regular"``, ``"bold"``.
        cap: Maximum thickness returned.

    Returns:
        Integer thickness clamped to ``[1, cap]``.
    """
    base_div = {"thin": 380, "regular": 320, "bold": 260}[style]  # ↑ bigger divisors = thinner
    t = int(round(H / base_div))
    return max(1, min(t, cap))
    
    
def compute_kernel_convex(kernel_size: int) -> np.ndarray:
    """Create an elliptical morphological structuring element.

    Used by the GI (gyrification index) pipeline to morphologically close
    sulci before computing the "outer" (convex) perimeter.

    Args:
        kernel_size: Diameter of the ellipse in pixels.

    Returns:
        A ``uint8`` structuring element suitable for ``cv2.morphologyEx``.
    """
    kernel_convex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    return kernel_convex

def defect_mm_per_px_and_fixed(
    start: tuple[float, float],
    end:   tuple[float, float],
    far:   tuple[float, float],
    sx: float,   # mm per pixel in x
    sz: float,   # mm per pixel in y (your z)
) -> tuple[float | None, float | None]:
    """
    Returns:
        mm_per_px     : mm per 1 pixel along the defect's normal direction
        mm_per_fixed  : mm per 1 raw 'd' unit (remember: depth_px = d/256)
    """
    a = np.asarray(start, float); b = np.asarray(end, float); f = np.asarray(far, float)
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 == 0.0:
        return None, None  # degenerate edge

    # projection of 'far' onto the (infinite) line through start-end
    t = float(np.dot(f - a, ab) / ab2)
    p = a + t * ab

    # normal vector in pixel space (direction of the defect)
    v = f - p
    n = float(np.hypot(v[0], v[1]))
    if n == 0.0:
        return 0.0, 0.0  # zero-depth defect

    ux, uy = v[0]/n, v[1]/n  # unit normal (pixels)
    mm_per_px = math.hypot(sx * ux, sz * uy)
    mm_per_fixed = mm_per_px / 256.0  # because OpenCV stores d in 8.8 fixed-point
    return mm_per_px, mm_per_fixed

def contours_exclude(contours: list, excluded_space: np.ndarray, image_shape: tuple[int, int]) -> list:
    # Needed to remove the red reference-cube contour from the brain contours
    # so that it does not pollute area / perimeter measurements.
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
    
def calc_scale(image_rgb: np.ndarray, cube_length: float) -> float | None:
    """
    Compute mm-per-pixel from a red reference cube drawn in the render.
    cube_length_mm: the real cube side length (x_length) in mm.
    """
    red_rect = np.where((image_rgb[:, :, 0] > 150) & (image_rgb[:, :, 1] < 50), 255, 0).astype("uint8")
    _, thresh_red = cv2.threshold(red_rect, 150, 255, 0)
    contours, _ = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[Scale] No red reference contour found; default scale 1.0 mm/px")
        return 1.0
    # Use the largest red blob as reference
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if float(w)>0:
        scale = (cube_length/float(w))
    else:
        print("[STL Scale] Cup length error: cup length not found!")
        return None
    return scale
    
def get_red_rect_offset(image_rgb: np.ndarray) -> np.ndarray:  # noqa: returning shape (2,)
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

def get_nifti_present_labels(path: str, cap: int = 5000) -> set[int] | None:
    """Return the set of unique integer labels present in a NIfTI file.

    Args:
        path: Path to the ``.nii`` / ``.nii.gz`` file.
        cap: Ignore label values above this threshold (avoids noise).

    Returns:
        A set of integer labels, or ``None`` on failure.
    """
    try:
        import nibabel as nib
        img = nib.load(path)
        data = img.get_fdata(dtype=float)

        arr_i = np.rint(data).astype(np.int32)
        uniq = np.unique(arr_i)
        uniq = uniq[(uniq >= 0) & (uniq <= cap)]
        uniq_list = set(uniq.tolist())
        print(f"Region labels: \n {uniq_list} \n")
        return uniq_list
    except Exception as ex:
        logger.warning("Could not detect labels: %s", ex)
        # Fall back to defaults
        return None
        
def _mm_per_pixel_x_for_axis(zooms: tuple, ax: int) -> float:
    """Return the in-plane mm/px for the displayed X axis given the slice axis."""
    # zooms is (z0,z1,z2) == voxel size along axes 0,1,2 in mm
    if ax == 0:               # slice is a[i, :, :]
        return float(zooms[2])  # X shows axis 2
    elif ax == 1:             # slice is a[:, i, :]
        return float(zooms[2])  # X shows axis 2
    else:                     # ax == 2: slice is a[:, :, i]
        return float(zooms[1])  # X shows axis 1

def add_scalebar(qimg: QImage, zooms: np.ndarray, ax: int) -> tuple[QImage, float, float]:
    """Draw a scalebar (mm) at the bottom-right of qimg and return it."""
    # QPainter needs a 32-bit RGB(A) surface for best compatibility
    if qimg.format() not in (QImage.Format_RGB32, QImage.Format_ARGB32):
        qimg = qimg.convertToFormat(QImage.Format_RGB32)

    w, h = qimg.width(), qimg.height()
    mm_per_px = _mm_per_pixel_x_for_axis(zooms, ax)

    # Pick a nice bar length (mm) that fits ~25% of the width
    max_px = int(w * 0.25)
    nice_lengths_mm = [100, 50, 25, 20, 10, 5]
    bar_mm = next((L for L in nice_lengths_mm if (L / mm_per_px) <= max_px and (L / mm_per_px) >= 30), None)
    if bar_mm is None:
        # fallback to whatever fits (at least 20 px)
        bar_mm = max(5, int(max_px * mm_per_px))
    bar_px = int(round(bar_mm / mm_per_px))

    margin = max(6, int(round(0.03 * min(w, h))))
    bar_thick = max(4, int(round(0.008 * min(w, h))))
    label_h = max(12, int(round(0.028 * min(w, h))))

    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # Backdrop for contrast
    pad = 6
    rect_w = bar_px + 2 * pad
    rect_h = bar_thick + label_h + 3 * pad
    rect_x = w - margin - rect_w
    rect_y = h - margin - rect_h
    painter.fillRect(rect_x, rect_y, rect_w, rect_h, QColor(0, 0, 0, 160))

    # Scalebar line (white)
    y_bar = rect_y + pad + bar_thick // 2
    pen = QPen(QColor(255, 255, 255))
    pen.setWidth(bar_thick)
    painter.setPen(pen)
    x1 = rect_x + pad
    x2 = x1 + bar_px
    painter.drawLine(x1, y_bar, x2, y_bar)

    # Text (e.g., "20 mm")
    painter.setPen(QColor(255, 255, 255))
    font = painter.font()
    font.setPointSizeF(max(8.0, 0.9 * label_h))
    painter.setFont(font)
    text_rect = QRectF(x1, y_bar + pad, bar_px, label_h + pad)
    painter.drawText(text_rect, Qt.AlignCenter, f"{int(round(bar_mm))} mm")

    painter.end()
    return qimg, mm_per_px, bar_mm

def _add_scalebar_on_annotated(
    annotated: np.ndarray,
    pixel_size: float,
    unit: str,
    add_scalebar: bool = True,
) -> np.ndarray:
    """Optionally draw a physical scale bar on an annotated BGR image."""
    if not add_scalebar or pixel_size <= 0:
        return annotated

    image_width_phys = annotated.shape[1] * pixel_size
    target = image_width_phys * 0.2
    magnitude = 10 ** int(np.floor(np.log10(max(target, 1e-9))))
    bar_phys = next(
        (magnitude * n for n in [1, 2, 5, 10] if magnitude * n >= target * 0.7),
        magnitude * 10,
    )
    bar_px = max(1, int(round(bar_phys / pixel_size)))
    return draw_new_scale_bar(annotated, bar_px, text=f"{bar_phys:g} {unit}")


def get_max_slice_thickness(path: str) -> float | None:
    """Return the smallest bounding-box dimension of an STL/VTK mesh.

    This gives the maximum sensible slice thickness for the mesh — slicing
    along the smallest axis with a larger step would produce no slices.

    Args:
        path: Path to an ``.stl`` or ``.vtk`` mesh file.

    Returns:
        The smallest extent (mm) or ``None`` for unsupported formats.
    """
    ext = Path(path).suffix.lower()
    mesh = pv.read(str(path))

    if ext in (".stl", ".vtk"):
        # ensure we have polygonal surface
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()

        x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
        dims = [x_max - x_min, y_max - y_min, z_max - z_min]
        return min(dims)

    return None


def slice_at(mesh: pv.DataSet, Slice_direction: str, s: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute the slicing normal and origin for a PyVista mesh.

    Args:
        mesh: Any PyVista dataset (will be converted to PolyData if needed).
        Slice_direction: ``"X"``, ``"Y"``, or ``"Z"``.
        s: Position along the slice axis.

    Returns:
        Tuple of ``(normal, origin)`` ready for ``mesh.slice()``.
    """
    # ensure polygonal surface
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    c = mesh.center  # (cx, cy, cz)

    axis = {
        "X": ( (1.0, 0.0, 0.0), lambda s: (float(s), c[1], c[2]) ),
        "Y": ( (0.0, 1.0, 0.0), lambda s: (c[0], float(s), c[2]) ),
        "Z": ( (0.0, 0.0, 1.0), lambda s: (c[0], c[1], float(s)) ),
    }.get(Slice_direction)

    if axis is None:
        raise ValueError("Slice_direction must be 'X','Y','Z'.")

    normal, origin_fn = axis
    origin = origin_fn(s)
    return normal, origin

def make_scale_cube(Slice_direction: str, cube_len: float, origin, s: float, offset: float = 50.0) -> pv.PolyData:
    """Create a thin red reference cube for scale calibration in renders.

    The cube is placed next to the slice cross-section.  Its known side length
    (``cube_len``) is later detected in the screenshot to compute mm-per-pixel.

    Args:
        Slice_direction: ``"X"``, ``"Y"``, or ``"Z"``.
        cube_len: Side length of the cube face parallel to the slice plane.
        origin: Mesh center ``(x, y, z)``.
        s: Current slice position along the slice axis.
        offset: Translation away from the mesh centre to avoid overlap.

    Returns:
        A ``pv.PolyData`` cube positioned beside the slice.
    """
    # choose thin axis and translation vector
    if Slice_direction == "X":
        c = (s, origin[1], origin[2])
        cube = pv.Cube(center = c, x_length=0.01, y_length=cube_len, z_length=cube_len)
        cube.translate((-0.05, offset, offset), inplace=True)
    elif Slice_direction == "Y":
        c = (origin[0],s, origin[2])
        cube = pv.Cube(center = c, x_length=cube_len, y_length=0.01, z_length=cube_len)
        cube.translate((offset,-0.05 , offset), inplace=True)
    elif Slice_direction == "Z":
        c = (origin[0], origin[1], s)
        cube = pv.Cube(center = c, x_length=cube_len, y_length=cube_len, z_length=0.01)
        cube.translate((offset, offset, -0.05), inplace=True)
    else:
        raise ValueError("Slice_direction must be 'X','Y','Z'.")

    return cube

def compactness_2D(area: float, perimeter: float) -> float:
    if perimeter == 0:
        return 0
    return (4 * 3.141592653589793 * area) / (perimeter ** 2)

def compactness_3D(volume: float, surface_area: float) -> float:
    if surface_area == 0:
        return 0
    return (36 * 3.141592653589793 * (volume ** 2)) / (surface_area ** 3)
