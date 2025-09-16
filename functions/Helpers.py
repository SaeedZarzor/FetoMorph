# helpers.py
from deps import *

def text_thickness(H, style="regular", cap=10):
    base_div = {"thin": 380, "regular": 320, "bold": 260}[style]  # ↑ bigger divisors = thinner
    t = int(round(H / base_div))
    return max(1, min(t, cap))
    
    
def compute_kernel_convex(kernel_size):

    kernel_convex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    return kernel_convex

def defect_mm_per_px_and_fixed(
    start: Tuple[float, float],
    end:   Tuple[float, float],
    far:   Tuple[float, float],
    sx: float,   # mm per pixel in x
    sz: float,   # mm per pixel in y (your z)
) -> Tuple[Optional[float], Optional[float]]:
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

def get_nifti_present_labels(path: str, cap: int = 5000)-> list[int]:
   
    try:
        if path is not None or data is None:
            import nibabel as nib
            img = nib.load(path or self.current_path)
            # Use dataobj (lazy) but rounding requires actual values; this will page from disk
            data = img.get_fdata(dtype=float)
            
        arr_i = np.rint(data).astype(np.int32)
        uniq = np.unique(arr_i)
        uniq = uniq[(uniq >= 0) & (uniq <= cap)]
        uniq_list = set(uniq.tolist())
        print(f"Region labels: \n {uniq_list} \n")
        return uniq_list
    except Exception as ex:
        print(f"[Regions] Could not detect labels: {ex}")
        # Fall back to defaults
        return None
        
def _mm_per_pixel_x_for_axis(zooms, ax):
    """Return the in-plane mm/px for the displayed X axis given the slice axis."""
    # zooms is (z0,z1,z2) == voxel size along axes 0,1,2 in mm
    if ax == 0:               # slice is a[i, :, :]
        return float(zooms[2])  # X shows axis 2
    elif ax == 1:             # slice is a[:, i, :]
        return float(zooms[2])  # X shows axis 2
    else:                     # ax == 2: slice is a[:, :, i]
        return float(zooms[1])  # X shows axis 1

def add_scalebar(qimg: QImage, zooms, ax) -> QImage:
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
