"""Aspect-fit image label with interactive measurement and selection modes.

Extends QLabel to draw a pixmap with preserved aspect ratio and provides
two interactive overlay modes: a click-drag line for scale-bar measurement
and a rubber-band rectangle for region-of-interest selection.  Persistent
rectangle annotations can also be added programmatically.
"""

from deps import *
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPolygonF


MASK_COLOR = QColor(255, 60, 60)
MASK_OPACITY = 0.45
MASK_UNDO_LIMIT = 30
SPLINE_COLOR = QColor(255, 220, 0)
SPLINE_HANDLE_PX = 9          # widget-px radius for grabbing/closing on a point
SPLINE_SAMPLES_PER_SEG = 24


class ScaledImageLabel(QLabel):
    """
    Aspect-fit image label with:
      - scalebar measurement mode (click-drag line, reports pixel length)
      - square selection mode with QRubberBand, returns image-rect + cropped QImage
      - mask painting mode (spline or brush/eraser in image pixels, image-sized overlay)
    It draws self._pix manually; QLabel's built-in pixmap is *not* used.
    """
    zoomChanged = Signal(float)
    maskBrushChanged = Signal(int)
    maskModeChanged = Signal(bool)

    def __init__(self, parent=None):
        """Initialise the scaled image label.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        # general appearance/behavior
        self._pix = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111; color:#ccc; border:1px solid #333;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(120, 90)
        self.setMouseTracking(True)            # we want move events while dragging
        self.setFocusPolicy(Qt.StrongFocus)    # so Esc cancels selection
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._zoom_factor: float = 1.0
        self._zoom_min: float = 0.1
        self._zoom_max: float = 8.0
        self._pan_active: bool = False
        self._pan_last_pos: QPoint | None = None
        self._pan_dx: int = 0
        self._pan_dy: int = 0

        # ---- scalebar measurement state ----
        self._measure_active: bool = False
        self._measure_p1_img: QPoint | None = None
        self._measure_p2_img: QPoint | None = None
        self._measure_finished_cb = None  # callable(pixel_length_px: float) -> None

        # ---- square selection state ----
        self._sel_active: bool = False
        self._sel_origin: QPoint | None = None     # widget coords (inside drawn rect)
        self._sel_callback = None                  # callable(QRect img_rect, QImage cropped)
        self._rubber: QRubberBand | None = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber.hide()
        self._annots: list[dict] = []

        # ---- straight-line measurement state ----
        self._line_measure_active: bool = False
        self._line_p1_img: QPoint | None = None
        self._line_p2_img: QPoint | None = None
        self._line_measure_cb = None  # callable(pixel_length, p1_img, p2_img)

        # ---- persistent line annotations ----
        self._line_annots: list[dict] = []  # [{p1, p2, label, color}]

        # ---- mask painting state ----
        # ARGB32 overlay with the same size as the pixmap; alpha > 0 = masked.
        self._mask_img: QImage | None = None
        self._mask_active: bool = False
        self._mask_erase: bool = False           # eraser toggle (right-drag always erases)
        self._mask_stroke_erase: bool = False    # erase state of the stroke in progress
        self._mask_brush_px: int = 10            # brush diameter in *image* pixels
        self._mask_last_img: QPointF | None = None
        self._mask_cursor_w: QPoint | None = None
        self._mask_undo: list[QImage] = []
        self._mask_backup: QImage | None = None  # restored on cancel
        self._mask_finished_cb = None            # callable(np.ndarray bool HxW) -> None
        self._mask_cancel_cb = None
        self._mask_tool: str = "spline"          # "spline" | "brush"
        self._spline_pts: list[QPointF] = []     # pending control points (image coords)
        self._spline_drag_idx: int | None = None
        self._spline_corner: list[bool] = []     # parallel to _spline_pts; True = sharp corner
        self._spline_closed_snapshot = None      # (pts, corners) of a shape closed by the last click

    # ---------------- public API ----------------
    def hasImage(self) -> bool:
        """Return True if a non-null pixmap is loaded."""
        return not self._pix.isNull()

    def imageSize(self) -> QSize:
        """Return the original (unscaled) size of the loaded pixmap."""
        return self._pix.size()

    def setImage(self, pm: QPixmap | None):
        """Replace the displayed pixmap and trigger a repaint.

        Args:
            pm: New pixmap, or None to clear.
        """
        self._pix = pm if pm is not None else QPixmap()
        self._zoom_factor = 1.0
        self._pan_dx = 0
        self._pan_dy = 0
        self._drop_mask()
        self.update()

    def clearImage(self):
        """Remove the displayed pixmap and repaint."""
        self._drop_mask()
        self._pix = QPixmap()
        self._zoom_factor = 1.0
        self._pan_dx = 0
        self._pan_dy = 0
        self.update()

    def set_zoom_factor(self, z: float):
        """Set absolute zoom factor."""
        self._zoom_factor = max(self._zoom_min, min(self._zoom_max, float(z)))
        if self._zoom_factor <= 1.0:
            self._pan_dx = 0
            self._pan_dy = 0
        self._clamp_pan_offsets()
        self.zoomChanged.emit(float(self._zoom_factor))
        self.update()

    def zoom_factor(self) -> float:
        """Return current zoom factor."""
        return float(self._zoom_factor)

    # --- scalebar measure mode ---
    def start_scalebar_measure(self, finished_cb):
        """Enable 'draw line' mode; call finished_cb(pixel_length_px) on release."""
        if self._pix.isNull():
            return
        self._measure_active = True
        self._measure_p1_img = None
        self._measure_p2_img = None
        self._measure_finished_cb = finished_cb
        self.setCursor(Qt.CrossCursor)
        self.update()

    def cancel_scalebar_measure(self):
        """Cancel an in-progress scalebar measurement and restore the cursor."""
        self._measure_active = False
        self._measure_p1_img = None
        self._measure_p2_img = None
        self._measure_finished_cb = None
        self.unsetCursor()
        self.update()

    # --- straight-line measure mode ---
    def start_line_measure(self, finished_cb):
        """Enable two-click line measure mode; calls finished_cb(pixel_length, p1_img, p2_img)."""
        if self._pix.isNull():
            return
        self._line_measure_active = True
        self._line_p1_img = None
        self._line_p2_img = None
        self._line_measure_cb = finished_cb
        self.setCursor(Qt.CrossCursor)
        self.update()

    def cancel_line_measure(self):
        """Cancel an in-progress line measurement and restore the cursor."""
        self._line_measure_active = False
        self._line_p1_img = None
        self._line_p2_img = None
        self._line_measure_cb = None
        self.unsetCursor()
        self.update()

    # --- persistent line annotations ---
    def add_line_annotation(self, p1_img: QPoint, p2_img: QPoint,
                            label: str = "", color: QColor = QColor(0, 200, 255)):
        """Store a persistent line overlay in image coordinates."""
        self._line_annots.append({
            "p1": QPoint(p1_img), "p2": QPoint(p2_img),
            "label": label, "color": QColor(color),
        })
        self.update()

    def clear_line_annotations(self):
        """Remove all persistent line annotations."""
        self._line_annots.clear()
        self.update()

    def remove_last_line_annotation(self):
        """Remove the most recently added line annotation."""
        if self._line_annots:
            self._line_annots.pop()
            self.update()

    # --- square selection mode ---
    def start_square_selection(self, on_done):
        """Enable one-shot square selection; calls on_done(QRect_in_image, QImage_crop)."""
        if self._pix.isNull():
            return
        self._sel_callback = on_done
        self._sel_active = True
        if self._rubber is None:
            self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber.hide()
        self.setCursor(Qt.CrossCursor)
        self.setFocus()

    def cancel_square_selection(self):
        """Cancel an in-progress square/rectangle selection."""
        self._sel_active = False
        if self._rubber:
            self._rubber.hide()
        self._sel_origin = None
        self._sel_callback = None
        self.unsetCursor()

   # ---------- NEW: annotation overlay API ----------
    def add_square_annotation(self, img_rect: QRect, label: str | None = None,
                              color: QColor | Qt.GlobalColor = Qt.green,
                              pen_width: int = 2, fill_alpha: int = 60):
        """Store a persistent square/rect overlay in *image* coordinates."""
        if self._pix.isNull() or img_rect.isNull():
            return
        # clamp to image bounds
        img_rect = img_rect.normalized().intersected(QRect(0, 0, self._pix.width(), self._pix.height()))
        if img_rect.isEmpty():
            return
        col = QColor(color) if not isinstance(color, QColor) else color
        self._annots.append({
            "rect": img_rect,
            "label": label,
            "color": col,
            "pen": int(max(1, pen_width)),
            "fill_alpha": int(max(0, min(255, fill_alpha)))
        })
        self.update()

    def clear_annotations(self):
        """Remove all persistent rectangle annotations."""
        self._annots.clear()
        self.update()

    def remove_last_annotation(self):
        """Remove the most recently added annotation, if any."""
        if self._annots:
            self._annots.pop()
            self.update()

    # --- mask painting mode ---
    def start_mask_paint(self, finished_cb, cancel_cb=None, initial_mask: np.ndarray | None = None):
        """Enable painting of an image-sized mask over the pixmap.

        Spline tool (default, ``S``): click to place points; a smooth closed
        curve runs through them. Drag a point to move it, right-click or
        Backspace removes the last one, double-clicking a point toggles it
        between smooth and a sharp corner, and clicking the first point or
        pressing Enter closes and fills the shape (or cuts it out when the
        eraser is on).
        Brush tool (``B``): left-drag paints (or erases when the eraser is
        on), right-drag always erases, ``[``/``]`` change the brush.
        Both: middle-drag pans when zoomed, ``E`` toggles the eraser, ``F``
        fills enclosed holes, Ctrl+Z undoes, Enter finishes and Esc cancels.

        Args:
            finished_cb: ``callable(mask)`` called on finish with a bool
                ``(H, W)`` array matching the pixmap size.
            cancel_cb: Optional ``callable()`` called when the user cancels.
            initial_mask: Optional bool ``(H, W)`` array to continue editing.
        """
        if self._pix.isNull():
            return
        w, h = self._pix.width(), self._pix.height()
        if initial_mask is not None and initial_mask.shape[:2] == (h, w):
            self._mask_img = self._array_to_mask_image(initial_mask)
        elif self._mask_img is None or self._mask_img.size() != self._pix.size():
            self._mask_img = QImage(w, h, QImage.Format_ARGB32)
            self._mask_img.fill(Qt.transparent)
        self._mask_backup = self._mask_img.copy()
        self._mask_undo.clear()
        self._mask_active = True
        self._mask_erase = False
        self._mask_last_img = None
        self._reset_spline()
        self._mask_finished_cb = finished_cb
        self._mask_cancel_cb = cancel_cb
        self.set_mask_brush_size(max(3, min(w, h) // 60))
        self._apply_mask_cursor()
        self.setFocus()
        self.maskModeChanged.emit(True)
        self.update()

    def finish_mask_paint(self):
        """Leave mask mode and hand the painted mask to the finish callback."""
        if not self._mask_active:
            return
        self.commit_spline()  # a shape still being drawn counts as part of the mask
        cb = self._mask_finished_cb
        mask = self.mask_array()
        self._end_mask_mode()
        if callable(cb) and mask is not None:
            cb(mask)

    def cancel_mask_paint(self):
        """Leave mask mode and restore the mask as it was before editing."""
        if not self._mask_active:
            return
        cb = self._mask_cancel_cb
        self._mask_img = self._mask_backup
        self._end_mask_mode()
        if callable(cb):
            cb()

    def clear_mask_overlay(self):
        """Cancel mask mode (without callbacks) and remove the mask overlay."""
        self._drop_mask()
        self.update()

    def is_mask_painting(self) -> bool:
        """Return True while mask painting mode is active."""
        return self._mask_active

    def mask_array(self) -> np.ndarray | None:
        """Return the current mask as a bool ``(H, W)`` array, or None."""
        if self._mask_img is None:
            return None
        img = self._mask_img.convertToFormat(QImage.Format_ARGB32)
        w, h = img.width(), img.height()
        buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes())
        argb = buf.reshape(h, img.bytesPerLine())[:, : w * 4].reshape(h, w, 4)
        # Format_ARGB32 is stored as B, G, R, A in memory on little-endian hosts.
        alpha = argb[..., 3] if sys.byteorder == "little" else argb[..., 0]
        return alpha >= 128

    def mask_brush_size(self) -> int:
        """Return the brush diameter in image pixels."""
        return int(self._mask_brush_px)

    def set_mask_brush_size(self, px: int):
        """Set the brush diameter in image pixels (clamped to 1..500)."""
        px = int(max(1, min(500, px)))
        if px != self._mask_brush_px:
            self._mask_brush_px = px
            self.maskBrushChanged.emit(px)
            self.update()

    def set_mask_erase(self, on: bool):
        """Toggle the eraser for left-drag strokes."""
        self._mask_erase = bool(on)

    def mask_erase(self) -> bool:
        """Return True when left-drag erases."""
        return self._mask_erase

    def mask_tool(self) -> str:
        """Return the active mask tool: ``"spline"`` or ``"brush"``."""
        return self._mask_tool

    def set_mask_tool(self, tool: str):
        """Switch between the ``"spline"`` and ``"brush"`` mask tools."""
        if tool not in ("spline", "brush") or tool == self._mask_tool:
            return
        self._mask_tool = tool
        self._reset_spline()
        self._mask_last_img = None
        if self._mask_active:
            self._apply_mask_cursor()
        self.maskBrushChanged.emit(self._mask_brush_px)  # lets toolbars resync
        self.update()

    def spline_point_count(self) -> int:
        """Return the number of pending (not yet closed) spline points."""
        return len(self._spline_pts)

    def remove_last_spline_point(self):
        """Drop the most recently placed spline point."""
        if self._spline_pts:
            self._spline_pts.pop()
            self._spline_corner.pop()
            self.update()

    def commit_spline(self) -> bool:
        """Close the pending spline and fill it into the mask (undoable).

        Fills the shape, or cuts it out when the eraser is on. Needs at least
        three points; returns True when something was committed.
        """
        if not self._mask_active or self._mask_img is None or len(self._spline_pts) < 3:
            return False
        curve = self.spline_curve(self._spline_pts, closed=True, corners=self._spline_corner)
        self._push_mask_undo()
        p = QPainter(self._mask_img)
        try:
            p.setRenderHint(QPainter.Antialiasing, False)  # keep the mask binary
            if self._mask_erase:
                p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.setPen(Qt.NoPen)
            p.setBrush(MASK_COLOR)
            p.drawPolygon(QPolygonF(curve), Qt.WindingFill)
        finally:
            p.end()
        self._reset_spline()
        self.update()
        return True

    def _reset_spline(self):
        self._spline_pts = []
        self._spline_corner = []
        self._spline_drag_idx = None
        self._spline_closed_snapshot = None

    def toggle_spline_corner(self, idx: int):
        """Switch pending point ``idx`` between smooth and a sharp corner."""
        if 0 <= idx < len(self._spline_corner):
            self._spline_corner[idx] = not self._spline_corner[idx]
            self.update()

    @staticmethod
    def spline_curve(pts: list[QPointF], closed: bool, corners: list[bool] | None = None,
                     samples: int = SPLINE_SAMPLES_PER_SEG) -> list[QPointF]:
        """Sample a centripetal Catmull-Rom spline through ``pts``.

        The curve passes through every control point; the centripetal
        parameterisation avoids the loops/cusps a uniform spline makes when
        points are unevenly spaced. Points flagged in ``corners`` are sharp:
        the neighbouring segments treat them as end points (mirrored phantom
        neighbour), so the curve meets there at an angle instead of smoothly.
        """
        n = len(pts)
        if n < 2:
            return list(pts)
        P = np.array([[q.x(), q.y()] for q in pts], dtype=float)
        if closed:
            ext = np.vstack([P[-1], P, P[0], P[1]])
            nseg = n
        else:
            # mirror the end points so the curve starts/ends on them
            ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
            nseg = n - 1
        out: list[QPointF] = []
        t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
        for i in range(nseg):
            p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
            if corners:
                if corners[i]:
                    p0 = 2 * p1 - p2
                if corners[(i + 1) % n]:
                    p3 = 2 * p2 - p1
            d01 = max(np.linalg.norm(p1 - p0) ** 0.5, 1e-6)
            d12 = max(np.linalg.norm(p2 - p1) ** 0.5, 1e-6)
            d23 = max(np.linalg.norm(p3 - p2) ** 0.5, 1e-6)
            # tangents of the centripetal Catmull-Rom (Barry-Goldman form),
            # rescaled to the [0, 1] segment parameter
            m1 = (p1 - p0) / d01 - (p2 - p0) / (d01 + d12) + (p2 - p1) / d12
            m2 = (p2 - p1) / d12 - (p3 - p1) / (d12 + d23) + (p3 - p2) / d23
            m1 *= d12
            m2 *= d12
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            seg = h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2
            out.extend(QPointF(float(x), float(y)) for x, y in seg)
        if not closed:
            out.append(QPointF(pts[-1]))
        return out

    def _spline_hit(self, pos: QPoint) -> int | None:
        """Index of the pending spline point under widget pos, if any."""
        best, best_d = None, float(SPLINE_HANDLE_PX)
        for i, q in enumerate(self._spline_pts):
            w = self._image_to_widget_f(q)
            d = ((w.x() - pos.x()) ** 2 + (w.y() - pos.y()) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = i, d
        return best

    def _image_to_widget_f(self, pt_img: QPointF) -> QPointF:
        tgt = self._scaled_target_rect()
        iw, ih = max(1, self._pix.width()), max(1, self._pix.height())
        return QPointF(tgt.x() + pt_img.x() * tgt.width() / iw,
                       tgt.y() + pt_img.y() * tgt.height() / ih)

    def _apply_mask_cursor(self):
        self.setCursor(Qt.CrossCursor if self._mask_tool == "spline" else Qt.BlankCursor)

    def undo_mask(self):
        """Undo the last spline point, or else the last stroke / fill / clear."""
        if self._mask_active and self._spline_pts:
            self.remove_last_spline_point()
            return
        if self._mask_active and self._mask_undo:
            self._mask_img = self._mask_undo.pop()
            self.update()

    def clear_mask(self):
        """Erase the whole mask (undoable)."""
        if self._mask_active and self._mask_img is not None:
            self._push_mask_undo()
            self._mask_img.fill(Qt.transparent)
            self.update()

    def fill_mask_holes(self):
        """Fill regions fully enclosed by painted pixels (undoable).

        Lets the user trace only the outline and fill the interior in one step.
        """
        mask = self.mask_array() if self._mask_active else None
        if mask is None or not mask.any():
            return
        h, w = mask.shape
        # Flood the background from outside a 1-px padded border; whatever
        # the flood can't reach is enclosed by the painted outline.
        padded = np.zeros((h + 2, w + 2), np.uint8)
        padded[1:-1, 1:-1] = mask.astype(np.uint8) * 255
        flood = padded.copy()
        cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
        filled = (flood[1:-1, 1:-1] != 128)
        self._push_mask_undo()
        self._mask_img = self._array_to_mask_image(filled)
        self.update()

    def _push_mask_undo(self):
        if self._mask_img is None:
            return
        self._mask_undo.append(self._mask_img.copy())
        if len(self._mask_undo) > MASK_UNDO_LIMIT:
            self._mask_undo.pop(0)

    def _end_mask_mode(self):
        self._mask_active = False
        self._mask_last_img = None
        self._mask_cursor_w = None
        self._mask_finished_cb = None
        self._mask_cancel_cb = None
        self._mask_backup = None
        self._mask_undo.clear()
        self._reset_spline()
        self.unsetCursor()
        self.maskModeChanged.emit(False)
        self.update()

    def _drop_mask(self):
        if self._mask_active:
            self._end_mask_mode()
        self._mask_img = None

    @staticmethod
    def _array_to_mask_image(mask: np.ndarray) -> QImage:
        h, w = mask.shape[:2]
        img = QImage(w, h, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        if mask.any():
            sel = QImage(np.ascontiguousarray(mask.astype(np.uint8) * 255).data,
                         w, h, w, QImage.Format_Alpha8).copy()
            p = QPainter(img)
            p.fillRect(img.rect(), MASK_COLOR)
            p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
            p.drawImage(0, 0, sel)
            p.end()
        return img

    def _widget_to_image_f(self, pt: QPoint) -> QPointF | None:
        """Map widget point -> fractional image coords (not clamped to the image)."""
        if self._pix.isNull():
            return None
        tgt = self._scaled_target_rect()
        if tgt.width() <= 0 or tgt.height() <= 0:
            return None
        return QPointF((pt.x() - tgt.x()) * self._pix.width() / tgt.width(),
                       (pt.y() - tgt.y()) * self._pix.height() / tgt.height())

    def _mask_stroke_to(self, pt_img: QPointF):
        """Paint (or erase) a round-capped segment from the last point to pt_img."""
        if self._mask_img is None:
            return
        p = QPainter(self._mask_img)
        try:
            p.setRenderHint(QPainter.Antialiasing, False)  # keep the mask binary
            if self._mask_stroke_erase:
                p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.setPen(QPen(MASK_COLOR, self._mask_brush_px, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            start = self._mask_last_img if self._mask_last_img is not None else pt_img
            if start == pt_img:
                p.drawPoint(pt_img)
            else:
                p.drawLine(start, pt_img)
        finally:
            p.end()
        self._mask_last_img = QPointF(pt_img)
        self.update()

    # ---------------- sizing ----------------
    def sizeHint(self) -> QSize:
        """Return the preferred size for layout negotiation."""
        return QSize(900, 600)

    def hasHeightForWidth(self) -> bool:
        """Signal that this widget uses aspect-ratio-based height."""
        return True

    def heightForWidth(self, w: int) -> int:
        """Return the ideal height for the given width, preserving aspect ratio."""
        if self._pix.isNull():
            return super().heightForWidth(w)
        ar = self._pix.height() / max(1, self._pix.width())
        return int(w * ar)

    # ---------------- painting ----------------
    def paintEvent(self, e):
        """Draw the scaled pixmap, measurement line overlay, and annotations."""
        super().paintEvent(e)
        if self._pix.isNull():
            return

        tgt = self._scaled_target_rect()
        if tgt.width() <= 0 or tgt.height() <= 0:
            return
        scaled = self._pix.scaled(tgt.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        p = QPainter(self)
        
        try:
            p.drawPixmap(tgt.topLeft(), scaled)

            # mask overlay (+ brush outline while painting)
            if self._mask_img is not None:
                p.save()
                p.setOpacity(MASK_OPACITY)
                p.drawImage(tgt, self._mask_img)
                p.restore()
            if self._mask_active and self._mask_tool == "spline" and self._spline_pts:
                self._paint_spline_preview(p)
            if self._mask_active and self._mask_tool == "brush" and self._mask_cursor_w is not None:
                r = self._mask_brush_px * tgt.width() / max(1, self._pix.width()) / 2.0
                p.save()
                p.setRenderHint(QPainter.Antialiasing, True)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(0, 0, 0), 3))
                p.drawEllipse(QPointF(self._mask_cursor_w), r, r)
                p.setPen(QPen(QColor(255, 255, 255) if not self._mask_erase else QColor(0, 200, 255), 1))
                p.drawEllipse(QPointF(self._mask_cursor_w), r, r)
                p.restore()

            # overlay the scalebar measurement line
            if self._measure_active and self._measure_p1_img is not None and self._measure_p2_img is not None:
                p1w = self._image_to_widget(self._measure_p1_img)
                p2w = self._image_to_widget(self._measure_p2_img)
                p.drawLine(p1w, p2w)
                dx = self._measure_p2_img.x() - self._measure_p1_img.x()
                dy = self._measure_p2_img.y() - self._measure_p1_img.y()
                pixlen = float((dx * dx + dy * dy) ** 0.5)
                tx = (p1w.x() + p2w.x()) // 2
                ty = (p1w.y() + p2w.y()) // 2 - 8
                p.drawText(tx, ty, f"{pixlen:.1f} px")

            # overlay the active straight-line measurement (cyan dashed)
            if self._line_measure_active and self._line_p1_img is not None and self._line_p2_img is not None:
                p1w = self._image_to_widget(self._line_p1_img)
                p2w = self._image_to_widget(self._line_p2_img)
                pen = QPen(QColor(0, 200, 255), 2, Qt.DashLine)
                p.setPen(pen)
                p.drawLine(p1w, p2w)
                # draw small circles at endpoints
                p.setBrush(QColor(0, 200, 255))
                p.drawEllipse(p1w, 4, 4)
                p.drawEllipse(p2w, 4, 4)
                # pixel length label
                dx = self._line_p2_img.x() - self._line_p1_img.x()
                dy = self._line_p2_img.y() - self._line_p1_img.y()
                pixlen = float((dx * dx + dy * dy) ** 0.5)
                tx = (p1w.x() + p2w.x()) // 2
                ty = (p1w.y() + p2w.y()) // 2 - 10
                p.setPen(QColor(0, 200, 255))
                p.drawText(tx, ty, f"{pixlen:.1f} px")

            # persistent line annotations
            for la in self._line_annots:
                p1w = self._image_to_widget(la["p1"])
                p2w = self._image_to_widget(la["p2"])
                col = la.get("color", QColor(0, 200, 255))
                pen = QPen(col, 2, Qt.SolidLine)
                p.setPen(pen)
                p.drawLine(p1w, p2w)
                p.setBrush(col)
                p.drawEllipse(p1w, 3, 3)
                p.drawEllipse(p2w, 3, 3)
                label = la.get("label", "")
                if label:
                    tx = (p1w.x() + p2w.x()) // 2
                    ty = (p1w.y() + p2w.y()) // 2 - 10
                    fm = p.fontMetrics()
                    tw = fm.horizontalAdvance(label) + 8
                    th = fm.height() + 4
                    bg = QRect(tx - 2, ty - th + 2, tw, th)
                    p.fillRect(bg, QColor(0, 0, 0, 160))
                    p.setPen(Qt.white)
                    p.drawText(bg, Qt.AlignCenter, label)
                
            # ✅ guard: only draw annotations if the attribute exists and is non-empty
            annots = getattr(self, "_annots", [])
            if annots:
                for a in annots:
                    # expect dicts like {"rect": QRect, "color": QColor/Qt.GlobalColor, "pen": int, "fill_alpha": int, "label": str|None}
                    r_img = a.get("rect")
                    if not isinstance(r_img, QRect) or r_img.isNull():
                        continue
                    wr = self._image_rect_to_widget(r_img)

                    col = a.get("color", Qt.green)
                    qcol = QColor(col) if not isinstance(col, QColor) else col

                    # fill
                    fill = QColor(qcol)
                    fill.setAlpha(int(a.get("fill_alpha", 60)))
                    p.fillRect(wr, fill)

                    # outline
                    pen = QPen(qcol)
                    pen.setWidth(int(a.get("pen", 2)))
                    p.setPen(pen)
                    p.drawRect(wr)

                    # optional label
                    label = a.get("label")
                    if label:
                        painter.setFont(self.font())
                        fm = painter.fontMetrics()
                        tw = fm.horizontalAdvance(label) + 12
                        th = fm.height() + 6
                        # top-left, clamped inside the annotation rect
                        L = wr.adjusted(1, 1, -1, -1)
                        cap = QRect(L.left(), L.top(), min(tw, L.width()), min(th, L.height()))
                        painter.fillRect(cap, QColor(0, 0, 0, 160))
                        painter.setPen(Qt.white)
                        painter.drawText(cap.adjusted(6, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, str(label))
        finally:
        # ✅ always end the painter to avoid QBackingStore complaints
            p.end()
            

    def _paint_spline_preview(self, p: QPainter):
        """Draw the pending spline: curve through the points (and the cursor),
        a dashed closing segment, and grab handles."""
        pts = list(self._spline_pts)
        cursor_img = None
        if self._mask_cursor_w is not None and self._spline_drag_idx is None:
            cursor_img = self._widget_to_image_f(self._mask_cursor_w)
            # snapping onto the first point means "close": don't add a ghost point
            if cursor_img is not None and len(pts) >= 3 and self._spline_hit(self._mask_cursor_w) == 0:
                cursor_img = None
        live = pts + ([cursor_img] if cursor_img is not None else [])
        live_corners = self._spline_corner + ([False] if cursor_img is not None else [])
        col = QColor(0, 200, 255) if self._mask_erase else SPLINE_COLOR
        p.save()
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            if len(live) >= 3:
                closed_w = [self._image_to_widget_f(q) for q in self.spline_curve(live, closed=True, corners=live_corners)]
                fill = QColor(col); fill.setAlpha(50)
                p.setPen(Qt.NoPen); p.setBrush(fill)
                p.drawPolygon(QPolygonF(closed_w))
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(col, 1, Qt.DashLine))
                p.drawPolyline(QPolygonF(closed_w))
            if len(live) >= 2:
                open_w = [self._image_to_widget_f(q)
                          for q in self.spline_curve(live, closed=False, corners=live_corners)]
                p.setPen(QPen(col, 1)); p.drawPolyline(QPolygonF(open_w))
            hover = self._spline_hit(self._mask_cursor_w) if self._mask_cursor_w is not None else None
            for i, q in enumerate(pts):
                w = self._image_to_widget_f(q)
                r = 4.5 if (i == hover or (i == 0 and len(pts) >= 3)) else 3.0
                p.setPen(QPen(QColor(0, 0, 0), 1))
                p.setBrush(QColor(255, 255, 255) if i == 0 else col)
                if self._spline_corner[i]:
                    p.drawRect(QRectF(w.x() - r, w.y() - r, 2 * r, 2 * r))  # corner = square
                else:
                    p.drawEllipse(w, r, r)
        finally:
            p.restore()

    # ---------------- coord mapping ----------------
    def _scaled_target_rect(self) -> QRect:
        """Where the pixmap is drawn inside the label (fit with aspect)."""
        rect = self.contentsRect()
        if self._pix.isNull():
            return rect
        iw, ih = self._pix.width(), self._pix.height()
        if iw <= 0 or ih <= 0:
            return rect
        s = min(rect.width() / iw, rect.height() / ih) * self._zoom_factor
        sw, sh = int(iw * s), int(ih * s)
        self._clamp_pan_offsets(sw=sw, sh=sh)
        x = rect.x() + (rect.width() - sw) // 2 + self._pan_dx
        y = rect.y() + (rect.height() - sh) // 2 + self._pan_dy
        return QRect(x, y, sw, sh)

    def _clamp_pan_offsets(self, sw: int | None = None, sh: int | None = None):
        """Clamp pan offsets so zoomed image always covers the viewport."""
        rect = self.contentsRect()
        if sw is None or sh is None:
            if self._pix.isNull():
                self._pan_dx = self._pan_dy = 0
                return
            iw, ih = self._pix.width(), self._pix.height()
            if iw <= 0 or ih <= 0:
                self._pan_dx = self._pan_dy = 0
                return
            s = min(rect.width() / iw, rect.height() / ih) * self._zoom_factor
            sw, sh = int(iw * s), int(ih * s)

        max_dx = max(0, (sw - rect.width()) // 2)
        max_dy = max(0, (sh - rect.height()) // 2)
        self._pan_dx = max(-max_dx, min(max_dx, int(self._pan_dx)))
        self._pan_dy = max(-max_dy, min(max_dy, int(self._pan_dy)))

    def _widget_to_image(self, pt: QPoint) -> QPoint | None:
        """Map widget point -> image pixel coords (int), or None if outside the drawn image."""
        if self._pix.isNull():
            return None
        tgt = self._scaled_target_rect()
        if not tgt.contains(pt):
            return None
        iw, ih = self._pix.width(), self._pix.height()
        x_img = (pt.x() - tgt.x()) * iw / max(1, tgt.width())
        y_img = (pt.y() - tgt.y()) * ih / max(1, tgt.height())
        return QPoint(int(round(x_img)), int(round(y_img)))

    def _image_to_widget(self, pt_img: QPoint) -> QPoint:
        """Map image pixel coords -> widget coords."""
        tgt = self._scaled_target_rect()
        iw, ih = max(1, self._pix.width()), max(1, self._pix.height())
        x = tgt.x() + pt_img.x() * tgt.width() / iw
        y = tgt.y() + pt_img.y() * tgt.height() / ih
        return QPoint(int(round(x)), int(round(y)))
        
    def _image_rect_to_widget(self, r_img: QRect) -> QRect:
        """Map an image-space rectangle to widget coordinates."""
        tl = self._image_to_widget(r_img.topLeft())
        br = self._image_to_widget(r_img.bottomRight())
        return QRect(tl, br).normalized()

    # ---------------- mouse / key handling ----------------
    def mousePressEvent(self, e):
        # pan start (only when zoomed in and no active measurement/selection
        # mode; while painting a mask the middle button pans instead)
        if self._zoom_factor > 1.0 and (
            (
                e.button() == Qt.LeftButton
                and not self._line_measure_active
                and not self._measure_active
                and not self._sel_active
                and not self._mask_active
            )
            or (e.button() == Qt.MiddleButton and self._mask_active)
        ):
            self._pan_active = True
            self._pan_last_pos = QPoint(e.pos())
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return

        # spline: left adds a point / grabs a point / clicks the first point
        # to close; right removes the last point
        if self._mask_active and self._mask_tool == "spline" and e.button() in (Qt.LeftButton, Qt.RightButton):
            if e.button() == Qt.RightButton:
                self.remove_last_spline_point()
            else:
                hit = self._spline_hit(e.pos())
                self._spline_closed_snapshot = None
                if hit == 0 and len(self._spline_pts) >= 3:
                    # kept so a double-click on the first point can reopen the
                    # shape and make that point a corner instead
                    snapshot = (list(self._spline_pts), list(self._spline_corner))
                    if self.commit_spline():
                        self._spline_closed_snapshot = snapshot
                elif hit is not None:
                    self._spline_drag_idx = hit
                else:
                    pt = self._widget_to_image_f(e.pos())
                    if pt is not None:
                        self._spline_pts.append(pt)
                        self._spline_corner.append(False)
                        self._spline_drag_idx = len(self._spline_pts) - 1  # drag right after placing
                self.update()
            e.accept()
            return

        # mask stroke start (left paints/erases per toggle, right always erases)
        if self._mask_active and e.button() in (Qt.LeftButton, Qt.RightButton):
            pt = self._widget_to_image_f(e.pos())
            if pt is not None:
                self._push_mask_undo()
                self._mask_stroke_erase = self._mask_erase or e.button() == Qt.RightButton
                self._mask_last_img = None
                self._mask_stroke_to(pt)
            e.accept()
            return

        # straight-line measure (two-click)
        if self._line_measure_active and e.button() == Qt.LeftButton:
            pt = self._widget_to_image(e.pos())
            if pt is not None:
                if self._line_p1_img is None:
                    self._line_p1_img = pt
                    self._line_p2_img = pt
                else:
                    self._line_p2_img = pt
                    dx = self._line_p2_img.x() - self._line_p1_img.x()
                    dy = self._line_p2_img.y() - self._line_p1_img.y()
                    pixlen = float((dx**2 + dy**2) ** 0.5)
                    cb = self._line_measure_cb
                    p1, p2 = QPoint(self._line_p1_img), QPoint(self._line_p2_img)
                    self.cancel_line_measure()
                    if cb and pixlen > 0:
                        cb(pixlen, p1, p2)
                self.update()
                e.accept()
                return

        # scalebar line start
        if self._measure_active and e.button() == Qt.LeftButton:
            pt = self._widget_to_image(e.pos())
            if pt is not None:
                self._measure_p1_img = pt
                self._measure_p2_img = pt
                self.update()
                e.accept()
                return

        # square selection start
        if self._sel_active and e.button() == Qt.LeftButton and self._rubber:
            r = self._scaled_target_rect()
            if not r.isNull():
                sx = max(r.left(), min(e.pos().x(), r.right()))
                sy = max(r.top(),  min(e.pos().y(), r.bottom()))
                self._sel_origin = QPoint(sx, sy)
                self._rubber.setGeometry(QRect(self._sel_origin, self._sel_origin))
                self._rubber.show()
                e.accept()
                return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        # pan update
        if self._pan_active and self._pan_last_pos is not None:
            delta = e.pos() - self._pan_last_pos
            self._pan_dx += int(delta.x())
            self._pan_dy += int(delta.y())
            self._pan_last_pos = QPoint(e.pos())
            self._clamp_pan_offsets()
            self.update()
            e.accept()
            return

        # mask brush cursor + stroke continuation
        if self._mask_active:
            self._mask_cursor_w = QPoint(e.pos())
            if self._spline_drag_idx is not None and e.buttons() & Qt.LeftButton:
                pt = self._widget_to_image_f(e.pos())
                if pt is not None and self._spline_drag_idx < len(self._spline_pts):
                    self._spline_pts[self._spline_drag_idx] = pt
            if self._mask_last_img is not None and e.buttons() & (Qt.LeftButton | Qt.RightButton):
                pt = self._widget_to_image_f(e.pos())
                if pt is not None:
                    self._mask_stroke_to(pt)
            self.update()
            e.accept()
            return

        # straight-line live tracking
        if self._line_measure_active and self._line_p1_img is not None:
            pt = self._widget_to_image(e.pos())
            if pt is not None:
                self._line_p2_img = pt
                self.update()
                e.accept()
                return

        # scalebar update
        if self._measure_active and self._measure_p1_img is not None:
            pt = self._widget_to_image(e.pos())
            if pt is not None:
                self._measure_p2_img = pt
                self.update()
                e.accept()
                return

        # square selection update
        if self._sel_active and self._rubber and self._sel_origin is not None:
            r = self._scaled_target_rect()
            if not r.isNull():
                x = max(r.left(), min(e.pos().x(), r.right()))
                y = max(r.top(),  min(e.pos().y(), r.bottom()))
                dx = x - self._sel_origin.x()
                dy = y - self._sel_origin.y()
                if e.modifiers() & Qt.ShiftModifier:
                    # square constraint while Shift is pressed
                    side = max(abs(dx), abs(dy))
                    x2 = self._sel_origin.x() + (side if dx >= 0 else -side)
                    y2 = self._sel_origin.y() + (side if dy >= 0 else -side)
                    rect = QRect(self._sel_origin, QPoint(x2, y2)).normalized().intersected(r)
                else:
                    # free rectangle
                    rect = QRect(self._sel_origin, QPoint(x, y)).normalized().intersected(r)
                
                self._rubber.setGeometry(rect)
                e.accept()
                return

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan_active and e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._pan_active = False
            self._pan_last_pos = None
            self.setCursor(Qt.BlankCursor if self._mask_active else Qt.ArrowCursor)
            e.accept()
            return

        # mask stroke end
        if self._mask_active and e.button() in (Qt.LeftButton, Qt.RightButton):
            self._mask_last_img = None
            self._spline_drag_idx = None
            e.accept()
            return

        # scalebar finish
        if self._measure_active and e.button() == Qt.LeftButton:    
            pt = self._widget_to_image(e.pos())
            if pt is not None:
                self._measure_p2_img = pt
            pixlen = 0.0
            if self._measure_p1_img is not None and self._measure_p2_img is not None:
                dx = self._measure_p2_img.x() - self._measure_p1_img.x()
                dy = self._measure_p2_img.y() - self._measure_p1_img.y()
                pixlen = float((dx * dx + dy * dy) ** 0.5)
            cb = self._measure_finished_cb
            self.cancel_scalebar_measure()
            if cb and pixlen > 0:
                cb(pixlen)
                e.accept()
                return

        # square selection finish
        if self._sel_active and self._rubber and self._sel_origin is not None:
            self._sel_active = False
            self.setCursor(Qt.ArrowCursor)
            rect_disp = self._rubber.geometry()
            self._rubber.hide()
            self._sel_origin = None

            if rect_disp.width() > 3 and rect_disp.height() > 3:
                tl_img = self._widget_to_image(rect_disp.topLeft())
                br_img = self._widget_to_image(rect_disp.bottomRight())
                if tl_img is not None and br_img is not None:
                    img_rect = QRect(tl_img, br_img).normalized()
                    cropped_qimg = self._pix.copy(img_rect).toImage()
                    cb = self._sel_callback
                    self._sel_callback = None
                    if callable(cb):
                        cb(img_rect, cropped_qimg)
            e.accept()
            return

        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        # double-click on a point toggles it between smooth and a sharp corner.
        # On empty space the first click already placed a point, so this makes
        # that new point a corner.
        if self._mask_active and self._mask_tool == "spline" and e.button() == Qt.LeftButton:
            self._spline_drag_idx = None
            if self._spline_closed_snapshot is not None:
                # the first click closed the shape on point 0 — reopen it
                pts, corners = self._spline_closed_snapshot
                self._spline_closed_snapshot = None
                if self._mask_undo:
                    self._mask_img = self._mask_undo.pop()
                self._spline_pts, self._spline_corner = pts, corners
                self.toggle_spline_corner(0)
            else:
                hit = self._spline_hit(e.pos())
                if hit is not None:
                    self.toggle_spline_corner(hit)
            e.accept()
            return
        super().mouseDoubleClickEvent(e)

    def leaveEvent(self, e):
        if self._mask_active:
            self._mask_cursor_w = None
            self.update()
        super().leaveEvent(e)

    def keyPressEvent(self, e):
        if self._mask_active:
            key = e.key()
            step = max(1, self._mask_brush_px // 5)
            if key == Qt.Key_Escape:
                if self._spline_pts:
                    self._reset_spline()           # drop the shape in progress only
                    self.update()
                else:
                    self.cancel_mask_paint()
            elif key in (Qt.Key_Return, Qt.Key_Enter):
                if not self.commit_spline():       # first Enter closes a pending spline
                    self.finish_mask_paint()
            elif key in (Qt.Key_Backspace, Qt.Key_Delete):
                self.remove_last_spline_point()
            elif key == Qt.Key_S:
                self.set_mask_tool("spline")
            elif key == Qt.Key_B:
                self.set_mask_tool("brush")
            elif key == Qt.Key_BracketLeft:
                self.set_mask_brush_size(self._mask_brush_px - step)
            elif key == Qt.Key_BracketRight:
                self.set_mask_brush_size(self._mask_brush_px + step)
            elif key == Qt.Key_E:
                self.set_mask_erase(not self._mask_erase)
                self.maskBrushChanged.emit(self._mask_brush_px)  # lets toolbars resync
                self.update()
            elif key == Qt.Key_F:
                self.fill_mask_holes()
            elif key == Qt.Key_Z and e.modifiers() & Qt.ControlModifier:
                self.undo_mask()
            else:
                super().keyPressEvent(e)
                return
            e.accept()
            return
        if e.key() == Qt.Key_Escape:
            if self._line_measure_active:
                self.cancel_line_measure()
                e.accept()
                return
            if self._sel_active:
                self.cancel_square_selection()
                e.accept()
                return
            if self._measure_active:
                self.cancel_scalebar_measure()
                e.accept()
                return
        super().keyPressEvent(e)

    def resizeEvent(self, e):
        # keep the rubber band inside when resizing
        if self._rubber and self._rubber.isVisible():
            r = self._scaled_target_rect()
            self._rubber.setGeometry(self._rubber.geometry().intersected(r))
        self._clamp_pan_offsets()
        super().resizeEvent(e)
