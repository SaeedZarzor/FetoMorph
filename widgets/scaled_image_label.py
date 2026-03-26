"""Aspect-fit image label with interactive measurement and selection modes.

Extends QLabel to draw a pixmap with preserved aspect ratio and provides
two interactive overlay modes: a click-drag line for scale-bar measurement
and a rubber-band rectangle for region-of-interest selection.  Persistent
rectangle annotations can also be added programmatically.
"""

from PySide6.QtCore import Qt, QSize, QPoint, QRect
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy, QRubberBand


class ScaledImageLabel(QLabel):
    """
    Aspect-fit image label with:
      - scalebar measurement mode (click-drag line, reports pixel length)
      - square selection mode with QRubberBand, returns image-rect + cropped QImage
    It draws self._pix manually; QLabel's built-in pixmap is *not* used.
    """
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
        self.update()

    def clearImage(self):
        """Remove the displayed pixmap and repaint."""
        self._pix = QPixmap()
        self.update()

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

        rect: QRect = self.contentsRect()
        scaled = self._pix.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2

        p = QPainter(self)
        
        try:
            p.drawPixmap(x, y, scaled)

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
            

    # ---------------- coord mapping ----------------
    def _scaled_target_rect(self) -> QRect:
        """Where the pixmap is drawn inside the label (fit with aspect)."""
        rect = self.contentsRect()
        if self._pix.isNull():
            return rect
        iw, ih = self._pix.width(), self._pix.height()
        if iw <= 0 or ih <= 0:
            return rect
        s = min(rect.width() / iw, rect.height() / ih)
        sw, sh = int(iw * s), int(ih * s)
        x = rect.x() + (rect.width() - sw) // 2
        y = rect.y() + (rect.height() - sh) // 2
        return QRect(x, y, sw, sh)

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

    def keyPressEvent(self, e):
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
        super().resizeEvent(e)
        
    
