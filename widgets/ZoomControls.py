"""Reusable zoom controls widget for image views."""

from deps import *


class ZoomControlsWidget(QWidget):
    """Compact control bar for zoom out / preset / zoom in."""

    zoomInRequested = Signal()
    zoomOutRequested = Signal()
    zoomTextChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target = None
        self._wheel_step = 1.25

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.zoom_label = QLabel("Zoom:")
        layout.addWidget(self.zoom_label)

        self.zoom_out_btn = QToolButton(self)
        self.zoom_out_btn.setText("−")
        self.zoom_out_btn.setToolTip("Zoom out")
        self.zoom_out_btn.clicked.connect(self._zoom_out)
        layout.addWidget(self.zoom_out_btn)

        self.zoom_combo = QComboBox(self)
        self.zoom_combo.setEditable(True)
        self.zoom_combo.setInsertPolicy(QComboBox.NoInsert)
        self.zoom_combo.lineEdit().setReadOnly(True)
        self.zoom_combo.addItems(["Fit", "50%", "75%", "100%", "125%", "150%", "200%", "300%"])
        self.zoom_combo.setCurrentText("Fit")
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_text_changed)
        layout.addWidget(self.zoom_combo)

        self.zoom_in_btn = QToolButton(self)
        self.zoom_in_btn.setText("+")
        self.zoom_in_btn.setToolTip("Zoom in")
        self.zoom_in_btn.clicked.connect(self._zoom_in)
        layout.addWidget(self.zoom_in_btn)

    def bind_image_label(self, image_label):
        """Attach controls to a ScaledImageLabel-like target."""
        if self._target is image_label:
            return
        if self._target is not None:
            try:
                self._target.removeEventFilter(self)
            except Exception:
                pass
            try:
                self._target.zoomChanged.disconnect(self._on_target_zoom_changed)
            except Exception:
                pass
        self._target = image_label
        if self._target is not None:
            self._target.installEventFilter(self)
            self._target.zoomChanged.connect(self._on_target_zoom_changed)
            self._on_target_zoom_changed(float(self._target.zoom_factor()))

    def set_zoom_text(self, text: str):
        """Set zoom combo text without re-emitting zoomTextChanged."""
        self.zoom_combo.blockSignals(True)
        self.zoom_combo.setCurrentText(text)
        self.zoom_combo.blockSignals(False)

    def _zoom_in(self):
        """Zoom in target view."""
        if self._target is None:
            return
        self.zoomInRequested.emit()
        self._target.set_zoom_factor(self._target.zoom_factor() * self._wheel_step)

    def _zoom_out(self):
        """Zoom out target view."""
        if self._target is None:
            return
        self.zoomOutRequested.emit()
        self._target.set_zoom_factor(self._target.zoom_factor() / self._wheel_step)

    def _on_zoom_text_changed(self, text: str):
        """Apply combo selection to target zoom."""
        self.zoomTextChanged.emit(text)
        if self._target is None:
            return
        t = (text or "").strip().lower()
        if t == "fit":
            self._target.set_zoom_factor(1.0)
            return
        if t.endswith("%"):
            try:
                v = float(t[:-1]) / 100.0
                self._target.set_zoom_factor(v)
            except Exception:
                pass

    def _on_target_zoom_changed(self, factor: float):
        """Keep combo text in sync with target zoom."""
        if abs(float(factor) - 1.0) < 1e-9:
            txt = "Fit"
        else:
            txt = f"{int(round(float(factor) * 100.0))}%"
        self.set_zoom_text(txt)

    def eventFilter(self, obj, event):
        """Handle wheel zoom on the bound image label."""
        if obj is self._target and event.type() == QtCore.QEvent.Wheel:
            if not self.isVisible():
                return False
            try:
                if hasattr(self._target, "hasImage") and not self._target.hasImage():
                    return False
            except Exception:
                return False
            dy = event.angleDelta().y()
            if dy > 0:
                self._zoom_in()
                event.accept()
                return True
            if dy < 0:
                self._zoom_out()
                event.accept()
                return True
        return super().eventFilter(obj, event)
