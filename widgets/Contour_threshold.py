"""Contour area threshold dialog widget.

Provides a slider-and-spinbox dialog for choosing a minimum contour area
(in pixels) used to filter small or noisy contours during image processing.
"""

from deps import *

class ContourThresholdDialog(QDialog):
    """Dialog for selecting a contour area threshold in pixels.

    Combines a horizontal slider with a double spin box so the user can
    quickly scrub to a rough value and then fine-tune it numerically.
    """

    def __init__(self, parent=None, initial: float = 50.0, minimum: float = 0.0, maximum: float = 100000.0):
        """Initialise the contour threshold dialog.

        Args:
            parent: Parent widget.
            initial: Starting threshold value in pixels.
            minimum: Lower bound of the allowed range.
            maximum: Upper bound of the allowed range.
        """
        super().__init__(parent)
        self.setWindowTitle("Contour Area Threshold (px)")
        self.setModal(True)

        initial = max(minimum, min(maximum, float(initial)))
        form = QFormLayout(self)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(int(minimum), int(maximum))
        self.slider.setSingleStep(10)
        self.slider.setPageStep(100)
        self.slider.setValue(int(initial))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.setMinimumWidth(280)
        self.slider.setStyleSheet("""
        QSlider::groove:horizontal { height:6px; background:#444; border-radius:3px; }
        QSlider::sub-page:horizontal { background:#2aa1ff; height:6px; border-radius:3px; }
        QSlider::add-page:horizontal { background:#333; height:6px; border-radius:3px; }
        QSlider::handle:horizontal { width:16px; background:#2aa1ff; border-radius:8px; margin:-5px 0; }
        """)

        self.spin = QDoubleSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(0)
        self.spin.setSingleStep(10)
        self.spin.setValue(initial)
        self.spin.setMinimumWidth(120)

        # sync slider <-> spin
        def _from_slider(v: int):
            if int(self.spin.value()) != v:
                self.spin.blockSignals(True); self.spin.setValue(v); self.spin.blockSignals(False)

        def _from_spin(v: float):
            iv = int(v)
            if self.slider.value() != iv:
                self.slider.blockSignals(True); self.slider.setValue(iv); self.slider.blockSignals(False)

        self.slider.valueChanged.connect(_from_slider)
        self.spin.valueChanged.connect(_from_spin)

        form.addRow("Threshold (px):", self.slider)
        form.addRow("Value:", self.spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def value(self) -> float:
        """Return the currently selected threshold value.

        Returns:
            The threshold in pixels as a float.
        """
        return float(self.spin.value())


