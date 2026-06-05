"""Morphological kernel size dialog widget."""

from deps import *

class KernelSizeDialog(QDialog):
    """Dialog for picking a morphology kernel diameter in millimetres."""

    def __init__(self, parent=None, initial: float = 5.0, minimum: float = 0.5, maximum: float = 25.0):
        """Initialise the kernel size dialog.

        Args:
            parent: Parent widget.
            initial: Starting kernel diameter in millimetres.
            minimum: Smallest allowed diameter in millimetres.
            maximum: Largest allowed diameter in millimetres.
        """
        super().__init__(parent)
        self.setWindowTitle("Morphology kernel diameter (mm)")
        self.setModal(True)

        minimum = float(minimum)
        maximum = float(maximum)
        initial = max(minimum, min(maximum, float(initial)))

        form = QFormLayout(self)

        self._scale = 2  # 0.5 mm steps
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(int(round(minimum * self._scale)), int(round(maximum * self._scale)))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(2)
        self.slider.setValue(int(round(initial * self._scale)))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimumWidth(280)
        self.slider.setTickInterval(2)
        self.slider.setStyleSheet("""
        QSlider::groove:horizontal {height: 6px; background: #444; border-radius: 3px;}
        QSlider::sub-page:horizontal {background: #2aa1ff; height: 6px; border-radius: 3px;}
        QSlider::add-page:horizontal {background: #333; height: 6px; border-radius: 3px;}
        QSlider::handle:horizontal {width: 16px; background: #2aa1ff; border-radius: 8px; margin: -5px 0;}
        """)


        self.spin = QDoubleSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(0.5)
        self.spin.setDecimals(1)
        self.spin.setSuffix(" mm")
        self.spin.setValue(initial)

        def _from_slider(v: int):
            mm = v / self._scale
            if abs(self.spin.value() - mm) > 1e-9:
                self.spin.blockSignals(True); self.spin.setValue(mm); self.spin.blockSignals(False)

        def _from_spin(v: float):
            slider_v = int(round(float(v) * self._scale))
            if self.slider.value() != slider_v:
                self.slider.blockSignals(True); self.slider.setValue(slider_v); self.slider.blockSignals(False)

        self.slider.valueChanged.connect(_from_slider)
        self.spin.valueChanged.connect(_from_spin)

        form.addRow("Kernel diameter:", self.slider)
        form.addRow("Value:", self.spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def value(self) -> float:
        """Return the selected kernel diameter in millimetres."""
        return float(self.spin.value())
