"""Slice thickness dialog widget.

Provides a slider-and-spinbox dialog for choosing a floating-point slice
thickness value, mapping a continuous range onto an integer slider for
smooth interaction.
"""

from deps import *

class SilceThicknessDialog(QDialog):
    """Dialog for selecting a slice thickness as a floating-point value.

    An integer QSlider is internally mapped to the float range
    [minimum, maximum] with the given step size.
    """

    def __init__(self, parent=None, initial: float = 0.5, minimum: float = 0.001, maximum: float = 10.0, step: float = 0.005):
        """Initialise the slice thickness dialog.

        Args:
            parent: Parent widget.
            initial: Starting thickness value.
            minimum: Lower bound of the allowed range.
            maximum: Upper bound of the allowed range.
            step: Step size for the slider and spin box.
        """
        super().__init__(parent)
        self.setWindowTitle("Set Slice Thickness")
        self.setModal(True)

        # clamp
        initial = max(minimum, min(maximum, initial))
        # map float range -> integer slider steps
        self._min = float(minimum)
        self._max = float(maximum)
        self._step = float(step)
        self._imax = int(round((self._max - self._min) / self._step))

        form = QFormLayout(self)

        # int slider, scaled to floats
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self._imax)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, int(round(0.1 / self._step))))  # ~0.1 step pages
        self.slider.setValue(self._to_int(initial))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimumWidth(280)
        self.slider.setTickInterval(max(1, int(round(0.05 / self._step))))  # tick ~0.05
        self.slider.setStyleSheet("""
        QSlider::groove:horizontal {height: 6px; background: #444; border-radius: 3px;}
        QSlider::sub-page:horizontal {background: #2aa1ff; height: 6px; border-radius: 3px;}
        QSlider::add-page:horizontal {background: #333; height: 6px; border-radius: 3px;}
        QSlider::handle:horizontal {width: 16px; background: #2aa1ff; border-radius: 8px; margin: -5px 0;}
        """)

        # float spin box
        self.spin = QDoubleSpinBox(self)
        self.spin.setRange(self._min, self._max)
        self.spin.setSingleStep(self._step)
        self.spin.setDecimals(max(0, min(6, self._decimals(self._step))))
        self.spin.setValue(initial)
        font = self.spin.font()
        font.setPointSize(14)  # increase number for larger text
        self.spin.setFont(font)
        self.spin.setMinimumWidth(100)  # optional, adjust width

        # sync both ways
        def _from_slider(i: int):
            f = self._to_float(i)
            if self.spin.value() != f:
                self.spin.blockSignals(True); self.spin.setValue(f); self.spin.blockSignals(False)

        def _from_spin(f: float):
            i = self._to_int(f)
            if self.slider.value() != i:
                self.slider.blockSignals(True); self.slider.setValue(i); self.slider.blockSignals(False)

        self.slider.valueChanged.connect(_from_slider)
        self.spin.valueChanged.connect(_from_spin)

        form.addRow("Slice Thickness:", self.slider)
        form.addRow("Value:", self.spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    # helpers
    def _to_int(self, val: float) -> int:
        """Convert a float value to its integer slider position."""
        return int(round((float(val) - self._min) / self._step))

    def _to_float(self, i: int) -> float:
        """Convert an integer slider position back to its float value."""
        return self._min + i * self._step

    @staticmethod
    def _decimals(step: float) -> int:
        s = f"{step:.10f}".rstrip("0").split(".")
        return len(s[1]) if len(s) == 2 else 0

    def value(self) -> float:
        """Return selected value as float"""
        return float(self.spin.value())
