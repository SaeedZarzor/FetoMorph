from deps import *

class KernelSizeDialog(QDialog):
    """Pick an odd kernel size for morphology."""
    def __init__(self, parent=None, initial: int = 5, minimum: int = 1, maximum: int = 201):
        super().__init__(parent)
        self.setWindowTitle("Morphology Kernel Size")
        self.setModal(True)

        # enforce sane bounds and odd default
        initial = max(minimum, min(maximum, initial))
        if initial % 2 == 0:
            initial += 1 if initial < maximum else -1

        form = QFormLayout(self)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(2)
        self.slider.setPageStep(2)
        self.slider.setValue(initial)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimumWidth(280)
        self.slider.setTickInterval(2)
        self.slider.setStyleSheet("""
        QSlider::groove:horizontal {height: 6px; background: #444; border-radius: 3px;}
        QSlider::sub-page:horizontal {background: #2aa1ff; height: 6px; border-radius: 3px;}
        QSlider::add-page:horizontal {background: #333; height: 6px; border-radius: 3px;}
        QSlider::handle:horizontal {width: 16px; background: #2aa1ff; border-radius: 8px; margin: -5px 0;}
        """)


        self.spin = QSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(2)
        self.spin.setValue(initial)

        # keep in sync and force odd values
        def _ensure_odd(v: int) -> int:
            if v % 2 == 0:
                v = v + 1 if v < maximum else v - 1
            return max(minimum, min(maximum, v))

        def _from_slider(v: int):
            v = _ensure_odd(v)
            if self.spin.value() != v:
                self.spin.blockSignals(True); self.spin.setValue(v); self.spin.blockSignals(False)

        def _from_spin(v: int):
            v = _ensure_odd(v)
            if self.slider.value() != v:
                self.slider.blockSignals(True); self.slider.setValue(v); self.slider.blockSignals(False)

        self.slider.valueChanged.connect(_from_slider)
        self.spin.valueChanged.connect(_from_spin)

        form.addRow("Kernel size (odd):", self.slider)
        form.addRow("Value:", self.spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def value(self) -> int:
        """Return the selected odd kernel size."""
        v = int(self.spin.value())
        return v if v % 2 == 1 else max(1, v - 1)

