from deps import *

class UnitScaleDialog(QDialog):
    """Ask for the length unit (e.g., mm) and pixel size (unit/pixel) in one window."""
    def __init__(self, parent=None, unit_init: str = "mm", pixel_size_init: float = 0.03):
        super().__init__(parent)
        self.setWindowTitle("Units & Pixel Size")
        self.setModal(True)

        form = QFormLayout(self)

        # Unit selector (editable combo with common presets)
        self.unit_box = QComboBox(self)
        self.unit_box.setEditable(True)
        self.unit_box.addItems(["mm", "µm", "cm", "m", "in"])
        # set initial
        idx = self.unit_box.findText(unit_init)
        if idx >= 0:
            self.unit_box.setCurrentIndex(idx)
        else:
            self.unit_box.setEditText(unit_init)

        # Pixel size (unit/pixel)
        self.scale_spin = QDoubleSpinBox(self)
        self.scale_spin.setRange(1e-12, 1e9)
        self.scale_spin.setDecimals(6)
        self.scale_spin.setValue(float(pixel_size_init))
        self.scale_spin.setMinimumWidth(160)
        self._update_suffix(self.unit_box.currentText())
        self.unit_box.currentTextChanged.connect(self._update_suffix)

        form.addRow("Length unit:", self.unit_box)
        form.addRow("Pixel size:", self.scale_spin)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _update_suffix(self, unit: str):
        unit = (unit or "mm").strip()
        self.scale_spin.setSuffix(f" {unit}/pixel")

    def values(self) -> tuple[str, float]:
        unit = (self.unit_box.currentText() or "mm").strip()
        return unit, float(self.scale_spin.value())

