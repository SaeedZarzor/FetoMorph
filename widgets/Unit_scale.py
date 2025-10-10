from deps import *

class UnitScaleDialog(QDialog):
    """Ask for the length unit (e.g., mm) and pixel size (unit/pixel) in one window."""
    def __init__(self, parent=None, unit_init: str = "mm", pixel_size_init: float = 0.03):
        super().__init__(parent)
        self.setWindowTitle("Units & Pixel Size")
        self.setModal(True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)  # ensure destruction on close
        self.status = False
        self.unit = unit_init
        self.scale = pixel_size_init
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
        
        btn_scalebar = btns.addButton("Set from scalebar…", QDialogButtonBox.ActionRole)
        btn_scalebar.clicked.connect(self._start_scalebar)

    def _update_suffix(self, unit: str):
        self.unit = (unit or "mm").strip()
        self.scale_spin.setSuffix(f" {unit}/pixel")
        self.scale = self.scale_spin.value()

    def values(self) -> tuple[str, float]:
        return self.unit, self.scale


    def _start_scalebar(self):
        self.status = True
        self.reject()       # closes the dialog; with WA_DeleteOnClose it will be destroyed
        self.deleteLater()  # safety

    def _get_status(self) -> bool:
        return self.status
