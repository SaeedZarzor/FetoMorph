"""Unit and pixel-size configuration dialog.

Allows the user to set the measurement unit (mm, um, cm, etc.) and the
physical size of one pixel in that unit, either manually or by launching
a scale-bar measurement workflow.
"""

from deps import *

class UnitScaleDialog(QDialog):
    """Dialog for setting the length unit and pixel size.

    Provides an editable combo box for the unit and a spin box for the
    pixel size.  An optional "Set from scalebar" button lets the user
    switch to an interactive measurement mode instead.
    """

    def __init__(self, parent=None, unit_init: str = "mm", pixel_size_init: float = 0.03):
        """Initialise the unit/scale dialog.

        Args:
            parent: Parent widget.
            unit_init: Initial unit string.
            pixel_size_init: Initial pixel size in unit/pixel.
        """
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
        """Update the spin box suffix and internal state when the unit changes."""
        self.unit = (unit or "mm").strip()
        self.scale_spin.setSuffix(f" {unit}/pixel")
        self.scale = self.scale_spin.value()

    def values(self) -> tuple[str, float]:
        """Return the selected unit and pixel size.

        Returns:
            A tuple of (unit_string, pixel_size_in_unit_per_pixel).
        """
        return self.unit, self.scale


    def _start_scalebar(self):
        """Close the dialog and signal that the scalebar workflow should begin."""
        self.status = True
        self.reject()       # closes the dialog; with WA_DeleteOnClose it will be destroyed
        self.deleteLater()  # safety

    def _get_status(self) -> bool:
        """Return True if the scalebar workflow was requested."""
        return self.status
