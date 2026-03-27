"""Scale-bar calibration dialog.

After the user draws a line on the image to measure a known real-world
distance in pixels, this dialog lets them enter the corresponding
physical length and unit so the application can compute pixels-per-unit.
"""

from PySide6.QtWidgets import (QDialog, QFormLayout, QLabel, QDoubleSpinBox,
                                QComboBox, QDialogButtonBox)

class ScalebarSetScaleDialog(QDialog):
    """Dialog for setting the image scale from a measured pixel length.

    Displays the pixel length that was measured and asks the user to
    provide the equivalent real-world length and unit.
    """

    def __init__(self, pixel_length: float, unit_init: str = "mm", parent=None):
        """Initialise the scale-bar calibration dialog.

        Args:
            pixel_length: Measured distance in pixels.
            unit_init: Initial measurement unit string.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Set Scale from Scalebar")
        self.setModal(True)
        lay = QFormLayout(self)

        # Show measured pixels (read-only)
        from PySide6.QtWidgets import QLabel
        self.lbl_px = QLabel(f"{pixel_length:.2f} px")
        lay.addRow("Measured length:", self.lbl_px)

        # Real length
        self.len_spin = QDoubleSpinBox(self)
        self.len_spin.setRange(1e-9, 1e12)
        self.len_spin.setDecimals(6)
        self.len_spin.setValue(1.0)
        self.len_spin.setMinimumWidth(160)

        # Unit
        self.unit_box = QComboBox(self)
        self.unit_box.setEditable(True)
        self.unit_box.addItems(["mm", "µm", "cm", "m", "in"])
        idx = self.unit_box.findText(unit_init)
        self.unit_box.setCurrentIndex(idx if idx >= 0 else 0)
        if idx < 0:
            self.unit_box.setEditText(unit_init)

        lay.addRow("Real-world length:", self.len_spin)
        lay.addRow("Unit:", self.unit_box)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addRow(btns)

        self._pxlen = float(pixel_length)

    def values(self) -> tuple[float, str]:
        """Returns (px_per_unit, unit)."""
        real = float(self.len_spin.value())
        unit = (self.unit_box.currentText() or "mm").strip()
        if real <= 0:
            raise ValueError("Real length must be > 0.")
        px_per_unit = self._pxlen / real
        return px_per_unit, unit
