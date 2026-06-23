"""Surface-connected cavity correction options dialog.

Combines the enable toggle and the area threshold into one widget so both
cavity-correction settings are configured from a single Adjustments action.
"""

from deps import *
from PySide6.QtWidgets import QCheckBox


class CavityOptionsDialog(QDialog):
    """Dialog to enable/disable the cavity correction and set its area threshold."""

    def __init__(self, parent=None, enabled: bool = True,
                 threshold_mm2: float = 0.0, maximum: float = 1_000_000.0):
        super().__init__(parent)
        self.setWindowTitle("Surface-Connected Cavities")
        self.setModal(True)

        form = QFormLayout(self)

        self.enable_check = QCheckBox("Enable cavity correction", self)
        self.enable_check.setChecked(bool(enabled))
        self.enable_check.setToolTip(
            "Subtract open (surface-connected) cavities from the 3-D volume and add "
            "their walls to the surface area; enclosed voids are left as solid.")

        self.spin = QDoubleSpinBox(self)
        self.spin.setRange(0.0, float(maximum))
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.5)
        self.spin.setValue(max(0.0, float(threshold_mm2)))
        self.spin.setMinimumWidth(120)
        self.spin.setToolTip("Ignore surface-connected cavities smaller than this area.")

        # The threshold only matters when the correction is enabled.
        self.spin.setEnabled(self.enable_check.isChecked())
        self.enable_check.toggled.connect(self.spin.setEnabled)

        form.addRow(self.enable_check)
        form.addRow("Min cavity area (mm²):", self.spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def enabled(self) -> bool:
        return bool(self.enable_check.isChecked())

    def threshold(self) -> float:
        return float(self.spin.value())
