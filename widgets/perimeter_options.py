"""Perimeter measurement options dialog."""

from deps import *
from PySide6.QtWidgets import QCheckBox


class PerimeterOptionsDialog(QDialog):
    """Dialog for binary-mask perimeter method and optional contour smoothing."""

    METHODS = {
        "arc_length": "Arc length contour perimeter",
        "crofton": "Crofton perimeter",
    }

    def __init__(
        self,
        parent=None,
        method: str = "arc_length",
        simplify: bool = False,
        epsilon: float = 0.5,
    ):
        super().__init__(parent)
        self.setWindowTitle("Perimeter Method")
        self.setModal(True)

        form = QFormLayout(self)

        self.method_combo = QComboBox(self)
        for key, label in self.METHODS.items():
            self.method_combo.addItem(label, key)
        idx = self.method_combo.findData(method if method in self.METHODS else "arc_length")
        self.method_combo.setCurrentIndex(max(0, idx))
        self.method_combo.setToolTip(
            "Crofton perimeter estimates boundary length using measurements in 4 directions: "
            "0 deg, 45 deg, 90 deg, and 135 deg. It is usually less biased than "
            "polygonal contour length for noisy binary masks. Applies to NIfTI and "
            "2D image masks."
        )

        self.simplify_check = QCheckBox("Simplify contours for arc length", self)
        self.simplify_check.setChecked(bool(simplify))
        self.simplify_check.setToolTip(
            "Apply cv2.approxPolyDP before arcLength. This does not affect Crofton."
        )

        self.epsilon_spin = QDoubleSpinBox(self)
        self.epsilon_spin.setRange(0.0, 100.0)
        self.epsilon_spin.setDecimals(2)
        self.epsilon_spin.setSingleStep(0.1)
        self.epsilon_spin.setValue(max(0.0, float(epsilon)))
        self.epsilon_spin.setMinimumWidth(120)
        self.epsilon_spin.setToolTip("Contour simplification epsilon in pixels.")

        form.addRow("Perimeter Measurement Method:", self.method_combo)
        form.addRow(self.simplify_check)
        form.addRow("Simplification epsilon (px):", self.epsilon_spin)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def method(self) -> str:
        return str(self.method_combo.currentData())

    def simplify(self) -> bool:
        return bool(self.simplify_check.isChecked())

    def epsilon(self) -> float:
        return float(self.epsilon_spin.value())
