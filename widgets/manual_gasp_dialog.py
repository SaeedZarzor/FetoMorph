"""Manual data-entry dialog for the GASP similarity profile.

Used when the user chooses to score a brain by typing in hallmark values
directly, without loading an image and running 'All hallmarks'.
"""

from __future__ import annotations

from deps import *
from PySide6.QtGui import QDoubleValidator, QIntValidator


FLOAT_KEYS: tuple[tuple[str, str], ...] = (
    ("Area", "Area"),
    ("Perimeter", "Perimeter"),
    ("LGI", "LGI"),
    ("Compactness", "Compactness"),
    ("PrimaryMeanDepth", "Primary mean depth"),
    ("SecondaryMeanDepth", "Secondary mean depth"),
    ("TertiaryMeanDepth", "Tertiary mean depth"),
    ("UnclassifiedMeanDepth", "Unclassified mean depth"),
)
INT_KEYS: tuple[tuple[str, str], ...] = (
    ("PrimarySulciCount", "Primary sulci count"),
    ("SecondarySulciCount", "Secondary sulci count"),
    ("TertiarySulciCount", "Tertiary sulci count"),
    ("UnclassifiedSulciCount", "Unclassified sulci count"),
)
AXES: tuple[str, ...] = ("axial", "coronal", "sagittal")


class ManualGASPDialog(QDialog):
    """Collect hallmark inputs + analysis metadata for a manual GASP run."""

    def __init__(
        self,
        parent=None,
        *,
        default_axis: str = "coronal",
        default_unit: str = "mm",
        default_kernel_size: int | None = 25,
        default_pixel_size: float | None = None,
        default_project_name: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Similarity Profile — Manual Entry")
        self.setModal(True)
        self.setMinimumWidth(440)

        outer = QVBoxLayout(self)

        info = QLabel(
            "Enter hallmark values for the brain to score against the "
            "gestational-week reference profiles.\n"
            "Leave a field blank to omit that metric from the GASP "
            "computation."
        )
        info.setWordWrap(True)
        outer.addWidget(info)

        form = QFormLayout()

        self.name_edit = QLineEdit("", self)
        self.name_edit.setText(default_project_name)
        self.name_edit.setPlaceholderText("Optional — used as the result folder name")
        form.addRow("Project / Result name:", self.name_edit)

        self.axis_combo = QComboBox(self)
        self.axis_combo.addItems(list(AXES))
        if default_axis in AXES:
            self.axis_combo.setCurrentText(default_axis)
        form.addRow("Axis:", self.axis_combo)

        outer.addLayout(form)

        # Hallmark inputs grouped for visual clarity.
        metrics_group = QGroupBox("Hallmark values")
        metrics_form = QFormLayout(metrics_group)

        self._float_inputs: dict[str, QLineEdit] = {}
        for key, label in FLOAT_KEYS:
            le = QLineEdit(self)
            le.setPlaceholderText("blank = omit")
            le.setValidator(QDoubleValidator(self))
            metrics_form.addRow(f"{label}:", le)
            self._float_inputs[key] = le

        self._int_inputs: dict[str, QLineEdit] = {}
        for key, label in INT_KEYS:
            le = QLineEdit(self)
            le.setPlaceholderText("blank = omit")
            iv = QIntValidator(self)
            iv.setBottom(0)
            le.setValidator(iv)
            metrics_form.addRow(f"{label}:", le)
            self._int_inputs[key] = le

        outer.addWidget(metrics_group)

        # Analysis metadata (kernel, pixel size, units).
        meta_group = QGroupBox("Analysis parameters (optional)")
        meta_form = QFormLayout(meta_group)

        self.kernel_edit = QLineEdit(self)
        if default_kernel_size is not None:
            self.kernel_edit.setText(str(int(default_kernel_size)))
        kv = QIntValidator(self)
        kv.setBottom(1)
        self.kernel_edit.setValidator(kv)
        meta_form.addRow("Kernel size:", self.kernel_edit)

        self.pxsize_edit = QLineEdit(self)
        if default_pixel_size is not None:
            self.pxsize_edit.setText(f"{float(default_pixel_size):.6g}")
        self.pxsize_edit.setValidator(QDoubleValidator(self))
        meta_form.addRow("Pixel size:", self.pxsize_edit)

        self.unit_edit = QLineEdit(default_unit, self)
        meta_form.addRow("Length unit:", self.unit_edit)

        outer.addWidget(meta_group)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

    def _on_accept(self) -> None:
        measured = self._collect_measured()
        if not measured:
            QMessageBox.warning(
                self, "Similarity Profile",
                "Please provide at least one hallmark value.")
            return
        self.accept()

    def _collect_measured(self) -> dict:
        out: dict = {}
        for key, _ in FLOAT_KEYS:
            t = self._float_inputs[key].text().strip()
            if not t:
                continue
            try:
                out[key] = float(t)
            except ValueError:
                continue
        for key, _ in INT_KEYS:
            t = self._int_inputs[key].text().strip()
            if not t:
                continue
            try:
                out[key] = int(t)
            except ValueError:
                continue
        return out

    def values(self) -> dict:
        """Return measured dict, axis, project name, and optional metadata."""
        kernel = None
        t = self.kernel_edit.text().strip()
        if t:
            try:
                kernel = int(t)
            except ValueError:
                pass
        pxsize = None
        t = self.pxsize_edit.text().strip()
        if t:
            try:
                pxsize = float(t)
            except ValueError:
                pass
        return {
            "project_name": self.name_edit.text().strip(),
            "axis": self.axis_combo.currentText(),
            "measured": self._collect_measured(),
            "kernel_size": kernel,
            "pixel_size": pxsize,
            "length_unit": self.unit_edit.text().strip() or None,
        }
