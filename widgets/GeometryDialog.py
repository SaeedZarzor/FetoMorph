# add to imports
from deps import *

class GeometryDialogWithAspect(QDialog):
    UNIT_FACTORS = {"mm": 1.0, "µm": 1e-3, "um": 1e-3, "cm": 10.0, "m": 1000.0}

    def __init__(self, parent=None, Lx=1.0, Ly=1.0, Lz=1.0, unit="mm", slice_dir="Y"):
        super().__init__(parent)
        self.setWindowTitle("Mesh Geometry")
        self.setWindowModality(Qt.ApplicationModal)

        # originals stored in mm
        f = self.UNIT_FACTORS.get(unit, 1.0)
        self._Lx0_mm = max(Lx * f, 1e-9)
        self._Ly0_mm = max(Ly * f, 1e-9)
        self._Lz0_mm = max(Lz * f, 1e-9)
        self._unit_mm_factor = f
        self._blocking = False

        # spin boxes
        self.x_sb = QDoubleSpinBox(self); self._setup_len_box(self.x_sb, Lx, unit)
        self.y_sb = QDoubleSpinBox(self); self._setup_len_box(self.y_sb, Ly, unit)
        self.z_sb = QDoubleSpinBox(self); self._setup_len_box(self.z_sb, Lz, unit)
        self.x_sb.valueChanged.connect(lambda v: self._scaled_update('x', v))
        self.y_sb.valueChanged.connect(lambda v: self._scaled_update('y', v))
        self.z_sb.valueChanged.connect(lambda v: self._scaled_update('z', v))

        # slice dir
        self.dir_cb = QComboBox(self); self.dir_cb.addItems(["X", "Y", "Z"])
        self.dir_cb.setCurrentText(slice_dir.upper() if slice_dir in ("X","Y","Z") else "Y")

        # unit
        self.unit_cb = QComboBox(self)
        self.unit_cb.setEditable(True)
        self.unit_cb.addItems(["mm", "µm", "cm", "m"])
        self.unit_cb.setCurrentText(unit if unit else "mm")
        self.unit_cb.currentTextChanged.connect(self._unit_changed)

        # layout
        form = QFormLayout()
        form.addRow("X length:", self.x_sb)
        form.addRow("Y length:", self.y_sb)
        form.addRow("Z length:", self.z_sb)
        form.addRow("Slice direction:", self.dir_cb)
        form.addRow("Unit:", self.unit_cb)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        # Original button
        self.orig_btn = QPushButton("Original")
        btns.addButton(self.orig_btn, QDialogButtonBox.ActionRole)
        self.orig_btn.clicked.connect(self._reset_to_original)

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(btns)

    def _setup_len_box(self, sb: QDoubleSpinBox, val: float, unit: str):
        sb.setDecimals(4)
        sb.setRange(1e-9, 1e12)
        sb.setSingleStep(max(val * 0.01, 0.01))
        sb.setValue(val)
        sb.setSuffix(f" {unit}")

    def _scaled_update(self, which: str, new_val: float):
        if self._blocking:
            return
        s = self.UNIT_FACTORS.get(self.unit_cb.currentText(), 1.0)
        new_mm = new_val * s
        if which == 'x':
            scale = new_mm / self._Lx0_mm
        elif which == 'y':
            scale = new_mm / self._Ly0_mm
        else:
            scale = new_mm / self._Lz0_mm
        if not (scale > 0):
            return
        self._apply(self._Lx0_mm * scale / s, self._Ly0_mm * scale / s, self._Lz0_mm * scale / s)

    def _apply(self, x, y, z):
        self._blocking = True
        self.x_sb.setValue(x)
        self.y_sb.setValue(y)
        self.z_sb.setValue(z)
        self._blocking = False

    def _unit_changed(self, new_unit: str):
        new_f = self.UNIT_FACTORS.get(new_unit, 1.0)
        old_f = self._unit_mm_factor
        if new_f == old_f:
            return
        ratio = old_f / new_f
        self._unit_mm_factor = new_f
        self._blocking = True
        self.x_sb.setValue(self.x_sb.value() * ratio)
        self.y_sb.setValue(self.y_sb.value() * ratio)
        self.z_sb.setValue(self.z_sb.value() * ratio)
        for sb in (self.x_sb, self.y_sb, self.z_sb):
            sb.setSuffix(f" {new_unit}")
        self._blocking = False

    def _reset_to_original(self):
        """Reset displayed lengths to original mesh bounds, in current unit."""
        s = self.UNIT_FACTORS.get(self.unit_cb.currentText(), 1.0)
        self._apply(self._Lx0_mm / s, self._Ly0_mm / s, self._Lz0_mm / s)

    def values(self) -> tuple[tuple[float, float, float], str, str]:
        return (
            (self.x_sb.value(), self.y_sb.value(), self.z_sb.value()),
            self.dir_cb.currentText().upper(),
            self.unit_cb.currentText().strip() or "mm",
        )
