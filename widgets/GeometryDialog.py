"""Mesh geometry dialog with interactive 3-D preview.

Lets the user inspect and rescale the X/Y/Z dimensions of a loaded mesh,
choose a slicing direction and measurement unit, and preview the result
in an embedded PyVista viewer.
"""

# add to imports
from deps import *
from pyvistaqt import QtInteractor
import pyvista as pv

class GeometryDialogWithAspect(QDialog):
    """Dialog for editing mesh geometry dimensions with a live 3-D preview.

    The three axis lengths are kept in sync so that changing one rescales
    the others proportionally.  A PyVista interactor on the right shows
    the mesh with an optional axis highlight for the chosen slice direction.
    """

    UNIT_FACTORS = {"mm": 1.0, "µm": 1e-3, "um": 1e-3, "cm": 10.0, "m": 1000.0}

    def __init__(self, parent=None, mesh=pv.DataSet, Lx=1.0, Ly=1.0, Lz=1.0, unit="mm", slice_dir="Y", flat_axis: int | None = None):
        """Initialise the geometry dialog.

        Args:
            parent: Parent widget.
            mesh: PyVista mesh dataset to display and measure.
            Lx: Initial X-axis length in the given unit.
            Ly: Initial Y-axis length in the given unit.
            Lz: Initial Z-axis length in the given unit.
            unit: Measurement unit string (e.g. "mm", "µm").
            slice_dir: Initial slicing axis ("X", "Y", or "Z").
            flat_axis: If the mesh is planar, the perpendicular axis index
                (0=X, 1=Y, 2=Z).  When set, the flat-axis spin box is
                disabled and the slice-direction combo is hidden.
        """
        super().__init__(parent)

        self.setWindowTitle("Mesh Geometry")
        self.setWindowModality(Qt.ApplicationModal)
        self.mesh = mesh
        self._flat_axis = flat_axis

        f = self.UNIT_FACTORS.get(unit, 1.0)
        self._Lx0_mm = max(Lx * f, 1e-9)
        self._Ly0_mm = max(Ly * f, 1e-9)
        self._Lz0_mm = max(Lz * f, 1e-9)
        self._unit_mm_factor = f
        self._blocking = False

        # --- left panel: controls
        left = QWidget(self)
        vleft = QVBoxLayout()          # no parent here
        left.setLayout(vleft)          # single layout for 'left'
        form = QFormLayout()           # no parent here


        self.x_sb = QDoubleSpinBox(self); self._setup_len_box(self.x_sb, Lx, unit)
        self.y_sb = QDoubleSpinBox(self); self._setup_len_box(self.y_sb, Ly, unit)
        self.z_sb = QDoubleSpinBox(self); self._setup_len_box(self.z_sb, Lz, unit)
        self.x_sb.valueChanged.connect(lambda v: self._scaled_update('x', v))
        self.y_sb.valueChanged.connect(lambda v: self._scaled_update('y', v))
        self.z_sb.valueChanged.connect(lambda v: self._scaled_update('z', v))

        self.dir_cb = QComboBox(self); self.dir_cb.addItems(["X", "Y", "Z"])
        self.dir_cb.currentTextChanged.connect(self._highlight_axis)

        self.unit_cb = QComboBox(self)
        self.unit_cb.setEditable(True)
        self.unit_cb.addItems(["mm", "µm", "cm", "m"])
        self.unit_cb.setCurrentText(unit if unit else "mm")
        self.unit_cb.currentTextChanged.connect(self._unit_changed)

        form.addRow("X length:", self.x_sb)
        form.addRow("Y length:", self.y_sb)
        form.addRow("Z length:", self.z_sb)

        # For planar meshes: disable flat-axis spin box, hide slice direction
        if flat_axis is not None:
            flat_sb = (self.x_sb, self.y_sb, self.z_sb)[flat_axis]
            flat_sb.setEnabled(False)
            flat_sb.setToolTip("Flat axis — not editable for planar meshes")
            # Auto-set slice direction to the flat axis
            self.dir_cb.setCurrentText(("X", "Y", "Z")[flat_axis])
            self.dir_cb.setVisible(False)
        else:
            self.dir_cb.setCurrentText(slice_dir.upper() if slice_dir in ("X","Y","Z") else "Y")
            form.addRow("Slice direction:", self.dir_cb)

        form.addRow("Unit:", self.unit_cb)

        self.orig_btn = QPushButton("Original", self)
        ok_btn = QPushButton("OK", self)
        cancel_btn = QPushButton("Cancel", self)
        self.orig_btn.clicked.connect(self._reset_to_original)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        # bottom row (right-aligned)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.orig_btn)
        button_row.addSpacing(10)
        button_row.addWidget(ok_btn)
        button_row.addWidget(cancel_btn)
        
        # assemble
        vleft.addLayout(form)          # add child layout to the ONLY parent layout
        vleft.addStretch(1)
        vleft.addLayout(button_row)

        # --- right panel: 3D view
        self.plot = QtInteractor(self)
        self._init_view()

        # --- layout
        root = QHBoxLayout(self)
        root.addWidget(left, 0)       # auto-size to form
        root.addWidget(self.plot, 1)  # take remaining space

        
    def _init_view(self):
        """Set up the 3-D preview with mesh, axes, and bounding box."""
        self.plot.clear()
        self.plot.add_mesh(self.mesh, style="surface", color='blue', show_edges=True, opacity=1.0)
        self.plot.add_axes(interactive=False)  # XYZ triad in corner
        self.plot.show_bounds(
            grid="back",
            location="outer",
            all_edges=True,
            xtitle="X",
            ytitle="Y",
            ztitle="Z",
        )
        self.plot.add_bounding_box(color="black", line_width=1)
        if hasattr(self.plot, "toolbar"):
            self.plot.toolbar.setVisible(False)
        if hasattr(self.plot, "statusBar"):
            self.plot.statusBar().setVisible(False)
            
#        self._add_axis_arrows()
        self._highlight_axis(self.dir_cb.currentText())
        self.plot.camera_position = "iso"
        self.plot.reset_camera()

#    def _add_axis_arrows(self):
#            cx, cy, cz = self.mesh.center
#            s = max(self._Lx0_mm, self._Ly0_mm, self._Lz0_mm) / self._unit_mm_factor * 0.2
#            self.plot.add_mesh(pv.Arrow((cx, cy, cz), (1, 0, 0), scale=s), color="red", name="axis_x")
#            self.plot.add_mesh(pv.Arrow((cx, cy, cz), (0, 1, 0), scale=s), color="green", name="axis_y")
#            self.plot.add_mesh(pv.Arrow((cx, cy, cz), (0, 0, 1), scale=s), color="blue", name="axis_z")

    def _highlight_axis(self, axis: str):
        """Brighten the selected axis arrow and dim the others."""
        if not hasattr(self, "plot"):
            return
        axis = axis.upper()
        for k in ("X", "Y", "Z"):
            actor = self.plot.renderer._actors.get(f"axis_{k}")
            if not actor:
                continue
            opacity = 1.0 if k == axis else 0.25
            actor.GetProperty().SetOpacity(opacity)
        self.plot.render()
        
    def _setup_len_box(self, sb: QDoubleSpinBox, val: float, unit: str):
        """Configure a length spin box with appropriate range and suffix."""
        sb.setDecimals(4)
        sb.setRange(1e-9, 1e12)
        sb.setSingleStep(max(val * 0.01, 0.01))
        sb.setValue(val)
        sb.setSuffix(f" {unit}")

    def _scaled_update(self, which: str, new_val: float):
        """Proportionally update all axes when one value changes."""
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
        new_x = self._Lx0_mm * scale / s
        new_y = self._Ly0_mm * scale / s
        new_z = self._Lz0_mm * scale / s
        # For planar meshes keep the flat axis value unchanged
        if self._flat_axis is not None:
            if self._flat_axis == 0:
                new_x = self.x_sb.value()
            elif self._flat_axis == 1:
                new_y = self.y_sb.value()
            else:
                new_z = self.z_sb.value()
        self._apply(new_x, new_y, new_z)

    def _apply(self, x, y, z):
        """Set all three spin box values without triggering re-entrant updates."""
        self._blocking = True
        self.x_sb.setValue(x)
        self.y_sb.setValue(y)
        self.z_sb.setValue(z)
        self._blocking = False

    def _unit_changed(self, new_unit: str):
        """Convert displayed values when the measurement unit changes."""
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
        xmin, xmax, ymin, ymax, zmin, zmax = self.mesh.bounds
        Lx_ori, Ly_ori, Lz_ori = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9), max(zmax - zmin, 1e-9)
        self._apply(Lx_ori / s, Ly_ori / s, Lz_ori / s)

    def values(self) -> tuple[tuple[float, float, float], str, str]:
        """Return the current geometry settings.

        Returns:
            A tuple of ((Lx, Ly, Lz), slice_direction, unit).
        """
        return (
            (self.x_sb.value(), self.y_sb.value(), self.z_sb.value()),
            self.dir_cb.currentText().upper(),
            self.unit_cb.currentText().strip() or "mm",
        )
