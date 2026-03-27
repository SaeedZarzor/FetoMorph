"""Unit and pixel-size configuration dialog.

Allows the user to set the measurement unit (mm, um, cm, etc.) and the
physical size of one pixel in that unit, either manually or by launching
a scale-bar measurement workflow.
"""

from deps import *

class UnitScaleDialog(QDialog):
    """Dialog for setting the length unit and pixel size.

    Provides an editable combo box for the unit and split fields for the
    pixel-size integer and fractional parts. The number of fractional
    digits is user-configurable. An optional "Set from scalebar" button
    lets the user switch to an interactive measurement mode instead.
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
        self._cached_unit = self.unit
        self._cached_scale = float(self.scale)
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

        # Pixel size (unit/pixel): integer and fractional inputs + decimals control.
        self.scale_int_spin = QSpinBox(self)
        self.scale_int_spin.setRange(0, 1_000_000_000)
        self.scale_int_spin.setMinimumWidth(120)

        self.decimals_spin = QSpinBox(self)
        self.decimals_spin.setRange(1, 12)
        self.decimals_spin.setValue(6)

        self.scale_frac_spin = QSpinBox(self)
        self.scale_frac_spin.setRange(0, 999999)
        self.scale_frac_spin.setMinimumWidth(120)

        self.unit_suffix_lbl = QLabel(self)
        self.full_value_lbl = QLabel(self)
        self.full_value_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Keep legacy attribute name used in older code paths.
        self.scale_spin = QWidget(self)
        parts_layout = QHBoxLayout(self.scale_spin)
        parts_layout.setContentsMargins(0, 0, 0, 0)
        parts_layout.setSpacing(6)
        parts_layout.addWidget(self.scale_int_spin)
        parts_layout.addWidget(QLabel(".", self))
        parts_layout.addWidget(self.scale_frac_spin)
        parts_layout.addWidget(QLabel("decimals:", self))
        parts_layout.addWidget(self.decimals_spin)
        parts_layout.addWidget(self.unit_suffix_lbl)
        parts_layout.addStretch(1)

        self._init_scale_parts(float(pixel_size_init))
        self.scale_int_spin.valueChanged.connect(self._sync_scale)
        self.scale_frac_spin.valueChanged.connect(self._sync_scale)
        self.decimals_spin.valueChanged.connect(self._on_decimals_changed)

        self._update_suffix(self.unit_box.currentText())
        self.unit_box.currentTextChanged.connect(self._update_suffix)

        form.addRow("Length unit:", self.unit_box)
        form.addRow("Pixel size:", self.scale_spin)
        form.addRow("Full value:", self.full_value_lbl)

        # OK / Cancel
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)
        
        btn_scalebar = btns.addButton("Set from scalebar…", QDialogButtonBox.ActionRole)
        btn_scalebar.clicked.connect(self._start_scalebar)

    def _update_suffix(self, unit: str):
        """Update the spin box suffix and internal state when the unit changes."""
        self.unit = (unit or "mm").strip()
        self.unit_suffix_lbl.setText(f"{self.unit}/pixel")
        self._sync_scale()

    def _init_scale_parts(self, pixel_size: float):
        """Initialise integer/fractional controls from a float."""
        val = max(float(pixel_size), 0.0)
        decimals = min(12, max(1, self._detect_decimals(val)))
        self.decimals_spin.blockSignals(True)
        self.decimals_spin.setValue(decimals)
        self.decimals_spin.blockSignals(False)

        int_part = int(math.floor(val))
        scale = 10 ** decimals
        frac_part = int(round((val - int_part) * scale))
        if frac_part >= scale:
            int_part += 1
            frac_part = 0

        self.scale_int_spin.setValue(int_part)
        self._set_frac_range(decimals)
        self.scale_frac_spin.setValue(frac_part)
        self._sync_scale()

    @staticmethod
    def _detect_decimals(val: float) -> int:
        """Guess a practical decimal precision from the initial value."""
        s = f"{val:.12f}".rstrip("0").rstrip(".")
        if "." not in s:
            return 6
        return len(s.split(".", 1)[1])

    def _set_frac_range(self, decimals: int):
        """Update fractional-part range according to decimal places."""
        decimals = min(decimals, 9)  # cap to avoid 32-bit int overflow
        max_frac = (10 ** decimals) - 1
        self.scale_frac_spin.setRange(0, max_frac)

    def _on_decimals_changed(self, new_decimals: int):
        """Rescale current fractional value when decimal precision changes."""
        old_decimals = getattr(self, "_last_decimals", self.decimals_spin.value())
        old_frac = self.scale_frac_spin.value()
        old_scale = 10 ** old_decimals
        frac_ratio = old_frac / old_scale

        self._set_frac_range(new_decimals)
        new_scale = 10 ** new_decimals
        new_frac = int(round(frac_ratio * new_scale))
        new_frac = max(0, min(new_frac, new_scale - 1))
        self.scale_frac_spin.setValue(new_frac)
        self._last_decimals = new_decimals
        self._sync_scale()

    def _sync_scale(self):
        """Combine integer and fractional fields into a single scale value."""
        decimals = self.decimals_spin.value()
        scale = 10 ** decimals
        self.scale = float(self.scale_int_spin.value()) + (float(self.scale_frac_spin.value()) / float(scale))
        self._last_decimals = decimals
        self._cached_unit = self.unit
        self._cached_scale = float(self.scale)
        self._update_full_value_preview()

    def _update_full_value_preview(self):
        """Show the fully combined number before user confirms the dialog."""
        decimals = getattr(self, "_last_decimals", self.decimals_spin.value())
        number_txt = f"{self.scale:.{decimals}f}"
        self.full_value_lbl.setText(f"{number_txt} {self.unit}/pixel")

    def _on_accept(self):
        """Cache values before closing the dialog."""
        try:
            self._sync_scale()
        except RuntimeError:
            # Widgets may already be tearing down; keep last cached values.
            pass
        self.accept()

    def values(self) -> tuple[str, float]:
        """Return the selected unit and pixel size.

        Returns:
            A tuple of (unit_string, pixel_size_in_unit_per_pixel).
        """
        return self._cached_unit, self._cached_scale


    def _start_scalebar(self):
        """Close the dialog and signal that the scalebar workflow should begin."""
        self.status = True
        self.reject()       # closes the dialog; with WA_DeleteOnClose it will be destroyed
        self.deleteLater()  # safety

    def _get_status(self) -> bool:
        """Return True if the scalebar workflow was requested."""
        return self.status
