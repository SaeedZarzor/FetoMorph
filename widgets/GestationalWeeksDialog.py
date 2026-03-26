"""Gestational weeks selector dialog widget.

Provides a slider-and-spinbox dialog for choosing a gestational age
in weeks, typically used to associate fetal brain data with a
developmental time point.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFormLayout, QSlider, QSpinBox,
                                QDialogButtonBox, QComboBox)


class GestationalWeeksDialog(QDialog):
    """Dialog for picking a gestational age in weeks.

    The slider and spin box are kept in sync within the specified range.
    """

    def __init__(self, parent=None, initial: int = 24, minimum: int = 24, maximum: int = 38):
        """Initialise the gestational weeks dialog.

        Args:
            parent: Parent widget.
            initial: Starting gestational week.
            minimum: Earliest allowed week.
            maximum: Latest allowed week.
        """
        super().__init__(parent)
        self.setWindowTitle("Gestational Age (Weeks)")
        self.setModal(True)

        initial = max(minimum, min(maximum, initial))

        form = QFormLayout(self)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(4)
        self.slider.setValue(initial)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimumWidth(280)
        self.slider.setTickInterval(1)
        self.slider.setStyleSheet("""
        QSlider::groove:horizontal {height: 6px; background: #444; border-radius: 3px;}
        QSlider::sub-page:horizontal {background: #2aa1ff; height: 6px; border-radius: 3px;}
        QSlider::add-page:horizontal {background: #333; height: 6px; border-radius: 3px;}
        QSlider::handle:horizontal {width: 16px; background: #2aa1ff; border-radius: 8px; margin: -5px 0;}
        """)

        self.spin = QSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(1)
        self.spin.setValue(initial)
        self.spin.setSuffix(" weeks")

        # keep slider and spin box in sync
        def _from_slider(v: int):
            if self.spin.value() != v:
                self.spin.blockSignals(True); self.spin.setValue(v); self.spin.blockSignals(False)

        def _from_spin(v: int):
            if self.slider.value() != v:
                self.slider.blockSignals(True); self.slider.setValue(v); self.slider.blockSignals(False)

        self.slider.valueChanged.connect(_from_slider)
        self.spin.valueChanged.connect(_from_spin)

        self.axis_combo = QComboBox(self)
        self.axis_combo.addItems(["Coronal", "Sagittal", "Axial"])
        self.axis_combo.setMinimumWidth(160)

        form.addRow("Gestational week:", self.slider)
        form.addRow("Value:", self.spin)
        form.addRow("Axis:", self.axis_combo)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def value(self) -> int:
        """Return the selected gestational week."""
        return int(self.spin.value())

    def axis(self) -> str:
        """Return the selected axis (Coronal, Sagittal, or Axial)."""
        return self.axis_combo.currentText()
