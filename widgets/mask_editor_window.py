"""Floating tool window that drives ScaledImageLabel's mask painting mode.

Shown while the user paints a mask over the current 2-D image. It floats
above the main window (Qt.Tool), can be dragged anywhere and remembers its
position for the session. Every button mirrors a keyboard shortcut handled
by the label itself, so the two stay in sync through ``maskBrushChanged``.
Closing the window cancels the edit.
"""

from deps import *
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import QGridLayout

from theme import ACCENT, ACCENT_TEXT, BORDER, DISABLED_TEXT


class MaskEditorWindow(QWidget):
    """Spline/brush tools, eraser, fill, undo, clear, save and cancel for mask painting."""

    def __init__(self, image_label, parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint
                         | Qt.CustomizeWindowHint)
        self._label = image_label
        self._placed = False  # first show is anchored to the image; later shows keep the user's spot
        self.setWindowTitle("Mask Tools")
        self.setAttribute(Qt.WA_ShowWithoutActivating)  # keep keyboard shortcuts on the image

        # make the active tool / eraser obvious
        self.setStyleSheet(
            f"QToolButton:checked {{ border:1px solid {ACCENT}; color:{ACCENT}; font-weight:600; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # --- tool ---
        tools = QActionGroup(self)
        tools.setExclusive(True)
        self.act_spline = QAction("Spline", self)
        self.act_spline.setCheckable(True)
        self.act_spline.setToolTip(
            "Click to place points along the outline — a smooth curve follows them (S).\n"
            "Drag a point to adjust it; double-click a point to make it a sharp corner (again to smooth it).\n"
            "Right-click / Backspace removes the last point.\n"
            "Click the first point or press Enter to close and fill.")
        self.act_spline.triggered.connect(lambda: self._set_tool("spline"))
        tools.addAction(self.act_spline)
        self.act_brush = QAction("Brush", self)
        self.act_brush.setCheckable(True)
        self.act_brush.setToolTip("Freehand painting: left-drag paints, right-drag erases (B)")
        self.act_brush.triggered.connect(lambda: self._set_tool("brush"))
        tools.addAction(self.act_brush)

        self.act_erase = QAction("Eraser", self)
        self.act_erase.setCheckable(True)
        self.act_erase.setToolTip("Spline shapes cut out and left-drag erases instead of painting (E).\n"
                                  "Right-drag always erases in Brush mode.")
        self.act_erase.toggled.connect(self._on_erase_toggled)

        lay.addWidget(self._section("Tool"))
        row = QHBoxLayout(); row.setSpacing(4)
        for act in (self.act_spline, self.act_brush, self.act_erase):
            row.addWidget(self._button(act))
        lay.addLayout(row)

        brush_row = QHBoxLayout(); brush_row.setSpacing(6)
        self.brush_label = QLabel("Brush (px)")
        brush_row.addWidget(self.brush_label)
        self.brush_spin = QSpinBox(self)
        self.brush_spin.setRange(1, 500)
        self.brush_spin.setToolTip("Brush diameter in image pixels  ( [ / ] )")
        self.brush_spin.valueChanged.connect(self._label.set_mask_brush_size)
        brush_row.addWidget(self.brush_spin, 1)
        lay.addLayout(brush_row)

        # --- edit ---
        lay.addWidget(self._section("Edit"))
        grid = QGridLayout(); grid.setSpacing(4)
        edits = [
            ("Close Shape", "Close the spline being drawn and fill it (Enter)", self._label.commit_spline),
            ("Fill Holes", "Fill areas fully enclosed by the mask (F)", self._label.fill_mask_holes),
            ("Undo", "Undo the last point / shape / stroke (Ctrl+Z)", self._label.undo_mask),
            ("Clear", "Erase the whole mask", self._label.clear_mask),
        ]
        for i, (text, tip, slot) in enumerate(edits):
            act = QAction(text, self)
            act.setToolTip(tip)
            act.triggered.connect(slot)
            grid.addWidget(self._button(act), i // 2, i % 2)
        lay.addLayout(grid)

        # --- finish ---
        act_save = QAction("Save Mask", self)
        act_save.setToolTip("Save the mask as a new image and measure on it (Enter)")
        act_save.triggered.connect(self._label.finish_mask_paint)
        act_cancel = QAction("Cancel", self)
        act_cancel.setToolTip("Discard this edit (Esc)")
        act_cancel.triggered.connect(self._label.cancel_mask_paint)
        row = QHBoxLayout(); row.setSpacing(4)
        save_btn = self._button(act_save)
        save_btn.setStyleSheet(
            f"QToolButton {{ background:{ACCENT}; color:{ACCENT_TEXT}; font-weight:600; border-radius:4px; }}")
        row.addWidget(save_btn)
        row.addWidget(self._button(act_cancel))
        lay.addSpacing(4)
        lay.addLayout(row)

        hint = QLabel("Double-click a point = corner\nEsc drops the shape, Esc again cancels")
        hint.setStyleSheet(f"color:{DISABLED_TEXT}; font-size:11px;")
        lay.addWidget(hint)

        self._label.maskBrushChanged.connect(self.sync_from_label)
        self._label.maskModeChanged.connect(self._on_mode_changed)
        self.adjustSize()
        self.setFixedSize(self.sizeHint())

    # ---------------- helpers ----------------
    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(f"color:{DISABLED_TEXT}; font-size:10px; font-weight:600; "
                          f"border-bottom:1px solid {BORDER}; padding-bottom:2px;")
        return lbl

    def _button(self, act: QAction) -> QToolButton:
        btn = QToolButton(self)
        btn.setDefaultAction(act)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setMinimumHeight(26)
        btn.clicked.connect(self._refocus_image)
        return btn

    def _refocus_image(self):
        # hand keyboard focus back so [ ] E F Enter Esc keep working on the image
        self._label.window().activateWindow()
        self._label.setFocus()

    # ---------------- state ----------------
    def sync_from_label(self, *_):
        """Refresh the controls from the label's current brush/eraser/tool state."""
        self.brush_spin.blockSignals(True)
        self.brush_spin.setValue(self._label.mask_brush_size())
        self.brush_spin.blockSignals(False)
        self.act_erase.blockSignals(True)
        self.act_erase.setChecked(self._label.mask_erase())
        self.act_erase.blockSignals(False)
        spline = self._label.mask_tool() == "spline"
        self.act_spline.setChecked(spline)
        self.act_brush.setChecked(not spline)
        self.brush_spin.setEnabled(not spline)
        self.brush_label.setEnabled(not spline)

    def _on_mode_changed(self, active: bool):
        if active:
            self.sync_from_label()
            if not self._placed:
                # top-right corner of the image area, inset a little
                top_right = self._label.mapToGlobal(QPoint(self._label.width(), 0))
                self.move(top_right.x() - self.width() - 16, top_right.y() + 16)
                self._placed = True
            self.show()
            self.raise_()
        else:
            self.hide()

    def _set_tool(self, tool: str):
        self._label.set_mask_tool(tool)

    def _on_erase_toggled(self, on: bool):
        self._label.set_mask_erase(on)
        self._label.update()

    def closeEvent(self, e):
        # the window's close button cancels the edit (which hides the window)
        if self._label.is_mask_painting():
            self._label.cancel_mask_paint()
        super().closeEvent(e)
