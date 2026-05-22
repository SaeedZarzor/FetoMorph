"""Tabbed Preferences dialog for visualization settings.

Reads from and writes to the live :class:`VisualizationSettings` instance.
``OK`` / ``Apply`` write through and emit ``settingsChanged``; ``Cancel``
restores a snapshot taken when the dialog opened.
"""

from __future__ import annotations

from deps import *
from PySide6.QtWidgets import QCheckBox, QColorDialog

from managers.visualization_settings import VisualizationSettings, defaults


# ---------------------------------------------------------------------------
# Color conversions between settings tuples and QColor
# ---------------------------------------------------------------------------

def _bgr_to_qcolor(bgr) -> QColor:
    b, g, r = (int(x) for x in bgr)
    return QColor(r, g, b)


def _qcolor_to_bgr(c: QColor) -> tuple:
    return (c.blue(), c.green(), c.red())


def _rgbf_to_qcolor(rgbf) -> QColor:
    r, g, b = (float(x) for x in rgbf)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def _qcolor_to_rgbf(c: QColor) -> tuple:
    return (c.red() / 255.0, c.green() / 255.0, c.blue() / 255.0)


# ---------------------------------------------------------------------------
# Reusable color-swatch button
# ---------------------------------------------------------------------------

class ColorButton(QPushButton):
    """Small swatch button that opens :class:`QColorDialog` on click."""

    colorChanged = Signal(QColor)

    def __init__(self, initial: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(initial)
        self.setFixedSize(48, 22)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._pick)
        self._refresh()

    def color(self) -> QColor:
        return QColor(self._color)

    def setColor(self, c: QColor) -> None:
        self._color = QColor(c)
        self._refresh()

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Choose color")
        if chosen.isValid():
            self.setColor(chosen)
            self.colorChanged.emit(self._color)

    def _refresh(self) -> None:
        self.setStyleSheet(
            f"QPushButton {{ background:{self._color.name()};"
            f" border:1px solid #555; }}"
        )


# ---------------------------------------------------------------------------
# Preferences dialog
# ---------------------------------------------------------------------------

class PreferencesDialog(QDialog):
    """Tabbed visualization preferences."""

    def __init__(
        self,
        viz: VisualizationSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.viz = viz
        self._snapshot = viz.snapshot()

        # The parent (MainWindow) owns ``draw_hallmarks_on_image`` and
        # ``contour_mode`` via property delegates to SettingsManager. Snapshot
        # both so Cancel can roll them back if the user changes them and aborts.
        self._draw_hallmarks_snapshot = bool(
            getattr(parent, "draw_hallmarks_on_image", True)
        ) if parent is not None else True
        self._contour_mode_snapshot = str(
            getattr(parent, "contour_mode", "outer")
        ) if parent is not None else "outer"

        self._build_ui()
        self._populate_from(viz)

    # ----- UI construction -----

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        root.addWidget(self.tabs)
        self.tabs.addTab(self._build_text_sizes_tab(), "Text and Sizes")
        self.tabs.addTab(self._build_colors_tab(), "Colors")
        self.tabs.addTab(self._build_view_tab(), "View")

        bb = QDialogButtonBox(self)
        self.btn_defaults = bb.addButton("Restore Defaults", QDialogButtonBox.ResetRole)
        bb.addButton(QDialogButtonBox.Cancel)
        bb.addButton(QDialogButtonBox.Apply)
        bb.addButton(QDialogButtonBox.Ok)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self._on_cancel)
        bb.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        self.btn_defaults.clicked.connect(self._on_restore_defaults)
        root.addWidget(bb)

    # ----- Tab builders -----

    @staticmethod
    def _add_row(lay: QFormLayout, label: str, widget: QWidget, tip: str) -> None:
        """Add a form row whose label and control both carry the same tooltip."""
        widget.setToolTip(tip)
        lay.addRow(label, widget)
        lbl = lay.labelForField(widget)
        if lbl is not None:
            lbl.setToolTip(tip)

    def _build_text_sizes_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        # --- Text ---
        grp_text = QGroupBox("Text")
        lay = QFormLayout(grp_text)
        lay.setLabelAlignment(Qt.AlignRight)

        self.spin_text_scale = QDoubleSpinBox()
        self.spin_text_scale.setRange(0.25, 4.0)
        self.spin_text_scale.setSingleStep(0.05)
        self.spin_text_scale.setDecimals(2)
        self._add_row(lay, "Text size multiplier:", self.spin_text_scale,
            "Overall scale for text drawn on images (hallmark values, sulcus depth labels, scalebar number).")

        self.spin_sulcus_label_scale = QDoubleSpinBox()
        self.spin_sulcus_label_scale.setRange(0.25, 2.0)
        self.spin_sulcus_label_scale.setSingleStep(0.05)
        self.spin_sulcus_label_scale.setDecimals(2)
        self._add_row(lay, "Sulcus label size multiplier:", self.spin_sulcus_label_scale,
            "Extra factor applied on top of Text size for the depth values printed next to sulcus markers.")

        self.btn_hallmark_text_color = ColorButton(QColor("white"))
        self._add_row(lay, "Hallmark text color:", self.btn_hallmark_text_color,
            "Color of the Area / Perimeter / lGI text printed in the hallmark box.")

        self.btn_scalebar_text_color = ColorButton(QColor("black"))
        self._add_row(lay, "Scalebar text color:", self.btn_scalebar_text_color,
            "Color of the scalebar number (e.g. '20 mm') and bar fill on NIfTI / measurement images.")

        outer.addWidget(grp_text)

        # --- Sizes ---
        grp_sizes = QGroupBox("Sizes")
        lay = QFormLayout(grp_sizes)
        lay.setLabelAlignment(Qt.AlignRight)

        self.spin_contour_thickness = QDoubleSpinBox()
        self.spin_contour_thickness.setRange(0.25, 4.0)
        self.spin_contour_thickness.setSingleStep(0.05)
        self.spin_contour_thickness.setDecimals(2)
        self._add_row(lay, "Contour line thickness ×:", self.spin_contour_thickness,
            "Multiplier on the auto-sized stroke width used for inner and outer brain contours.")

        self.spin_marker_radius = QDoubleSpinBox()
        self.spin_marker_radius.setRange(0.25, 4.0)
        self.spin_marker_radius.setSingleStep(0.05)
        self.spin_marker_radius.setDecimals(2)
        self._add_row(lay, "Marker radius ×:", self.spin_marker_radius,
            "Multiplier on the auto-sized radius of the circular sulcus markers.")

        self.spin_scalebar_thickness = QDoubleSpinBox()
        self.spin_scalebar_thickness.setRange(0.25, 4.0)
        self.spin_scalebar_thickness.setSingleStep(0.05)
        self.spin_scalebar_thickness.setDecimals(2)
        self._add_row(lay, "Scalebar thickness ×:", self.spin_scalebar_thickness,
            "Multiplier on the scalebar bar height (auto-derived from image size).")

        outer.addWidget(grp_sizes)
        outer.addStretch(1)
        return page

    def _build_colors_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)

        # --- Contours & measurements ---
        grp_contours = QGroupBox("Contours && measurements")
        lay = QFormLayout(grp_contours)
        lay.setLabelAlignment(Qt.AlignRight)

        self.btn_contour_inner = ColorButton(QColor("red"))
        self._add_row(lay, "Inner contour:", self.btn_contour_inner,
            "Color of the inner brain outline drawn on annotated images.")

        self.btn_contour_outer = ColorButton(QColor("green"))
        self._add_row(lay, "Outer contour:", self.btn_contour_outer,
            "Color of the convex-hull outline drawn outside the inner contour.")

        self.btn_measurement_line = ColorButton(QColor("blue"))
        self._add_row(lay, "Measurement line:", self.btn_measurement_line,
            "Color of measurement strokes (convexity-defect rays).")

        outer.addWidget(grp_contours)

        # --- Sulcus markers ---
        grp_sulci = QGroupBox("Sulcus markers")
        lay = QFormLayout(grp_sulci)
        lay.setLabelAlignment(Qt.AlignRight)

        self.btn_sulcus_primary = ColorButton(QColor("blue"))
        self._add_row(lay, "Primary:", self.btn_sulcus_primary,
            "Marker color for primary (deepest fraction range) sulci.")

        self.btn_sulcus_secondary = ColorButton(QColor("gold"))
        self._add_row(lay, "Secondary:", self.btn_sulcus_secondary,
            "Marker color for secondary sulci.")

        self.btn_sulcus_tertiary = ColorButton(QColor("cyan"))
        self._add_row(lay, "Tertiary:", self.btn_sulcus_tertiary,
            "Marker color for tertiary sulci.")

        self.btn_sulcus_unclassified = ColorButton(QColor("lightgray"))
        self._add_row(lay, "Unclassified:", self.btn_sulcus_unclassified,
            "Marker color for depths inside the filter window but outside every classified range.")

        outer.addWidget(grp_sulci)

        # --- 3-D viewer ---
        grp_vtk = QGroupBox("3-D viewer")
        lay = QFormLayout(grp_vtk)
        lay.setLabelAlignment(Qt.AlignRight)

        self.btn_vtk_bg = ColorButton(QColor("black"))
        self._add_row(lay, "Background:", self.btn_vtk_bg,
            "Background color of the 3-D viewer canvas.")

        self.btn_vtk_surface = ColorButton(QColor("lightblue"))
        self._add_row(lay, "Default surface:", self.btn_vtk_surface,
            "Default color applied to loaded VTK / STL surface meshes when no per-vertex coloring is set.")

        outer.addWidget(grp_vtk)
        outer.addStretch(1)
        return page

    def _build_view_tab(self) -> QWidget:
        page = QWidget()
        lay = QFormLayout(page)
        lay.setLabelAlignment(Qt.AlignRight)

        self.chk_draw_hallmarks = QCheckBox()
        self._add_row(lay, "Draw hallmarks on image:", self.chk_draw_hallmarks,
            "When on, the Area / Perimeter / lGI text block is drawn in the corner of annotated images.")

        self.cmb_contour_mode = QComboBox()
        # itemData carries the canonical mode string; itemText is the label
        self.cmb_contour_mode.addItem("Outer contours only", "outer")
        self.cmb_contour_mode.addItem("Subtract internal contours", "subtract")
        self.cmb_contour_mode.addItem("Internal contours only", "internal_only")
        self._add_row(lay, "Contour accounting:", self.cmb_contour_mode,
            "How VTK Area / Volume / Compactness use nested contours (e.g. ventricles). "
            "`Outer only` measures the brain outline (default). `Subtract internal` subtracts "
            "the area of qualifying internal contours from the brain area. `Internal only` "
            "measures only the internal contours. Internal contours must still pass the "
            "filtered-area threshold.")

        self.chk_label_overlay = QCheckBox()
        self._add_row(lay, "Show label overlay:", self.chk_label_overlay,
            "When on, NIfTI region labels are blended as colored overlays onto 2-D slices.")

        self.chk_zoom_controls = QCheckBox()
        self._add_row(lay, "Show zoom controls:", self.chk_zoom_controls,
            "When on, the image zoom toolbar is visible above the viewer.")
        return page

    # ----- (de)serialization between widgets and settings -----

    def _populate_from(self, vs) -> None:
        self.spin_text_scale.setValue(float(vs.text_scale_multiplier))
        self.btn_hallmark_text_color.setColor(_bgr_to_qcolor(vs.hallmark_text_color_bgr))
        self.spin_sulcus_label_scale.setValue(float(vs.sulcus_label_scale_multiplier))
        self.btn_scalebar_text_color.setColor(_bgr_to_qcolor(vs.scalebar_text_color_bgr))

        self.btn_contour_inner.setColor(_bgr_to_qcolor(vs.contour_inner_color_bgr))
        self.btn_contour_outer.setColor(_bgr_to_qcolor(vs.contour_outer_color_bgr))
        self.btn_measurement_line.setColor(_bgr_to_qcolor(vs.measurement_line_color_bgr))
        self.btn_sulcus_primary.setColor(_bgr_to_qcolor(vs.sulcus_primary_color_bgr))
        self.btn_sulcus_secondary.setColor(_bgr_to_qcolor(vs.sulcus_secondary_color_bgr))
        self.btn_sulcus_tertiary.setColor(_bgr_to_qcolor(vs.sulcus_tertiary_color_bgr))
        self.btn_sulcus_unclassified.setColor(_bgr_to_qcolor(vs.sulcus_unclassified_color_bgr))

        self.spin_contour_thickness.setValue(float(vs.contour_thickness_multiplier))
        self.spin_marker_radius.setValue(float(vs.marker_radius_multiplier))
        self.spin_scalebar_thickness.setValue(float(vs.scalebar_thickness_multiplier))

        self.chk_label_overlay.setChecked(bool(vs.show_label_overlay))
        self.chk_zoom_controls.setChecked(bool(vs.show_zoom_controls))
        self.btn_vtk_bg.setColor(_rgbf_to_qcolor(vs.vtk_background_rgbf))
        self.btn_vtk_surface.setColor(_rgbf_to_qcolor(vs.vtk_surface_rgbf))

        # Mirror the live ``draw_hallmarks_on_image`` from the main window.
        # On Restore Defaults (vs is a VizDefaults instance) fall back to the
        # factory defaults: draw_hallmarks=on, contour_mode="outer".
        if not isinstance(vs, type(defaults())):
            parent = self.parent()
            self.chk_draw_hallmarks.setChecked(
                bool(getattr(parent, "draw_hallmarks_on_image", True))
            )
            mode = str(getattr(parent, "contour_mode", "outer"))
        else:
            self.chk_draw_hallmarks.setChecked(True)
            mode = "outer"
        idx = self.cmb_contour_mode.findData(mode)
        if idx < 0:
            idx = 0  # fall back to first item ("outer")
        self.cmb_contour_mode.setCurrentIndex(idx)

    def _collect(self) -> dict:
        return {
            "text_scale_multiplier": float(self.spin_text_scale.value()),
            "hallmark_text_color_bgr": _qcolor_to_bgr(self.btn_hallmark_text_color.color()),
            "sulcus_label_scale_multiplier": float(self.spin_sulcus_label_scale.value()),
            "scalebar_text_color_bgr": _qcolor_to_bgr(self.btn_scalebar_text_color.color()),
            "contour_inner_color_bgr": _qcolor_to_bgr(self.btn_contour_inner.color()),
            "contour_outer_color_bgr": _qcolor_to_bgr(self.btn_contour_outer.color()),
            "measurement_line_color_bgr": _qcolor_to_bgr(self.btn_measurement_line.color()),
            "sulcus_primary_color_bgr": _qcolor_to_bgr(self.btn_sulcus_primary.color()),
            "sulcus_secondary_color_bgr": _qcolor_to_bgr(self.btn_sulcus_secondary.color()),
            "sulcus_tertiary_color_bgr": _qcolor_to_bgr(self.btn_sulcus_tertiary.color()),
            "sulcus_unclassified_color_bgr": _qcolor_to_bgr(self.btn_sulcus_unclassified.color()),
            "contour_thickness_multiplier": float(self.spin_contour_thickness.value()),
            "marker_radius_multiplier": float(self.spin_marker_radius.value()),
            "scalebar_thickness_multiplier": float(self.spin_scalebar_thickness.value()),
            "show_label_overlay": bool(self.chk_label_overlay.isChecked()),
            "show_zoom_controls": bool(self.chk_zoom_controls.isChecked()),
            "vtk_background_rgbf": _qcolor_to_rgbf(self.btn_vtk_bg.color()),
            "vtk_surface_rgbf": _qcolor_to_rgbf(self.btn_vtk_surface.color()),
        }

    # ----- button slots -----

    @staticmethod
    def _sync_contour_mode_menu(parent, mode: str) -> None:
        """Tick the matching radio in the Adjustments contour-mode submenu."""
        attr = {
            "outer":         "act_contour_outer",
            "subtract":      "act_contour_subtract",
            "internal_only": "act_contour_internal_only",
        }.get(mode)
        if attr is None:
            return
        group = getattr(parent, "contour_mode_group", None)
        act = getattr(parent, attr, None)
        if group is not None:
            group.blockSignals(True)
        if act is not None:
            act.setChecked(True)
        if group is not None:
            group.blockSignals(False)

    def _on_apply(self) -> None:
        self.viz.apply(self._collect())
        self.viz.save()
        parent = self.parent()
        if parent is not None and hasattr(parent, "draw_hallmarks_on_image"):
            parent.draw_hallmarks_on_image = bool(self.chk_draw_hallmarks.isChecked())
        if parent is not None and hasattr(parent, "contour_mode"):
            new_mode = str(self.cmb_contour_mode.currentData() or "outer")
            parent.contour_mode = new_mode
            self._sync_contour_mode_menu(parent, new_mode)

    def _on_ok(self) -> None:
        self._on_apply()
        self.accept()

    def _on_cancel(self) -> None:
        self.viz.restore(self._snapshot)
        parent = self.parent()
        if parent is not None and hasattr(parent, "draw_hallmarks_on_image"):
            parent.draw_hallmarks_on_image = self._draw_hallmarks_snapshot
        if parent is not None and hasattr(parent, "contour_mode"):
            parent.contour_mode = self._contour_mode_snapshot
            self._sync_contour_mode_menu(parent, self._contour_mode_snapshot)
        self.reject()

    def _on_restore_defaults(self) -> None:
        self._populate_from(defaults())
