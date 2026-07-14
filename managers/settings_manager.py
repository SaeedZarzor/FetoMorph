"""Settings manager -- calibration, units, and processing parameters.

Extracted from MainWindow to keep settings-related state and dialogs
in a single, cohesive module.
"""

from __future__ import annotations

from deps import *
from typing import TYPE_CHECKING
from constants import (
    DEFAULT_PIXEL_SIZE, DEFAULT_CNT_THRESHOLD, DEFAULT_SULCUS_DEPTH_THRESHOLD,
    DEFAULT_KERNEL_SIZE_MM, DEFAULT_SLICE_THICKNESS,
    DEFAULT_SCALEBAR_MM,
    DEFAULT_CAVITY_CORRECTION_ENABLED, DEFAULT_CAVITY_AREA_THRESHOLD_MM2,
    DEFAULT_FILL_CROSS_SECTION,
    DEFAULT_PERIMETER_METHOD, DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER,
    DEFAULT_CONTOUR_SIMPLIFY_EPSILON,
)
from helpers.helpers import get_max_slice_thickness
from widgets.contour_threshold import ContourThresholdDialog
from widgets.kernel_size import KernelSizeDialog
from widgets.scalebar_set_scale import ScalebarSetScaleDialog
from widgets.slice_thickness import SliceThicknessDialog
from widgets.unit_scale import UnitScaleDialog
from widgets.geometry_dialog import GeometryDialogWithAspect
from widgets.cavity_options import CavityOptionsDialog
from widgets.perimeter_options import PerimeterOptionsDialog

if TYPE_CHECKING:
    from FetoMorph import MainWindow

logger = logging.getLogger("fetomorph")


class SettingsManager:
    """Owns calibration state and dialogs for units, pixel scale, kernel, etc."""

    def __init__(self, mw: MainWindow) -> None:
        self.mw = mw

        # ---- state ----
        self.units_length: str | None = None
        self.pixel_size_default: float = DEFAULT_PIXEL_SIZE
        self.pixel_size: float = self.pixel_size_default
        self.image_scales: dict[str, float] = {}
        self.image_scale_from_scalebar: dict[str, bool] = {}
        self.draw_hallmarks_on_image: bool = True
        self.cnt_threshold: float = DEFAULT_CNT_THRESHOLD
        self.kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM
        self.perimeter_method: str = DEFAULT_PERIMETER_METHOD
        self.simplify_contours_for_perimeter: bool = DEFAULT_SIMPLIFY_CONTOURS_FOR_PERIMETER
        self.contour_simplify_epsilon: float = DEFAULT_CONTOUR_SIMPLIFY_EPSILON
        # Contour-accounting mode for VTK area / volume / compactness:
        #   "outer"         — measure outer brain contour only (default)
        #   "subtract"      — outer minus area of internal contours (holes)
        #   "internal_only" — measure only the internal contour areas
        self.contour_mode: str = "outer"
        self.slice_thickness: float = DEFAULT_SLICE_THICKNESS
        # Surface-connected cavity correction (3D volume / surface area). When on,
        # cavities that open onto the outer surface have their area removed from
        # the volume integral and their wall added to the surface area; enclosed
        # voids are left as solid. Only cavities above the area threshold count.
        self.cavity_correction_enabled: bool = DEFAULT_CAVITY_CORRECTION_ENABLED
        self.cavity_area_threshold_mm2: float = DEFAULT_CAVITY_AREA_THRESHOLD_MM2
        # Render thin surface-mesh section curves as filled solid faces (like VTK).
        self.fill_cross_section: bool = DEFAULT_FILL_CROSS_SECTION
        self.mm_per_px_bar: float = 0
        self.bar_mm: float = DEFAULT_SCALEBAR_MM
        self.custom_label: str | None = None
        self.physical_dim: tuple[int, int, int] = (0, 0, 0)
        self.slice_direction: Literal["X", "Y", "Z"] = "Y"
        self._flat_axis: int | None = None

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

    def kernel_size_px(self, pixel_size_mm: float | None = None) -> int:
        """Return the current morphology kernel diameter in pixels."""
        try:
            px_mm = float(pixel_size_mm if pixel_size_mm is not None else self.pixel_size)
        except (TypeError, ValueError):
            px_mm = self.pixel_size_default
        px_mm = max(px_mm, 1e-9)
        return max(3, int(round(float(self.kernel_size_mm) / px_mm)))

    @property
    def kernel_size(self) -> int:
        """Legacy pixel accessor for code that has not moved to mm yet."""
        return self.kernel_size_px(self.pixel_size)

    @kernel_size.setter
    def kernel_size(self, value: float) -> None:
        self.kernel_size_mm = float(value)

    def set_custom_label(self) -> None:
        """Prompt the user to enter a custom label string for the current file."""
        val, ok = QInputDialog.getText(
            self.mw,
            "Set Custom Label",
            "Enter label:",
            QLineEdit.Normal,
        )
        self.custom_label = val if ok else None

    def ensure_units(self) -> str:
        """Ensure a length unit string is set, prompting once per session if needed.

        Returns:
            The active length unit string (e.g. "mm", "um", "cm").
        """
        if self.units_length:
            return self.units_length

        val, ok = QInputDialog.getText(
            self.mw,
            "Set Units",
            "Length unit (e.g., mm, µm, cm):",
            text="mm",
        )
        if not ok or not val.strip():
            QMessageBox.information(
                self.mw,
                "No Input",
                "You closed the window without entering any values. Default unit will be used.",
            )
            val = "mm"
        self.units_length = val.strip()
        return self.units_length

    def ensure_calibrated(self) -> tuple[str, float] | None:
        """Ensure units and pixel scale are set for the current file.

        Returns:
            ``(unit, px_size)`` on success, or ``None`` if the user cancelled.
        """
        while True:
            if not self.units_length or self.mw.current_path not in self.image_scales:
                ok = self.set_image_scale()
                if ok:
                    break
                else:
                    return None
            else:
                break
        u = self.ensure_units()
        px_size = self.image_scales.get(self.mw.current_path, self.pixel_size)
        return (u, px_size)

    def load_mesh_and_ask_geometry(self) -> bool:
        """Read the current VTK mesh bounds and prompt the user for physical dimensions.

        Returns:
            True if the user accepted the dialog, False otherwise.
        """
        if self.mw.is_vtk:
            mesh = pv.read(str(self.mw.current_path))
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            if all(abs(v) > 1e-9 for v in self.physical_dim):
                Lx0, Ly0, Lz0 = self.physical_dim
            else:
                Lx0, Ly0, Lz0 = xmax - xmin, ymax - ymin, zmax - zmin
                Lx0, Ly0, Lz0 = max(Lx0, 1e-9), max(Ly0, 1e-9), max(Lz0, 1e-9)

            unit = self.ensure_units()
            slice_dir = self.slice_direction

            dlg = GeometryDialogWithAspect(
                self.mw, mesh=mesh, Lx=Lx0, Ly=Ly0, Lz=Lz0,
                unit=unit, slice_dir=slice_dir, flat_axis=self._flat_axis,
            )
            if dlg.exec() == QDialog.Accepted:
                (Lx, Ly, Lz), slice_dir, unit = dlg.values()
            else:
                QMessageBox.information(
                    self.mw,
                    "No Input",
                    "You closed the window without entering any values. Default dimensions will be used.",
                )
                (Lx, Ly, Lz) = (Lx0, Ly0, Lz0)

            self.physical_dim = (Lx, Ly, Lz)
            self.slice_direction = slice_dir
            self.units_length = unit

            # Dimensions changed → keep slice thickness (and slice count) valid.
            self.clamp_slice_thickness_to_mesh()

            print(f"[Geometry] from mesh={self.mw.current_path}")
            print(f"  bounds=({xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax})")
            print(f"  physical_dim={self.physical_dim} {unit}, slice_direction={slice_dir}")

            return True
        else:
            return False

    def set_image_scale(self) -> bool:
        """Set the length unit and pixel size for the current file via a single dialog.

        Returns:
            True if a valid scale was set, False if cancelled.
        """
        if not self.mw.current_path:
            QMessageBox.information(self.mw, "Set Units & Pixel Size", "Load a file first.")
            return False

        unit_init = self.units_length or "mm"
        px_init = float(self.image_scales.get(self.mw.current_path, self.pixel_size))

        dlg = UnitScaleDialog(self.mw, unit_init=unit_init, pixel_size_init=px_init)

        if dlg.exec() != QDialog.Accepted:
            if dlg._get_status():
                ok = self.set_scale_from_scalebar()
                return ok
            else:
                return False

        unit, scale = dlg.values()
        if not (scale > 0):
            QMessageBox.warning(self.mw, "Invalid value", "Pixel size must be a positive number.")
            return False

        self.units_length = unit
        self.image_scales[self.mw.current_path] = scale
        self.image_scale_from_scalebar[self.mw.current_path] = False
        self.pixel_size = scale

        print(f"[Units] {unit}  |  [Scale] {scale} {unit}/pixel  —  {os.path.basename(self.mw.current_path)}")
        return True

    def set_scale_from_scalebar(self) -> bool:
        """Activate line measurement on the image and compute scale from the drawn line.

        Returns:
            True if the scale was successfully set, False otherwise.
        """
        if self.mw.current_kind != "image":
            QMessageBox.information(self.mw, "Set Scale", "Open a 2D image to set scale from a scalebar.")
            return False
        if self.mw.image_label is None or self.mw.image_label._pix.isNull():
            QMessageBox.information(self.mw, "Set Scale", "No image visible.")
            return False
        print("[Scale] Draw a line over the scalebar: click, drag, release.")

        loop = QEventLoop(self.mw)
        result = {"ok": False}

        def _cb(px_len):
            try:
                result["ok"] = self._finish_scalebar_scale(px_len)
            finally:
                loop.quit()

        self.mw.image_label.start_scalebar_measure(_cb)
        loop.exec()
        return result["ok"]

    def _finish_scalebar_scale(self, pixel_length: float) -> bool:
        """Called after the user drags a line; asks for real length & unit, computes px/unit."""
        try:
            unit_init = self.units_length or "mm"
            dlg = ScalebarSetScaleDialog(pixel_length, unit_init=unit_init, parent=self.mw)
            if dlg.exec() != QDialog.Accepted:
                print("[Scale] Canceled.")
                return False

            px_per_unit, unit = dlg.values()
            if px_per_unit <= 0:
                QMessageBox.warning(self.mw, "Set Scale", "Scale must be positive.")
                return False

            self.units_length = unit
            mm_per_px = 1.0 / px_per_unit
            self.pixel_size = mm_per_px
            self.image_scales[self.mw.current_path] = float(mm_per_px)
            self.image_scale_from_scalebar[self.mw.current_path] = True

            label_text = self.mw.get_label_for_cropped_path(self.mw.last_annotated_path)
            print(
                f"[Scale] {pixel_length:.2f} px = {px_per_unit:.6f} px/{unit}  "
                f"→ pixel size {mm_per_px:.6f} {unit}/pixel for {os.path.basename(self.mw.current_path)}"
            )
            return True

        except Exception as ex:
            logger.error("Set Scale failed: %s", ex)
            QMessageBox.critical(self.mw, "Set Scale Failed", f"{type(ex).__name__}: {ex}")
            return False

    def mesh_shortest_dimension(self) -> float | None:
        """Return the mesh's shortest physical dimension (mm).

        Uses the user-set ``physical_dim`` when available (VTK meshes scaled to
        real-world size) so the value tracks any dimension changes; otherwise
        falls back to the raw mesh bounding box (STL / un-scaled VTK).
        """
        pdim = self.physical_dim
        # physical_dim only applies to VTK meshes scaled to real-world size; for
        # STL (and un-scaled VTK) use the raw mesh bounding box.
        if getattr(self.mw, "is_vtk", False) and pdim and all(abs(float(v)) > 1e-9 for v in pdim):
            return float(min(abs(float(v)) for v in pdim))
        return get_max_slice_thickness(self.mw.current_path)

    def clamp_slice_thickness_to_mesh(self) -> None:
        """Clamp the stored slice thickness to the current mesh (≤ half shortest).

        Call this whenever the mesh or its physical dimensions change so the
        thickness — and therefore the slice count — stays valid for the new
        geometry.
        """
        shortest = self.mesh_shortest_dimension()
        if not shortest or shortest <= 0:
            return
        max_thickness = shortest / 2
        if self.slice_thickness > max_thickness:
            self.slice_thickness = max_thickness
            unit = self.units_length or "mm"
            print(f"[Slice Thickness] Clamped to {self.slice_thickness:.4g} {unit} "
                  f"(max for shortest dim {shortest:.3g} {unit})")

    def set_slice_thickness_dialog(self) -> None:
        """Open a dialog to set the inter-slice distance for 3-D measurements."""
        shortest = self.mesh_shortest_dimension()
        if not shortest or shortest <= 0:
            QMessageBox.information(
                self.mw, "Slice thickness",
                "Could not determine the mesh dimensions for this file.")
            return
        unit = self.units_length or "mm"
        dlg = SliceThicknessDialog(
            self.mw,
            initial=min(self.slice_thickness, shortest / 2),
            maximum=(shortest / 2),
            reference_length=shortest,
            unit=unit,
        )
        if dlg.exec() == QDialog.Accepted:
            k = dlg.value()
            self.slice_thickness = k
            print(f"[Slice Thickness] Set Slice Thickness to {k} {unit} "
                  f"(≈ {dlg.number_of_slices()} slices over shortest dim {shortest:.3g} {unit})")

    def set_kernel_dialog(self) -> None:
        """Open dialog to set morphology kernel size in millimetres."""
        dlg = KernelSizeDialog(self.mw, initial=self.kernel_size_mm)
        if dlg.exec() == QDialog.Accepted:
            k = dlg.value()
            self.kernel_size_mm = k
            print(f"[Kernel] Set morphology kernel diameter to {k:g} mm")

    def set_cnt_threshold_dialog(self) -> None:
        """Open a dialog to set the minimum contour area threshold in mm²."""
        dlg = ContourThresholdDialog(self.mw, initial=self.cnt_threshold)
        if dlg.exec() == QDialog.Accepted:
            val = dlg.value()
            self.cnt_threshold = max(0.0, float(val))
            print(f"[Threshold] Contour area threshold set to {self.cnt_threshold:.2f} mm²")

    def set_sulcus_depth_threshold_dialog(self) -> None:
        """Open a dialog to set the minimum sulcus depth (mm) kept when measuring.

        Stored on the shared :class:`VisualizationSettings` singleton so every
        measurement function honours it via ``helpers.sulcus_depth_min()``.
        """
        from PySide6.QtWidgets import QInputDialog
        from managers.visualization_settings import get_active
        vs = get_active()
        current = float(getattr(vs, "sulcus_depth_threshold",
                                DEFAULT_SULCUS_DEPTH_THRESHOLD))
        val, ok = QInputDialog.getDouble(
            self.mw, "Sulcus Depth Threshold",
            "Minimum sulcus depth to keep (mm):", current, 0.0, 100.0, 3)
        if ok:
            vs.apply({"sulcus_depth_threshold": max(0.0, float(val))})
            vs.save()
            print(f"[Threshold] Sulcus depth threshold set to {float(val):.3f} mm")

    def set_slice_kind_override_dialog(self) -> None:
        """Manually override the automatic slice-kind classification.

        "Auto (classifier)" restores the ONNX classifier; any explicit kind is
        forced for every measurement via ``classify_slice_kind``'s override
        check. Stored on the shared :class:`VisualizationSettings` singleton.
        """
        from PySide6.QtWidgets import QInputDialog
        from managers.visualization_settings import get_active
        vs = get_active()
        labels = [
            "Auto (classifier)",
            "Sagittal",
            "Coronal",
            "Axial",
            "Not full slice (cropped)",
        ]
        values = ["auto", "sagittal", "coronal", "axial", "not_full_slice"]
        current = str(getattr(vs, "slice_kind_override", "auto")).strip().lower()
        idx = values.index(current) if current in values else 0
        choice, ok = QInputDialog.getItem(
            self.mw, "Slice Kind",
            "Force the slice kind used by all measurements:",
            labels, idx, False)
        if ok:
            selected = values[labels.index(choice)]
            vs.apply({"slice_kind_override": selected})
            vs.save()
            print(f"[Slice Kind] Override set to {selected}")

    def set_cavity_options_dialog(self) -> None:
        """Open a dialog to enable/disable the surface-connected cavity correction
        and set its area threshold (both in one widget)."""
        dlg = CavityOptionsDialog(
            self.mw,
            enabled=self.cavity_correction_enabled,
            threshold_mm2=self.cavity_area_threshold_mm2,
        )
        if dlg.exec() == QDialog.Accepted:
            self.cavity_correction_enabled = dlg.enabled()
            self.cavity_area_threshold_mm2 = max(0.0, dlg.threshold())
            print(f"[Cavity] correction "
                  f"{'ON' if self.cavity_correction_enabled else 'OFF'}; "
                  f"area threshold {self.cavity_area_threshold_mm2:.2f} mm²")

    def set_perimeter_options_dialog(self) -> None:
        """Open dialog for binary-mask perimeter measurement options."""
        dlg = PerimeterOptionsDialog(
            self.mw,
            method=self.perimeter_method,
            simplify=self.simplify_contours_for_perimeter,
            epsilon=self.contour_simplify_epsilon,
        )
        if dlg.exec() == QDialog.Accepted:
            method = dlg.method()
            if method not in {"arc_length", "crofton"}:
                method = DEFAULT_PERIMETER_METHOD
            self.perimeter_method = method
            self.simplify_contours_for_perimeter = dlg.simplify()
            self.contour_simplify_epsilon = max(0.0, dlg.epsilon())
            print(
                "[Perimeter] method="
                f"{self.perimeter_method}; simplify="
                f"{'on' if self.simplify_contours_for_perimeter else 'off'}; "
                f"epsilon={self.contour_simplify_epsilon:.2f} px"
            )
