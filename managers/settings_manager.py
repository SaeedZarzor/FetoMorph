"""Settings manager -- calibration, units, and processing parameters.

Extracted from MainWindow to keep settings-related state and dialogs
in a single, cohesive module.
"""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Literal

from deps import (
    QDialog, QEventLoop, QInputDialog, QLineEdit, QMessageBox,
)
from constants import (
    DEFAULT_PIXEL_SIZE, DEFAULT_CNT_THRESHOLD,
    DEFAULT_KERNEL_SIZE, DEFAULT_SLICE_THICKNESS,
    DEFAULT_SCALEBAR_MM,
)
from helpers.Helpers import get_max_slice_thickness
from widgets.Contour_threshold import ContourThresholdDialog
from widgets.Kernel_size import KernelSizeDialog
from widgets.Scalebar_set_scale import ScalebarSetScaleDialog
from widgets.Slice_thickness import SliceThicknessDialog
from widgets.Unit_scale import UnitScaleDialog
from widgets.GeometryDialog import GeometryDialogWithAspect

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
        self.kernel_size: int = DEFAULT_KERNEL_SIZE
        self.slice_thickness: float = DEFAULT_SLICE_THICKNESS
        self.mm_per_px_bar: float = 0
        self.bar_mm: float = DEFAULT_SCALEBAR_MM
        self.custom_label: str | None = None
        self.physical_dim: tuple[int, int, int] = (0, 0, 0)
        self.slice_direction: Literal["X", "Y", "Z"] = "Y"
        self._flat_axis: int | None = None

    # ------------------------------------------------------------------
    # Dialogs
    # ------------------------------------------------------------------

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
            import pyvista as pv
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

    def set_slice_thickness_dialog(self) -> None:
        """Open a dialog to set the inter-slice distance for 3-D measurements."""
        dlg = SliceThicknessDialog(
            self.mw,
            initial=self.slice_thickness,
            maximum=(get_max_slice_thickness(self.mw.current_path) / 2),
        )
        if dlg.exec() == QDialog.Accepted:
            k = dlg.value()
            self.slice_thickness = k
            print(f"[Slice Thickness] Set Slice Thickness to {k}")

    def set_kernel_dialog(self) -> None:
        """Open dialog to set morphology kernel size (odd)."""
        dlg = KernelSizeDialog(self.mw, initial=self.kernel_size)
        if dlg.exec() == QDialog.Accepted:
            k = dlg.value()
            self.kernel_size = k
            print(f"[Kernel] Set morphology kernel size to {k}")

    def set_cnt_threshold_dialog(self) -> None:
        """Open a dialog to set the minimum contour area threshold in pixels."""
        dlg = ContourThresholdDialog(self.mw, initial=self.cnt_threshold)
        if dlg.exec() == QDialog.Accepted:
            val = dlg.value()
            self.cnt_threshold = max(0.0, float(val))
            print(f"[Threshold] Contour area threshold set to {self.cnt_threshold:.0f} px")
