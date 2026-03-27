"""View manager -- display, slice navigation, and pixmap carousel.

Extracted from MainWindow to consolidate all view/display logic.
"""

from __future__ import annotations

import os
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from deps import (
    QColor, QIcon, QImage, QPainter, QPen, QPixmap, QTextCursor,
    QWidget, Qt,
    vtkNIFTIImageReader,
)
from helpers.Helpers import add_scalebar

if TYPE_CHECKING:
    from FetoMorph import MainWindow

logger = logging.getLogger("fetomorph")


class ViewManager:
    """Owns display state and slice/pixmap navigation logic."""

    def __init__(self, mw: MainWindow) -> None:
        self.mw = mw

        # ---- state ----
        self._pm_index: int = 0
        self._pms: list[QPixmap] = []

        self.slice_nav_mode: str | None = None       # None | "nifti" | "png" | "vtk"
        self.slice_nav_items: list[str] = []
        self.slice_nav_index_map: list[int] = []

        self.nifti_axis: int = 1          # 0=sagittal, 1=coronal, 2=axial
        self.nifti_depth: int = 0
        self.label_overlay_enabled: bool = True
        self.nifti_selected_regions_default: set[int] = set()
        self.nifti_selected_regions: set[int] = set()
        self.nifti_label_lut: dict[int, QColor] = {}
        self.labels_available: set[int] = set()

    # ------------------------------------------------------------------
    # Pixmap helpers
    # ------------------------------------------------------------------

    @staticmethod
    def np_bgr_to_qpixmap(arr: np.ndarray) -> QPixmap:
        """Convert a NumPy BGR/BGRA/grayscale image to a QPixmap."""
        if arr.dtype != np.uint8:
            raise ValueError("Expected uint8 array.")
        if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            raise ValueError("Expected HxWx1/3/4 array.")

        h, w = arr.shape[:2]

        if arr.shape[2] == 3:  # BGR -> RGB
            rgb = arr[:, :, ::-1].copy(order="C")
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            qimg._np_ref = rgb
        elif arr.shape[2] == 4:  # BGRA -> RGBA
            rgba = arr[:, :, [2, 1, 0, 3]].copy(order="C")
            qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
            qimg._np_ref = rgba
        else:  # 1 channel grayscale
            gray = arr[:, :, 0].copy(order="C")
            qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
            qimg._np_ref = gray

        return QPixmap.fromImage(qimg)

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def show_widget(self, w: QWidget) -> None:
        """Show only *w* (image_label or vtk_view) and update ``_active_view``."""
        mw = self.mw
        mw.image_label.setVisible(False)
        mw.vtk_view.setVisible(False)
        w.setVisible(True)
        mw._active_view = "image" if w is mw.image_label else "vtk"

    def append_progress(self, text: str) -> None:
        """Append *text* to the progress console and scroll to the bottom."""
        mw = self.mw
        mw.progress_edit.moveCursor(QTextCursor.End)
        mw.progress_edit.insertPlainText(text)
        mw.progress_edit.moveCursor(QTextCursor.End)

    # ------------------------------------------------------------------
    # Pixmap carousel
    # ------------------------------------------------------------------

    def next_pm(self) -> None:
        """Cycle forward through the pixmap carousel (Ctrl+M)."""
        if not self._pms:
            return
        self._pm_index = (self._pm_index + 1) % len(self._pms)
        self.mw.image_label.setImage(self._pms[self._pm_index])

    def prev_pm(self) -> None:
        """Cycle backward through the pixmap carousel (Ctrl+Shift+M)."""
        if not self._pms:
            return
        self._pm_index = (self._pm_index - 1) % len(self._pms)
        self.mw.image_label.setImage(self._pms[self._pm_index])

    # ------------------------------------------------------------------
    # Slice controls
    # ------------------------------------------------------------------

    def sync_slice_controls(self) -> None:
        """Synchronise the slice slider range with the VTK viewer's current volume."""
        mw = self.mw
        if mw.vtk_view.has_slice():
            lo, hi = mw.vtk_view.slice_range()
            mw.slice_slider.blockSignals(True)
            mw.slice_slider.setMinimum(lo)
            mw.slice_slider.setMaximum(hi)
            mw.slice_slider.setValue((lo + hi) // 2)
            mw.slice_slider.blockSignals(False)
            self.set_slice_controls(True)
            self._update_slice_readout()
            mw.vtk_view.set_slice((lo + hi) // 2)
        else:
            self.set_slice_controls(False)

    def set_slice_controls(self, vis: bool) -> None:
        """Toggle visibility of all slice-navigation widgets."""
        mw = self.mw
        for w in (mw.slice_slider, mw.orient_combo, mw.slice_caption, mw.slice_value_label):
            w.setVisible(vis)
        if not vis:
            mw.slice_value_label.setText("—")

    def _update_slice_readout(self) -> None:
        """Refresh the slice index / mm label next to the slider."""
        mw = self.mw
        if not mw.slice_caption.isVisible():
            mw.slice_value_label.setText("—")
            return
        lo = mw.slice_slider.minimum()
        hi = mw.slice_slider.maximum()
        idx = mw.slice_slider.value()
        pos_mm = mw.vtk_view.slice_index_to_mm(idx)
        mw.slice_value_label.setText(f"{idx}/{hi}  ({pos_mm:.2f} mm)")

    def set_zoom_controls_visible(self, visible: bool) -> None:
        """Show or hide image zoom controls in the navigation toolbar."""
        self.mw.zoom_controls.setVisible(bool(visible))

    # ------------------------------------------------------------------
    # Slice slider handler
    # ------------------------------------------------------------------

    def on_slice_slider_changed(self, v: int) -> None:
        """Single handler for the slice slider (works for NIfTI, PNG, and VTK)."""
        mw = self.mw
        if self.slice_nav_mode == "png" and self.slice_nav_items:
            idx = max(0, min(v, len(self.slice_nav_items) - 1))
            path = self.slice_nav_items[idx]
            self._show_png_on_image_label(path)
            self._update_slice_readout()
        elif self.slice_nav_mode == "nifti":
            self.show_nifti_slice(v)
            mw._active_view = "image"
            self._update_slice_readout()
        elif self.slice_nav_mode == "vtk":
            mw._active_view = "vtk"
            idx = max(0, min(v, len(self.slice_nav_items) - 1))
            self.show_widget(mw.vtk_view)
            mw.vtk_view.delete_slice_section()
            mw.vtk_view.show_slice_with_mesh(
                mesh_file=mw.current_path,
                slice_file=mw.current_output_3D_slices,
                slice_value=idx,
            )
            self._update_slice_readout()
        else:
            mw.vtk_view.set_slice(v)
            self._update_slice_readout()

    # ------------------------------------------------------------------
    # Orientation / view mode changes
    # ------------------------------------------------------------------

    def on_orientation_changed(self, text: str) -> None:
        """Handle a change in the orientation combo box."""
        mw = self.mw
        mw.vtk_view.set_orientation(text)
        if mw.vtk_view.has_slice():
            lo, hi = mw.vtk_view.slice_range()
            mw.slice_slider.blockSignals(True)
            mw.slice_slider.setMinimum(lo)
            mw.slice_slider.setMaximum(hi)
            mw.slice_slider.setValue(max(lo, min(hi, mw.slice_slider.value())))
            mw.slice_slider.blockSignals(False)
            self._update_slice_readout()
        if self.slice_nav_mode == "nifti":
            self.nifti_set_orientation(text)
            self._update_slice_readout()

    def on_view_changed(self, text: str, path: str | None = None) -> None:
        """Switch between 2-D and 3-D display modes for NIfTI or mesh data."""
        mw = self.mw
        if mw.current_kind == "nifti":
            if text == "3D":
                self.slice_nav_mode = None
                rdr = vtkNIFTIImageReader()
                rdr.SetFileName(mw.current_path if path is None else path)
                rdr.Update()
                img = rdr.GetOutput()
                mw.vtk_view.show_image2d(img)
                self.show_widget(mw.vtk_view)
                self.sync_slice_controls()
                self.on_orientation_changed(mw.orient_combo.currentText())
            elif text == "2D":
                self.slice_nav_mode = "nifti"
                self.nifti_set_orientation(mw.orient_combo.currentText(), path)
        elif mw.is_vtk or mw.current_kind == "stl":
            if text == "3D":
                self.show_widget(mw.vtk_view)
                self.slice_nav_mode = "vtk"
                mw.vtk_view.show_slice_with_mesh(
                    mesh_file=mw.current_path,
                    slice_file=mw.current_output_3D_slices,
                    slice_value=mw.slice_slider.value(),
                )
            elif text == "2D":
                mw.vtk_view.delete_slice_section()
                self.show_widget(mw.image_label)
                self.slice_nav_mode = "png"
                self.on_slice_slider_changed(mw.slice_slider.value())

    # ------------------------------------------------------------------
    # PNG on image label
    # ------------------------------------------------------------------

    def _show_png_on_image_label(self, png_path: str) -> None:
        """Load a PNG file and display it on the image label widget."""
        pm = QPixmap(png_path)
        if pm.isNull():
            print(f"Could not load preview image: {png_path}")
            return
        self.mw.image_label.setImage(pm)
        if hasattr(self.mw, "zoom_controls") and self.mw.zoom_controls is not None:
            self.mw.zoom_controls.set_zoom_text("Fit")
        self.show_widget(self.mw.image_label)
        self.mw._active_view = "image"

    # ------------------------------------------------------------------
    # NIfTI orientation / slice rendering
    # ------------------------------------------------------------------

    def nifti_set_orientation(self, view: str, path: str | None = None) -> None:
        """Set the NIfTI slice axis from a label and reconfigure the slider."""
        import nibabel as nib

        mw = self.mw
        img = nib.load(mw.current_path if path is None else path)
        vol = img.get_fdata(dtype=float)
        if vol is None:
            print("[NIfTI] No data loaded.")
            return

        a = np.asarray(vol)
        if a.ndim == 4:
            a = a[..., 0]

        axis_map = {"Sagittal (X)": 0, "Coronal (Y)": 1, "Axial (Z)": 2}
        self.nifti_axis = axis_map.get(view, 2)

        self.nifti_depth = int(a.shape[self.nifti_axis])
        mid = max(0, self.nifti_depth // 2)

        if hasattr(mw, "slice_slider"):
            mw.slice_slider.blockSignals(True)
            mw.slice_slider.setMinimum(0)
            mw.slice_slider.setMaximum(max(0, self.nifti_depth - 1))
            mw.slice_slider.setValue(mid)
            mw.slice_slider.blockSignals(False)

        self.show_nifti_slice(mid)

    def show_nifti_slice(self, idx: int, axis: int | None = None) -> None:
        """Render a single NIfTI slice on the image label with optional label overlay."""
        import nibabel as nib

        mw = self.mw
        img = nib.load(mw.current_path)
        vol = img.get_fdata(dtype=float)
        if vol is None:
            return

        a = np.asarray(vol)
        if a.ndim == 4:
            a = a[..., 0]

        ax = self.nifti_axis if axis is None else int(axis)
        depth = a.shape[ax]
        i = max(0, min(int(idx), depth - 1))

        sl = a[i, :, :] if ax == 0 else (a[:, i, :] if ax == 1 else a[:, :, i])

        f = sl.astype(np.float32, copy=False)
        lo, hi = np.percentile(f, (1, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
        gray = (np.clip((f - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
        gray = np.ascontiguousarray(gray)

        qimg = None
        if self.label_overlay_enabled and self.nifti_selected_regions:
            L = getattr(mw, "nifti_label_data", None) or a
            if L.ndim == 4:
                L = L[..., 0]
            label2d = np.rint(
                L[i, :, :] if ax == 0 else (L[:, i, :] if ax == 1 else L[:, :, i])
            ).astype(np.int32)

            rgb = np.dstack([gray, gray, gray])
            rgb = np.ascontiguousarray(rgb)
            qimg = self._compose_label_overlay(rgb, label2d, self.nifti_selected_regions)
            mw._last_frame_rgb = rgb
        else:
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, gray.strides[0], QImage.Format_Grayscale8)
            mw._last_frame_gray = gray

        zooms = img.header.get_zooms()[:3]
        qimg, mw.mm_per_px_bar, mw.bar_mm = add_scalebar(qimg, zooms, ax)
        mw.image_label.setImage(QPixmap.fromImage(qimg))
        self.show_widget(mw.image_label)

    # ------------------------------------------------------------------
    # Label overlay helpers
    # ------------------------------------------------------------------

    def _compose_label_overlay(
        self,
        img2d: np.ndarray,
        label2d: np.ndarray,
        selected: set[int],
        alpha: float = 0.5,
    ) -> QImage:
        """Blend coloured label regions onto a grayscale or RGB base image."""
        if img2d.ndim == 3 and img2d.shape[-1] == 3:
            f = (0.299 * img2d[..., 0] + 0.587 * img2d[..., 1] + 0.114 * img2d[..., 2]).astype(np.float32, copy=False)
        else:
            f = img2d.astype(np.float32, copy=False)

        lo, hi = np.percentile(f, (1, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0

        gray = (np.clip((f - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = np.dstack([gray, gray, gray]).astype(np.float32, copy=False)

        if selected:
            for lab in selected:
                mask = (label2d == lab)
                if not np.any(mask):
                    continue
                c = self.nifti_label_lut.get(lab, self._color_for_label(lab))
                overlay_color = np.array([float(c.red()), float(c.green()), float(c.blue())], dtype=np.float32)
                rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * overlay_color[None, :]

        rgb_u8 = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
        h, w, _ = rgb_u8.shape
        qimg = QImage(rgb_u8.data, w, h, rgb_u8.strides[0], QImage.Format_RGB888)
        return qimg.copy()

    def _color_for_label(self, lab: int) -> QColor:
        """Return a deterministic vivid QColor for a given integer label."""
        from colorsys import hsv_to_rgb
        hue = (lab * 0.61803398875) % 1.0
        r, g, b = hsv_to_rgb(hue, 0.75, 0.95)
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    def _color_square_icon(self, col: QColor, size: int = 12) -> QIcon:
        """Create a small square icon filled with *col*."""
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        try:
            p.fillRect(0, 0, size, size, col)
            p.setPen(QPen(Qt.black, 1))
            p.drawRect(0, 0, size - 1, size - 1)
        finally:
            p.end()
        return QIcon(pm)

    # ------------------------------------------------------------------
    # PNG navigation
    # ------------------------------------------------------------------

    def enable_png_navigation(
        self,
        png_paths: list[str],
        slice_indices: list[int] | None = None,
        start_index: int | None = None,
    ) -> None:
        """Switch the slice slider to browse a list of PNG previews."""
        if not png_paths:
            return
        mw = self.mw
        self.slice_nav_mode = "png"
        self.slice_nav_items = list(png_paths)
        self.slice_nav_index_map = list(slice_indices) if slice_indices is not None else [None] * len(png_paths)

        mw.nav_tb.show()
        mw.slice_slider.setEnabled(True)

        mw.slice_slider.blockSignals(True)
        mw.slice_slider.setMinimum(slice_indices[0])
        mw.slice_slider.setMaximum(slice_indices[-1])
        mw.slice_slider.setSingleStep(1)
        mw.slice_slider.setPageStep(5)
        init = start_index if isinstance(start_index, int) else len(self.slice_nav_items) // 2
        init = max(0, min(init, len(self.slice_nav_items) - 1))
        mw.slice_slider.setValue(init)
        mw.slice_slider.blockSignals(False)
        mw.orient_combo.setEnabled(False)

        if mw.current_kind != "stl":
            self.on_slice_slider_changed(init)
            self.show_widget(mw.image_label)
            mw.view_mode.setEnabled(False)

    def reset_png_navigation(self) -> None:
        """Return the slider to normal NIfTI navigation."""
        mw = self.mw
        self.slice_nav_mode = "nifti" if mw.current_kind == "nifti" else None
        self.slice_nav_items = []
        self.slice_nav_index_map = []
        if self.slice_nav_mode == "nifti" and hasattr(mw, "nifti_depth"):
            mw.slice_slider.blockSignals(True)
            mw.slice_slider.setMinimum(0)
            mw.slice_slider.setMaximum(max(0, self.nifti_depth - 1))
            mw.slice_slider.blockSignals(False)
            mw.view_mode.setEnabled(True)
            mw.orient_combo.setEnabled(True)

    def disable_png_navigation(self) -> None:
        """Exit PNG navigation mode and reset slice controls."""
        if getattr(self, "slice_nav_mode", None) != "png":
            return
        mw = self.mw
        self.slice_nav_mode = None
        self.slice_nav_items = []
        self.slice_nav_index_map = []
        if hasattr(mw, "slice_slider"):
            mw.slice_slider.setEnabled(False)
        if hasattr(mw, "slice_value_label"):
            mw.slice_value_label.clear()
        if hasattr(mw, "image_label"):
            mw.image_label.clear()

    def two_mode_view(self, out_dir: str, saved_pngs: list[str], valid_slices: list[int]) -> None:
        """Set up dual 2-D/3-D navigation for STL or VTK measurement results."""
        mw = self.mw
        mw.current_output_3D_slices = os.path.join(out_dir, "all_slices_mesh.vtk")
        self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
        mw.nav_tb.show()
        mw.slice_slider.setEnabled(True)
        mw.view_mode.setEnabled(True)
        mw.view_mode.setCurrentText("2D")
        mid = len(saved_pngs) // 2
        if mw.view_mode.currentText() == "2D":
            self.on_slice_slider_changed(mid)
        elif mw.view_mode.currentText() == "3D":
            mw.vtk_view.show_slice_with_mesh(
                mesh_file=mw.current_path,
                slice_file=mw.current_output_3D_slices,
                slice_value=mid,
            )
