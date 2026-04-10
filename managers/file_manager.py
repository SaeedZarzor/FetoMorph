"""File manager -- import, load, save, and close operations.

Extracted from MainWindow to consolidate all file I/O logic
in a single module.
"""

from __future__ import annotations

from deps import *
from typing import TYPE_CHECKING
from helpers.helpers import get_nifti_present_labels
from widgets.recent_paths import RecentPaths, populate_recent_menu

if TYPE_CHECKING:
    from FetoMorph import MainWindow

logger = logging.getLogger("fetomorph")


class FileManager:
    """Owns file-related state and import/load/save/close methods."""

    def __init__(self, mw: MainWindow) -> None:
        self.mw = mw

        # ---- state ----
        self.last_dir: str = (
            os.path.expanduser("~/Documents")
            if os.path.isdir(os.path.expanduser("~/Documents"))
            else os.path.expanduser("~")
        )
        self.current_output_dir: str | None = None
        self.current_output_3D_slices: str | None = None
        self.last_annotated_path: str | None = None
        self.recent = RecentPaths("YourOrg", "YourApp")

    # ------------------------------------------------------------------
    # Import dialogs
    # ------------------------------------------------------------------

    def import_image(self) -> None:
        """Open a file dialog for image files and load the selected image."""
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(
            self.mw, "Import Image", start,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif)",
        )
        self.open_path(path)
        if not path:
            return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing image: {path}")
        self.load_image(path)

    def import_vtk(self) -> None:
        """Open a file dialog for VTK legacy files and load the selected mesh/image."""
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(
            self.mw, "Import .vtk", start, "VTK Legacy (*.vtk)",
        )
        self.open_path(path)
        if not path:
            return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing VTK: {path}")
        self.load_vtk(path)

    def import_stl(self) -> None:
        """Open a file dialog for STL mesh files and load the selected surface."""
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(
            self.mw, "Import .stl", start, "STL Mesh (*.stl)",
        )
        self.open_path(path)
        if not path:
            return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing STL: {path}")
        self.load_stl(path)

    def import_nifti(self) -> None:
        """Open a file dialog for NIfTI volumes and load the selected scan."""
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(
            self.mw, "Import NIfTI", start, "NIfTI (*.nii *.nii.gz)",
        )
        self.open_path(path)
        if not path:
            return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing NIfTI: {path}")
        self.load_nifti(path)

    # ------------------------------------------------------------------
    # Save / export
    # ------------------------------------------------------------------

    def save_view(self) -> None:
        """Ask path & save exactly what is displayed."""
        mw = self.mw
        if mw._active_view not in ("image", "vtk"):
            QMessageBox.information(mw, "Save View", "Nothing to save.")
            return
        base = "view"
        if mw.current_path:
            base = os.path.splitext(os.path.basename(mw.current_path))[0] + "_view"
        default_name = os.path.join(self.last_dir, base + ".png")
        path, _ = QFileDialog.getSaveFileName(
            mw, "Save View As…", default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
        )
        if not path:
            return
        extn = os.path.splitext(path)[1].lower()
        if extn not in (".png", ".jpg", ".jpeg"):
            path += ".png"
            extn = ".png"
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            reply = QMessageBox.question(
                mw, "Create Folder?",
                f"The folder\n\n{folder}\n\ndoes not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                try:
                    os.makedirs(folder, exist_ok=True)
                    print(f"Created folder: {folder}")
                except Exception as ex:
                    logger.error("Error creating folder: %s", ex)
                    QMessageBox.critical(mw, "Save Failed", f"Could not create folder:\n{ex}")
                    return
            else:
                return
        self.last_dir = folder or self.last_dir
        try:
            if mw._active_view == "image":
                pm = mw.image_label.grab()
                ok = pm.save(path)
                if not ok:
                    raise RuntimeError("Failed to save image widget snapshot.")
            else:
                rw = mw.vtk_view.vtkWidget.GetRenderWindow()
                rw.Render()
                w2i = vtkWindowToImageFilter()
                w2i.SetInput(rw)
                w2i.ReadFrontBufferOff()
                w2i.Update()
                writer = vtkJPEGWriter() if extn in (".jpg", ".jpeg") else vtkPNGWriter()
                writer.SetFileName(path)
                writer.SetInputConnection(w2i.GetOutputPort())
                writer.Write()
            print(f"Saved view to: {path}")
        except Exception as ex:
            logger.error("Error saving view: %s", ex)
            QMessageBox.critical(mw, "Save Failed", f"{type(ex).__name__}: {ex}")

    def save_data_as(self) -> None:
        """Export the current results folder or the loaded data file."""
        mw = self.mw
        src_folder = self.current_output_dir
        has_folder = bool(
            src_folder and os.path.isdir(src_folder)
            and any(files for _, _, files in os.walk(src_folder))
        )

        src_file = mw.current_path if (mw.current_path and os.path.isfile(mw.current_path)) else None
        has_file = bool(src_file)

        if not has_folder and not has_file:
            QMessageBox.information(mw, "Export", "There is nothing to export yet.")
            return

        mode = None
        if has_folder and has_file:
            m = QMessageBox(mw)
            m.setWindowTitle("Export")
            m.setText("What do you want to export?")
            btn_folder = m.addButton("Results Folder", QMessageBox.AcceptRole)
            btn_file = m.addButton("Current File Only", QMessageBox.ActionRole)
            m.addButton(QMessageBox.Cancel)
            m.exec()
            if m.clickedButton() is btn_folder:
                mode = "folder"
            elif m.clickedButton() is btn_file:
                mode = "file"
            else:
                return
        elif has_folder:
            mode = "folder"
        else:
            mode = "file"

        start_dir = self.last_dir if os.path.isdir(self.last_dir) else os.path.expanduser("~")

        if mode == "folder":
            dest_root = QFileDialog.getExistingDirectory(
                mw, "Choose Destination Folder", start_dir,
                QFileDialog.Option.ShowDirsOnly,
            )
            if not dest_root:
                return
            folder_name = os.path.basename(os.path.normpath(src_folder))
            target = os.path.join(dest_root, folder_name)
            try:
                if os.path.exists(target):
                    reply = QMessageBox.question(
                        mw, "Folder Exists",
                        f"{target}\n\nalready exists. Merge into it?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                    )
                    if reply != QMessageBox.Yes:
                        return
                    shutil.copytree(src_folder, target, dirs_exist_ok=True)
                else:
                    shutil.copytree(src_folder, target)
            except Exception as ex:
                QMessageBox.critical(mw, "Export Failed", f"{type(ex).__name__}: {ex}")
                return
            print(f"[Export] Folder → {target}")
            self.last_dir = dest_root
            try:
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(target))
            except Exception:
                pass
            QMessageBox.information(mw, "Export Complete", f"Exported folder to:\n{target}")
            return

        # mode == "file"
        base = os.path.basename(src_file)
        kind = (mw.current_kind or "").lower()
        if kind == "stl":
            filt = "STL Mesh (*.stl)"
            suggested = os.path.join(start_dir, base if base.lower().endswith(".stl") else os.path.splitext(base)[0] + ".stl")
        elif kind == "nifti":
            filt = "NIfTI (*.nii *.nii.gz)"
            suggested = os.path.join(start_dir, base if (base.lower().endswith(".nii") or base.lower().endswith(".nii.gz")) else os.path.splitext(base)[0] + ".nii.gz")
        elif kind == "image":
            filt = "Image (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            suggested = os.path.join(start_dir, base)
        else:
            filt = "All Files (*.*)"
            suggested = os.path.join(start_dir, base)

        path, _ = QFileDialog.getSaveFileName(mw, "Save Current File As…", suggested, filt)
        if not path:
            return

        if kind == "stl" and not path.lower().endswith(".stl"):
            path += ".stl"
        if kind == "nifti" and not (path.lower().endswith(".nii") or path.lower().endswith(".nii.gz")):
            path += ".nii.gz"

        try:
            shutil.copy2(src_file, path)
            print(f"[Export] File → {path}")
            self.last_dir = os.path.dirname(path)
            QMessageBox.information(mw, "Export Complete", f"Saved:\n{path}")
        except Exception as ex:
            QMessageBox.critical(mw, "Export Failed", f"{type(ex).__name__}: {ex}")

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close_current(self) -> None:
        """Close the currently loaded file and reset the display to a blank state."""
        mw = self.mw
        for meth in ("clear_annotations", "clear_line_annotations"):
            if hasattr(mw.image_label, meth):
                getattr(mw.image_label, meth)()
        mw.image_label.clearImage()
        mw.view.show_widget(mw.image_label)
        mw.view.set_slice_controls(False)
        mw.act_choose_regions.setEnabled(False)
        mw.act_annotate_square.setEnabled(False)
        mw.nav_tb.hide()
        mw._set_current(None, None)
        print("\n Closed current file and reset view.")
        mw.statusBar().showMessage("Closed current file and reset view.", 3000)

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def open_path(self, path: str) -> None:
        """Route a file path to the appropriate loader based on its extension."""
        ext = Path(path).suffix.lower()
        try:
            if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}:
                self.load_image(path)
                print(f"Importing image: {path}")
            elif ext == ".vtk":
                self.load_vtk(path)
                print(f"Importing VTK: {path}")
            elif ext == ".nii" or ext == ".gz":
                self.load_nifti(path)
                print(f"Importing NIfTI: {path}")
            elif ext == ".stl":
                self.load_stl(path)
                print(f"Importing STL: {path}")
            else:
                logger.warning("Unknown file type: %s", ext)

            self.last_dir = os.path.dirname(path) or self.last_dir
        except Exception as e:
            logger.error("Failed to open %s: %s", path, e)

        self.recent.add(path)
        populate_recent_menu(self.mw.menu_recent, self.recent, self.open_path)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> None:
        """Load and display a 2-D raster image."""
        mw = self.mw
        pm = QPixmap(path)
        if pm.isNull():
            QMessageBox.critical(mw, "Open Failed", "Could not read image file.")
            return
        mw.image_label.setImage(pm)
        mw.view.show_widget(mw.image_label)
        mw.view.set_slice_controls(False)
        if hasattr(mw, "zoom_controls"):
            mw.zoom_controls.set_zoom_text("Fit")
        print(f"Loaded image: {path}  size={pm.width()}x{pm.height()}")
        mw._set_current("image", path)
        mw.statusBar().showMessage(f"{mw.current_path} image is loaded", 5000)

    def load_nifti(self, path: str) -> None:
        """Load a NIfTI volume and display it in 2-D or 3-D mode."""
        mw = self.mw
        mw._set_current("nifti", path)
        rdr = vtkNIFTIImageReader()
        rdr.SetFileName(path)
        rdr.Update()
        img = rdr.GetOutput()
        print(
            f"NIfTI loaded:\n"
            f"Extent={img.GetExtent()} \n Spacing={img.GetSpacing()} \n  Range={img.GetScalarRange()}"
        )
        mw.labels_available = get_nifti_present_labels(path)
        mw.statusBar().showMessage(f"{mw.current_path} nifti file is loaded", 5000)
        if mw.view_mode.currentText() == "3D":
            mw.slice_nav_mode = None
            mw.vtk_view.show_image2d(img)
            mw.view.show_widget(mw.vtk_view)
            mw.view.sync_slice_controls()
        elif mw.view_mode.currentText() == "2D":
            mw.slice_nav_mode = "nifti"
            mw.view.nifti_set_orientation(mw.orient_combo.currentText())

    def load_stl(self, path: str) -> None:
        """Load an STL mesh and render it in the VTK viewer."""
        mw = self.mw
        r = vtkSTLReader()
        r.SetFileName(path)
        r.Update()
        poly = r.GetOutput()
        if not poly or poly.GetNumberOfPoints() == 0:
            QMessageBox.critical(mw, "Open Failed", "Empty or invalid .stl file.")
            return
        mw.vtk_view.show_polydata(poly)
        mw.view.show_widget(mw.vtk_view)
        mw.view.set_slice_controls(False)
        print(
            f"STL loaded:\n"
            f"  File: {path}\n"
            f"  Points: {poly.GetNumberOfPoints():,}\n"
            f"  Polys:  {poly.GetNumberOfPolys():,}"
        )
        mw._set_current("stl", path)

    @staticmethod
    def _polydata_is_planar(pd: vtkPolyData, tol: float = 1e-6) -> int | None:
        """Return the flat axis index if all points lie in a single plane, else None."""
        n = pd.GetNumberOfPoints()
        if n < 3:
            return 2
        bounds = pd.GetBounds()
        for i in range(3):
            if bounds[2 * i + 1] - bounds[2 * i] < tol:
                return i
        return None

    def load_vtk(self, path: str) -> None:
        """Load a VTK legacy file (image data or polydata) and display it."""
        mw = self.mw
        dsr = vtkDataSetReader()
        dsr.SetFileName(path)
        dsr.Update()
        ds = dsr.GetOutput()
        if ds is None:
            gr = vtkGenericDataObjectReader()
            gr.SetFileName(path)
            gr.Update()
            ds = gr.GetOutput()
        if isinstance(ds, vtkImageData):
            mw._flat_axis = None
            mw.vtk_view.show_image2d(ds)
            mw.view.show_widget(mw.vtk_view)
            mw.view.sync_slice_controls()
            print(f"Legacy VTK image loaded. Extent={ds.GetExtent()}  Range={ds.GetScalarRange()}")
            mw._set_current("vtk_image", path)
            return
        if isinstance(ds, vtkPolyData) and ds.GetNumberOfPoints() > 0:
            flat_axis = self._polydata_is_planar(ds)
            if flat_axis is not None:
                mw._flat_axis = flat_axis
                mw.slice_direction = ("X", "Y", "Z")[flat_axis]
                mw.vtk_view.show_polydata_2d(ds, flat_axis)
                mw.view.show_widget(mw.vtk_view)
                mw.view.set_slice_controls(False)
                print(f"Legacy VTK polydata (planar/2D). Points={ds.GetNumberOfPoints()}  Polys={ds.GetNumberOfPolys()}")
                mw._set_current("vtk_image", path)
                return
            mw._flat_axis = None
            mw.vtk_view.show_polydata(ds)
            mw.view.show_widget(mw.vtk_view)
            mw.view.set_slice_controls(False)
            print(f"Legacy VTK polydata loaded. Points={ds.GetNumberOfPoints()}  Polys={ds.GetNumberOfPolys()}")
            mw._set_current("vtk_poly", path)
            return
        surf = vtkDataSetSurfaceFilter()
        surf.SetInputData(ds)
        surf.Update()
        poly = surf.GetOutput()
        if poly and poly.GetNumberOfPoints() > 0:
            flat_axis = self._polydata_is_planar(poly)
            if flat_axis is not None:
                mw._flat_axis = flat_axis
                mw.slice_direction = ("X", "Y", "Z")[flat_axis]
                mw.vtk_view.show_polydata_2d(poly, flat_axis)
                mw.view.show_widget(mw.vtk_view)
                mw.view.set_slice_controls(False)
                print(f"Legacy VTK dataset surfaced (planar/2D). Points={poly.GetNumberOfPoints()}  Polys={poly.GetNumberOfPolys()}")
                mw._set_current("vtk_image", path)
                return
            mw._flat_axis = None
            mw.vtk_view.show_polydata(poly)
            mw.view.show_widget(mw.vtk_view)
            mw.view.set_slice_controls(False)
            print(f"Legacy VTK dataset surfaced. Points={poly.GetNumberOfPoints()}  Polys={poly.GetNumberOfPolys()}")
            mw._set_current("vtk_surface", path)
            return
        QMessageBox.critical(mw, "Open Failed", "Unsupported or empty .vtk dataset (no points after surface extraction).")
