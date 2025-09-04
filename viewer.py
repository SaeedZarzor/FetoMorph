from deps import *

# ---- Measurements (your OpenCV routine, now pure) ----
from functions.measurements_image import *
from functions.pial_to_stl import *
from functions.measurements_Nifti import *
from functions.Fetal_brain_GI_stl import compute_stl_lGI
from widgets.scaled_image_label import ScaledImageLabel
from widgets.Contour_threshold import ContourThresholdDialog
from widgets.Scalebar_set_scale import ScalebarSetScaleDialog
from widgets.Unit_scale import UnitScaleDialog
from widgets.VTK_Viewer import VTKViewer
from widgets.Kernel_size import KernelSizeDialog
from ribbon import *


# ---------------------------
# Supported extensions
# ---------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
NIFTI_EXTS = {".nii", ".nii.gz"}
APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"

def ext(path: str) -> str:
    low = path.lower()
    if low.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(low)[1]

# ---------------------------
# Console capture
# ---------------------------
class QtConsole(QObject):
    text = Signal(str)
    def write(self, s: str):
        if s:
            self.text.emit(str(s))
    def flush(self): pass

class TeeStream:
    def __init__(self, a, b): self.a, self.b = a, b
    def write(self, s):
        for t in (self.a, self.b):
            try: t.write(s)
            except Exception: pass
    def flush(self):
        for t in (self.a, self.b):
            try: t.flush()
            except Exception: pass

class QtVTKOutputWindow(vtkOutputWindow):
    def __init__(self, sink: QtConsole):
        super().__init__(); self._sink = sink
    def DisplayText(self, txt): self._sink.write(txt)
    def DisplayErrorText(self, txt): self._sink.write("VTK ERROR: " + txt)
    def DisplayWarningText(self, txt): self._sink.write("VTK WARNING: " + txt)
    def DisplayGenericWarningText(self, txt): self._sink.write("VTK WARNING: " + txt)
    def DisplayDebugText(self, txt): self._sink.write("VTK DEBUG: " + txt)


# ---------------------------
# Main Window
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified Image / VTK / NIfTI Viewer (PySide6 + VTK)")
        self.resize(1200, 900)

        # State
        self.current_path: str | None = None
        self.current_kind: str | None = None  # "image" | "nifti" | "vtk_image" | "vtk_poly" | "vtk_surface" | "stl"
        self._active_view = "image"
        self.last_dir = os.path.expanduser("~/Documents") if os.path.isdir(os.path.expanduser("~/Documents")) else os.path.expanduser("~")

        # Temp working directory for processing (no persistent saves here)
        self.temp_dir = tempfile.mkdtemp(prefix="FetoMorph_")
        print(f"[Temp] Working directory: {self.temp_dir}")

        # store metics
        self.metrics: dict[str, list[dict]] = {}  # per-path grouping {path: {"File":..., "Kind":..., "Area":..., "Volume":..., "SulciDepth_P1":..., "SulciDepth_P2":..., "SulciDepth_P3":..., "LGI":...}}
        self.current_output_dir = None
        self.last_annotated_path: str | None = None
        self.annotation_records: list[dict] = []          # flat list of all annotations
        self.annotations_by_source: dict[str, list[dict]] = {}  # per-image grouping
        self._roi_counter_by_source: dict[str, int] = {}  # for auto names if user leaves blank
        self.annotation_labels_by_path: dict[str, str] = {}  # save the label for each RIO path

        
        # Params for measurements
        self.units_length = None          # e.g., "mm" (set by user prompt)
        self.pixel_size_default = 0.01
        self.pixel_size = self.pixel_size_default       # current working scale (units/pixel)
        self.image_scales = {}                          # per-file scale: {path: float}
        self.cnt_threshold = 200
        self.kernel_size = 5  # default (pixels)
        self.slice_thickness = 0.5


        # Slice navigation mode/state
        self.slice_nav_mode = None           # None | "nifti" | "png"
        self.slice_nav_items = []            # list[str] when mode=="png" (PNG paths)
        self.slice_nav_index_map = []        # list[int] original slice indices (optional label)

        # View widgets
        self.image_label = ScaledImageLabel()
        self.vtk_view = VTKViewer()

        self.display_box = QHBoxLayout(); self.display_box.setContentsMargins(0,0,0,0)
        self.display_box.addWidget(self.image_label); self.display_box.addWidget(self.vtk_view)
        display_container = QWidget(); display_container.setLayout(self.display_box)

        # Progress console
        self.progress_group = QGroupBox("Progress"); pg = QVBoxLayout(self.progress_group)
        self.progress_edit = QPlainTextEdit(); self.progress_edit.setReadOnly(True)
        self.progress_edit.setMaximumBlockCount(10000); self.progress_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.progress_edit.setStyleSheet("background:#0b0b0b; color:#d0d0d0; border:1px solid #333;")
        pg.addWidget(self.progress_edit)

        # Splitter
        self.splitter = QSplitter(Qt.Vertical); self.splitter.addWidget(display_container); self.splitter.addWidget(self.progress_group)
        self.splitter.setSizes([700, 200]); self.progress_group.setMinimumHeight(100); self.progress_group.setMaximumHeight(500)

        self.container = QWidget(); self.vbox = QVBoxLayout(self.container); self.vbox.setContentsMargins(8,8,8,8); self.vbox.addWidget(self.splitter)
        self.setCentralWidget(self.container)
        self._show_widget(self.image_label)
        
        # Menus — File (with Import submenu)
        file_menu = self.menuBar().addMenu("File")
        import_menu = file_menu.addMenu("Import")

        self.act_imp_img = QAction("Image…", self); self.act_imp_img.setShortcut(QKeySequence.Open); self.act_imp_img.triggered.connect(self.import_image); import_menu.addAction(self.act_imp_img)
        self.act_imp_vtk = QAction(".vtk file…", self); self.act_imp_vtk.setShortcut(QKeySequence("Ctrl+Shift+V")); self.act_imp_vtk.triggered.connect(self.import_vtk); import_menu.addAction(self.act_imp_vtk)
        self.act_imp_stl = QAction(".stl file…", self); self.act_imp_stl.setShortcut(QKeySequence("Ctrl+Shift+L")); self.act_imp_stl.triggered.connect(self.import_stl); import_menu.addAction(self.act_imp_stl)
        self.act_imp_nii = QAction("NIfTI…", self); self.act_imp_nii.setShortcut(QKeySequence("Ctrl+Shift+N")); self.act_imp_nii.triggered.connect(self.import_nifti); import_menu.addAction(self.act_imp_nii)

        file_menu.addSeparator()
        self.act_save = QAction("Save View As…", self); self.act_save.setShortcut(QKeySequence("Ctrl+V")); self.act_save.triggered.connect(self.save_view); file_menu.addAction(self.act_save)
        self.act_save_data = QAction("Save Data As…", self); self.act_save_data.setShortcut(QKeySequence.SaveAs); self.act_save_data.triggered.connect(self.save_data_as); file_menu.addAction(self.act_save_data)
        self.act_export_metrics = QAction("Export Metrics to Excel…", self); self.act_export_metrics.setShortcut(QKeySequence("Ctrl+E")); self.act_export_metrics.triggered.connect(self.export_metrics_excel); file_menu.addAction(self.act_export_metrics)
        self.act_Reset= QAction("Reset view…", self); self.act_Reset.setShortcut(QKeySequence("Ctrl+R")); self.act_Reset.setToolTip("Return to original view and clear on-screen annotations"); self.act_Reset.triggered.connect(self.reset_view); file_menu.addAction(self.act_Reset)
        self.act_close = QAction("Close", self); self.act_close.setShortcut(QKeySequence.Close); self.act_close.triggered.connect(self.close_current); file_menu.addAction(self.act_close)
        
        file_menu.addSeparator()
        self.act_quit = QAction("Quit", self); self.act_quit.setShortcut(QKeySequence.Quit); self.act_quit.setMenuRole(QAction.MenuRole.QuitRole); self.act_quit.triggered.connect(self.quit_app); file_menu.addAction(self.act_quit)

        # Process menu (auto-enabled by file type)
        process_menu = self.menuBar().addMenu("Process"); self.process_menu = process_menu
        measures_menu = process_menu.addMenu("Measure")
        self.act_meas_allmarks = QAction("All hallmarks", self); self.act_meas_allmarks.triggered.connect(self.on_measure_allmarks); measures_menu.addAction(self.act_meas_allmarks)
        self.act_meas_volumes = QAction("Volumes", self); self.act_meas_volumes.triggered.connect(self.on_measure_volumes); measures_menu.addAction(self.act_meas_volumes)
        self.act_meas_area = QAction("Area", self); self.act_meas_area.triggered.connect(self.on_measure_area); measures_menu.addAction(self.act_meas_area)
        self.act_meas_perimeter = QAction("Perimeter", self); self.act_meas_perimeter.triggered.connect(self.on_measure_perimeter); measures_menu.addAction(self.act_meas_perimeter)
        self.act_meas_lgi = QAction("LGI", self); self.act_meas_lgi.triggered.connect(self.on_measure_lgi); measures_menu.addAction(self.act_meas_lgi); self.act_meas_lgi.setToolTip("Compute Local Gyrification Index")
        self.act_meas_sulci = QAction("Sulci Depth", self); self.act_meas_sulci.triggered.connect(self.on_measure_sulci_depth); measures_menu.addAction(self.act_meas_sulci)
        process_menu.addSeparator()
        self.act_optimization = QAction("Optimization", self); self.act_optimization.triggered.connect(self.on_optimization); process_menu.addAction(self.act_optimization)
        process_menu.addSeparator()
        self.act_pial_to_stl = QAction("Pial → STL…", self); self.act_pial_to_stl.triggered.connect(self.on_pial_to_stl); process_menu.addAction(self.act_pial_to_stl)
        self.act_pial_merge = QAction("Combined STL…", self); self.act_pial_merge.triggered.connect(self.on_combined_stl); process_menu.addAction(self.act_pial_merge)

        # Setting menu
        Setting_menu = self.menuBar().addMenu("Setting"); self.Setting_menu = Setting_menu
        self.act_set_image_scale = QAction("Set Image Scale…", self); self.act_set_image_scale.triggered.connect(self.set_image_scale); Setting_menu.addAction(self.act_set_image_scale)
        self.act_set_scale = QAction("Set Scale From Scalebar…", self);self.act_set_scale.triggered.connect(self.set_scale_from_scalebar);
        Setting_menu.addAction(self.act_set_scale)
        self.act_kernel_size = QAction("Set Kernel Size…", self); self.act_kernel_size.triggered.connect(self.set_kernel_dialog); Setting_menu.addAction(self.act_kernel_size)
        self.act_cnt_threshold = QAction("Set Contour Threshold…", self); self.act_cnt_threshold.setShortcut(QKeySequence("Ctrl+T")); self.act_cnt_threshold.triggered.connect(self.set_cnt_threshold_dialog); Setting_menu.addAction(self.act_cnt_threshold)
        self.act_annotate_square = QAction("Annotation…", self); self.act_annotate_square.setShortcut(QKeySequence("Ctrl+Shift+A"));self.act_annotate_square.setToolTip("Drag a square on the image and save the crop to the temp folder"); self.act_annotate_square.triggered.connect(self.annotate_square)


        # Disable initially
        self.act_Reset.setEnabled(False); self.act_close.setEnabled(False); self.act_save.setEnabled(False); self.act_close.setEnabled(False); self.act_export_metrics.setEnabled(False);self.act_save_data.setEnabled(False)  # will enable for STL/polydata
        for a in (self.act_meas_allmarks, self.act_meas_volumes, self.act_meas_area, self.act_meas_perimeter, self.act_meas_lgi, self.act_meas_sulci, self.act_optimization):
            a.setEnabled(False)

        # drag & drop
        self.setAcceptDrops(True)

        # Console hooking
        self._orig_stdout = sys.stdout; self._orig_stderr = sys.stderr
        self._qt_console = QtConsole(); self._qt_console.text.connect(self._append_progress)
        sys.stdout = TeeStream(self._orig_stdout, self._qt_console); sys.stderr = TeeStream(self._orig_stderr, self._qt_console)
        vtk_output = QtVTKOutputWindow(self._qt_console); vtkOutputWindow.SetInstance(vtk_output)
        print("Application started. Progress output will appear here.")

        self._update_process_actions()
    

        self.ribbon = RibbonBar(self, icon_size=QSize(20, 20))
        self.ribbon.set_icon_size(QSize(20, 20))
        self.ribbon_tb = QToolBar("Ribbon", self)
        self.ribbon_tb.setMovable(False)
        self.ribbon_tb.setFloatable(False)
        self.ribbon_tb.addWidget(self.ribbon)
        self.addToolBar(Qt.TopToolBarArea, self.ribbon_tb)
        
        self.act_nav_import = QAction("Import", self)
        self.act_nav_import.setToolTip("Go to Import tools")
        self.act_nav_import.triggered.connect(lambda: self.ribbon.set_current_tab("Import"))

        self.act_nav_export = QAction("Export", self)
        self.act_nav_export.setToolTip("Go to Export tools")
        self.act_nav_export.triggered.connect(lambda: self.ribbon.set_current_tab("Export"))
        
        self.act_nav_import.setIcon(QIcon(str(ASSETS / "icons/import.png")))
        self.act_nav_export.setIcon(QIcon(str(ASSETS / "icons/export.png")))
        self.act_Reset.setIcon(QIcon(str(ASSETS / "icons/rest.png")))
        self.act_close.setIcon(QIcon(str(ASSETS / "icons/close.png")))
        self.act_quit.setIcon(QIcon(str(ASSETS / "icons/quit.png")))
        self.act_imp_img.setIcon(QIcon(str(ASSETS / "icons/image.png")))
        self.act_imp_vtk.setIcon(QIcon(str(ASSETS / "icons/vtk.png")))
        self.act_imp_stl.setIcon(QIcon(str(ASSETS / "icons/stl.png")))
        self.act_imp_nii.setIcon(QIcon(str(ASSETS / "icons/nifti.png")))
        self.act_save.setIcon(QIcon(str(ASSETS / "icons/screenshot.png")))
        self.act_save_data.setIcon(QIcon(str(ASSETS / "icons/save.png")))
        self.act_export_metrics.setIcon(QIcon(str(ASSETS / "icons/export_to_excel.png")))
        self.act_set_image_scale.setIcon(QIcon(str(ASSETS / "icons/scale.png")))
        self.act_set_scale.setIcon(QIcon(str(ASSETS / "icons/scale_bar.png")))
        self.act_kernel_size.setIcon(QIcon(str(ASSETS / "icons/kernel.png")))
        self.act_annotate_square.setIcon(QIcon(str(ASSETS / "icons/crop_1.png")))
        self.act_cnt_threshold.setIcon(QIcon(str(ASSETS / "icons/threshold.png")))
        self.act_meas_allmarks.setIcon(QIcon(str(ASSETS / "icons/hallmarks.png")))
        self.act_meas_area.setIcon(QIcon(str(ASSETS / "icons/area.png")))
        self.act_meas_volumes.setIcon(QIcon(str(ASSETS / "icons/volume.png")))
        self.act_meas_perimeter.setIcon(QIcon(str(ASSETS / "icons/Perimeter.png")))
        self.act_meas_lgi.setIcon(QIcon(str(ASSETS / "icons/LGI.png")))
        self.act_meas_sulci.setIcon(QIcon(str(ASSETS / "icons/depth.png")))
        
        
        self.ribbon.add_action("Home", self.act_nav_import)
        self.ribbon.add_action("Home", self.act_nav_export)
        self.ribbon.add_action("Home", self.act_Reset)
        self.ribbon.add_action("Home", self.act_close)
        self.ribbon.add_action("Home", self.act_quit)
    
        self.ribbon.add_action("Import", self.act_imp_img)
        self.ribbon.add_action("Import", self.act_imp_vtk)
        self.ribbon.add_action("Import", self.act_imp_stl)
        self.ribbon.add_action("Import", self.act_imp_nii)
        
        self.ribbon.add_action("Export", self.act_save)
        self.ribbon.add_action("Export", self.act_save_data)
        self.ribbon.add_action("Export", self.act_export_metrics)

        self.ribbon.add_action("Measure", self.act_meas_allmarks)
        self.ribbon.add_action("Measure", self.act_meas_perimeter)
        self.ribbon.add_action("Measure", self.act_meas_area)
        self.ribbon.add_action("Measure", self.act_meas_volumes)
        self.ribbon.add_action("Measure", self.act_meas_lgi)
        self.ribbon.add_action("Measure", self.act_meas_sulci)

        self.ribbon.add_action("Settings", self.act_set_image_scale)
        self.ribbon.add_action("Settings", self.act_set_scale)
        self.ribbon.add_action("Settings", self.act_kernel_size)
        self.ribbon.add_action("Settings", self.act_cnt_threshold)
        self.ribbon.add_action("Settings", self.act_annotate_square)





        self.addToolBarBreak(Qt.TopToolBarArea)


        # --- Navigation toolbar (goes BELOW the ribbon) ---
        self.nav_tb = QToolBar("Navigation", self)
        self.nav_tb.setIconSize(QSize(20, 20))
        self.nav_tb.setMovable(True)
        self.addToolBar(Qt.TopToolBarArea, self.nav_tb)

        self.nav_tb.addSeparator()

        self.orient_combo = QComboBox()
        self.orient_combo.addItems(["Axial (Z)", "Coronal (Y)", "Sagittal (X)"])
        self.orient_combo.currentTextChanged.connect(self._on_orientation_changed)
        self.nav_tb.addWidget(self.orient_combo)
        self.orient_combo.setVisible(False)
        
#        slef.act_view_sagittal=QAction("Sagittal", self); self.act_view_sagittal.triggered.connect(lambda: self._on_view_button("sagittal")); self.nav_tb.addAction(self.act_view_sagittal)
#        self.act_view_coronal=QAction("Coronal", self); self.act_view_coronal.triggered.connect(lambda: self._on_view_button("coronal")); self.nav_tb.addAction(self.act_view_coronal)
#        self.act_view_axial=QAction("Axial", self); self.act_view_axial.triggered.connect(lambda: self._on_view_button("axial")); self.nav_tb.addAction(self.act_view_axial)

        

        self.slice_caption = QLabel("Section:")
        self.nav_tb.addWidget(self.slice_caption)
        self.slice_caption.setVisible(False)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setPageStep(5)
        self.slice_slider.valueChanged.connect(self.on_slice_slider_changed)
        self.nav_tb.addWidget(self.slice_slider)
        self.slice_slider.setVisible(False)

        self.slice_value_label = QLabel("—")
        self.nav_tb.addWidget(self.slice_value_label)
        self.slice_value_label.setVisible(False)

    # ---------- Import handlers ----------
    def import_image(self):
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Import Image", start, "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif)")
        if not path: return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing image: {path}"); self.load_image(path)

    def import_vtk(self):
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Import .vtk", start, "VTK Legacy (*.vtk)")
        if not path: return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing VTK: {path}"); self.load_vtk(path)

    def import_stl(self):
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Import .stl", start, "STL Mesh (*.stl)")
        if not path: return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing STL: {path}"); self.load_stl(path)

    def import_nifti(self):
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Import NIfTI", start, "NIfTI (*.nii *.nii.gz)")
        if not path: return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing NIfTI: {path}"); self.load_nifti(path)

    # ---------- File menu ----------
    def save_view(self):
        """Ask path & save exactly what is displayed (no auto-saving during processing)."""
        if self._active_view not in ("image", "vtk"):
            QMessageBox.information(self, "Save View", "Nothing to save."); return
        base = "view"
        if self.current_path: base = os.path.splitext(os.path.basename(self.current_path))[0] + "_view"
        default_name = os.path.join(self.last_dir, base + ".png")
        path, _ = QFileDialog.getSaveFileName(self, "Save View As…", default_name, "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)")
        if not path: return
        extn = os.path.splitext(path)[1].lower()
        if extn not in (".png", ".jpg", ".jpeg"): path += ".png"; extn = ".png"
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            reply = QMessageBox.question(self, "Create Folder?", f"The folder\n\n{folder}\n\ndoes not exist. Create it?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                try: os.makedirs(folder, exist_ok=True); print(f"Created folder: {folder}")
                except Exception as ex: print(f"ERROR creating folder: {ex}"); QMessageBox.critical(self, "Save Failed", f"Could not create folder:\n{ex}"); return
            else: return
        self.last_dir = folder or self.last_dir
        try:
            if self._active_view == "image":
                pm = self.image_label.grab(); ok = pm.save(path)
                if not ok: raise RuntimeError("Failed to save image widget snapshot.")
            else:
                rw = self.vtk_view.vtkWidget.GetRenderWindow(); rw.Render()
                w2i = vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.ReadFrontBufferOff(); w2i.Update()
                writer = vtkJPEGWriter() if extn in (".jpg", ".jpeg") else vtkPNGWriter()
                writer.SetFileName(path); writer.SetInputConnection(w2i.GetOutputPort()); writer.Write()
            print(f"Saved view to: {path}")
        except Exception as ex:
            print(f"ERROR saving view: {ex}"); QMessageBox.critical(self, "Save Failed", f"{type(ex).__name__}: {ex}")
            
    def save_data_as(self):
        """
        Export either:
          • the current results folder (e.g., nifti_area_{uid}), or
          • ONLY the currently loaded data file (e.g., .stl, .nii.gz, image).
        Prompts the user when both are available.
        """

        src_folder = getattr(self, "current_output_dir", None)
        has_folder = bool(src_folder and os.path.isdir(src_folder) and any(files for _,_,files in os.walk(src_folder)))

        src_file = self.current_path if (self.current_path and os.path.isfile(self.current_path)) else None
        has_file = bool(src_file)

        if not has_folder and not has_file:
            QMessageBox.information(self, "Export", "There is nothing to export yet.")
            return

        # Decide mode: ask if both exist
        mode = None
        if has_folder and has_file:
            m = QMessageBox(self)
            m.setWindowTitle("Export")
            m.setText("What do you want to export?")
            btn_folder = m.addButton("Results Folder", QMessageBox.AcceptRole)
            btn_file   = m.addButton("Current File Only", QMessageBox.ActionRole)
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
            # Pick destination directory and copy the whole results folder into it.
            dest_root = QFileDialog.getExistingDirectory(self, "Choose Destination Folder", start_dir,
                                                         QFileDialog.Option.ShowDirsOnly)
            if not dest_root:
                return
            folder_name = os.path.basename(os.path.normpath(src_folder))
            target = os.path.join(dest_root, folder_name)
            try:
                if os.path.exists(target):
                    reply = QMessageBox.question(self, "Folder Exists",
                                                 f"{target}\n\nalready exists. Merge into it?",
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply != QMessageBox.Yes:
                        return
                    shutil.copytree(src_folder, target, dirs_exist_ok=True)
                else:
                    shutil.copytree(src_folder, target)
            except Exception as ex:
                QMessageBox.critical(self, "Export Failed", f"{type(ex).__name__}: {ex}")
                return
            print(f"[Export] Folder → {target}")
            self.last_dir = dest_root
            try:
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(target))
            except Exception:
                pass
            QMessageBox.information(self, "Export Complete", f"Exported folder to:\n{target}")
            return

        # mode == "file": save ONLY the current data file (e.g., STL), like before.
        base = os.path.basename(src_file)
        # Choose a sensible filter by kind
        kind = (self.current_kind or "").lower()
        if kind == "stl":
            filt = "STL Mesh (*.stl)"; suggested = os.path.join(start_dir, base if base.lower().endswith(".stl") else os.path.splitext(base)[0] + ".stl")
        elif kind == "nifti":
            filt = "NIfTI (*.nii *.nii.gz)"; suggested = os.path.join(start_dir, base if (base.lower().endswith(".nii") or base.lower().endswith(".nii.gz")) else os.path.splitext(base)[0] + ".nii.gz")
        elif kind == "image":
            # just copy the source image; if you want re-encode, add logic here
            filt = "Image (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"; suggested = os.path.join(start_dir, base)
        else:
            # generic data
            filt = "All Files (*.*)"; suggested = os.path.join(start_dir, base)

        path, _ = QFileDialog.getSaveFileName(self, "Save Current File As…", suggested, filt)
        if not path:
            return

        # Ensure extension for some types
        if kind == "stl" and not path.lower().endswith(".stl"):
            path += ".stl"
        if kind == "nifti" and not (path.lower().endswith(".nii") or path.lower().endswith(".nii.gz")):
            path += ".nii.gz"

        try:
            shutil.copy2(src_file, path)
            print(f"[Export] File → {path}")
            self.last_dir = os.path.dirname(path)
            QMessageBox.information(self, "Export Complete", f"Saved:\n{path}")
        except Exception as ex:
            QMessageBox.critical(self, "Export Failed", f"{type(ex).__name__}: {ex}")


    def close_current(self):
        self.image_label.clearImage(); self._show_widget(self.image_label); self._set_slice_controls(False)
        self._set_current(None, None); print("Closed current file and reset view.")

    def quit_app(self):
        print("Quitting application."); self.close()
        

    def _ensure_metric_row(
        self,
        path: Optional[str],
        kind: Optional[str],
        label: Optional[str] = None,
        *,
        pixel_size: Optional[float] = None,
        pixel_size_units: Optional[str] = None,
        kernel_size: Optional[float] = None,
        unite:  Optional[str] = None,
        slice_thickness: Optional[float] = None,
        new_on_param_change: bool = False,
    ):
        """
        Ensure a metrics row exists for (path, label).

        - Storage: self.metrics: dict[str, list[dict]]  # path -> list of row dicts
        - If new_on_param_change=True and any of PixelSize/PixelSizeUnits/KernelSize
          differs from the most recent row for this (path, label), append a NEW row.
        - Returns the (existing or newly created) row dict.
        """
        if not path:
            return None

        if not hasattr(self, "metrics") or self.metrics is None:
            self.metrics = {}

        rows = self.metrics.get(path)

        # Backward compatibility: single dict -> list[dict]
        if isinstance(rows, dict):
            rows = [rows]
            self.metrics[path] = rows
        elif rows is None:
            rows = []
            self.metrics[path] = rows

        # Most recent row for this label (if any)
        last = next((r for r in reversed(rows) if r.get("Label") == label), None)

        def differs(key: str, new_val):
            if new_val is None:
                return False
            if last is None:
                return True  # creating first row and user provided a value
            return new_val != last.get(key)

        make_new = (
            last is None
            or (new_on_param_change and any([
                differs("PixelSize",       pixel_size),
                differs("PixelSizeUnits",  pixel_size_units),
                differs("KernelSize",      kernel_size),
                differs("SliceThickness",      slice_thickness),
                differs("LengthUnit",      unite),
                
            ]))
        )

        if make_new:
            row = {
                "File": os.path.basename(path),
                "Kind": kind,
                "Label": label,
                "Area": None,
                "PixelSize":       pixel_size if pixel_size is not None else (last.get("PixelSize") if last else None),
                "PixelSizeUnits":  pixel_size_units if pixel_size_units is not None else (last.get("PixelSizeUnits") if last else None),
                "KernelSize":      kernel_size if kernel_size is not None else (last.get("KernelSize") if last else None),
                "LengthUnit":      unite if unite is not None else (last.get("LengthUnit") if last else None),
                "SliceThickness":  slice_thickness if slice_thickness is not None else (last.get("SliceThickness") if last else None),
                "Volume": None,
                "Perimeter": None,
                "Perimeter_convex": None,
                "SulciCount": None,
                "MinDepth": None,
                "MaxDpeth": None,
                "MeanDepth": None,
                "LGI": None,
            }
            rows.append(row)
            return row

        # Otherwise: refresh/update the latest row and return it
        last["File"] = os.path.basename(path)
        if kind is not None:
            last["Kind"] = kind
        if label is not None:
            last["Label"] = label
        if pixel_size is not None:
            last["PixelSize"] = pixel_size
        if pixel_size_units is not None:
            last["PixelSizeUnits"] = pixel_size_units
        if kernel_size is not None:
            last["KernelSize"] = kernel_size
        if unite is not None:
            last["LengthUnit"] = unite
        if slice_thickness is not None:
            last["SliceThickness"] = slice_thickness
        return last

        
    def _record_metric_for(self, path: str, label: Optional[str] = None, **vals):
        if not path:
            return
        kind = getattr(self, "current_kind", None)

        # Extract the triple so _ensure_metric_row can decide whether to create a new row
        psize = vals.pop("pixel_size", None)
        punit = vals.pop("pixel_size_units", None)
        ksize = vals.pop("kernel_size", None)
        thicsl = vals.pop("slice_thickness", None)
        uni = vals.pop("unite", None)
        row = self._ensure_metric_row(
            path, kind, label,
            pixel_size=psize,
            pixel_size_units=punit,
            kernel_size=ksize,
            unite = uni,
            slice_thickness = thicsl,
            new_on_param_change=True,
        )

        # Optional: accept 'sulci_depth' as a 3-tuple/list
        sd = vals.pop("sulci_depth", None)
        if sd is not None:
            if isinstance(sd, (list, tuple)):
                n = len(sd)
                if n == 0:
                    row["SulciCount"] = row["MinDepth"] = row["MaxDpeth"] = row["MeanDepth"]= None
                else:
                    row["SulciCount"] = n
                    row["MinDepth"] = min(sd)
                    row["MaxDpeth"] = max(sd)
                    row["MeanDepth"] = sum(sd)/n
                    
            else:
                raise ValueError("sulci_depth must be an iterable")

        # Map remaining friendly keys to columns
        keymap = {
            "area": "Area",
            "volume": "Volume",
            "perimeter": "Perimeter",
            "perimeter_convex": "Perimeter_convex",
            "lgi": "LGI",
            # triple handled above
        }
        for k, v in vals.items():
            col = keymap.get(k.lower(), k)
            row[col] = v


    # ---------- Loaders ----------
    def load_image(self, path: str):
        pm = QPixmap(path)
        if pm.isNull(): QMessageBox.critical(self, "Open Failed", "Could not read image file."); return
        self.image_label.setImage(pm); self._show_widget(self.image_label); self._set_slice_controls(False)
        print(f"Loaded image: {path}  size={pm.width()}x{pm.height()}"); self._set_current("image", path)

    def load_nifti(self, path: str):
        rdr = vtkNIFTIImageReader(); rdr.SetFileName(path); rdr.Update(); img = rdr.GetOutput()
        self.vtk_view.show_image2d(img); self._show_widget(self.vtk_view); self._sync_slice_controls()
        print(f"NIfTI loaded:\n"
              f"Extent={img.GetExtent()} \n Spacing={img.GetSpacing()} \n  Range={img.GetScalarRange()}")
        self._set_current("nifti", path)

    def load_stl(self, path: str):
        r = vtkSTLReader(); r.SetFileName(path); r.Update(); poly = r.GetOutput()
        if not poly or poly.GetNumberOfPoints()==0:
            QMessageBox.critical(self,"Open Failed","Empty or invalid .stl file."); return
        self.vtk_view.show_polydata(poly); self._show_widget(self.vtk_view); self._set_slice_controls(False)
        print(f"STL loaded:\n"
              f"  File: {path}\n"
              f"  Points: {poly.GetNumberOfPoints():,}\n"
              f"  Polys:  {poly.GetNumberOfPolys():,}")
        self._set_current("stl", path)

    def load_vtk(self, path: str):
        dsr = vtkDataSetReader(); dsr.SetFileName(path); dsr.Update(); ds = dsr.GetOutput()
        if ds is None:
            gr = vtkGenericDataObjectReader(); gr.SetFileName(path); gr.Update(); ds = gr.GetOutput()
        if isinstance(ds, vtkImageData):
            self.vtk_view.show_image2d(ds); self._show_widget(self.vtk_view); self._sync_slice_controls()
            print(f"Legacy VTK image loaded. Extent={ds.GetExtent()}  Range={ds.GetScalarRange()}"); self._set_current("vtk_image", path); return
        if isinstance(ds, vtkPolyData) and ds.GetNumberOfPoints()>0:
            self.vtk_view.show_polydata(ds); self._show_widget(self.vtk_view); self._set_slice_controls(False)
            print(f"Legacy VTK polydata loaded. Points={ds.GetNumberOfPoints()}  Polys={ds.GetNumberOfPolys()}"); self._set_current("vtk_poly", path); return
        surf = vtkDataSetSurfaceFilter(); surf.SetInputData(ds); surf.Update(); poly = surf.GetOutput()
        if poly and poly.GetNumberOfPoints()>0:
            self.vtk_view.show_polydata(poly); self._show_widget(self.vtk_view); self._set_slice_controls(False)
            print(f"Legacy VTK dataset surfaced. Points={poly.GetNumberOfPoints()}  Polys={poly.GetNumberOfPolys()}"); self._set_current("vtk_surface", path); return
        QMessageBox.critical(self, "Open Failed", "Unsupported or empty .vtk dataset (no points after surface extraction).")
        
    # -------------- Export to Excel -------------------------
    def export_metrics_excel(self):
        """Export collected metrics (File, Kind, Label, PixelSize, PixelSizeUnits, KernelSize,
        Area, Volume, Perimeter, Perimeter_convex, SulciDepth_P1..3, LGI) to an Excel .xlsx file.
        Works with: self.metrics: dict[str, list[dict]]
        """
        if not getattr(self, "metrics", None):
            QMessageBox.information(self, "Export Metrics", "No metrics to export yet.")
            return

        # Define columns in the order you want them in Excel
        base_cols = ["File", "Kind"]
        metric_cols = [
            "Label",
            "PixelSize", "PixelSizeUnits", "KernelSize","LengthUnit",
            "Area", "Volume", "Perimeter", "Perimeter_convex",
            "SulciCount", "MinDepth", "MaxDpeth","MeanDepth",
            "LGI",
        ]
        
        cols = base_cols + metric_cols

        # Flatten: path -> [rows]  ==>  list of row dicts
        flat_rows = []
        for _path, rows in self.metrics.items():
            if rows is None:
                continue
            if isinstance(rows, dict):
                rows = [rows]  # backward-compat safeguard
            for row in rows:
                # Keep only known columns; missing keys become NaN in DataFrame
                flat_rows.append({c: row.get(c) for c in cols})

        if not flat_rows:
            QMessageBox.information(self, "Export Metrics", "No metrics to export yet.")
            return

        # Build DataFrame (requires pandas)
        try:
            import pandas as pd
        except Exception:
            QMessageBox.critical(
                self,
                "Export Metrics",
                "Pandas is required to export to Excel.\nInstall with:\n  pip install pandas openpyxl"
            )
            return

        df = pd.DataFrame(flat_rows, columns=cols)

        # Keep only rows that have at least one *real* metric filled in
        # (exclude Label and PixelSizeUnits; keep PixelSize/KernelSize and numeric metrics)
        real_metric_cols = [
            "PixelSize", "KernelSize", "LengthUnit",
            "Area", "Volume", "Perimeter", "Perimeter_convex",
            "SulciCount", "MinDepth", "MaxDpeth","MeanDepth",
            "LGI",
        ]
        has_any_metric = df[real_metric_cols].notna().any(axis=1)
        df = df.loc[has_any_metric].copy()

        if df.empty:
            QMessageBox.information(self, "Export Metrics", "No non-empty metrics to export yet.")
            return

        # Drop metric columns that are entirely empty across remaining rows
        drop_all_null = [c for c in real_metric_cols + ["Label", "PixelSizeUnits"] if c in df.columns and df[c].isna().all()]
        if drop_all_null:
            df.drop(columns=drop_all_null, inplace=True)

        # Choose file path
        default_name = os.path.join(getattr(self, "last_dir", os.getcwd()), "metrics.xlsx")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics to Excel…", default_name, "Excel Workbook (*.xlsx)"
        )
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            reply = QMessageBox.question(
                self, "Create Folder?",
                f"The folder\n\n{folder}\n\ndoes not exist. Create it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                try:
                    os.makedirs(folder, exist_ok=True)
                    print(f"Created folder: {folder}")
                except Exception as ex:
                    print(f"ERROR creating folder: {ex}")
                    QMessageBox.critical(self, "Export Failed", f"Could not create folder:\n{ex}")
                    return
            else:
                return

        # Write Excel
        try:
            df.to_excel(path, index=False)  # uses openpyxl/xlsxwriter if installed
            print(f"Exported metrics to: {path}")
            if folder:
                self.last_dir = folder
        except Exception as ex:
            print(f"ERROR exporting metrics: {ex}")
            QMessageBox.critical(self, "Export Failed", f"{type(ex).__name__}: {ex}")

    # ---------- Process menu (stubs) ----------
    def on_measure_allmarks(self):
        """Process → Measures → All hallmarks: compute and show annotated result WITHOUT saving."""
        if not self.current_path or not os.path.isfile(self.current_path):
            print("[All hallmarks] No image file is loaded."); return
        if self.current_kind == "image":
            try:
                if not self.units_length or self.current_path not in self.image_scales:
                    self.set_image_scale()  # pops the single dialog
                u = self.ensure_units()                # <-- get unit (e.g., 'mm')
                px_size = self.image_scales.get(self.current_path, self.pixel_size)

                print(f"[All hallmarks] Measuring: {self.current_path}")
                print(f"[All hallmarks] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.current_path
                if self.last_annotated_path is not None:
                    image_path = self.last_annotated_path
                    
                area, perimeter, perimeter_convex, lGI, depth, annotated_bgr = measure_image_allmarks(
                    image_path,
                    pixel_size=px_size,
                    kernel_size= self.kernel_size,
                    cnt_threshold=self.cnt_threshold,
                    unit = u,
                )
                
                print(f"Annotated area = {area:.2f} {u}^2.")
                print(f"Annotated Perimeter = {perimeter:.2f} {u}.")
                print(f"Convex Perimeter = {perimeter_convex:.2f} {u}.")
                print(f"LGI (Convex Perimeter/ Perimeter) = {lGI:.2f} .")
                print(f"Sulci Depth = {depth[0]:.2f}, {depth[1]:.2f}, {depth[2]:.2f} {u}.")
                
                # Convert BGR ndarray → QPixmap and show (no disk write)
                label_text = self.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")


                pm = self._np_bgr_to_qpixmap(annotated_bgr)
                self.image_label.setImage(pm)
                self.image_label.remove_last_annotation()
                self._show_widget(self.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self._active_view = "image"
                # Ensure File/Process actions stay enabled
                self._set_current("image", self.current_path)
                self._record_metric_for(
                    self.current_path,
                    label = label_text,
                    pixel_size_units = f"{self.units_length}/pixel",
                    unite = self.units_length,
                    pixel_size = self.pixel_size,
                    kernel_size = self.kernel_size,
                    area=area,
                    perimeter=perimeter,
                    perimeter_convex = perimeter_convex,
                    lgi=lGI,
                    sulci_depth = depth)

            except Exception as ex:
                print(f"[All hallmarks] ERROR: {ex}")
                QMessageBox.critical(self, "All hallmarks Failed", f"{type(ex).__name__}: {ex}")
        elif self.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.current_path
                print(f"[NIfTI] Computing area/perimeter from: {nif_path}")

                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"nifti_allmarks_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                area, volume, gi, depth, saved_pngs, valid_slices = compute_nifti_allmarks(file_path=nif_path,
                out_dir=out_dir, min_contour_area=self.cnt_threshold, kernel_size = self.kernel_size)
            
                if area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    kernel_size = self.kernel_size,
                    unite = "cm",
                    volume=volume,
                    area=area,
                    lgi=gi,
                    sulci_depth = depth)
                self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.on_slice_slider_changed(mid)
                
                print(f"[NIfTI hallmarks]:")
                print("The Brain Volume Result = {volume:.2f} cm^3.")
                print(f"The Brain Outer Surface Area Result = {area:.2f} cm^2.")
                print(f"The Brain GI (Convex surface area/ surfacearea) = {gi:.2f} .")
                print(f"The Maximum Sulci Depth = {depth[0]:.2f}, {depth[1]:.2f}, {depth[2]:.2f} cm.")
                dt = time.time() - t0
                print(f"[NIfTI Area] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[NIfTI hallmarks] ERROR: {ex}")
                QMessageBox.critical(self, "NIfTI Area Failed", f"{type(ex).__name__}: {ex}")
            return
            
    def on_measure_volumes(self):
        if not self.current_path or not os.path.isfile(self.current_path):
            print("[Volume] No file is loaded."); return
        if self.current_kind == "image":
            print("[Volume] Implemented for 3D objects only."); return
        
        elif self.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.current_path
                print(f"[NIfTI] Computing Volume from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"nifti_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                volume,saved_pngs, valid_slices = compute_nifti_volume(file_path=nif_path, out_dir=out_dir,)
            
                if volume is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(self.current_path, unite="cm",volume = volume,)

                self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.on_slice_slider_changed(mid)
                
                print(f"[NIfTI Volume] The Brain Volume Result = {volume:.2f} cm^3. ")
                dt = time.time() - t0
                print(f"[NIfTI Volume] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"ERROR (NIfTI Volume): {ex}")
                QMessageBox.critical(self, "NIfTI Volume Failed", f"{type(ex).__name__}: {ex}")
            return
        
        else:
            print("[Volume] Unsupported current kind. Open an image or NIfTI first.")

    
    def on_measure_perimeter(self):
        """Process → Measures → Perimeter: compute and show annotated result WITHOUT saving."""
        if not self.current_path or not os.path.isfile(self.current_path):
            print("[Perimeter] No file is loaded."); return
        
        if self.current_kind == "image":
            try:
                if not self.units_length or self.current_path not in self.image_scales:
                    self.set_image_scale()  # pops the single dialog
                u = self.ensure_units()                # <-- get unit (e.g., 'mm')
                px_size = self.image_scales.get(self.current_path, self.pixel_size)
                
                print(f"[Perimeter] Measuring: {self.current_path}")
                print(f"[Perimeter] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.current_path
                if self.last_annotated_path is not None:
                    image_path = self.last_annotated_path
                
                perimeter, annotated_bgr = measure_image_perimeter(
                    image_path,
                    pixel_size = px_size,
                    cnt_threshold = self.cnt_threshold,
                    unit = u,
                )
                print(f"Annotated perimeter = {perimeter:.2f} {u}.")
                
                label_text = self.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")
                # Convert BGR ndarray → QPixmap and show (no disk write)
                pm = self._np_bgr_to_qpixmap(annotated_bgr)
                self.image_label.setImage(pm)
                self.image_label.remove_last_annotation()
                self._show_widget(self.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self._active_view = "image"
                # Ensure File/Process actions stay enabled
                self._set_current("image", self.current_path)
                self._record_metric_for(self.current_path,label=label_text,
                pixel_size_units = f"{self.units_length}/pixel",
                unite= self.units_length,
                pixel_size = self.pixel_size,
                perimeter=perimeter)

            except Exception as ex:
                print(f"[Perimeter] ERROR: {ex}")
                QMessageBox.critical(self, "Perimeter Failed", f"{type(ex).__name__}: {ex}")
            
        else:
            return
            
    def on_measure_lgi(self):
        """Process → Measures → lGI: compute and show annotated result WITHOUT saving."""
        
        if not self.current_path or not os.path.isfile(self.current_path):
            print("[lGI] No file is loaded."); return
        if self.current_kind == "image":
            try:
            
                if not self.units_length or self.current_path not in self.image_scales:
                    self.set_image_scale()  # pops the single dialog
                u = self.ensure_units()                # <-- get unit (e.g., 'mm')
                px_size = self.image_scales.get(self.current_path, self.pixel_size)
                
                print(f"[lGI] Measuring: {self.current_path}")

                image_path = self.current_path
                if self.last_annotated_path is not None:
                    image_path = self.last_annotated_path
                lGI,perimeter, perimeter_convex, annotated_bgr = measure_image_lGI(
                    image_path,
                    pixel_size = px_size,
                    kernel_size= self.kernel_size,
                    cnt_threshold=self.cnt_threshold,
                    unit = u
                )
                print(f"lGI = {lGI:.2f}.")

                label_text = self.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")
                # Convert BGR ndarray → QPixmap and show (no disk write)
                pm = self._np_bgr_to_qpixmap(annotated_bgr)
                self.image_label.setImage(pm)
                self.image_label.remove_last_annotation()
                self._show_widget(self.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self._active_view = "image"
                # Ensure File/Process actions stay enabled
                self._set_current("image", self.current_path)
                self._record_metric_for(self.current_path, label=label_text,
                pixel_size_units = f"{self.units_length}/pixel",
                unite = self.units_length,
                pixel_size = self.pixel_size,
                kernel_size = self.kernel_size,
                perimeter=perimeter, perimeter_convex=perimeter_convex, lgi=lGI)

            except Exception as ex:
                print(f"[lGI] ERROR: {ex}")
                QMessageBox.critical(self, "lGI Failed", f"{type(ex).__name__}: {ex}")
                
        elif self.current_kind == "nifti":
            t0 = time.time()
            reply = QMessageBox.question(self,"Enhance measurement",
            "For accurate LGI computation, please provide the FreeSurfer pial surfaces for both hemispheres (lh.pial and rh.pial). Do you have these files?",   # message
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.No:
                QMessageBox.warning(self, "LGI Input Missing",
                    "The LGI can be computed based on the NIfTI file alone, but the accuracy of the results is not guaranteed.")
                
                try:
                    nif_path = self.current_path
                    print(f"[NIfTI] Computing lGI from: {nif_path}")


                    uid = uuid.uuid4().hex[:8]
                    out_dir = os.path.join(self.temp_dir, f"nifti_lGI_{uid}")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    self.current_output_dir = out_dir
                    lGI,saved_pngs, valid_slices = compute_nifti_lGI(file_path=nif_path, out_dir=out_dir,min_contour_area=self.cnt_threshold, kernel_size= self.kernel_size,)
                
                    if lGI is None:
                        return

                    # record metrics (consistent with your global export; units in mm unless noted)
                    self._record_metric_for(self.current_path, kernel_size= self.kernel_size ,lgi = lGI,)

                    self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                    
                    mid = len(saved_pngs) // 2
                    self.on_slice_slider_changed(mid)
                    
                    print(f"[NIfTI lGI] The Brain GI (Convex surface area/ surfacearea) = {lGI:.2f}. ")
                    dt = time.time() - t0
                    print(f"[NIfTI lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                          f"Use File → Save Data As… to copy outputs you want to keep.")

                except Exception as ex:
                    print(f"[NIfTI lGI] ERROR: {ex}")
                    QMessageBox.critical(self, "NIfTI lGI Failed", f"{type(ex).__name__}: {ex}")
                return
        
            elif reply == QMessageBox.Yes:
                nif_path = self.current_path
#                QTimer.singleShot(0, self.on_combined_stl)
                self.on_combined_stl()
                stl_path = self.current_path if (self.current_path and os.path.isfile(self.current_path)) else None
                
                try:
                    print(f"[NIfTI] Computing lGI from: {nif_path} based on rh & lh .pial")

                    uid = uuid.uuid4().hex[:8]
                    out_dir = os.path.join(self.temp_dir, f"STL_lGI_{uid}")
                    os.makedirs(out_dir, exist_ok=True)
                    
                    self.current_output_dir = out_dir
                    lGI,saved_pngs, valid_slices =compute_stl_lGI(
                        file_path=stl_path,
                        out_dir=out_dir,
                        cnt_threshold=self.cnt_threshold,
                        kernel_size=self.kernel_size,
                        slice_thickness=self.slice_thickness,
                        build_solid=False,   # keep False for stability
                    )
                                        
                
                    if lGI is None:
                        return

                    # record metrics (consistent with your global export; units in mm unless noted)
                    self._record_metric_for(self.current_path,
                            slice_thickness =self.slice_thickness,
                            kernel_size =self.kernel_size, lgi = lGI,)

                    self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                    
                    mid = len(saved_pngs) // 2
                    self.on_slice_slider_changed(mid)
                    
                    print(f"[STL lGI] The Brain GI (Convex surface area/ surfacearea) = {lGI:.2f}. ")
                    dt = time.time() - t0
                    print(f"[STL lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                          f"Use File → Save Data As… to copy outputs you want to keep.")

                except Exception as ex:
                    print(f"[STL lGI] ERROR: {ex}")
                    QMessageBox.critical(self, "STL lGI Failed", f"{type(ex).__name__}: {ex}")
                return
            
        else:
            print("[lGI] Unsupported current kind. Open an image or NIfTI first.")

            
        
    
    def on_measure_sulci_depth(self):
        """Process → Measures → All hallmarks for 2D images: compute and show annotated result WITHOUT saving."""
        if not self.current_path or not os.path.isfile(self.current_path):
            print("[Sulci depth] No file is loaded."); return
            
        if self.current_kind == "image":

            try:
                if not self.units_length or self.current_path not in self.image_scales:
                    self.set_image_scale()  # pops the single dialog
                u = self.ensure_units()                # <-- get unit (e.g., 'mm')
                px_size = self.image_scales.get(self.current_path, self.pixel_size)
                
                print(f"[Sulci depth] Measuring: {self.current_path}")
                print(f"[Sulci depth] Measuring with pixel size = {px_size} {u}/pixel")

                image_path = self.current_path
                if self.last_annotated_path is not None:
                    image_path = self.last_annotated_path
                depth, annotated_bgr = measure_image_sulci_depth(
                    image_path,
                    pixel_size = px_size,
                    cnt_threshold=self.cnt_threshold,
                )
                print(f"Sulci Depth = {depth[0]:.2f}, {depth[1]:.2f}, {depth[2]:.2f}. {u}")
                
                label_text = self.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")


                # Convert BGR ndarray → QPixmap and show (no disk write)
                pm = self._np_bgr_to_qpixmap(annotated_bgr)
                self.image_label.setImage(pm)
                self.image_label.remove_last_annotation()
                self._show_widget(self.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self._active_view = "image"
                # Ensure File/Process actions stay enabled
                self._set_current("image", self.current_path)
                self._record_metric_for(self.current_path, label=label_text,
                    pixel_size_units = f"{self.units_length}/pixel",
                    unite = self.units_length,
                    pixel_size = self.pixel_size,
                    sulci_depth = depth)

            except Exception as ex:
                print(f"[Sulci depth] ERROR: {ex}")
                QMessageBox.critical(self, "Sulci depth Failed", f"{type(ex).__name__}: {ex}")
        
        elif self.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.current_path
                print(f"[NIfTI] Computing Volume from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"nifti_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                depth,saved_pngs, valid_slices = compute_nifti_sulci_depth(file_path=nif_path, out_dir=out_dir, min_contour_area=self.cnt_threshold)
            
                if depth is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(self.current_path, unite ="mm",sulci_depth = depth,)

                self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.on_slice_slider_changed(mid)
                
                print(f"[NIfTI Sulci depth] The max Brain Sulci depth across slices = {depth[0]:.2f}, {depth[1]:.2f}, {depth[2]:.2f}. mm")
                dt = time.time() - t0
                print(f"[NIfTI Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[NIfTI Sulci depth] ERROR: {ex}")
                QMessageBox.critical(self, "NIfTI Sulci depth Failed", f"{type(ex).__name__}: {ex}")
            return
        else:
            print("[Sulci depth] Unsupported current kind. Open an image or NIfTI first.")

            
            
    
    def on_measure_area(self):
        """Process → Measures → Area: compute and show annotated result WITHOUT saving."""
        if self.current_kind == "image":
            if not self.current_path or not os.path.isfile(self.current_path):
                print("[Area] No image file is loaded."); return
            try:
                if not self.units_length or self.current_path not in self.image_scales:
                    self.set_image_scale()  # pops the single dialog
                u = self.ensure_units()                # <-- get unit (e.g., 'mm')
                px_size = self.image_scales.get(self.current_path, self.pixel_size)

                print(f"[Area] Measuring: {self.current_path}")
                print(f"[Area] Measuring with pixel size = {px_size} {u}/pixel")
                
                image_path = self.current_path
                if self.last_annotated_path is not None:
                    image_path = self.last_annotated_path
                    
                area, annotated_bgr = measure_image_area(
                    image_path,
                    pixel_size=px_size,
                    cnt_threshold=self.cnt_threshold,
                    unit = u,
                )
                
                label_text = self.get_label_for_cropped_path(image_path)
                if label_text:
                    annotated_bgr = put_label_on_bgr(annotated_bgr, label_text, pos="topleft")
                    
                print(f"[Area] Result = {area:.2f} {u}^2.")
                # Convert BGR ndarray → QPixmap and show (no disk write)
                pm = self._np_bgr_to_qpixmap(annotated_bgr)
                self.image_label.setImage(pm)
                self.image_label.remove_last_annotation()
                self._show_widget(self.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self._active_view = "image"
                # Ensure File/Process actions stay enabled
                self._set_current("image", self.current_path)
                self._record_metric_for(self.current_path, label=label_text ,
                pixel_size_units = f"{self.units_length}/pixel",
                pixel_size = self.pixel_size,
                unite = self.units_length,
                area=area)

            except Exception as ex:
                print(f"[Area] ERROR : {ex}")
                QMessageBox.critical(self, "[Area] Failed", f"{type(ex).__name__}: {ex}")
        elif self.current_kind == "nifti":
            t0 = time.time()
            try:
                nif_path = self.current_path
                print(f"[NIfTI] Computing area/perimeter from: {nif_path}")


                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"nifti_area_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                area,saved_pngs, valid_slices = compute_nifti_arae(file_path=nif_path, out_dir=out_dir, min_contour_area=self.cnt_threshold,)
            
                if area == 0:
                    QMessageBox.information(self, "NIfTI Area", "All slices were filtered out (too small).")
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(self.current_path, unite="cm",area = area,)

                self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.on_slice_slider_changed(mid)
                
                print(f"[NIfTI Area] The Brain Outer Surface Area Result = {area:.2f} cm^2. ")
                dt = time.time() - t0
                print(f"[NIfTI Area] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[NIfTI Area] ERROR: {ex}")
                QMessageBox.critical(self, "[NIfTI Area] Failed", f"{type(ex).__name__}: {ex}")
            return

        else:
            print("[Area] Unsupported current kind. Open an image or NIfTI first.")

    def on_optimization(self): pass
            
    def on_pial_to_stl(self):
        """Pick one .pial, convert to STL in TEMP, show it, and keep source in metrics."""
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        pial, _ = QFileDialog.getOpenFileName(self, "Select FreeSurfer Pial Surface",
                                              start, "FreeSurfer Surface (*.pial);;All Files (*)")
        if not pial:
            return

        # Save to TEMP (don’t pester user yet)
        uid = uuid.uuid4().hex[:8]
        base = os.path.splitext(os.path.basename(pial))[0]
        temp_out = os.path.join(self.temp_dir, f"{base}_{uid}.stl")

        try:
            print(f"[Pial → STL] TEMP output: {temp_out}")
            saved = pial_to_stl(pial, temp_out)

            # Show it immediately
            self.load_stl(saved)                   # shows in VTK window
            self._ensure_metric_row(saved, "stl")  # track result row, path = temp file
            self.metrics[saved]["SourcePial"] = pial
            print("[Pial → STL] Hint: use File → Save Data As… to keep a permanent copy.")
        except Exception as ex:
            print(f"[Pial → STL] ERROR: {ex}")
            QMessageBox.critical(self, "Pial → STL", f"{type(ex).__name__}: {ex}")
        
    def on_combined_stl(self):
        """Pick rh & lh .pial, convert + merge in TEMP, show combined STL, record provenance."""
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        rh, _ = QFileDialog.getOpenFileName(self, "Select RIGHT hemisphere (rh.pial)",
                                            start, "FreeSurfer Surface (*.pial);;All Files (*)")
        if not rh:
            return
        lh, _ = QFileDialog.getOpenFileName(self, "Select LEFT hemisphere (lh.pial)",
                                            os.path.dirname(rh), "FreeSurfer Surface (*.pial);;All Files (*)")
        if not lh:
            return

        # TEMP output
        uid = uuid.uuid4().hex[:8]
        temp_out = os.path.join(self.temp_dir, f"brain_both_{uid}.stl")

        try:
            print(f"[Combined STL] TEMP output (combined): {temp_out}")
            saved = pial_pair_to_combined_stl(rh, lh, temp_out)

            # Show combined STL
            self.load_stl(saved)
            self._ensure_metric_row(saved, "stl")
            self.metrics[saved]["SourcePial_RH"] = rh
            self.metrics[saved]["SourcePial_LH"] = lh
            print("[Combined STL] Combined STL loaded. Use File → Save Data As… to export.")
        except Exception as ex:
            print(f"[Combined STL] ERROR: {ex}")
            QMessageBox.critical(self, "Pial (rh & lh) → Combined STL", f"{type(ex).__name__}: {ex}")
        
    # -------------- Setting functions ---------------
    def ensure_units(self) -> str:
        """
        Ensure a length unit string exists (e.g., 'mm').
        Prompts the user once per session. Also records unit columns for the current file's metrics.
        """
        if self.units_length:
            return self.units_length

        val, ok = QInputDialog.getText(
            self,
            "Set Units",
            "Length unit (e.g., mm, µm, cm):",
            text="mm",
        )
        if not ok or not val.strip():
            # if user cancels, keep default
            val = "mm"
        self.units_length = val.strip()

        print(f"[Units] Using {self.units_length}")
        return self.units_length

    
    
    def set_image_scale(self):
        """
        One dialog to set BOTH the length unit (e.g., mm) and pixel size (unit/pixel)
        for the *current file*. Updates per-file scale, current scale, and metrics.
        """
        if not self.current_path:
            QMessageBox.information(self, "Set Units & Pixel Size", "Load a file first.")
            return

        unit_init = self.units_length or "mm"
        px_init = float(self.image_scales.get(self.current_path, getattr(self, "pixel_size", 0.03)))

        dlg = UnitScaleDialog(self, unit_init=unit_init, pixel_size_init=px_init)
        if dlg.exec() != QDialog.Accepted:
            return

        unit, scale = dlg.values()
        if not (scale > 0):
            QMessageBox.warning(self, "Invalid value", "Pixel size must be a positive number.")
            return

        # Store for this file and as current
        self.units_length = unit
        self.image_scales[self.current_path] = scale
        self.pixel_size = scale

        # Track in metrics (so Excel shows context)
        label_text = self.get_label_for_cropped_path(self.last_annotated_path)
        self._record_metric_for (self.current_path, label= label_text  ,pixel_size= scale, pixel_size_units=f"{unit}/pixel" )
#        row = self._ensure_metric_row(self.current_path, self.current_kind)
#        row = self.metrics[self.current_path]
#        row["PixelSize"] = scale
#        row["PixelSizeUnits"] = f"{unit}/pixel"

        print(f"[Units] {unit}  |  [Scale] {scale} {unit}/pixel  —  {os.path.basename(self.current_path)}")

    def set_scale_from_scalebar(self):
        """Activate line measurement on the image; on release we ask real length+unit."""
        if self.current_kind != "image":
            QMessageBox.information(self, "Set Scale", "Open a 2D image to set scale from a scalebar.")
            return
        if self.image_label is None or self.image_label._pix.isNull():
            QMessageBox.information(self, "Set Scale", "No image visible.")
            return
        print("[Scale] Draw a line over the scalebar: click, drag, release.")
        self.image_label.start_scalebar_measure(self._finish_scalebar_scale)

    def _finish_scalebar_scale(self, pixel_length: float):
        """Called after the user drags a line; asks for real length & unit, computes px/unit."""
        try:
            unit_init = self.units_length or "mm"
            dlg = ScalebarSetScaleDialog(pixel_length, unit_init=unit_init, parent=self)
            if dlg.exec() != QDialog.Accepted:
                print("[Scale] Canceled.")
                return
            px_per_unit, unit = dlg.values()  # e.g., px/mm
            if px_per_unit <= 0:
                QMessageBox.warning(self, "Set Scale", "Scale must be positive.")
                return

            # Store: keep px/mm per file; keep working mm/pixel for algorithms
            self.units_length = unit
            mm_per_px = 1.0 / px_per_unit
            self.pixel_size = mm_per_px
            self.image_scales[self.current_path] = float(mm_per_px)      # unit/px


            # Record in metrics
            label_text = self.get_label_for_cropped_path(self.last_annotated_path)
            self._record_metric_for (self.current_path, label = label_text, pixel_size= mm_per_px, pixel_size_units=f"{unit}/pixel" )

            print(f"[Scale] {pixel_length:.2f} px = {px_per_unit:.6f} px/{unit}  "
                  f"→ pixel size {mm_per_px:.6f} {unit}/pixel for {os.path.basename(self.current_path)}")
        except Exception as ex:
            print(f"ERROR (Set Scale): {ex}")
            QMessageBox.critical(self, "Set Scale Failed", f"{type(ex).__name__}: {ex}")



    def set_kernel_dialog(self):
        """Open dialog to set morphology kernel size (odd)."""
        dlg = KernelSizeDialog(self, initial=getattr(self, "kernel_size", 5))
        if dlg.exec() == QDialog.Accepted:
            k = dlg.value()
            self.kernel_size = k
            print(f"[Kernel] Set morphology kernel size to {k}")
            # Record in metrics
            if self.current_path:
                label_text = self.get_label_for_cropped_path(self.last_annotated_path)
                self._record_metric_for(self.current_path, label = label_text, kernel_size= self.kernel_size)
    
    def set_cnt_threshold_dialog(self):
        dlg = ContourThresholdDialog(self, initial=getattr(self, "cnt_threshold", 50.0))
        if dlg.exec() == QDialog.Accepted:
            val = dlg.value()
            self.cnt_threshold = max(0.0, float(val))
            print(f"[Threshold] Contour area threshold set to {self.cnt_threshold:.0f} px")



    # ---------- Utils ----------
    def _np_bgr_to_qpixmap(self, arr: np.ndarray) -> QPixmap:
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected HxWx3 BGR array.")
        h, w, _ = arr.shape
        rgb = arr[:, :, ::-1].copy(order="C")
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # ---------- Plumbing ----------
    def _show_widget(self, w: QWidget):
        self.image_label.setVisible(False); self.vtk_view.setVisible(False); w.setVisible(True)
        self._active_view = "image" if w is self.image_label else "vtk"

    def _append_progress(self, text: str):
        self.progress_edit.moveCursor(QTextCursor.End); self.progress_edit.insertPlainText(text); self.progress_edit.moveCursor(QTextCursor.End)

    def closeEvent(self, e):
        sys.stdout = self._orig_stdout; sys.stderr = self._orig_stderr
        try: shutil.rmtree(self.temp_dir, ignore_errors=True); print(f"[Temp] Cleaned: {self.temp_dir}")
        except Exception as ex: print(f"[Temp] Cleanup error: {ex}")
        super().closeEvent(e)

    # ----- slice controls -----
    def _sync_slice_controls(self):
        if self.vtk_view.has_slice():
            lo, hi = self.vtk_view.slice_range()
            self.slice_slider.blockSignals(True); self.slice_slider.setMinimum(lo); self.slice_slider.setMaximum(hi)
            self.slice_slider.setValue((lo+hi)//2); self.slice_slider.blockSignals(False)
            self._set_slice_controls(True); self._update_slice_readout(); self.vtk_view.set_slice((lo+hi)//2)
        else: self._set_slice_controls(False)
    def _set_slice_controls(self, vis: bool):
        for w in (self.slice_slider, self.orient_combo, self.slice_caption, self.slice_value_label): w.setVisible(vis)
        if not vis: self.slice_value_label.setText("—")
    def _update_slice_readout(self):
        if not self.vtk_view.has_slice(): self.slice_value_label.setText("—"); return
        lo, hi = self.vtk_view.slice_range(); idx = self.slice_slider.value(); pos_mm = self.vtk_view.slice_index_to_mm(idx)
        self.slice_value_label.setText(f"{idx}/{hi}  ({pos_mm:.2f} mm)")
        
    def on_slice_slider_changed(self, v: int):
        """Single handler for the slice slider (works for both NIfTI and PNG preview)."""
        if self.slice_nav_mode == "png" and self.slice_nav_items:
            idx = max(0, min(v, len(self.slice_nav_items) - 1))
            path = self.slice_nav_items[idx]
            # Show the PNG on the image pane
            self._show_png_on_image_label(path)
            self._update_slice_readout()
        else:
            self.vtk_view.set_slice(v)
            self._update_slice_readout()

                
    def _on_orientation_changed(self, text: str):
        self.vtk_view.set_orientation(text)
        if self.vtk_view.has_slice():
            lo, hi = self.vtk_view.slice_range()
            self.slice_slider.blockSignals(True); self.slice_slider.setMinimum(lo); self.slice_slider.setMaximum(hi)
            self.slice_slider.setValue(max(lo, min(hi, self.slice_slider.value()))); self.slice_slider.blockSignals(False)
            self._update_slice_readout()
            
            
            
    def _show_png_on_image_label(self, png_path: str):
        from PySide6.QtGui import QPixmap
        pm = QPixmap(png_path)
        if pm.isNull():
            print(f"[NIfTI] Could not load preview image: {png_path}")
            return
        self.image_label.setImage(pm)
        self._show_widget(self.image_label)   # show the image pane, keep kind='nifti'
        self._active_view = "image"

    # ----- DnD -----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls: return
        local = urls[0].toLocalFile()
        if not local: return
        print(f"Dropped: {local}")
        eext = ext(local)
        if eext in IMAGE_EXTS: self.load_image(local)
        elif eext in NIFTI_EXTS: self.load_nifti(local)
        elif eext == ".stl": self.load_stl(local)
        elif eext == ".vtk": self.load_vtk(local)
        elif eext == ".vti":
            rdr = vtkXMLImageDataReader(); rdr.SetFileName(local); rdr.Update(); img = rdr.GetOutput()
            self.vtk_view.show_image2d(img); self._show_widget(self.vtk_view); self._sync_slice_controls()
            print(f"VTI loaded (drop). Extent={img.GetExtent()} Spacing={img.GetSpacing()} Range={img.GetScalarRange()}"); self._set_current("vtk_image", local)
        else:
            QMessageBox.information(self, "Unsupported", f"Unsupported file: {local}")

    # ----- Action state mgmt -----
    def _set_current(self, kind: str | None, path: str | None):
        self.current_kind = kind
        self.current_path = path
        has_file = kind is not None
        self.act_save.setEnabled(has_file)
        self.act_save_data.setEnabled(has_file)
        self.act_close.setEnabled(has_file)
        self.act_export_metrics.setEnabled(has_file)
        self._update_process_actions()
        self.act_close.setEnabled(has_file)
        self.act_Reset.setEnabled(has_file)
        self.reset_png_navigation()
        if path and kind:
            self._ensure_metric_row(path, kind)
        if kind in ("stl", "vtk_poly", "vtk_surface", "nifti"):
            self.slice_slider.setEnabled(True)
        else:
            self.slice_slider.setEnabled(False)


    def _update_process_actions(self):
        for a in (self.act_meas_allmarks, self.act_meas_volumes, self.act_meas_area, self.act_meas_perimeter, self.act_meas_lgi, self.act_meas_sulci, self.act_optimization):
            a.setEnabled(False)

        kind = self.current_kind
        
        self.act_pial_to_stl.setEnabled(True)
        self.act_pial_merge.setEnabled(True)

        if kind in ("stl", "vtk_poly", "vtk_surface", "nifti"):
            self.act_meas_area.setEnabled(True)
            self.act_meas_perimeter.setEnabled(False)
            self.act_meas_lgi.setEnabled(True)
            self.act_meas_sulci.setEnabled(True)
            self.act_meas_volumes.setEnabled(True)
            self.act_meas_allmarks.setEnabled(True)
            
            
        elif kind == "image":
            self.act_meas_area.setEnabled(True)
            self.act_meas_perimeter.setEnabled(True)
            self.act_meas_lgi.setEnabled(True)
            self.act_meas_volumes.setEnabled(False)
            self.act_meas_sulci.setEnabled(True)
            self.act_optimization.setEnabled(True)
            self.act_meas_allmarks.setEnabled(True)
            self.act_annotate_square.setEnabled(True)


    def enable_png_navigation(self, png_paths: list[str], slice_indices: list[int] | None = None, start_index: int | None = None):
        """Switch the slice slider to browse a list of PNG previews."""
        if not png_paths:
            return
        self.slice_nav_mode = "png"
        self.slice_nav_items = list(png_paths)
        self.slice_nav_index_map = list(slice_indices) if slice_indices is not None else [None] * len(png_paths)

        self.slice_slider.blockSignals(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(len(self.slice_nav_items) - 1)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setPageStep(5)
        init = start_index if isinstance(start_index, int) else len(self.slice_nav_items) // 2
        init = max(0, min(init, len(self.slice_nav_items) - 1))
        self.slice_slider.setValue(init)
        self.slice_slider.blockSignals(False)

        # Show initial PNG
        self.on_slice_slider_changed(init)
        # Make sure the image pane is visible
        self._show_widget(self.image_label)

    def reset_png_navigation(self):
        """Return the slider to normal NIfTI navigation."""
        self.slice_nav_mode = "nifti" if self.current_kind == "nifti" else None
        self.slice_nav_items = []
        self.slice_nav_index_map = []
        # You may want to re-range the slider back to your NIfTI volume depth here.
        if self.slice_nav_mode == "nifti" and hasattr(self, "nifti_depth"):
            self.slice_slider.blockSignals(True)
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(max(0, self.nifti_depth - 1))
            self.slice_slider.blockSignals(False)


    def _dir_has_files(self, d: str) -> bool:
        if not d or not os.path.isdir(d):
            return False
        for _, _, files in os.walk(d):
            if files:
                return True
        return False
        
    def reset_view(self):
        """Clear on-screen annotations and reload the current item from disk."""
#        import os
#        from PySide6.QtWidgets import QMessageBox
#        from PySide6.QtGui import QPixmap

        # 1) cancel active modes
        if hasattr(self.image_label, "cancel_square_selection"):
            self.image_label.cancel_square_selection()
        if hasattr(self.image_label, "cancel_scalebar_measure"):
            self.image_label.cancel_scalebar_measure()

        # 2) clear overlays (data stays in memory for export)
        if hasattr(self.image_label, "clear_annotations"):
            self.image_label.clear_annotations()

        # 3) reload from disk depending on kind
        kind = self.current_kind
        path = self.current_path
        self.last_annotated_path = None

        if not path or not os.path.exists(path):
            # nothing to reload; just reset camera/labels if possible
#            if kind in ("stl", "vtk", "vtk_poly", "vtk_surface") and hasattr(self, "_vtk_view_isometric"):
#                self._vtk_view_isometric()
            self.statusBar().showMessage("No file path to reload.", 3000)
            return

        try:
            if kind == "image":
                pm = QPixmap(path)
                if pm.isNull():
                    QMessageBox.warning(self, "Reset View", f"Failed to open image:\n{path}")
                    return
                self.image_label.setImage(pm)
                self._active_view = "image"
                # hide NIfTI nav UI if it was visible
                for w in (getattr(self, "orient_combo", None),
                          getattr(self, "slice_caption", None),
                          getattr(self, "slice_slider", None),
                          getattr(self, "slice_value_label", None)):
                    if w: w.setVisible(False)

#            elif kind == "nifti":
#                # reinitialize from file (reloads header/data and resets slider/orientation)
#                if hasattr(self, "_init_nifti"):
#                    self._init_nifti(path)
#                elif hasattr(self, "_nifti_set_orientation"):
#                    # fallback if you don’t have _init_nifti; keep current axis
#                    axis_name = {0: "sagittal", 1: "coronal", 2: "axial"}.get(getattr(self, "nifti_axis", 1), "coronal")
#                    self._nifti_set_orientation(axis_name)
#
#            elif kind in ("stl", "vtk", "vtk_poly", "vtk_surface"):
#                # You usually don't need to re-read the mesh to "reset view".
#                # Just reset the camera. If you do want to re-read, call your existing loader here.
#                if hasattr(self, "_vtk_view_isometric"):
#                    self._vtk_view_isometric()
#                elif hasattr(self, "_vtk_set_view"):
#                    self._vtk_set_view("coronal", flip=False, ortho=True)

            # If any PNG navigation mode is on, turn it off
            if hasattr(self, "disable_png_navigation"):
                self.disable_png_navigation()

            self._append_progress("\n [View] Reloaded from disk and cleared on-screen annotations.")
            self.statusBar().showMessage("\n Reloaded from path; annotations cleared (data kept).", 3000)

        except Exception as ex:
            QMessageBox.warning(self, "Reset View", f"Could not reload:\n{ex}")


        
# ---------------- annotate square ----------

    def annotate_square(self):
        """Start a one-shot square selection and save the crop to temp/roi_crops."""
        # must be showing an image
        if not self.image_label.hasImage():
            QMessageBox.warning(self, "Annotate", "No image is visible to annotate.")
            return

        out_dir = os.path.join(self.temp_dir, "roi_crops")
        os.makedirs(out_dir, exist_ok=True)
        
        from datetime import datetime

        def _on_done(img_rect, cropped_qimg):
            base_name = os.path.basename(self.current_path)
            default_n = self._roi_counter_by_source.get(base_name, 0) + 1
            default_label = f"ROI-{default_n}"
            label_text, ok = QInputDialog.getText(self, "Annotation Label", f"Label for {base_name}:", text=default_label)
            if not ok or not str(label_text).strip():
                text = default_label  # keep going even if dialog canceled

            self.image_label.add_square_annotation(img_rect, color=Qt.yellow, pen_width=2, fill_alpha=0)
                        
            out_dir = os.path.join(self.temp_dir, "roi_crops"); os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}_roi_{ts}.png")
            ok_save = cropped_qimg.save(out_path)
            self.annotation_records.append(str(label_text))
            self.annotations_by_source.setdefault(base_name, []).append(str(label_text))
            self._roi_counter_by_source[base_name] = default_n

            if ok_save:
                # keep the path(s) for later use
                self.last_annotated_path = out_path
                self.annotation_labels_by_path[out_path] = str(label_text or "")

                print(f"[Annotate] Saved ROI → {out_path}")
                self._append_progress(f"[Annotate] ROI {img_rect.getRect()} → {out_path}")
                self.current_output_dir = out_dir
                try:
                    self.statusBar().showMessage(f"Saved ROI: {out_path}", 5000)
                except Exception:
                    pass
            else:
                QMessageBox.warning(self, "Annotate", "Failed to save cropped image.")

        # show hint in the status bar (this line needs to be in a method -> self is defined)
        try:
            self.statusBar().showMessage("Drag to select a square region… (Esc to cancel)")
        except Exception:
            pass
        
        # start selection on the image widget
        
        self.image_label.start_square_selection(_on_done)
        self.image_label.setFocus()
    

    def get_label_for_cropped_path(self, path: str) -> str | None:
        """Return the annotation label for a saved cropped image path, or None."""
        if not path:
            return None
        # normalize to avoid mismatches
        key = os.path.abspath(os.path.expanduser(path))
        # fast path: direct dict
        if key in self.annotation_labels_by_path:
            return self.annotation_labels_by_path[key]
        # fallback: try raw key and search records if you keep them
        else:
            return None


# ---------------------------
# Entry point
# ---------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FetoMorph")
    app.setApplicationDisplayName("FetoMorph")

    win = MainWindow();
    win.setWindowTitle("FetoMorph")
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
