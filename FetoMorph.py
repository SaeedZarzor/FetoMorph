from deps import *
from constants import DEFAULT_NIFTI_REGIONS
from functions.measurements_image import *
from functions.measurements_Nifti import *
from functions.measurements_stl import *
from functions.measurements_vtk import *
from functions.measurement_Batch import *
from functions.pial_to_stl import *
from functions.Nifti2image import nifti_slice_to_image
from functions.curvature import compute_curvature_profile, save_curvature_plot
from functions.hausdorff import calculate_hausdorff_distance, convert_image
from functions.nii_extractor import nifti_extractor
from functions.optimization import optimization
from helpers.Read_Excel import conver_excel
from helpers.Helpers import get_nifti_present_labels, add_scalebar, get_max_slice_thinckness
from widgets.scaled_image_label import ScaledImageLabel
from widgets.Contour_threshold import ContourThresholdDialog
from widgets.Scalebar_set_scale import ScalebarSetScaleDialog
from widgets.Unit_scale import UnitScaleDialog
from widgets.VTK_Viewer import VTKViewer
from widgets.Kernel_size import KernelSizeDialog
from widgets.optimization_widgets import OptimizationOptionsDialog
from widgets.Slice_thickness import SilceThicknessDialog
from widgets.OptionsDialog import ProcessingOptionsDialog
from widgets.Recent_paths import RecentPaths, populate_recent_menu
from widgets.GeometryDialog import GeometryDialogWithAspect
from widgets.RegionDock import *
from ribbon import *
from icons import set_icons

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
        self.current_output_3D_slices = None
        self.last_annotated_path: str | None = None
        self.annotation_records: list[dict] = []          # flat list of all annotations
        self.annotations_by_source: dict[str, list[dict]] = {}  # per-image grouping
        self._roi_counter_by_source: dict[str, int] = {}  # for auto names if user leaves blank
        self.annotation_labels_by_path: dict[str, str] = {}  # save the label for each RIO path
        # NIfTI viewing state (axis: 0=sagittal, 1=coronal, 2=axial)
        self.nifti_axis: int = 1         # default = coronal
        self.nifti_depth: int = 0        # number of slices along current axis
        self.nifti_selected_regions_default = DEFAULT_NIFTI_REGIONS
        self.labels_available : set[int] ={}
        self.nifti_label_lut: dict[int, QColor] = {}   # label -> color
        self.nifti_selected_regions: set[int] = set()
        self.Freesurfer_record: List[Dict[str, str]] = []
        self.label_overlay_enabled: bool = True
#        self.label_overlay_alpha: float = 0.5
        
        # Params for measurements
        self.units_length = None          # e.g., "mm" (set by user prompt)
        self.pixel_size_default = 0.01
        self.pixel_size = self.pixel_size_default       # current working scale (units/pixel)
        self.image_scales = {}                          # per-file scale: {path: float}
        self.cnt_threshold = 100
        self.kernel_size = 5  # default (pixels)
        self.slice_thickness = 0.5
        self.mm_per_px_bar = 0
        self.bar_mm = 25
        self.custom_label: str = None
        self.physical_dim: tuple[int, int, int] = (0, 0, 0)
        self.slice_direction: Literal["X", "Y", "Z"] = "Y"
        
        # Params for optimization
        self.optimization_objectives: list[str] = []
        self.optimization_constraints: dict[str, float] = {}
        self.optimization_algorithms: str = "NSGA-III"
        self.optimization_n_gen: int = 200
        self.optimization_objective_directions: dict[str, str] = {}
            
        # Slice navigation mode/state
        self.slice_nav_mode = None           # None | "nifti" | "png"
        self.slice_nav_items = []            # list[str] when mode=="png" (PNG paths)
        self.slice_nav_index_map = []        # list[int] original slice indices (optional label)
        self._init_metrics_dock()

        self._pm_index = 0
        self._pms = []
        QShortcut(QKeySequence("Ctrl+M"), self).activated.connect(self._next_pm)
        QShortcut(QKeySequence("Ctrl+Shift+M"), self).activated.connect(self._prev_pm)
        self._resume_sc = QShortcut(QKeySequence("Shift+Alt+E"), self)
        self._resume_sc.setContext(Qt.ApplicationShortcut)

        
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
        
        self.recent = RecentPaths("YourOrg", "YourApp")
        self.menu_recent = file_menu.addMenu("Recent")
        populate_recent_menu(self.menu_recent, self.recent, self.open_path)
        self.act_show_results = QAction("Show Results…", self); self.act_show_results.triggered.connect(self.show_results_dock); file_menu.addAction(self.act_show_results)
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
        self.act_meas_sulci = QAction("Sulci Depth", self); self.act_meas_sulci.triggered.connect(self.on_measure_sulci_depth); measures_menu.addAction(self.act_meas_sulci)
        process_menu.addSeparator()
        
        analysis_menu = process_menu.addMenu("Analysis")
        self.act_meas_lgi = QAction("LGI", self); self.act_meas_lgi.triggered.connect(self.on_measure_lgi); analysis_menu.addAction(self.act_meas_lgi); self.act_meas_lgi.setToolTip("Compute Local Gyrification Index")
        self.act_meas_curvature = QAction("Curvature", self); self.act_meas_curvature.triggered.connect(self.on_measure_curvature); analysis_menu.addAction(self.act_meas_curvature)
        self.act_hausdorf = QAction("Hausdorff distance", self); self.act_hausdorf.triggered.connect(self.on_measure_hausdorff); analysis_menu.addAction(self.act_hausdorf)
        process_menu.addSeparator()
        
        self.act_img_batch = QAction("Process images batch", self); self.act_img_batch.triggered.connect(self.on_process_batch); process_menu.addAction(self.act_img_batch)
        self.act_optimization = QAction("Optimization", self); self.act_optimization.triggered.connect(self.on_optimization); process_menu.addAction(self.act_optimization)
        process_menu.addSeparator()
        self.act_nitfi2png = QAction("Nifti masking…", self); self.act_nitfi2png.triggered.connect(self.Nifti_to_png); process_menu.addAction(self.act_nitfi2png)
        self.act_niftiextractor = QAction("Nifti extract regions…", self); self.act_niftiextractor.triggered.connect(self.Nifti_extractor); process_menu.addAction(self.act_niftiextractor)

        # Setting menu
        Setting_menu = self.menuBar().addMenu("Adjustments"); self.Setting_menu = Setting_menu
        self.act_set_custom_label = QAction("Custom label…", self); self.act_set_custom_label.triggered.connect(self.set_custom_label); Setting_menu.addAction(self.act_set_custom_label)
        self.act_set_image_scale = QAction("Set Image Scale…", self); self.act_set_image_scale.triggered.connect(self.set_image_scale); Setting_menu.addAction(self.act_set_image_scale)
        self.act_set_scale = QAction("Set Scale From Scalebar…", self);self.act_set_scale.triggered.connect(self.set_scale_from_scalebar);
        Setting_menu.addAction(self.act_set_scale)
        self.act_kernel_size = QAction("Set Kernel Size…", self); self.act_kernel_size.triggered.connect(self.set_kernel_dialog); Setting_menu.addAction(self.act_kernel_size)
        self.act_slice_thickness = QAction("Set Slice Thikcness…", self); self.act_slice_thickness.triggered.connect(self.set_slice_thickness_dialog); Setting_menu.addAction(self.act_slice_thickness); self.act_slice_thickness.setToolTip("Set the distance between slices")
        self.act_cnt_threshold = QAction("Set filtered Threshold…", self); self.act_cnt_threshold.setShortcut(QKeySequence("Ctrl+T")); self.act_cnt_threshold.triggered.connect(self.set_cnt_threshold_dialog); Setting_menu.addAction(self.act_cnt_threshold)
        self.act_annotate_square = QAction("Annotation…", self); self.act_annotate_square.setShortcut(QKeySequence("Ctrl+Shift+A"));self.act_annotate_square.setToolTip("Drag a square on the image and save the crop to the temp folder"); self.act_annotate_square.triggered.connect(self.annotate_square); Setting_menu.addAction(self.act_annotate_square)
        self.act_choose_regions = QAction("ROI selection…", self); self.act_choose_regions.setShortcut(QKeySequence("Ctrl+Shift+R"));self.act_choose_regions.setToolTip("Pick label IDs to include when processing NIfTI Hallmarks"); self.act_choose_regions.triggered.connect(self.choose_regions_dock);Setting_menu.addAction(self.act_choose_regions)
        self.act_set_physical_dim = QAction("Mesh dimensions…", self);self.act_set_physical_dim.setToolTip("Define the physical dimensions of the VTK mesh."); self.act_set_physical_dim.triggered.connect(self.load_mesh_and_ask_geometry);Setting_menu.addAction(self.act_set_physical_dim)
    
        Freesurfer_menu = self.menuBar().addMenu("Freesurfer Viewer")
        self.act_view_surfacses = QAction("Surfaces…", self); self.act_view_surfacses.setToolTip("Display the brain surface reconstructed with FreeSurfer (e.g. pial, white)."); self.act_view_surfacses.triggered.connect(self.view_freesurfer_surfaces); Freesurfer_menu.addAction(self.act_view_surfacses)
        self.act_view_morph_map = QAction("Morph maps…", self); self.act_view_morph_map.setToolTip("Display the morph map of a brain surface reconstructed with FreeSurfer (e.g. slucs, thickness, curve)."); self.act_view_morph_map.triggered.connect(self.view_morph_map); Freesurfer_menu.addAction(self.act_view_morph_map)
        self.act_pial_to_stl = QAction("Pial → STL…", self); self.act_pial_to_stl.triggered.connect(self.on_pial_to_stl); Freesurfer_menu.addAction(self.act_pial_to_stl)
        self.act_pial_merge = QAction("Combined STL…", self); self.act_pial_merge.triggered.connect(self.on_combined_stl); Freesurfer_menu.addAction(self.act_pial_merge)

        # Disable initially
        for action in [
            self.act_Reset,
            self.act_close,
            self.act_save,
            self.act_export_metrics,
            self.act_save_data,
            self.act_choose_regions,
            self.act_annotate_square,
            self.act_nitfi2png,
            self.act_slice_thickness,
            self.act_meas_curvature,
            self.act_set_physical_dim,
            self.act_show_results,
            self.act_niftiextractor,
            self.act_set_image_scale,
            self.act_set_scale,
            self.act_kernel_size,
            self.act_cnt_threshold,
            self.act_set_custom_label,
            self.act_meas_allmarks,
            self.act_meas_volumes,
            self.act_meas_area,
            self.act_meas_perimeter,
            self.act_meas_lgi,
            self.act_meas_sulci,
        ]:
            action.setEnabled(False)

        # drag & drop
        self.setAcceptDrops(True)

        # Console hooking
        self._orig_stdout = sys.stdout; self._orig_stderr = sys.stderr
        self._qt_console = QtConsole(); self._qt_console.text.connect(self._append_progress)
        sys.stdout = TeeStream(self._orig_stdout, self._qt_console); sys.stderr = TeeStream(self._orig_stderr, self._qt_console)
        vtk_output = QtVTKOutputWindow(self._qt_console); vtkOutputWindow.SetInstance(vtk_output)
        print("Application started. Progress output will appear here.")

        self.all_actions = {self.act_show_results, self.act_Reset, self.act_close, self.act_quit, self.act_imp_img, self.act_imp_vtk, self.act_imp_stl, self.act_imp_nii, self.act_save, self.act_save_data, self.act_export_metrics, self.act_meas_allmarks, self.act_meas_perimeter, self.act_meas_area, self.act_meas_volumes, self.act_meas_lgi, self.act_meas_sulci, self.act_meas_curvature, self.act_hausdorf, self.act_set_custom_label,  self.act_set_image_scale, self.act_set_scale,  self.act_kernel_size, self.act_slice_thickness,  self.act_cnt_threshold, self.act_annotate_square, self.act_choose_regions, self.act_optimization, self.act_nitfi2png, self.act_niftiextractor, self.act_pial_to_stl, self.act_pial_merge, self.act_img_batch, self.act_set_physical_dim}
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
        
        ASSETS = Path(__file__).resolve().parent / "assets"
        set_icons(self, ASSETS)


        self.ribbon.add_action("Home", self.act_nav_import)
        self.ribbon.add_action("Home", self.act_nav_export)
        self.ribbon.add_action("Home", self.act_show_results)
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
        self.ribbon.add_action("Measure", self.act_meas_sulci)
        
        self.ribbon.add_action("Analysis", self.act_meas_lgi)
        self.ribbon.add_action("Analysis", self.act_meas_curvature)
        self.ribbon.add_action("Analysis", self.act_hausdorf)
        
        self.ribbon.add_action("Process", self.act_img_batch)
        self.ribbon.add_action("Process", self.act_optimization)
        self.ribbon.add_action("Process", self.act_nitfi2png)
        self.ribbon.add_action("Process", self.act_niftiextractor)


        self.ribbon.add_action("Adjustments", self.act_set_custom_label)
        self.ribbon.add_action("Adjustments", self.act_set_image_scale)
        self.ribbon.add_action("Adjustments", self.act_set_scale)
        self.ribbon.add_action("Adjustments", self.act_kernel_size)
        self.ribbon.add_action("Adjustments", self.act_slice_thickness)
        self.ribbon.add_action("Adjustments", self.act_cnt_threshold)
        self.ribbon.add_action("Adjustments", self.act_annotate_square)
        self.ribbon.add_action("Adjustments", self.act_choose_regions)
        self.ribbon.add_action("Adjustments", self.act_set_physical_dim)
        
        self.ribbon.add_action("Freesurfer Viewer", self.act_view_surfacses)
        self.ribbon.add_action("Freesurfer Viewer", self.act_view_morph_map)
        self.ribbon.add_action("Freesurfer Viewer", self.act_pial_to_stl)
        self.ribbon.add_action("Freesurfer Viewer", self.act_pial_merge)

        self.addToolBarBreak(Qt.TopToolBarArea)


        # --- Navigation toolbar (goes BELOW the ribbon) ---
        self.nav_tb = QToolBar("Navigation", self)
        self.nav_tb.setIconSize(QSize(20, 20))
        self.nav_tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.nav_tb)
        self. nav_tb.hide()
        
        self.nav_tb.addSeparator()

        self.orient_combo = QComboBox()
        self.orient_combo.addItems(["Axial (Z)", "Coronal (Y)", "Sagittal (X)"])
        self.orient_combo.currentTextChanged.connect(self._on_orientation_changed)
        self.nav_tb.addWidget(self.orient_combo)
#        self.orient_combo.setVisible(False)
        
        self.view_mode = QComboBox()
        self.view_mode.addItems(["2D", "3D"])
        self.view_mode.setCurrentText("2D")
        self.view_mode.currentTextChanged.connect(self._on_view_changed)
        self.nav_tb.addWidget(self.view_mode)
#        self.view_mode.setVisible(False)
        
        self.nav_tb.addSeparator()

        self.slice_caption = QLabel("Section:")
        self.nav_tb.addWidget(self.slice_caption)
#        self.slice_caption.setVisible(False)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setPageStep(5)
        self.slice_slider.valueChanged.connect(self.on_slice_slider_changed)
        self.nav_tb.addWidget(self.slice_slider)
#        self.slice_slider.setVisible(False)

        self.slice_value_label = QLabel("—")
        self.nav_tb.addWidget(self.slice_value_label)
#        self.slice_value_label.setVisible(False)

    # ---------- Import handlers ----------
    def import_image(self):
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Import Image", start, "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif)")
        self.open_path(path)
        if not path: return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing image: {path}"); self.load_image(path)

    def import_vtk(self):
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Import .vtk", start, "VTK Legacy (*.vtk)")
        self.open_path(path)
        if not path: return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing VTK: {path}"); self.load_vtk(path)

    def import_stl(self):
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Import .stl", start, "STL Mesh (*.stl)")
        self.open_path(path)
        if not path: return
        self.last_dir = os.path.dirname(path) or self.last_dir
        print(f"Importing STL: {path}"); self.load_stl(path)

    def import_nifti(self):
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        path, _ = QFileDialog.getOpenFileName(self, "Import NIfTI", start, "NIfTI (*.nii *.nii.gz)")
        self.open_path(path)
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
        self.image_label.clearImage(); self._show_widget(self.image_label); self._set_slice_controls(False);self.act_choose_regions.setEnabled(False); self.act_annotate_square.setEnabled(False)
        self.nav_tb.hide()
        self._set_current(None, None); print("\n Closed current file and reset view.")
        self.statusBar().showMessage("Closed current file and reset view.", 3000)


    def quit_app(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True

        # Block user input
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Disable all actions
        acts = set(self.findChildren(QAction))
        mb = self.menuBar()
        if mb:
            for menu in mb.findChildren(QMenu):
                acts.update(menu.actions())
        for tb in self.findChildren(QToolBar):
            acts.update(tb.actions())
        for a in acts:
            a.setEnabled(False)
            a.setVisible(False)

        # Stop timers
        for t in getattr(self, "_timers", []):
            try: t.stop()
            except Exception as e: print(f"Warning: timer stop failed: {e}")

        # Stop threads/workers
        for w in getattr(self, "_workers", []):
            try:
                if hasattr(w, "stop"): w.stop()
                if hasattr(w, "quit"): w.quit()
                if hasattr(w, "wait"): w.wait()
            except Exception as e: print(f"Warning: worker cleanup failed: {e}")

        # Finish
        QApplication.restoreOverrideCursor()
        print("Quitting application.")
        self.close()
        

    def _ensure_metric_row(
        self,
        path: Optional[str],
        kind: Optional[str],
        label: Optional[str] = None,
        annotation:Optional[str] = None,
        source: Optional[str] = None,
        direction: Optional[str] = None,
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
        last = next((r for r in reversed(rows) if r.get("Annotation") == annotation), None)
        
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
                differs("SliceThickness",  slice_thickness),
                differs("LengthUnit",      unite),
                differs("SliceDirection",  direction),
            ]))
        )

        if make_new:
            row = {
                "File": os.path.basename(path),
                "Kind": kind,
                "Label": label,
                "Annotation": annotation,
                "Source": source,
                "SliceDirection": direction,
                "Area": None,
                "PixelSize":       pixel_size if pixel_size is not None else (last.get("PixelSize") if last else None),
                "PixelSizeUnits":  pixel_size_units if pixel_size_units is not None else (last.get("PixelSizeUnits") if last else None),
                "KernelSize":      kernel_size if kernel_size is not None else (last.get("KernelSize") if last else None),
                "LengthUnit":      unite if unite is not None else (last.get("LengthUnit") if last else None),
                "SliceThickness":  slice_thickness if slice_thickness is not None else (last.get("SliceThickness") if last else None),
                "Length(PA)": None,
                "Width(LR)": None,
                "Hight(IS)": None,
                "Volume": None,
                "Perimeter": None,
                "Perimeter_convex": None,
                "SulciCount": None,
                "MinDepth": None,
                "MaxDepth": None,
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
        if direction is not None:
            last["SliceDirection"] = direction
        return last

        
    def _record_metric_for(self, path: str, annotation: Optional[str] = None, source: Optional[str] = None ,**vals):
        if not path:
            return
        kind = getattr(self, "current_kind", None)
        label = getattr(self, "custom_label", None)

        # Extract the triple so _ensure_metric_row can decide whether to create a new row
        psize = vals.pop("pixel_size", None)
        punit = vals.pop("pixel_size_units", None)
        ksize = vals.pop("kernel_size", None)
        thicsl = vals.pop("slice_thickness", None) 
        direction = vals.pop("direction", None)
        uni = vals.pop("unite", None)
        row = self._ensure_metric_row(
            path, kind, label, annotation,
            source, direction,
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
                    row["SulciCount"] = row["MinDepth"] = row["MaxDepth"] = row["MeanDepth"]= None
                else:
                    row["SulciCount"] = n
                    row["MinDepth"] = min(sd)
                    row["MaxDepth"] = max(sd)
                    row["MeanDepth"] = sum(sd)/n
                    
            else:
                raise ValueError("sulci_depth must be an iterable")

        ld = vals.pop("dimensions", None)
        if ld is not None:
            if isinstance(ld, (list, tuple)):
                nl = len(ld)
                if nl == 3:
                    row["Length(PA)"] = ld[0]
                    row["Width(LR)"] = ld[1]
                    row["Hight(IS)"] = ld[2]
                else:
                    raise ValueError("one or more dimensions are missing")
                    
            else:
                raise ValueError("dimensions must be an iterable")
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
            
        self._metrics_rebuild_for_current()


    def open_path(self, path: str):
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
                print("The file type unknown")
           
            self.last_dir = os.path.dirname(path) or self.last_dir
        except Exception as e:
            print("Failed to open", path, e)

        # Update recent list + menu
        self.recent.add(path)
        populate_recent_menu(self.menu_recent, self.recent, self.open_path)
    # --------- Loaders ----------
    def load_image(self, path: str):
        pm = QPixmap(path)
        if pm.isNull(): QMessageBox.critical(self, "Open Failed", "Could not read image file."); return
        self.image_label.setImage(pm); self._show_widget(self.image_label); self._set_slice_controls(False)
        print(f"Loaded image: {path}  size={pm.width()}x{pm.height()}"); self._set_current("image", path)
        self.statusBar().showMessage(f"{self.current_path} image is loaded", 5000)

    def load_nifti(self, path: str):
        self._set_current("nifti", path)
        rdr = vtkNIFTIImageReader(); rdr.SetFileName(path); rdr.Update(); img = rdr.GetOutput()
        print(f"NIfTI loaded:\n"
              f"Extent={img.GetExtent()} \n Spacing={img.GetSpacing()} \n  Range={img.GetScalarRange()}")
        self.labels_available = get_nifti_present_labels(path)
        self.statusBar().showMessage(f"{self.current_path} nifti file is loaded", 5000)
        if self.view_mode.currentText() == "3D":
            self.slice_nav_mode = None
            self.vtk_view.show_image2d(img);  self._show_widget(self.vtk_view); self._sync_slice_controls()
        elif self.view_mode.currentText() == "2D":
            self.slice_nav_mode = "nifti"
            self._nifti_set_orientation(self.orient_combo.currentText());

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
            print(f"Legacy VTK image loaded. Extent={ds.GetExtent()}  Range={ds.GetScalarRange()}"); self._set_current("vtk", path); return
        if isinstance(ds, vtkPolyData) and ds.GetNumberOfPoints()>0:
            self.vtk_view.show_polydata(ds); self._show_widget(self.vtk_view); self._set_slice_controls(False)
            print(f"Legacy VTK polydata loaded. Points={ds.GetNumberOfPoints()}  Polys={ds.GetNumberOfPolys()}"); self._set_current("vtk", path); return
        surf = vtkDataSetSurfaceFilter(); surf.SetInputData(ds); surf.Update(); poly = surf.GetOutput()
        if poly and poly.GetNumberOfPoints()>0:
            self.vtk_view.show_polydata(poly); self._show_widget(self.vtk_view); self._set_slice_controls(False)
            print(f"Legacy VTK dataset surfaced. Points={poly.GetNumberOfPoints()}  Polys={poly.GetNumberOfPolys()}"); self._set_current("vtk", path); return
        QMessageBox.critical(self, "Open Failed", "Unsupported or empty .vtk dataset (no points after surface extraction).")
        
        
    def set_custom_label(self):
        val, ok = QInputDialog.getText(
        self,
        "Set Custom Label",
        "Enter label:",              # <--- mandatory label text
        QLineEdit.Normal
        )
        self.custom_label = val if ok else None
            
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
            "Label", "Annotation", "Source", "SliceDirection",
            "PixelSize", "PixelSizeUnits", "KernelSize","LengthUnit", "SliceThickness",
            "Length(PA)", "Width(LR)", "Hight(IS)",
            "Area", "Volume", "Perimeter", "Perimeter_convex",
            "SulciCount", "MinDepth", "MaxDepth","MeanDepth",
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
            "PixelSize", "KernelSize", "SliceThickness",
            "Length(PA)", "Width(LR)", "Hight(IS)",
            "Area", "Volume", "Perimeter", "Perimeter_convex",
            "SulciCount", "MinDepth", "MaxDepth","MeanDepth",
            "LGI",
        ]
        has_any_metric = df[real_metric_cols].notna().any(axis=1)
        df = df.loc[has_any_metric].copy()

        if df.empty:
            QMessageBox.information(self, "Export Metrics", "No non-empty metrics to export yet.")
            return

        # Drop metric columns that are entirely empty across remaining rows
        drop_all_null = [c for c in real_metric_cols + ["Label", "Annotation", "Source", "SliceDirection","LengthUnit","PixelSizeUnits"] if c in df.columns and df[c].isna().all()]
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
                while True:
                    if not self.units_length or self.current_path not in self.image_scales:
                        ok = self.set_image_scale()  # pops the single dialog
                        if ok:
                            break
                        else:
                            return
                    else:
                        break
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
                    annotation = label_text,
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
                
                labels = self.nifti_selected_regions if self.nifti_selected_regions else self.labels_available
                dims, area, volume, gi, depth, saved_pngs, valid_slices = compute_nifti_allmarks(self, file_path=nif_path,
                out_dir=out_dir,valid_labels = labels, min_contour_area=self.cnt_threshold, kernel_size = self.kernel_size)
            
                if area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    kernel_size = self.kernel_size,
                    dimensions = dims,
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
                print(f"[NIfTI hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[NIfTI hallmarks] ERROR: {ex}")
                QMessageBox.critical(self, "NIfTI All hallmarks Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.current_kind == "stl":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"STL_allmarks_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                source_label, dims, area, volume, gi, depth, saved_pngs, valid_slices = compute_stl_allmarks(self, file_path=self.current_path,     out_dir=out_dir, min_contour_area=self.cnt_threshold,
                kernel_size = self.kernel_size, slice_thickness=self.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    source = source_label,
                    kernel_size = self.kernel_size,
                    dimensions = dims,
                    unite = "cm",
                    slice_thickness= self.slice_thickness,
                    volume=volume,
                    area=area,
                    sulci_depth = depth,
                    lgi=gi)

                self.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"[STL hallmarks]:")
                print(f"STL mesh Volume Result = {volume:.2f} cm^3.")
                print(f"STL mesh Outer Surface Area Result = {area:.2f} cm^2.")
                print(f"STL mesh GI (Convex surface area/ surfacearea) = {gi:.2f} .")
                print(f"The Maximum Grooves Depth = {depth[0]:.2f}, {depth[1]:.2f}, {depth[2]:.2f} cm.")

                dt = time.time() - t0
                print(f"[STL hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
            
            except Exception as ex:
                print(f"[STL hallmarks] ERROR: {ex}")
                QMessageBox.critical(self, "STL hallmarks Failed", f"{type(ex).__name__}: {ex}")
                return
        
        elif self.current_kind == "vtk":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"VTL_allmarks_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                
                if all(v == 0 for v in self.physical_dim):
                    self.load_mesh_and_ask_geometry()

                u = self.units_length
                area, volume, gi, depth, saved_pngs, valid_slices = compute_vtk_allmarks(self, file_path=self.current_path, out_dir=out_dir, min_contour_area=self.cnt_threshold,
                    kernel_size = self.kernel_size, Slice_direction = self.slice_direction, Physical_dim= self.physical_dim, unit = u, slice_thickness=self.slice_thickness)
            
                if area is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    direction = self.slice_direction,
                    kernel_size = self.kernel_size,
                    unite = u,
                    dimensions = self.physical_dim,
                    slice_thickness= self.slice_thickness,
                    volume=volume,
                    area=area,
                    sulci_depth = depth,
                    lgi=gi)
                    
                
                self.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"[VTK hallmarks]:")
                print(f"VTK mesh Volume Result = {volume:.2f} {u}^3.")
                print(f"VTK mesh Outer Surface Area Result = {area:.2f} {u}^2.")
                print(f"VTK mesh GI (Convex surface area/ surfacearea) = {gi:.2f} .")
                if len(depth)>=3:
                    print(f"The Maximum Grooves Depth = {depth[0]:.2f}, {depth[1]:.2f}, {depth[2]:.2f} {u}.")

                dt = time.time() - t0
                print(f"[VTK hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                print(f"[VTK hallmarks] ERROR: {ex}")
                QMessageBox.critical(self, "VTK hallmarks Failed", f"{type(ex).__name__}: {ex}")
                return
            
        else:
            print("[All hallmarks] Unsupported current kind.")


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
                labels = self.nifti_selected_regions if self.nifti_selected_regions else self.labels_available

                dims, volume,saved_pngs, valid_slices = compute_nifti_volume(self, file_path=nif_path, out_dir=out_dir, valid_labels = labels)
            
                if volume is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(self.current_path, unite="cm", dimensions = dims, volume = volume,)

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
        elif self.current_kind == "stl":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"STL_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                source_label, dims,volume, saved_pngs, valid_slices = compute_stl_volume(self, file_path=self.current_path,     out_dir=out_dir, min_contour_area=self.cnt_threshold, slice_thickness=self.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif volume is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    source = source_label,
                    slice_thickness= self.slice_thickness,
                    dimensions = dims,
                    unite = "cm",
                    volume=volume)

                self.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"STL mesh Volume Result = {volume:.2f} cm^3.")


                dt = time.time() - t0
                print(f"[STL Volume] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[STL Volume] ERROR: {ex}")
                QMessageBox.critical(self, "STL Volume Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.current_kind == "vtk":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"VTL_volume_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                
                if all(v == 0 for v in self.physical_dim):
                    self.load_mesh_and_ask_geometry()

                u = self.units_length
                volume, saved_pngs, valid_slices = compute_vtk_volume(self, file_path=self.current_path, out_dir=out_dir, min_contour_area=self.cnt_threshold,
                    Slice_direction = self.slice_direction, Physical_dim= self.physical_dim, unit = u, slice_thickness=self.slice_thickness)
            
                if volume is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    direction = self.slice_direction,
                    unite = u,
                    dimensions = self.physical_dim,
                    slice_thickness= self.slice_thickness,
                    volume=volume)
                    
                self.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"VTK mesh Volume Result = {volume:.2f} {u}^3.")

                dt = time.time() - t0
                print(f"[VTK hallmarks] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                print(f"[VTK Volume] ERROR: {ex}")
                QMessageBox.critical(self, "VTK hallmarks Failed", f"{type(ex).__name__}: {ex}")
                return
                
        else:
            print("[Volume] Unsupported current kind. Open an image, NIfTI or STL file.")

    def on_measure_perimeter(self):
        """Process → Measures → Perimeter: compute and show annotated result WITHOUT saving."""
        if not self.current_path or not os.path.isfile(self.current_path):
            print("[Perimeter] No file is loaded."); return
        
        if self.current_kind == "image":
            try:
                while True:
                    if not self.units_length or self.current_path not in self.image_scales:
                        ok = self.set_image_scale()  # pops the single dialog
                        if ok:
                            break
                        else:
                            return
                    else:
                        break
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
                while True:
                    if not self.units_length or self.current_path not in self.image_scales:
                        ok = self.set_image_scale()  # pops the single dialog
                        if ok:
                            break
                        else:
                            return
                    else:
                        break
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
                self._record_metric_for(self.current_path, annotation=label_text,
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
                    labels = self.nifti_selected_regions if self.nifti_selected_regions else self.labels_available

                    lGI,saved_pngs, valid_slices = compute_nifti_lGI(self, file_path=nif_path, out_dir=out_dir, valid_labels = labels, min_contour_area=self.cnt_threshold, kernel_size= self.kernel_size,)
                
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
                    source_label, dims, gi, saved_pngs, valid_slices =compute_stl_lGI(
                        self,
                        file_path=stl_path,
                        out_dir=out_dir,
                        min_contour_area=self.cnt_threshold,
                        kernel_size=self.kernel_size,
                        slice_thickness=self.slice_thickness,
                        build_solid=False,   # keep False for stability
                    )
                                        
                
                    if gi is None:
                        return

                    # record metrics (consistent with your global export; units in mm unless noted)
                    self._record_metric_for(
                        self.current_path,
                        source = source_label,
                        kernel_size = self.kernel_size,
                        dimensions = dims,
                        unite = "cm",
                        slice_thickness= self.slice_thickness,
                        lgi=gi)
                        
                    self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                    
                    mid = len(saved_pngs) // 2
                    self.on_slice_slider_changed(mid)
                    
                    print(f"[STL lGI] The Brain GI (Convex surface area/ surfacearea) = {gi:.2f}. ")
                    dt = time.time() - t0
                    print(f"[STL lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                          f"Use File → Save Data As… to copy outputs you want to keep.")

                except Exception as ex:
                    print(f"[STL lGI] ERROR: {ex}")
                    QMessageBox.critical(self, "STL lGI Failed", f"{type(ex).__name__}: {ex}")
                return
                
        elif self.current_kind == "stl":
        
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"STL_lgi_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                source_label, dims, gi, saved_pngs, valid_slices = compute_stl_lGI(self, file_path=self.current_path,     out_dir=out_dir, min_contour_area=self.cnt_threshold,
                kernel_size = self.kernel_size, slice_thickness=self.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif gi is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    source = source_label,
                    kernel_size = self.kernel_size,
                    dimensions = dims,
                    unite = "cm",
                    slice_thickness= self.slice_thickness,
                    lgi=gi)

                self.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"STL mesh GI (Convex surface area/ surfacearea) = {gi:.2f} .")

                dt = time.time() - t0
                print(f"[STL lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[STL lGI] ERROR: {ex}")
                QMessageBox.critical(self, "STL lGI Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.current_kind == "vtk":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"VTL_lGI_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                
                if all(v == 0 for v in self.physical_dim):
                    self.load_mesh_and_ask_geometry()

                u = self.units_length
                gi, saved_pngs, valid_slices = compute_vtk_lGI(self, file_path=self.current_path, out_dir=out_dir, min_contour_area=self.cnt_threshold,
                    kernel_size = self.kernel_size, Slice_direction = self.slice_direction, Physical_dim= self.physical_dim, unit = u, slice_thickness=self.slice_thickness)
            
                if gi is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    direction = self.slice_direction,
                    kernel_size = self.kernel_size,
                    unite = u,
                    dimensions = self.physical_dim,
                    slice_thickness= self.slice_thickness,
                    lgi=gi)
                    
                
                self.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"VTK mesh GI (Convex surface area/ surfacearea) = {gi:.2f} .")

                dt = time.time() - t0
                print(f"[VTK lGI] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                print(f"[VTK lGI] ERROR: {ex}")
                QMessageBox.critical(self, "VTK lGI Failed", f"{type(ex).__name__}: {ex}")
                return
            
        else:
            print("[lGI] Unsupported current kind.")

            
        
    
    def on_measure_sulci_depth(self):
        """Process → Measures → All hallmarks for 2D images: compute and show annotated result WITHOUT saving."""
        def _depth_summary(vals, unit: str) -> str:
            if not isinstance(vals, (list, tuple)) or len(vals) == 0:
                return f"No sulci depth detected ({unit})."
            return ", ".join(f"{float(v):.2f}" for v in vals[:3]) + f" {unit}"

        if not self.current_path or not os.path.isfile(self.current_path):
            print("[Sulci depth] No file is loaded."); return
            
        if self.current_kind == "image":
            try:
                while True:
                    if not self.units_length or self.current_path not in self.image_scales:
                        ok = self.set_image_scale()  # pops the single dialog
                        if ok:
                            break
                        else:
                            return
                    else:
                        break
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
                    unit = u,
                )
                print(f"Sulci Depth = {_depth_summary(depth, u)}")
                
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
                self._record_metric_for(self.current_path, annotation=label_text,
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
                labels = self.nifti_selected_regions if self.nifti_selected_regions else self.labels_available

                dims, depth,saved_pngs, valid_slices = compute_nifti_sulci_depth(self, file_path=nif_path, out_dir=out_dir, valid_labels = labels, min_contour_area=self.cnt_threshold)
            
                if depth is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(self.current_path, unite ="mm", dimensions = dims, sulci_depth = depth,)

                self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
                
                mid = len(saved_pngs) // 2
                self.on_slice_slider_changed(mid)
                
                print(f"[NIfTI Sulci depth] The max Brain Sulci depth across slices = {_depth_summary(depth, 'mm')}")
                dt = time.time() - t0
                print(f"[NIfTI Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[NIfTI Sulci depth] ERROR: {ex}")
                QMessageBox.critical(self, "NIfTI Sulci depth Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.current_kind == "stl":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"STL_sulic_depth_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                source_label, dims, depth, saved_pngs, valid_slices = compute_stl_sulci_depth (self, file_path=self.current_path, out_dir=out_dir, min_contour_area=self.cnt_threshold, slice_thickness=self.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif depth is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(self.current_path, source = source_label, slice_thickness= self.slice_thickness,
                    dimensions = dims,unite ="mm", sulci_depth = depth)

                self.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"[STL Sulci depth] The max Brain Sulci depth across slices = {_depth_summary(depth, 'mm')}")
                dt = time.time() - t0
                print(f"[STL Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[STL Sulci depth] ERROR: {ex}")
                QMessageBox.critical(self, "STL Sulci depth Failed", f"{type(ex).__name__}: {ex}")
            return
            
        elif self.current_kind == "vtk":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"VTL_sulic_depth_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                
                if all(v == 0 for v in self.physical_dim):
                    self.load_mesh_and_ask_geometry()

                u = self.units_length
                depth, saved_pngs, valid_slices = compute_vtk_sulci_depth(self, file_path=self.current_path, out_dir=out_dir, min_contour_area=self.cnt_threshold,
                Slice_direction = self.slice_direction, Physical_dim= self.physical_dim, unit = u, slice_thickness=self.slice_thickness)
            
                if depth is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    direction = self.slice_direction,
                    unite = u,
                    dimensions = self.physical_dim,
                    slice_thickness= self.slice_thickness,
                    sulci_depth = depth)
                                    
                self.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                if isinstance(depth, (list, tuple)) and len(depth) > 0:
                    print("[VTK Sulci depth]")
                    print(f"The Maximum Grooves Depth = {_depth_summary(depth, u)}")

                dt = time.time() - t0
                print(f"[VTK Sulci depth] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                print(f"[VTK Sulci depth] ERROR: {ex}")
                QMessageBox.critical(self, "VTK Sulci depth Failed", f"{type(ex).__name__}: {ex}")
                return
        else:
            print("[Sulci depth] Unsupported current kind.")

            
            
    def on_measure_area(self):
        """Process → Measures → Area: compute and show annotated result WITHOUT saving."""
        if self.current_kind == "image":
            if not self.current_path or not os.path.isfile(self.current_path):
                print("[Area] No image file is loaded."); return
            try:
                while True:
                    if not self.units_length or self.current_path not in self.image_scales:
                        ok = self.set_image_scale()  # pops the single dialog
                        if ok:
                            break
                        else:
                            return
                    else:
                        break
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
                self._record_metric_for(self.current_path, annotation=label_text ,
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
                labels = self.nifti_selected_regions if self.nifti_selected_regions else self.labels_available

                dims, area,saved_pngs, valid_slices = compute_nifti_area(self, file_path=nif_path, out_dir=out_dir, valid_labels = labels, min_contour_area=self.cnt_threshold,)
            
                if area == 0:
                    QMessageBox.information(self, "NIfTI Area", "All slices were filtered out (too small).")
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(self.current_path, unite="cm", dimensions = dims, area = area,)

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
            
        elif self.current_kind == "stl":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"STL_area_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                source_label, dims,area, saved_pngs, valid_slices = compute_stl_area(self, file_path=self.current_path,     out_dir=out_dir, min_contour_area=self.cnt_threshold, slice_thickness=self.slice_thickness)
            
                if source_label == "not_brain":
                    QMessageBox.warning(self, "Mesh ignored", "The computation has been canceled")
                    return
                elif area is None:
                    return

                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    source = source_label,
                    slice_thickness= self.slice_thickness,
                    dimensions = dims,
                    unite = "cm",
                    area=area)
  
                self.two_mode_view(out_dir, saved_pngs, valid_slices)

                
                print(f"STL mesh Area Result = {area:.2f} cm^2.")


                dt = time.time() - t0
                print(f"[STL Area] Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")

            except Exception as ex:
                print(f"[STL Area] ERROR: {ex}")
                QMessageBox.critical(self, "STL Area Failed", f"{type(ex).__name__}: {ex}")
            return
        
        elif self.current_kind == "vtk":
            t0 = time.time()
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"VTL_area_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                
                self.current_output_dir = out_dir
                
                if all(v == 0 for v in self.physical_dim):
                    self.load_mesh_and_ask_geometry()

                u = self.units_length
                area, saved_pngs, valid_slices = compute_vtk_area(self, file_path=self.current_path, out_dir=out_dir, min_contour_area=self.cnt_threshold, Slice_direction = self.slice_direction, Physical_dim= self.physical_dim, unit = u, slice_thickness=self.slice_thickness)
            
                if area is None:
                    return
       
                # record metrics (consistent with your global export; units in mm unless noted)
                self._record_metric_for(
                    self.current_path,
                    direction = self.slice_direction,
                    unite = u,
                    dimensions = self.physical_dim,
                    slice_thickness= self.slice_thickness,
                    area=area)
                    
                
                self.two_mode_view(out_dir, saved_pngs, valid_slices)
                
                print(f"VTK mesh Outer Surface Area Result = {area:.2f} {u}^2.")


                dt = time.time() - t0
                print(f" Done in {dt:.2f}s. Results live in TEMP.\n"
                      f"Use File → Save Data As… to copy outputs you want to keep.")
                      
            except Exception as ex:
                print(f"[VTK Area] ERROR: {ex}")
                QMessageBox.critical(self, "VTK area Failed", f"{type(ex).__name__}: {ex}")
                return
    
        else:
            print("[Area] Unsupported current kind. Open an image, NIfTI, or STL file.")

    def on_process_batch(self):
        """Process → Measure hallmarks for a set of images simultaneously """
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Choose a folder", start,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if not dir_path:
            return  # user canceled
            
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        imgs = sorted(
        (e.path for e in os.scandir(dir_path) if e.is_file()
         and os.path.splitext(e.name.lower())[1] in exts),
        key=lambda p: os.path.basename(p).lower())
        if not imgs:
            QMessageBox.warning(self, "No images", "The selected folder contains no image files.")
            return

        self.last_dir = dir_path
        first_pm = QPixmap(imgs[0])
        if first_pm.isNull():
            QMessageBox.critical(
                self,
                "Process Batch Failed",
                f"Could not open image file:\n{imgs[0]}",
            )
            return
        self.load_image(imgs[0])  # show first image

        self._enter_adjustment_mode()
        self.statusBar().showMessage("Adjust the image now and then press Shift+Alt+E to continue.")
        print("[Process Batch] Adjust the image now and then press Shift+Alt+E to continue.")
        self.wait_for_resume()   # blocks here; resumes after key press
        self.statusBar().clearMessage()
        self._exit_adjustment_mode()

        btn = QMessageBox.warning(self,
                    "Processing Images Batch",
                    "All images must share the same resolution (pixel spacing) and measurement unit.",
                    QMessageBox.Ok | QMessageBox.Cancel)
        if btn == QMessageBox.Cancel:
            return

        while True:
            if not self.units_length or self.current_path not in self.image_scales:
                ok = self.set_image_scale()  # pops the single dialog
                if not ok:
                    return
                break
            else:
                break
        u = self.ensure_units()                # <-- get unit (e.g., 'mm')
        px_size = self.image_scales.get(self.current_path, self.pixel_size)

        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.temp_dir, f"Process_images_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.current_output_dir = out_dir
        print(f"[Process Batch] TEMP output: {out_dir}")

        self.reset_view()


        try:
            # Fail fast: stop the whole batch if any image cannot be opened.
            for img_path in imgs:
                if cv2.imread(img_path) is None:
                    raise ValueError(f"Could not read image: {img_path}")

            valid_slices, saved_pngs = process_on_images_batch(dir_path, out_dir, pixel_size=px_size, kernel_size= self.kernel_size,
                cnt_threshold = self.cnt_threshold, unit = u)
                
            self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)

            mid = len(saved_pngs) // 2
            self.on_slice_slider_changed(mid)
            
        except Exception as ex:
            print(f"[Process Batch] ERROR: {ex}")
            QMessageBox.critical(self, "Process Batch Failed", f"{type(ex).__name__}: {ex}")
            return

    def on_measure_curvature(self):
        if self.current_kind == "image":
            try:
                uid = uuid.uuid4().hex[:8]
                out_dir = os.path.join(self.temp_dir, f"Curvature_{uid}")
                os.makedirs(out_dir, exist_ok=True)
                self.current_output_dir = out_dir

                mask, edge_pixels, curvature_values,curvature_values_s  = compute_curvature_profile(path =self.current_path, min_area = self.cnt_threshold)
                
                print(f"[Curvature] Analysis completed for image {self.current_path}")
                # Convert BGR ndarray → QPixmap and show (no disk write)
                
                img = save_curvature_plot(out_dir,  mask, edge_pixels, curvature_values)
                img2 = save_curvature_plot(out_dir,  mask, edge_pixels, curvature_values_s, filename="curvature_plot_2.png")
                pm = self._np_bgr_to_qpixmap(img)
                pm2 = self._np_bgr_to_qpixmap(img2)
                
                self._pms = [pm, pm2]
                self._pm_index = 0
                self.image_label.setImage(self._pms[self._pm_index])
                self._show_widget(self.image_label)
                # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                self.statusBar().showMessage("Use Ctrl+M to toggle between the two modes.")

                self._active_view = "image"
                # Ensure File/Process actions stay enabled
                self._set_current("image", self.current_path)
            except Exception as ex:
                print(f"[Curvature] ERROR: {ex}")
                QMessageBox.critical(self, "[Curvature] Failed", f"{type(ex).__name__}: {ex}")
            return
            
        else:
            print("[Curvature] Unsupported current kind. Open an image first.")

    def on_optimization(self): 
        # TEMP output
        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.temp_dir, f"Optimization_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.current_output_dir = out_dir
        print(f"[Optimization] TEMP output: {out_dir}")
        
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        while True:
            excel_files, _ = QFileDialog.getOpenFileNames(self, "Select one or multiple Excel files",
                    start, "Excel Files (*.xlsx *.xls)")
            if not excel_files:
                reply = QMessageBox.question(self, "No files selected",
                            "No Excel files were selected. Would you like to try again?",
                            QMessageBox.Retry | QMessageBox.Cancel)
                if reply == QMessageBox.Cancel:
                    return
                continue
            break

        try:
            df1, max_sulci, max_cell_density = conver_excel(excel_files)
            if df1 is None or df1.empty:
                QMessageBox.warning(self, "Optimization Failed", "No valid rows were found in the selected Excel files.")
                return

            self.last_dir = os.path.dirname(excel_files[0]) or self.last_dir
            opt_dialog = OptimizationOptionsDialog(
                self,
                max_sulci_count=max_sulci,
                max_cell_density=max_cell_density,
            )
            if not opt_dialog.exec():
                return

            self.optimization_objectives = opt_dialog.get_selected_objectives()
            self.optimization_objective_directions = opt_dialog.get_objective_directions()
            self.optimization_constraints = opt_dialog.get_constraints()
            self.optimization_algorithms = opt_dialog.get_selected_algorithms()
            self.optimization_n_gen = opt_dialog.get_termination_criterion()

            if max_sulci is not None:
                print(f"[Optimization] Max SulciCount in selected files: {max_sulci}")
            if max_cell_density is not None:
                print(f"[Optimization] Max CellDensity in selected files: {max_cell_density}")

            results, saved_pngs, n_optimal_results = optimization(
                self,
                df1,
                out_dir,
                objectives=self.optimization_objectives,
                objective_directions=self.optimization_objective_directions,
                constraints=self.optimization_constraints,
                algorithms=self.optimization_algorithms,
                n_gen=self.optimization_n_gen,
            )
            if results is not None:
                print(f"[Optimization] Optimization completed. Results saved in {out_dir}.")
                print(f"[Optimization] Number of optimal results: {n_optimal_results}")
                print("Use File → Save Data As… to copy outputs you want to keep.")

                if isinstance(results, pd.DataFrame) and not results.empty:
                    obj_to_column = {
                        "perimeter_rate": "LGI",
                        "cell_density": "CellDensity",
                        "min_d_value": "MinDepth",
                        "max_min_d_value": "MinDepth",
                        "mean_d_value": "MeanDepth",
                        "max_d_value": "MaxDepth",
                        "area": "area",
                    }
                    objective_cols = []
                    for obj in self.optimization_objectives:
                        col = obj_to_column.get(obj, obj)
                        if col in results.columns and col not in objective_cols:
                            objective_cols.append(col)
                    cols_to_print = [c for c in ["File"] + objective_cols if c in results.columns]
                    if cols_to_print:
                        print("[Optimization] Pareto results:")
                        print(results[cols_to_print].to_string(index=False))

                    source_paths_seen = []
                    for idx, r in results.reset_index(drop=True).iterrows():
                        metric_path = r.get("__source_excel_path")
                        if not metric_path or not isinstance(metric_path, str):
                            metric_path = excel_files[0]
                        source_paths_seen.append(metric_path)
                        self._record_metric_for(
                            path=metric_path,
                            annotation=f"pareto_optimal_{idx + 1}",
                            source=metric_path,
                            area=r.get("area"),
                            volume=r.get("Volume"),
                            perimeter=r.get("Perimeter"),
                            perimeter_convex=r.get("Perimeter_convex"),
                            lgi=r.get("LGI"),
                            File=r.get("File", f"index_{idx}"),
                            SulciCount=r.get("SulciCount"),
                            MinDepth=r.get("MinDepth"),
                            MaxDepth=r.get("MaxDepth"),
                            MeanDepth=r.get("MeanDepth"),
                        )
                    if source_paths_seen:
                        self._set_current("Optimization", source_paths_seen[0])
                    self._metrics_rebuild_for_current()
            else:
                print(f"[Optimization] Optimization failed or was canceled.")
                QMessageBox.warning(self, "Optimization Failed", "Optimization failed or was canceled.")

            if len(saved_pngs) == 1:
                img_array = cv2.imread(saved_pngs[0])
                pm = self._np_bgr_to_qpixmap(img_array)
                self.image_label.setImage(pm)
                self._show_widget(self.image_label)
                self._active_view = "image"

            elif saved_pngs and len(saved_pngs) > 1:
                # Provide a default list of indices if valid_slices is not available
                indices = list(range(len(saved_pngs)))
                self.enable_png_navigation(saved_pngs, slice_indices=indices)
                mid = len(saved_pngs) // 2
                self.on_slice_slider_changed(mid)


    
        except Exception as ex:
            print(f"[Optimization] ERROR: {ex}")
            QMessageBox.critical(self, "Optimization Failed", f"{type(ex).__name__}: {ex}")
            return  
        

    def on_measure_hausdorff(self):
        """Pick first & second images, convert and save in TEMP, compute hausdorff distance and show the plot."""

        # TEMP output
        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.temp_dir, f"Huasdorff_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.current_output_dir = out_dir
        print(f"[Hausdorff] TEMP output: {out_dir}")

        if  self.current_kind !="image" and self.current_path is None:
            start = self.last_dir if os.path.isdir(self.last_dir) else ""
            First, _ = QFileDialog.getOpenFileName(self, "Select the first image",
                                            start, "Images (*.png *.jpg *.jpeg )")
            if not First:
                return
            
            self.last_dir = os.path.dirname(First)
            self.load_image(First)
        else:
            First = self.current_path
        
        self._enter_adjustment_mode()
        self.statusBar().showMessage("Adjust now. Press Shift+Alt+E to continue.")
        print("[Hausdorff] Adjust now. Press Shift+Alt+E to continue.")
        self.wait_for_resume()   # blocks here; resumes after key press
        self.statusBar().clearMessage()

        while True:
            if not self.units_length or self.current_path not in self.image_scales:
                ok = self.set_image_scale()  # pops the single dialog
                if ok:
                    break
                else:
                    return
            else:
                break
        u1 = self.ensure_units()                # <-- get unit (e.g., 'mm')
        px_size_1 = self.image_scales.get(self.current_path, self.pixel_size)

                        
        annotated1, basename1, First_array, label1 =self.annotation_con(out_dir)
        
        self.reset_view()
        
        self.statusBar().showMessage("Select two images to measure the Hausdorff distance")
        print("[Hausdorff] Select two images to measure the Hausdorff distance.")
        
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        Second, _ =  QFileDialog.getOpenFileName(self, "Select the second image",
                                            start, "Images (*.png *.jpg *.jpeg )")
        if not Second:
            return
        self.load_image(Second)
        self.last_dir = os.path.dirname(Second)

        
        self.statusBar().clearMessage()
        
        self._enter_adjustment_mode()
        self.statusBar().showMessage("Adjust now. Press Shift+Alt+E to continue.")
        print("[Hausdorff] Adjust now. Press Shift+Alt+E to continue.")
        self.wait_for_resume()   # blocks here; resumes after key press
        self.statusBar().clearMessage()
        
        while True:
            if not self.units_length or self.current_path not in self.image_scales:
                ok = self.set_image_scale()  # pops the single dialog
                if ok:
                    break
                else:
                    return
            else:
                break
        u2 = self.ensure_units()                # <-- get unit (e.g., 'mm')
        px_size_2 = self.image_scales.get(self.current_path, self.pixel_size)
                
        while True:
            if u1 == u2:
                break
            else:
                btn = QMessageBox.warning(
                    self,
                    "Hausdorff distance",
                    "Both images must use the same measuring units to compute the Hausdorff distance.",
                    QMessageBox.Ok | QMessageBox.Cancel)
                if btn == QMessageBox.Cancel:
                    return
                ok = self.set_image_scale()
                if ok:
                    u2 = self.ensure_units()                 # return unit string or None on cancel
                    px_size_2 = self.image_scales.get(self.current_path, self.pixel_size)
                else:
                    return
            # Units mismatch: ask to retry or cancel


        annotated2,basename2, Second_array, label2= self.annotation_con(out_dir)
        self.reset_view()
        self._exit_adjustment_mode()

        mode = self.ask_align_direction()
    
        try:
            img, hd, d12, d21 = calculate_hausdorff_distance(First_array, Second_array, First_label= label1 or "First", Second_label = label2 or "Second", align_mode= mode,  out_dir=out_dir )
            
            pm = self._np_bgr_to_qpixmap(img)
            pm2 = self._np_bgr_to_qpixmap(annotated1)
            pm3 = self._np_bgr_to_qpixmap(annotated2)
                
            self._pms = [pm, pm2, pm3]
            self._pm_index = 0
            self.image_label.setImage(self._pms[self._pm_index])
            self._show_widget(self.image_label)
            # Keep kind/path as the ORIGINAL file; Save View As… will ask user where to save what they see.
                
            self._active_view = "image"
            # Ensure File/Process actions stay enabled
            self._set_current("image", str(First))

            print("[Hausdorff] The Hausdorff distance results:")
            print(f"Between {basename1} and {basename2}: {d12} {u1}")
            print(f"Between {basename2} and {basename1}: {d21} {u1}")
            print(f"Maximum distance: {hd} {u1}")
            
            self.statusBar().showMessage("Use Ctrl+M and Ctrl+Shift+M to switch between images.")

        except Exception as ex:
            print(f"[Hausdorff] ERROR: {ex}")
            QMessageBox.critical(self, "Hausdorff distance", f"{type(ex).__name__}: {ex}")
    
    def on_pial_to_stl(self):
        """Pick one .pial, convert to STL in TEMP, show it, and keep source in metrics."""
        pial = None
        if not self.current_kind == "Freesurfer":
            start = self.last_dir if os.path.isdir(self.last_dir) else ""
            pial, _ = QFileDialog.getOpenFileName(self, "Select FreeSurfer Pial Surface",
                                                  start, "FreeSurfer Surface (*.pial);;All Files (*)")
            if not pial:
                return
        
            self.last_dir = os.path.dirname(pial)
            
            
        elif len(self.Freesurfer_record) == 2:
            self.on_combined_stl()
            return
        
        else:
            pial = self.Freesurfer_record[0]['path']

        # Save to TEMP (don’t pester user yet)
        uid = uuid.uuid4().hex[:8]
        base = os.path.splitext(os.path.basename(pial))[0]
        temp_out = os.path.join(self.temp_dir, f"{base}_{uid}.stl")

        try:
            print(f"[Pial → STL] TEMP output: {temp_out}")
            saved = pial_to_stl(pial, temp_out)

            # Show it immediately
            self.load_stl(saved)                   # shows in VTK window
            print("[Pial → STL] Hint: use File → Save Data As… to keep a permanent copy.")
        except Exception as ex:
            print(f"[Pial → STL] ERROR: {ex}")
            QMessageBox.critical(self, "Pial → STL", f"{type(ex).__name__}: {ex}")
        
    def on_combined_stl(self):
        """Pick rh & lh .pial, convert + merge in TEMP, show combined STL, record provenance."""
        rh, lh = None, None
        if self.current_kind != "Freesurfer" or  len(self.Freesurfer_record) == 1:
            start = self.last_dir if os.path.isdir(self.last_dir) else ""
            while True:
                files, _ = QFileDialog.getOpenFileNames(self, "Select Both hemisphere (e.g. rh.pial, lh.pial)",
                                                    start, "FreeSurfer Surface (*.pial *.white *.inflated);;All Files (*)")
                                                    
                if not files:
                    return
                
                self.last_dir = os.path.dirname(files[0])
                if len(files) != 2:
                    QMessageBox.warning(self, "Invalid selection", "You must select exactly two files.")
                    continue
                
                names=set()
                exts=set()
                for f in files:
                    base = os.path.basename(f)
                    name, ext = os.path.splitext(base)
                    names.add(name)
                    exts.add(ext)
                if not {"lh", "rh"}.issubset(names):
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "You must select both 'lh' and 'rh' files (e.g., lh.pial and rh.pial)."
                    )
                    continue
                
                if len(exts) != 1:
                    reply = QMessageBox.question(
                    self,
                    "Confirm",
                    "You have selected two different file types. Would you like to proceed?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                   )
                    if reply == QMessageBox.Yes:
                        break
                    else:
                        continue
                
                rh = files[0]; lh = files[1]
                break
        else:
            rh = self.Freesurfer_record[0]['path']
            lh = self.Freesurfer_record[1]['path']

        # TEMP output
        uid = uuid.uuid4().hex[:8]
        temp_out = os.path.join(self.temp_dir, f"brain_both_{uid}.stl")

        try:
            print(f"[Combined STL] TEMP output (combined): {temp_out}")
            saved = pial_pair_to_combined_stl(rh, lh, temp_out)

            # Show combined STL
            self.load_stl(saved)
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
            QMessageBox.information(
                    self,
                    "No Input",
                    "You closed the window without entering any values. Default unit will be used.")
            val = "mm"
        self.units_length = val.strip()

        print(f"[Units] Using {self.units_length}")
        return self.units_length
        

    def load_mesh_and_ask_geometry(self) -> bool:
        """
        Reads a VTK mesh, derives bounds, opens one window to adjust
        dimensions with aspect lock and live unit conversion.
        Returns (physical_dim, slice_direction, unit).
        """
        if self.current_kind == "vtk":
            mesh = pv.read(str(self.current_path))
            xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
            if all(abs(v) > 1e-9 for v in self.physical_dim):
                Lx0, Ly0, Lz0 = self.physical_dim
            else:
                Lx0, Ly0, Lz0 = xmax - xmin, ymax - ymin, zmax - zmin
                Lx0, Ly0, Lz0 = max(Lx0, 1e-9), max(Ly0, 1e-9), max(Lz0, 1e-9)

            
            unit = self.ensure_units()
            slice_dir = getattr(self, "slice_direction", "Y").upper()


            dlg = GeometryDialogWithAspect(self, mesh = mesh, Lx=Lx0, Ly=Ly0, Lz=Lz0, unit=unit, slice_dir=slice_dir)
            if dlg.exec() == QDialog.Accepted:
                (Lx, Ly, Lz), slice_dir, unit = dlg.values()
            else:
                QMessageBox.information(
                    self,
                    "No Input",
                    "You closed the window without entering any values. Default dimensions will be used.")
                (Lx, Ly, Lz) = (Lx0, Ly0, Lz0)

            self.physical_dim = (Lx, Ly, Lz)
            self.slice_direction = slice_dir
            self.units_length = unit

            print(f"[Geometry] from mesh={self.current_path}")
            print(f"  bounds=({xmin}, {xmax}, {ymin}, {ymax}, {zmin}, {zmax})")
            print(f"  physical_dim={self.physical_dim} {unit}, slice_direction={slice_dir}")

            return True
        else:
            return False

    
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
            if dlg._get_status():
                ok= self.set_scale_from_scalebar()
                return ok
            else:
                return False
        
        unit, scale = dlg.values()
        if not (scale > 0):
            QMessageBox.warning(self, "Invalid value", "Pixel size must be a positive number.")
            return False

        # Store for this file and as current
        self.units_length = unit
        self.image_scales[self.current_path] = scale
        self.pixel_size = scale

        # Track in metrics (so Excel shows context)
#        label_text = self.get_label_for_cropped_path(self.last_annotated_path)
#        self._record_metric_for (self.current_path, label= label_text  ,pixel_size= scale, pixel_size_units=f"{unit}/pixel" )
#        row = self._ensure_metric_row(self.current_path, self.current_kind)
#        row = self.metrics[self.current_path]
#        row["PixelSize"] = scale
#        row["PixelSizeUnits"] = f"{unit}/pixel"

        print(f"[Units] {unit}  |  [Scale] {scale} {unit}/pixel  —  {os.path.basename(self.current_path)}")
        return True
            
    def set_scale_from_scalebar(self):
        ok = False
        """Activate line measurement on the image; on release we ask real length+unit."""
        if self.current_kind != "image":
            QMessageBox.information(self, "Set Scale", "Open a 2D image to set scale from a scalebar.")
            return
        if self.image_label is None or self.image_label._pix.isNull():
            QMessageBox.information(self, "Set Scale", "No image visible.")
            return
        print("[Scale] Draw a line over the scalebar: click, drag, release.")

        loop = QEventLoop(self)
        result = {"ok": False}

        def _cb(px_len):
            try:
                result["ok"] = self._finish_scalebar_scale(px_len)  # returns True/False
            finally:
                loop.quit()

        self.image_label.start_scalebar_measure(_cb)  # callback will be called later
        loop.exec()                                   # wait here
        return result["ok"]
        
        
    def _finish_scalebar_scale(self, pixel_length: float) -> bool:
        """Called after the user drags a line; asks for real length & unit, computes px/unit."""
        try:
            unit_init = self.units_length or "mm"
            dlg = ScalebarSetScaleDialog(pixel_length, unit_init=unit_init, parent=self)
            if dlg.exec() != QDialog.Accepted:
                print("[Scale] Canceled.")
                return False
                
            px_per_unit, unit = dlg.values()  # e.g., px/mm
            if px_per_unit <= 0:
                QMessageBox.warning(self, "Set Scale", "Scale must be positive.")
                return False

            # Store: keep px/mm per file; keep working mm/pixel for algorithms
            self.units_length = unit
            mm_per_px = 1.0 / px_per_unit
            self.pixel_size = mm_per_px
            self.image_scales[self.current_path] = float(mm_per_px)      # unit/px


            # Record in metrics
            label_text = self.get_label_for_cropped_path(self.last_annotated_path)
#            self._record_metric_for (self.current_path, label = label_text, pixel_size= mm_per_px, pixel_size_units=f"{unit}/pixel" )
            print(f"[Scale] {pixel_length:.2f} px = {px_per_unit:.6f} px/{unit}  "
                  f"→ pixel size {mm_per_px:.6f} {unit}/pixel for {os.path.basename(self.current_path)}")
            return True
                              
        except Exception as ex:
            print(f"ERROR (Set Scale): {ex}")
            QMessageBox.critical(self, "Set Scale Failed", f"{type(ex).__name__}: {ex}")
            return False


    def set_slice_thickness_dialog(self):
        """Open dialog to set slice khikcness (odd)."""
        dlg = SilceThicknessDialog(self, initial=getattr(self, "slice_thickness", 0.5), maximum=(get_max_slice_thinckness(self.current_path)/2))
        if dlg.exec() == QDialog.Accepted:
            k = dlg.value()
            self.slice_thickness = k
            print(f"[Slice Thickness] Set Slice Thickness to {k}")

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
#                self._record_metric_for(self.current_path, label = label_text, kernel_size= self.kernel_size)
    
    def set_cnt_threshold_dialog(self):
        dlg = ContourThresholdDialog(self, initial=getattr(self, "cnt_threshold", 50.0))
        if dlg.exec() == QDialog.Accepted:
            val = dlg.value()
            self.cnt_threshold = max(0.0, float(val))
            print(f"[Threshold] Contour area threshold set to {self.cnt_threshold:.0f} px")



    # ---------- Utils ----------
    def _np_bgr_to_qpixmap(self, arr: np.ndarray) -> QPixmap:
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
        self.statusBar().clearMessage()

    def _next_pm(self):
        if not self._pms:
            return
        self._pm_index = (self._pm_index + 1) % len(self._pms)
        self.image_label.setImage(self._pms[self._pm_index])

    def _prev_pm(self):
        if not self._pms:
            return
        self._pm_index = (self._pm_index - 1) % len(self._pms)
        self.image_label.setImage(self._pms[self._pm_index])
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
        if not self.slice_caption.isVisible(): self.slice_value_label.setText("—"); return
        lo = self.slice_slider.minimum(); hi = self.slice_slider.maximum(); idx = self.slice_slider.value(); pos_mm = self.vtk_view.slice_index_to_mm(idx)
        self.slice_value_label.setText(f"{idx}/{hi}  ({pos_mm:.2f} mm)")
        
    def on_slice_slider_changed(self, v: int):
        """Single handler for the slice slider (works for both NIfTI and PNG preview)."""
        if self.slice_nav_mode == "png" and self.slice_nav_items:
            idx = max(0, min(v, len(self.slice_nav_items) - 1))
            path = self.slice_nav_items[idx]
            # Show the PNG on the image pane
            self._show_png_on_image_label(path)
            self._update_slice_readout()
        elif self.slice_nav_mode == "nifti":
            self.show_nifti_slice(v)
            self._active_view = "image"
            self._update_slice_readout()
        elif self.slice_nav_mode == "vtk":
            self._active_view = "vtk"
            idx = max(0, min(v, len(self.slice_nav_items) - 1))
            self._show_widget(self.vtk_view)
            self.vtk_view.delete_slice_section()
            self.vtk_view.show_slice_with_mesh (mesh_file=self.current_path,
                slice_file= self.current_output_3D_slices,
                slice_value=idx)
            self._update_slice_readout()
        else:
            self.vtk_view.set_slice(v)
            self._update_slice_readout()

                
    def _on_orientation_changed(self, text: str):
        self.vtk_view.set_orientation(text)
        if self.vtk_view.has_slice():
            lo, hi = self.vtk_view.slice_range()
            self.slice_slider.blockSignals(True); self.slice_slider.setMinimum(lo); self.slice_slider.setMaximum(hi)
            self.slice_slider.setValue(max(lo, min(hi, self.slice_slider.value())))
            self.slice_slider.blockSignals(False)
            self._update_slice_readout()
        if self.slice_nav_mode == "nifti":
            self._nifti_set_orientation(text);
            self._update_slice_readout()
                    
    def _on_view_changed(self, text: str, path: Optional[str] = None):
        if self.current_kind == "nifti":
            if text == "3D":
                self.slice_nav_mode = None
                rdr = vtkNIFTIImageReader(); rdr.SetFileName(self.current_path if path is None else path); rdr.Update(); img = rdr.GetOutput()
                self.vtk_view.show_image2d(img);  self._show_widget(self.vtk_view); self._sync_slice_controls()
                self._on_orientation_changed(self.orient_combo.currentText())
            elif text == "2D":
                self.slice_nav_mode = "nifti"
                self._nifti_set_orientation(self.orient_combo.currentText(), path);
        elif self.current_kind in ["vtk", "stl"]:
            if text == "3D":
                self._show_widget(self.vtk_view)
                self.slice_nav_mode = "vtk"
                self.vtk_view.show_slice_with_mesh(mesh_file=self.current_path, slice_file= self.current_output_3D_slices, slice_value= self.slice_slider.value())
            elif text == "2D":
                self.vtk_view.delete_slice_section()
                self._show_widget(self.image_label)
                self.slice_nav_mode = "png"
                self.on_slice_slider_changed(self.slice_slider.value())
                
            
            
    def _show_png_on_image_label(self, png_path: str):
        from PySide6.QtGui import QPixmap
        pm = QPixmap(png_path)
        if pm.isNull():
            print(f"Could not load preview image: {png_path}")
            return
        self.image_label.setImage(pm)
        self._show_widget(self.image_label)   # show the image pane, keep kind='nifti'
        self._active_view = "image"


    def _nifti_set_orientation(self, view: str, path: Optional[str] = None):
        """
        Set slice axis from a name and reconfigure the slider + view.
        view in {'sagittal','coronal','axial'}
        """
        import nibabel as nib

        img = nib.load(self.current_path if path is None else path)
            # Use dataobj (lazy) but rounding requires actual values; this will page from disk
        vol = img.get_fdata(dtype=float)
        if vol is None:
            print("[NIfTI] No data loaded."); return

        a = np.asarray(vol)
        if a.ndim == 4:
            a = a[..., 0]
#        Axial (Z)", "Coronal (Y)", "Sagittal (X)
        axis_map = {"Sagittal (X)": 0, "Coronal (Y)": 1, "Axial (Z)": 2}
        self.nifti_axis = axis_map.get(view, 2)

        self.nifti_depth = int(a.shape[self.nifti_axis])
        mid = max(0, self.nifti_depth // 2)

        if hasattr(self, "slice_slider"):
            self.slice_slider.blockSignals(True)
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(max(0, self.nifti_depth - 1))
            self.slice_slider.setValue(mid)
            self.slice_slider.blockSignals(False)

        self.show_nifti_slice(mid)

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
            print(f"VTI loaded (drop). Extent={img.GetExtent()} Spacing={img.GetSpacing()} Range={img.GetScalarRange()}"); self._set_current("vtk", local)
        else:
            QMessageBox.information(self, "Unsupported", f"Unsupported file: {local}")

    # ----- Action state mgmt -----
    def _set_current(self, kind: str | None, path: str | None):
        self.current_kind = kind
        self.current_path = path
        has_file = kind is not None
        
        self.act_save.setEnabled(has_file)
        self.act_close.setEnabled(has_file)

        if self.current_kind == "Optimization":
            allowed_actions = {
                # Import actions
                self.act_imp_img,
                self.act_imp_vtk,
                self.act_imp_stl,
                self.act_imp_nii,
                # Export actions
                self.act_save,
                self.act_save_data,
                self.act_export_metrics,
                # App-level
                self.act_close,
                self.act_quit,
                self.act_show_results,
            }
            for action in self.all_actions:
                action.setEnabled(action in allowed_actions)
            self.menu_recent.setEnabled(False)
            self.reset_png_navigation()
            self._update_process_actions()
            return

        self.menu_recent.setEnabled(True)
        
        if not self.current_kind == "Freesurfer":
            for action in [
                self.act_save_data,
                self.act_export_metrics,
                self.act_show_results,
                self.act_Reset,
                self.act_kernel_size,
                self.act_cnt_threshold,
                self.act_set_custom_label,
                self.act_set_image_scale,
                self.act_set_scale,
                self.act_meas_allmarks,
                self.act_meas_volumes,
                self.act_meas_area,
                self.act_meas_perimeter,
                self.act_meas_lgi,
                self.act_meas_sulci,
                self.act_optimization,
                self.act_slice_thickness,
            ]:
                action.setEnabled(has_file)

        self.reset_png_navigation()
        self._update_process_actions()


    def _update_process_actions(self):

        kind = self.current_kind

        if kind in ("stl", "vtk"):
            self.act_meas_area.setEnabled(True)
            self.act_meas_perimeter.setEnabled(False)
            self.act_meas_lgi.setEnabled(True)
            self.act_meas_sulci.setEnabled(True)
            self.act_meas_volumes.setEnabled(True)
            self.act_meas_allmarks.setEnabled(True)
            self.act_nitfi2png.setEnabled(False)
            self.act_meas_curvature.setEnabled(False)
            self.act_slice_thickness.setEnabled(True)
            self.act_set_image_scale.setEnabled(False)
            self.act_niftiextractor.setEnabled(False)
            self.act_set_scale.setEnabled(False)
            self.slice_slider.setEnabled(False)
            self.orient_combo.setEnabled(False)
            self.view_mode.setEnabled(False)
            self.act_set_image_scale.setEnabled(False)
            self.act_set_scale.setEnabled(False)
            self.nav_tb.hide()

            
        if kind == "nifti":
            self.act_meas_area.setEnabled(True)
            self.act_meas_perimeter.setEnabled(False)
            self.act_meas_lgi.setEnabled(True)
            self.act_meas_sulci.setEnabled(True)
            self.act_meas_volumes.setEnabled(True)
            self.act_meas_allmarks.setEnabled(True)
            self.act_choose_regions.setEnabled(True)
            self.label_overlay_enabled = True
            self.act_nitfi2png.setEnabled(True)
            self.act_meas_curvature.setEnabled(False)
            self.act_set_image_scale.setEnabled(False)
            self.act_set_scale.setEnabled(False)
            self.act_slice_thickness.setEnabled(False)
            self.act_niftiextractor.setEnabled(True)
            self.slice_slider.setEnabled(True)
            self.orient_combo.setEnabled(True)
            self.view_mode.setEnabled(True)
            self.act_set_image_scale.setEnabled(False)
            self.act_set_scale.setEnabled(False)
            self.nav_tb.show()

        elif kind == "image":
            self.act_meas_area.setEnabled(True)
            self.act_meas_perimeter.setEnabled(True)
            self.act_meas_lgi.setEnabled(True)
            self.act_meas_volumes.setEnabled(False)
            self.act_meas_sulci.setEnabled(True)
            self.act_optimization.setEnabled(True)
            self.act_meas_allmarks.setEnabled(True)
            self.act_annotate_square.setEnabled(True)
            self.act_nitfi2png.setEnabled(False)
            self.act_hausdorf.setEnabled(True)
            self.act_meas_curvature.setEnabled(True)
            self.act_slice_thickness.setEnabled(False)
            self.act_niftiextractor.setEnabled(False)
            self.slice_slider.setEnabled(False)
            self.orient_combo.setEnabled(False)
            self.view_mode.setEnabled(False)
            self.act_set_image_scale.setEnabled(True)
            self.act_set_scale.setEnabled(True)
            self.nav_tb.hide()

        if kind == "vtk":
            self.act_set_physical_dim.setEnabled(True)
        
        else:
            self.act_set_physical_dim.setEnabled(False)
            
            

    def enable_png_navigation(self, png_paths: list[str], slice_indices: list[int] | None = None, start_index: int | None = None):
        """Switch the slice slider to browse a list of PNG previews."""
        if not png_paths:
            return
        self.slice_nav_mode = "png"
        self.slice_nav_items = list(png_paths)
        self.slice_nav_index_map = list(slice_indices) if slice_indices is not None else [None] * len(png_paths)

        self.nav_tb.show()
        self.slice_slider.setEnabled(True)
        
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMinimum(slice_indices[0])
        self.slice_slider.setMaximum(slice_indices[-1])
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setPageStep(5)
        init = start_index if isinstance(start_index, int) else len(self.slice_nav_items) // 2
        init = max(0, min(init, len(self.slice_nav_items) - 1))
        self.slice_slider.setValue(init)
        self.slice_slider.blockSignals(False)
        self.orient_combo.setEnabled(False)
        
        if self.current_kind != "stl":
            # Show initial PNG
            self.on_slice_slider_changed(init)
            # Make sure the image pane is visible
            self._show_widget(self.image_label)
            self.view_mode.setEnabled(False)


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
            self.view_mode.setEnabled(True)
            self.orient_combo.setEnabled(True)

    def _dir_has_files(self, d: str) -> bool:
        if not d or not os.path.isdir(d):
            return False
        for _, _, files in os.walk(d):
            if files:
                return True
        return False
        
        
    def disable_png_navigation(self):
        """Exit PNG navigation mode and reset slice controls."""
        if getattr(self, "slice_nav_mode", None) != "png":
            return
        self.slice_nav_mode = None
        self.slice_nav_items = []
        self.slice_nav_index_map = []
        if hasattr(self, "slice_slider"):
            self.slice_slider.setEnabled(False)
        if hasattr(self, "slice_value_label"):
            self.slice_value_label.clear()
        if hasattr(self, "image_label"):
            self.image_label.clear()
      
      
    def two_mode_view(self, out_dir, saved_pngs, valid_slices):
        self.current_output_3D_slices = os.path.join(out_dir, "all_slices_mesh.vtk")
        self.enable_png_navigation(saved_pngs, slice_indices=valid_slices)
        self.nav_tb.show()
        self.slice_slider.setEnabled(True)
        self.view_mode.setEnabled(True)
        self.view_mode.setCurrentText("2D")
        mid = len(saved_pngs) // 2
        if self.view_mode.currentText() ==  "2D":
            self.on_slice_slider_changed(mid)
        elif self.view_mode.currentText() ==  "3D":
            self.vtk_view.show_slice_with_mesh(
            mesh_file=self.current_path,
            slice_file= self.current_output_3D_slices,
                    slice_value=mid)
        
    
    def reset_view(self):
        for meth in ("cancel_square_selection", "cancel_scalebar_measure", "clear_annotations"):
            if hasattr(self.image_label, meth):
                getattr(self.image_label, meth)()

        # 2) path check
        kind = self.current_kind
        path = self.current_path
        self.last_annotated_path = None
        self.nav_tb.hide()
        self.vtk_view.delete_slice_section()
        if not path or not os.path.exists(path):
            self.statusBar().showMessage("No file path to reload.", 3000)
            return

        # helpers
        def _hide_image_widgets():
            for w in (getattr(self, "orient_combo", None),
                      getattr(self, "slice_caption", None),
                      getattr(self, "slice_slider", None),
                      getattr(self, "slice_value_label", None)):
                if w: w.setVisible(False)

        def _read_mesh(p: str):
            ext = os.path.splitext(p)[1].lower()
            if ext == ".stl":
                r = vtkSTLReader(); r.SetFileName(p); r.Update(); return r.GetOutput()
            if ext == ".vtp":
                r = vtkXMLPolyDataReader(); r.SetFileName(p); r.Update(); return r.GetOutput()
            if ext == ".obj":
                r = vtkOBJReader(); r.SetFileName(p); r.Update(); return r.GetOutput()
            if ext == ".vtk":
                # try polydata first
                r1 = vtkPolyDataReader(); r1.SetFileName(p); r1.Update()
                poly = r1.GetOutput()
                if poly and poly.GetNumberOfPoints() > 0:
                    return poly
                # fallback: legacy dataset -> extract surface
                r2 = vtkDataSetReader(); r2.SetFileName(p); r2.Update()
                ds = r2.GetOutput()
                f = vtkDataSetSurfaceFilter(); f.SetInputData(ds); f.Update()
                return f.GetOutput()
            # last resort
            raise ValueError(f"Unsupported mesh format: {ext}")

        try:
            if kind == "image":
                pm = QPixmap(path)
                if pm.isNull():
                    QMessageBox.warning(self, "Reset View", f"Failed to open image:\n{path}")
                    return
                self._show_widget(self.image_label)
                self.image_label.setImage(pm)
                self._active_view = "image"
                _hide_image_widgets()  # hides slider etc., label shown above via _show_widget

            elif kind == "nifti":
                # reuse your existing loader to reset orientation/slider
                self._set_current("nifti", path)
                self._on_view_changed(self.view_mode.currentText())

            elif kind in ("stl", "vtk"):
                poly = _read_mesh(path)
                # show in your VTK view
                self._show_widget(self.vtk_view)
                self.vtk_view.show_polydata(poly)
                # hide image UI
                _hide_image_widgets()
                self._active_view = "vtk"

            # turn off any PNG navigation
            if hasattr(self, "disable_png_navigation"):
                self.disable_png_navigation()

            self._append_progress("\n[View] Reloaded and cleared annotations.\n")
            self.statusBar().showMessage("Reloaded; annotations cleared.", 3000)

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

#--------------- select and show labels -------------------------------

    def _color_for_label(self, lab: int) -> QColor:
        # deterministic vivid color
        from colorsys import hsv_to_rgb
        hue = (lab * 0.61803398875) % 1.0
        r, g, b = hsv_to_rgb(hue, 0.75, 0.95)
        return QColor(int(r*255), int(g*255), int(b*255))

    def _color_square_icon(self, col: QColor, size: int = 12) -> QIcon:
        pm = QPixmap(size, size); pm.fill(Qt.transparent)
        p = QPainter(pm)
        try:
            p.fillRect(0, 0, size, size, col)
            p.setPen(QPen(Qt.black, 1)); p.drawRect(0, 0, size-1, size-1)
        finally:
            p.end()
        return QIcon(pm)
        
    def choose_regions_dock(self):
        """
        Dock-UI version of choose_regions_dialog().
        Creates/shows a dock on the right with the same controls.
        Stores result in self.nifti_selected_regions (on Apply or live while toggling).
        """
        # Setup context same as before
        self.slice_nav_mode = "nifti"
        idx = int(self.slice_slider.value()) if hasattr(self, "slice_slider") else 0
        self.view_mode.setCurrentText("2D")
        self.view_mode.setEnabled(False)
        self._nifti_set_orientation(self.orient_combo.currentText())
        self.on_slice_slider_changed(idx)

        # Determine labels
        labels_available = sorted(set(int(x) for x in self.labels_available))
        if not labels_available:
            QMessageBox.warning(self, "Regions", "No discrete labels detected in this NIfTI.")
            return

        # Prepare colors LUT
        if not hasattr(self, "nifti_label_lut"):
            self.nifti_label_lut = {}
        for lab in labels_available:
            self.nifti_label_lut.setdefault(lab, self._color_for_label(lab))

        # Current & defaults
        current = set(getattr(self, "nifti_selected_regions", self.nifti_selected_regions_default))
        defaults = set(getattr(self, "nifti_selected_regions_default", set()))

        # Create dock if needed
        if not hasattr(self, "_regions_dock") or self._regions_dock is None:
            self._regions_dock = RegionsDock(self)
            self.addDockWidget(Qt.RightDockWidgetArea, self._regions_dock)
            # When user presses Apply
            def on_apply(selected: set[int]):
                if not selected:
                    QMessageBox.warning(self, "Regions", "Please select at least one label.")
                    return
                self.nifti_selected_regions = set(selected)
                try:
                    self._append_progress(f"[Regions] Selected labels: {sorted(selected)} \n")
                    self.statusBar().showMessage(f"Regions set: {sorted(selected)}", 3000)
                except Exception:
                    print("[Regions] Selected:", sorted(selected))
                # Optional: refresh display
                if hasattr(self, "show_nifti_slice"):
                    idx2 = int(self.slice_slider.value()) if hasattr(self, "slice_slider") else 0
                    self.show_nifti_slice(idx2)
            self._regions_dock.applied.connect(on_apply)

            # If dock is closed, re-enable view mode
            def on_closed():
                self.view_mode.setEnabled(True)
            self._regions_dock.closed.connect(on_closed)

        # Populate/refresh dock content every time we open it
        self._regions_dock.populate(labels_available, current, self.nifti_label_lut)
        self._regions_dock.set_defaults(defaults)
        self._regions_dock.show()
        self._regions_dock.raise_()

    def _compose_label_overlay(
        self,
        img2d: np.ndarray,          # can be (H,W) grayscale OR (H,W,3) RGB
        label2d: np.ndarray,        # (H,W) integer labels
        selected: set[int],
        alpha: float = 0.5
    ) -> QImage:
        # --- make a grayscale base in [0,255] ---
        if img2d.ndim == 3 and img2d.shape[-1] == 3:
            # convert RGB to luma for percentile windowing
            f = (0.299 * img2d[..., 0] + 0.587 * img2d[..., 1] + 0.114 * img2d[..., 2]).astype(np.float32, copy=False)
        else:
            f = img2d.astype(np.float32, copy=False)

        lo, hi = np.percentile(f, (1, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0

        gray = (np.clip((f - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)

        # base RGB made from grayscale
        rgb = np.dstack([gray, gray, gray]).astype(np.float32, copy=False)

        # --- overlay only selected labels ---
        if selected:
            for lab in selected:
                mask = (label2d == lab)
                if not np.any(mask):
                    continue
                c = self.nifti_label_lut.get(lab, self._color_for_label(lab))
                overlay_color = np.array([float(c.red()), float(c.green()), float(c.blue())], dtype=np.float32)
                # blend on the masked pixels; rgb[mask] is (N,3)
                rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * overlay_color[None, :]

        # --- to QImage ---
        rgb_u8 = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
        h, w, _ = rgb_u8.shape
        qimg = QImage(rgb_u8.data, w, h, rgb_u8.strides[0], QImage.Format_RGB888)
        return qimg.copy()  # detach from NumPy buffer

    def show_nifti_slice(self, idx, axis=None):
        img = nib.load(self.current_path)
        vol = img.get_fdata(dtype=float)
        if vol is None:
            return

        a = np.asarray(vol)
        if a.ndim == 4:
            a = a[..., 0]  # take first volume

        ax = self.nifti_axis if axis is None else int(axis)
        depth = a.shape[ax]
        i = max(0, min(int(idx), depth - 1))

        # Slice
        sl = a[i, :, :] if ax == 0 else (a[:, i, :] if ax == 1 else a[:, :, i])

        # Normalize to [0,255]
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
            # Prepare label overlay
            L = getattr(self, "nifti_label_data", None) or a
            if L.ndim == 4:
                L = L[..., 0]
            label2d = np.rint(
                L[i, :, :] if ax == 0 else (L[:, i, :] if ax == 1 else L[:, :, i])
            ).astype(np.int32)

            # Expand grayscale to RGB for overlay
            rgb = np.dstack([gray, gray, gray])
            rgb = np.ascontiguousarray(rgb)
            qimg = self._compose_label_overlay(rgb, label2d, self.nifti_selected_regions)
            self._last_frame_rgb = rgb  # keep alive
        else:
            # Just grayscale, no overlay
            h, w = gray.shape
            qimg = QImage(gray.data, w, h, gray.strides[0], QImage.Format_Grayscale8)
            self._last_frame_gray = gray  # keep alive

        zooms = img.header.get_zooms()[:3]  # (z0,z1,z2) voxel sizes in mm
        qimg, self.mm_per_px_bar , self.bar_mm = add_scalebar(qimg, zooms, ax)
        self.image_label.setImage(QPixmap.fromImage(qimg))
        self._show_widget(self.image_label)
#        if hasattr(self, "_update_slice_label"):
#            self._update_slice_label(i, depth, mode="nifti")

    def ask_processing_options(self):
        dlg = ProcessingOptionsDialog(self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return {
                "unify_color": dlg.unify_color(),       # BGR tuple or None
                "add_scale_bar": dlg.add_scale_bar(),   # bool
                "smooth_kind": dlg.smooth_kind(),       # "none", "gaussian", "median", "bilateral"
                "smooth_strength": dlg.smooth_strength()# int
            }
        return None
        
    def Nifti_to_png(self):
        """Ask path & save exactly what is displayed (no auto-saving during processing)."""
        if not (self.current_kind == "nifti"
                and self.view_mode.currentText() == "2D"):
            QMessageBox.information(self, "Nifti to Png", "This function only works for Nifti file with 2D view mode"); return
        base = "view"
        if self.current_path: base = os.path.splitext(os.path.basename(self.current_path))[0]
        uid = uuid.uuid4().hex[:8]
        folder = os.path.join(self.temp_dir, f"nifti_slice_{uid}")
        os.makedirs(folder, exist_ok=True)
        
        path= os.path.join (folder, base + "_view.png")
        out_path = os.path.join(folder, base + "_section.png")
        self.current_output_dir = folder
        pm = self.image_label.grab(); ok = pm.save(path)
        if not ok: raise RuntimeError("Failed to save nifti slice.")
        options= self.ask_processing_options()
      
        length_px = nifti_slice_to_image(path, out_path,
        unify_color = options["unify_color"],
        label_text = f"{self.bar_mm} mm",
        scale_bar= options["add_scale_bar"],
        smooth = options["smooth_kind"],
        smooth_strength = options["smooth_strength"])
        self.pixel_size = self.bar_mm/ length_px
        self.load_image(out_path)

# -------------------------------------------------------

    def Nifti_extractor(self):
    
        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.temp_dir, f"nifti_allmarks_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.current_output_dir = out_dir
    
        labels = self.nifti_selected_regions if self.nifti_selected_regions else self.labels_available

        nii_output = nifti_extractor (self, self.current_path, out_dir, valid_labels = labels)
                
        self.load_nifti(nii_output)
# ---- Metrics Dock (per-path, reads from self.metrics) -----------------------

    def _init_metrics_dock(self):
        # Ensure container exists
        if not hasattr(self, "metrics") or not isinstance(self.metrics, dict):
            self.metrics = {}  # {path: [dict, ...]}

        self.metricsDock = QDockWidget("Metrics", self)
        self.metricsDock.setObjectName("MetricsDock")
        self.metricsDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        host = QWidget()
        v = QVBoxLayout(host); v.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        tb = QToolBar()
        act_copy   = QAction("Copy", self);   act_copy.setShortcut(QKeySequence.Copy)
        act_export = QAction("Export Excel…", self)
        act_clear  = QAction("Clear (this file)", self)
        tb.addAction(act_copy); tb.addAction(act_export); tb.addSeparator(); tb.addAction(act_clear)
        v.addWidget(tb)

        # Table
        self.metricsView = QTableView()
        self.metricsView.setSortingEnabled(True)
        self.metricsView.horizontalHeader().setStretchLastSection(True)
        self.metricsView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        v.addWidget(self.metricsView)

        self.metricsDock.setWidget(host)
        self.addDockWidget(Qt.RightDockWidgetArea, self.metricsDock)
        self.metricsDock.hide()

        # Model
        self._metrics_model = QStandardItemModel(0, 0, self)
        self.metricsView.setModel(self._metrics_model)

        # Wire actions
        act_copy.triggered.connect(self._metrics_copy_selection)
        # Your existing exporter should read from self.metrics; leave as-is:
        act_export.triggered.connect(self.export_metrics_excel)
        act_clear.triggered.connect(self._metrics_clear_current_file)

    def _metrics_headers(self):
        # Adjust/extend columns as you like; keys should match your records
        return [
            "File", "Kind", "Label", "Annotation", "Source", "SliceDirection",
            "PixelSize", "PixelSizeUnits", "KernelSize","LengthUnit", "SliceThickness",
            "Length(PA)", "Width(LR)", "Hight(IS)",
            "Area", "Volume", "Perimeter", "Perimeter_convex",
            "SulciCount", "MinDepth", "MaxDepth","MeanDepth",
            "LGI"
        ]

    def show_results_dock(self):
        self._metrics_rebuild_for_current()
        self.metricsDock.show()
        self.metricsDock.raise_()

    def _metrics_rebuild_for_current(self):
        """Rebuild the table for the currently open file from self.metrics."""
        headers = self._metrics_headers()
        m = self._metrics_model
        m.clear()
        m.setHorizontalHeaderLabels(headers)

        cur_path = getattr(self, "current_path", None)
        rows = []
        if cur_path and cur_path in self.metrics:
            for rec in (self.metrics.get(cur_path) or []):
                if not isinstance(rec, dict):
                    continue
                rows.append([rec.get(h, "") for h in headers])

        # Append rows
        for row in rows:
            items = []
            for val in row:
                txt = "" if val is None else str(val)
                it = QStandardItem(txt)
                # right-align numeric-looking cells
                try:
                    f = float(txt)
                    it.setText(f"{f:.3f}")
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                except Exception:
                    pass
                items.append(it)
            m.appendRow(items)

        # Show/hide dock depending on content
#        if rows:
#            self.metricsDock.show()
#        else:
#            self.metricsDock.hide()

    def _metrics_append_record(self):
        """
        Append the last record of the current file (from self.metrics) into the table.
        Call this right after you add to self.metrics[current_path].
        """
        cur_path = getattr(self, "current_path", None)
        if not cur_path or cur_path not in self.metrics:
            return
        seq = self.metrics[cur_path]
        if not seq:
            return
        rec = seq[-1]
        if not isinstance(rec, dict):
            return

        headers = self._metrics_headers()
        # Ensure model columns exist; if not, rebuild fully once
        if self._metrics_model.columnCount() != len(headers):
            self._metrics_rebuild_for_current()
            return

        row_vals = [rec.get(h, "") for h in headers]
        items = []
        for val in row_vals:
            txt = "" if val is None else str(val)
            it = QStandardItem(txt)
            try:
                f = float(txt)
                it.setText(f"{f:.3f}")
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            except Exception:
                pass
            items.append(it)

        self._metrics_model.appendRow(items)

    def _metrics_copy_selection(self):
        """Copy selected table cells as CSV (with header)."""
        sel = self.metricsView.selectionModel()
        if not sel or not sel.hasSelection():
            return
        idxs = sorted(sel.selectedIndexes(), key=lambda i: (i.row(), i.column()))
        # Build a dict row -> {col: text}
        rows = {}
        model = self._metrics_model
        for i in idxs:
            rows.setdefault(i.row(), {})[i.column()] = model.item(i.row(), i.column()).text()

        header = ",".join(self._metrics_headers())
        lines = [header]
        for r in sorted(rows):
            cols = []
            for c in range(model.columnCount()):
                cols.append(rows[r].get(c, ""))
            lines.append(",".join(cols))
        QApplication.clipboard().setText("\n".join(lines))

    def _metrics_clear_current_file(self):
        """Clear in-memory metrics for the current file and refresh the table."""
        cur_path = getattr(self, "current_path", None)
        if not cur_path:
            return
        self.metrics[cur_path] = []
        self._metrics_rebuild_for_current()
        try:
            self._append_progress(f"[Metrics] Cleared metrics for {cur_path} \n")
        except Exception:
            pass


# ------------------ hausdorff helpers ------------------

    def wait_for_resume(self):
        loop = QEventLoop(self)
        # unique connect; if already connected, reconnect
        try:
            self._resume_sc.activated.connect(loop.quit, Qt.UniqueConnection)
        except TypeError:
            try: self._resume_sc.activated.disconnect(loop.quit)
            except Exception: pass
            self._resume_sc.activated.connect(loop.quit)
        loop.exec()
        # cleanup for next time
        try: self._resume_sc.activated.disconnect(loop.quit)
        except Exception: pass

    def _enter_adjustment_mode(self):
        allow = { self.act_annotate_square, self.act_cnt_threshold, self.act_set_scale, self.act_set_image_scale, self.act_set_custom_label, self.act_kernel_size}

        for a in self.all_actions:
            if a in allow:
                a.setEnabled(True)
            else:
                a.setEnabled(False)

    def _exit_adjustment_mode(self):
        for a in self.all_actions:
            a.setEnabled(True)
            self._update_process_actions()


    def annotation_con(self, out_dir):
                    
        image_path = self.current_path
        if self.last_annotated_path is not None:
            image_path = self.last_annotated_path

        annotated, basename, array = convert_image(image_path, out_dir, pixel_spacing= self.pixel_size, min_contour_area=self.cnt_threshold)

        label_text = self.get_label_for_cropped_path(image_path)
        if label_text:
            annotated_bgr = put_label_on_bgr(annotated, label_text, pos="topleft")
    
        self._record_metric_for(self.current_path, label=label_text ,
                pixel_size_units = f"{self.units_length}/pixel",
                pixel_size = self.pixel_size,
                unite = self.units_length)
        label = getattr(self, "custom_label", None)
        return annotated, basename, array, label
        
    def ask_align_direction(self):
        box = QMessageBox(self)
        box.setWindowTitle("Align direction")
        box.setText("Which direction do you want to align?")
        btn_rb = box.addButton("Right + Down", QMessageBox.AcceptRole)
        btn_lt = box.addButton("Left + Up", QMessageBox.AcceptRole)
        btn_lr = box.addButton("Left", QMessageBox.AcceptRole)
        btn_ud = box.addButton("Up", QMessageBox.AcceptRole)
        btn_none = box.addButton("No alignment", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)

        box.exec()

        b = box.clickedButton()
        if b is btn_rb:   return "right_bottom"
        if b is btn_lt:   return "left_top"
        if b is btn_lr:   return "left"
        if b is btn_ud:   return "up"
        if b is btn_none: return "none"
        return None  # Cance
        

# ================= Freesurfer ============================

    def view_freesurfer_surfaces(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Number of files?")
        msg.setText("Would you like to display one hemisphere or both?")

        one_btn = msg.addButton("One", QMessageBox.ActionRole)
        both_btn = msg.addButton("Both", QMessageBox.ActionRole)

        msg.exec_()
        
        self.Freesurfer_record =[]
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        if msg.clickedButton() == one_btn:
            surf_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select FreeSurfer surface (lh.pial / rh.pial)",
                    start,
                    "FreeSurfer surface (*.pial *.white *.inflated);;All files (*)",
                )
            if not surf_path:
                return
            
            name, ext = os.path.splitext(os.path.basename(surf_path))
            print (f"[Load Freesurfer file] the {ext} map of {os.path.dirname(surf_path)} imported successfully" )
            File_info = {'name': name, 'ext': ext  ,'path': surf_path}
            self.Freesurfer_record.append(File_info)
            self._show_widget(self.vtk_view)
            self.vtk_view.show_pial_surface(surf_path)
            self.last_dir = os.path.dirname(surf_path)
            
        elif msg.clickedButton() == both_btn:
            while True:
                surf_paths, _ = QFileDialog.getOpenFileNames(
                    self,
                    "Select FreeSurfer surfaces (lh.pial / rh.pial)",
                    start,
                    "FreeSurfer surface (*.pial *.white *.inflated);;All files (*)",
                )

                if not surf_paths:  # user canceled
                    return
                
                self.last_dir = os.path.dirname(surf_paths[0])
                if len(surf_paths) != 2:
                    QMessageBox.warning(self, "Invalid selection", "You must select exactly two files.")
                    continue

                files = []
                exts = set()
                lh_file, rh_file = None, None
                
                for path in surf_paths:
                    base = os.path.basename(path)
                    name, ext = os.path.splitext(base)
                    ext = ext.lstrip(".").lower()
                    exts.add(ext)
                    files.append({'name': name, 'ext': ext, 'path': path})
                    if name == "lh":
                        lh_file = path
                    elif name == "rh":
                        rh_file = path
                # both must share same extension
                if len(exts) != 1:
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "Both files must have the same extension (e.g., both .pial)."
                    )
                    continue

                # must include both hemispheres
                if not rh_file or not lh_file:
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "You must select both 'lh' and 'rh' files (e.g., lh.pial and rh.pial)."
                    )
                    continue

                print(f"[Load FreeSurfer file] Both {exts.pop()} surfaces imported successfully from {os.path.dirname(surf_paths[0])}")

                self.Freesurfer_record.extend(files)
                break
            self._show_widget(self.vtk_view)
            self.vtk_view.show_pial_both(lh_file, rh_file)

        self._set_current("Freesurfer", self.Freesurfer_record[0]['path'])
        
        
    def view_morph_map(self):
        if self.current_kind != "Freesurfer" or len(self.Freesurfer_record) == 0:
            QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "This function supports only FreeSurfer files. Please load FreeSurfer surfaces first."
                    )
            return
            
        start = self.last_dir if os.path.isdir(self.last_dir) else ""
        if len(self.Freesurfer_record) == 1:
            name1 = self.Freesurfer_record[0]['name']
            path = self.Freesurfer_record[0]['path']
            while True:
                # ask for sulc/thickness file
                morph_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select FreeSurfer morph file (sulc / thickness)",
                    start,
                    "FreeSurfer morph (*.sulc *.thickness *.curv);;All files (*)",
                )
                if not morph_path:
                    return
                
                self.last_dir = os.path.dirname(morph_path)
                name2 = os.path.splitext(os.path.basename(morph_path))[0].lower()
                ext = os.path.splitext(morph_path)[1].lstrip(".").lower()

                if not name1 == name2:
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "Both files must belong to the same hemisphere."
                    )
                    continue
                
                if not os.path.dirname(morph_path) == os.path.dirname(path):
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "The selected morph belongs to a different brain or gestational week."
                    )
                    continue
                print (f"[Load Freesurfer file] the {ext} map of {os.path.dirname(morph_path)} imported successfully" )
                break

            # call the viewer function
            self._show_widget(self.vtk_view)
            self.vtk_view.show_freesurfer_morph(path, morph_path)
            
        elif len(self.Freesurfer_record) == 2:
            for f in self.Freesurfer_record:
                if f['name'] == "lh":
                    lh_file = f['path']
                elif  f['name'] == "rh":
                    rh_file = f['path']

            while True:
                morph_paths, _ = QFileDialog.getOpenFileNames(
                    self,
                    "Select FreeSurfer morph file (sulc / thickness)",
                    start,
                    "FreeSurfer morph (*.sulc *.thickness *.curv);;All files (*)",
                )

                # if user cancels, abort cleanly
                if not morph_paths:
                    return
                    
                self.last_dir = os.path.dirname(morph_paths[0])
                if len(morph_paths) != 2:
                    QMessageBox.warning(self, "Invalid selection", "You must select exactly two files.")
                    continue  # re-open dialog

                lh_morph, rh_morph = None, None
                exts = []

                for f in morph_paths:
                    name = os.path.basename(f).lower()
                    ext = os.path.splitext(name)[1].lstrip(".").lower()
                    exts.append(ext)
                    if name.startswith("lh"):
                        lh_morph = f
                    elif name.startswith("rh"):
                        rh_morph = f
                        
                if len(set(exts)) != 1:
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "Both files must have the same extension (e.g., both .sulcs)."
                    )
                    continue
                    
                if not lh_morph or not rh_morph:
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "Both 'lh' and 'rh' files must be selected (e.g., lh.sulcs and rh.sulcs)."
                    )
                    continue  # re-open dialog
                
                if not os.path.dirname(lh_morph) == os.path.dirname(rh_file):
                    QMessageBox.warning(
                        self,
                        "Invalid selection",
                        "The selected morph belongs to a different brain or gestational week."
                    )
                    continue
                print (f"[Load Freesurfer file] both {exts[0]} map of {os.path.dirname(lh_morph)} imported successfully" )
                # valid selection → exit loop
                break
                
            self._show_widget(self.vtk_view)
            self.vtk_view.show_freesurfer_morph_both(lh_file,lh_morph, rh_file,rh_morph)

        else:
            return
    
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
