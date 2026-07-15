"""FetoMorph -- a desktop application for morphometric analysis of fetal brain data.

Supports 2-D histological images (PNG/JPEG/TIFF), NIfTI volumetric scans,
VTK legacy meshes, and STL surface meshes.  The GUI is built with PySide6
and VTK, exposing measurement routines (area, volume, perimeter, sulci
depth, LGI, curvature, Hausdorff distance) through both menus and a
ribbon toolbar.  Results are collected in an in-memory metrics store and
can be exported to Excel.

Typical workflow:
    1. Import a file (image, NIfTI, VTK, or STL).
    2. Adjust parameters (pixel scale, kernel size, ROI selection, etc.).
    3. Run one or more measurements from the Process menu.
    4. Review results in the Metrics dock and export to Excel.
"""

from deps import *
from constants import (DEFAULT_NIFTI_REGIONS, WINDOW_WIDTH, WINDOW_HEIGHT,
                       CONSOLE_MAX_BLOCKS)
from functions.nifti_to_image import nifti_slice_to_image
from functions.hausdorff import convert_image
from functions.measurements_image import put_label_on_bgr
from functions.nii_extractor import nifti_extractor
from widgets.scaled_image_label import ScaledImageLabel
from widgets.vtk_viewer import VTKViewer
from widgets.options_dialog import ProcessingOptionsDialog
from widgets.recent_paths import populate_recent_menu
from widgets.region_dock import RegionsDock
from widgets.gestational_weeks_dialog import GestationalWeeksDialog
from widgets.image_browser_dialog import ImageBrowserDialog
from widgets.zoom_controls import ZoomControlsWidget
from ribbon import *
from icons import set_icons
from managers.metrics_store import MetricsStore
from managers.settings_manager import SettingsManager
from managers.file_manager import FileManager
from managers.view_manager import ViewManager
from managers.measurement_dispatcher import MeasurementDispatcher
from managers.visualization_settings import VisualizationSettings, set_active as set_active_viz
from widgets.preferences_dialog import GASPOptionsDialog, PreferencesDialog

import logging

logger = logging.getLogger("fetomorph")

# ---------------------------
# Supported extensions
# ---------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
NIFTI_EXTS = {".nii", ".nii.gz"}
APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "assets"

def ext(path: str) -> str:
    """Return the lowercase file extension, with special handling for .nii.gz.

    Args:
        path: Filesystem path (absolute or relative).

    Returns:
        Lowercase extension string, e.g. ".png" or ".nii.gz".
    """
    low = path.lower()
    if low.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(low)[1]

# ---------------------------
# Console capture
# ---------------------------
class QtConsole(QObject):
    """Qt-aware console sink that re-emits written text as a Signal.

    Used to redirect stdout/stderr into the in-app progress panel so the
    user can follow long-running computations without a terminal.
    """

    text = Signal(str)

    def write(self, s: str):
        """Emit non-empty strings through the ``text`` signal.

        Args:
            s: Text to emit.
        """
        if s:
            self.text.emit(str(s))

    def flush(self):
        """No-op flush to satisfy the file-like interface."""
        pass

class TeeStream:
    """Duplicate writes to two streams, like the Unix ``tee`` command.

    Errors on either stream are silently swallowed so that a broken
    console never crashes the application.
    """

    def __init__(self, a, b):
        """Initialise with two file-like destinations.

        Args:
            a: Primary stream (typically the original sys.stdout/stderr).
            b: Secondary stream (typically a QtConsole instance).
        """
        self.a, self.b = a, b

    def write(self, s):
        """Write *s* to both underlying streams.

        Args:
            s: Text to write.
        """
        for t in (self.a, self.b):
            try: t.write(s)
            except Exception: logger.debug("TeeStream write failed", exc_info=True)

    def flush(self):
        """Flush both underlying streams."""
        for t in (self.a, self.b):
            try: t.flush()
            except Exception: logger.debug("TeeStream flush failed", exc_info=True)

class QtVTKOutputWindow(vtkOutputWindow):
    """Redirect VTK's internal logging into the application's QtConsole.

    Without this, VTK prints warnings and errors to the C-level stderr
    which is invisible inside a GUI application.
    """

    def __init__(self, sink: QtConsole):
        """Initialise with the application's QtConsole sink.

        Args:
            sink: QtConsole instance whose ``text`` signal feeds the
                progress panel.
        """
        super().__init__(); self._sink = sink

    def DisplayText(self, txt):
        """Forward plain VTK text.

        Args:
            txt: Message text.
        """
        self._sink.write(txt)

    def DisplayErrorText(self, txt):
        """Forward VTK error messages with an ERROR prefix.

        Args:
            txt: Error message text.
        """
        self._sink.write("VTK ERROR: " + txt)

    def DisplayWarningText(self, txt):
        """Forward VTK warning messages with a WARNING prefix.

        Args:
            txt: Warning message text.
        """
        self._sink.write("VTK WARNING: " + txt)

    def DisplayGenericWarningText(self, txt):
        """Forward VTK generic warning messages with a WARNING prefix.

        Args:
            txt: Warning message text.
        """
        self._sink.write("VTK WARNING: " + txt)

    def DisplayDebugText(self, txt):
        """Forward VTK debug messages with a DEBUG prefix.

        Args:
            txt: Debug message text.
        """
        self._sink.write("VTK DEBUG: " + txt)


# ---------------------------
# Main Window
# ---------------------------
class MainWindow(QMainWindow):
    """Central application window for FetoMorph.

    Manages file import (images, NIfTI, VTK, STL), interactive display
    via a ScaledImageLabel (2-D) or VTKViewer (3-D), measurement
    dispatching for area/volume/perimeter/sulci-depth/LGI/curvature/
    Hausdorff, annotation cropping, FreeSurfer surface viewing, and
    metrics collection with Excel export.  A ribbon toolbar mirrors the
    menu actions for quick access.
    """

    # ---- Property delegates to SettingsManager ----
    # These let existing code keep using ``self.pixel_size`` etc. while
    # the actual state lives in ``self.settings``.

    def _settings_prop(attr):  # noqa: N805 – deliberate factory
        """Create a read/write property that delegates to ``self.settings.<attr>``."""
        return property(
            lambda self: getattr(self.settings, attr),
            lambda self, v: setattr(self.settings, attr, v),
        )

    units_length       = _settings_prop("units_length")
    pixel_size_default = _settings_prop("pixel_size_default")
    pixel_size         = _settings_prop("pixel_size")
    image_scales       = _settings_prop("image_scales")
    image_scale_from_scalebar = _settings_prop("image_scale_from_scalebar")
    draw_hallmarks_on_image   = _settings_prop("draw_hallmarks_on_image")
    cnt_threshold      = _settings_prop("cnt_threshold")
    kernel_size_mm     = _settings_prop("kernel_size_mm")
    kernel_size        = _settings_prop("kernel_size")
    contour_mode               = _settings_prop("contour_mode")
    slice_thickness    = _settings_prop("slice_thickness")
    mm_per_px_bar      = _settings_prop("mm_per_px_bar")
    bar_mm             = _settings_prop("bar_mm")
    custom_label       = _settings_prop("custom_label")
    physical_dim       = _settings_prop("physical_dim")
    slice_direction    = _settings_prop("slice_direction")
    _flat_axis         = _settings_prop("_flat_axis")

    del _settings_prop  # remove helper from class namespace

    # ---- Property delegates to FileManager ----
    def _file_prop(attr):  # noqa: N805
        return property(
            lambda self: getattr(self.file_mgr, attr),
            lambda self, v: setattr(self.file_mgr, attr, v),
        )

    last_dir               = _file_prop("last_dir")
    current_output_dir     = _file_prop("current_output_dir")
    current_output_3D_slices = _file_prop("current_output_3D_slices")
    last_annotated_path    = _file_prop("last_annotated_path")
    recent                 = _file_prop("recent")

    del _file_prop

    # ---- Property delegates to ViewManager ----
    def _view_prop(attr):  # noqa: N805
        return property(
            lambda self: getattr(self.view, attr),
            lambda self, v: setattr(self.view, attr, v),
        )

    _pm_index              = _view_prop("_pm_index")
    _pms                   = _view_prop("_pms")
    slice_nav_mode         = _view_prop("slice_nav_mode")
    slice_nav_items        = _view_prop("slice_nav_items")
    slice_nav_index_map    = _view_prop("slice_nav_index_map")
    nifti_axis             = _view_prop("nifti_axis")
    nifti_depth            = _view_prop("nifti_depth")
    label_overlay_enabled  = _view_prop("label_overlay_enabled")
    nifti_selected_regions_default = _view_prop("nifti_selected_regions_default")
    nifti_selected_regions = _view_prop("nifti_selected_regions")
    nifti_label_lut        = _view_prop("nifti_label_lut")
    labels_available       = _view_prop("labels_available")

    del _view_prop

    def __init__(self):
        """Initialise the main window, menus, toolbar, state variables, and console hooks."""
        super().__init__()
        self.setWindowTitle("Unified Image / VTK / NIfTI Viewer (PySide6 + VTK)")
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # ---- Core state ----
        self.current_path: str | None = None
        self.current_kind: str | None = None  # "image" | "nifti" | "vtk_image" | "vtk_poly" | "vtk_surface" | "stl"
        self._active_view = "image"

        # Temp working directory for processing (no persistent saves here)
        self.temp_dir = tempfile.mkdtemp(prefix="FetoMorph_")
        print(f"[Temp] Working directory: {self.temp_dir}")

        # Managers
        self.metrics_store = MetricsStore(self)
        self.file_mgr = FileManager(self)
        self.view = ViewManager(self)
        self.view.nifti_selected_regions_default = DEFAULT_NIFTI_REGIONS
        self.dispatcher = MeasurementDispatcher(self)
        self.Freesurfer_record: List[Dict[str, str]] = []

        # Params for measurements (owned by SettingsManager)
        self.settings = SettingsManager(self)

        # Visualization preferences (text/colors/sizes/view toggles, persisted to QSettings)
        self.viz = VisualizationSettings(self)
        self.viz.load()
        set_active_viz(self.viz)
        self.viz.settingsChanged.connect(self._on_viz_settings_changed)

        # Params for optimization
        self.optimization_objectives: list[str] = []
        self.optimization_constraints: list[dict] = []
        self.optimization_algorithms: str = "NSGA-III"
        self.optimization_n_gen: int = 200
        self.optimization_objective_directions: dict[str, str] = {}

        # Pixmap carousel shortcuts
        QShortcut(QKeySequence("Ctrl+M"), self).activated.connect(self.view.next_pm)
        QShortcut(QKeySequence("Ctrl+Shift+M"), self).activated.connect(self.view.prev_pm)
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
        self.progress_edit.setMaximumBlockCount(CONSOLE_MAX_BLOCKS); self.progress_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.progress_edit.setStyleSheet("background:#0b0b0b; color:#d0d0d0; border:1px solid #333;")
        pg.addWidget(self.progress_edit)

        # Splitter
        self.splitter = QSplitter(Qt.Vertical); self.splitter.addWidget(display_container); self.splitter.addWidget(self.progress_group)
        self.splitter.setSizes([700, 200]); self.progress_group.setMinimumHeight(100); self.progress_group.setMaximumHeight(500)

        self.container = QWidget(); self.vbox = QVBoxLayout(self.container); self.vbox.setContentsMargins(8,8,8,8); self.vbox.addWidget(self.splitter)
        self.setCentralWidget(self.container)
        self.view.show_widget(self.image_label)
        
        # Menus — File (with Import submenu)
        file_menu = self.menuBar().addMenu("File")
        import_menu = file_menu.addMenu("Import")

        self.act_imp_img = QAction("Image…", self); self.act_imp_img.setShortcut(QKeySequence.Open); self.act_imp_img.triggered.connect(self.file_mgr.import_image); import_menu.addAction(self.act_imp_img)
        self.act_imp_vtk = QAction(".vtk file…", self); self.act_imp_vtk.setShortcut(QKeySequence("Ctrl+Shift+V")); self.act_imp_vtk.triggered.connect(self.file_mgr.import_vtk); import_menu.addAction(self.act_imp_vtk)
        self.act_imp_stl = QAction(".stl file…", self); self.act_imp_stl.setShortcut(QKeySequence("Ctrl+Shift+L")); self.act_imp_stl.triggered.connect(self.file_mgr.import_stl); import_menu.addAction(self.act_imp_stl)
        self.act_imp_nii = QAction("NIfTI…", self); self.act_imp_nii.setShortcut(QKeySequence("Ctrl+Shift+N")); self.act_imp_nii.triggered.connect(self.file_mgr.import_nifti); import_menu.addAction(self.act_imp_nii)
        file_menu.addSeparator()
        
        self.menu_recent = file_menu.addMenu("Recent")
        populate_recent_menu(self.menu_recent, self.recent, self.file_mgr.open_path)
        self.act_open_home_folder = QAction("Open Home Folder…", self); self.act_open_home_folder.setToolTip("Open your home folder in the system file browser"); self.act_open_home_folder.triggered.connect(self.open_home_folder); file_menu.addAction(self.act_open_home_folder)
        self.act_open_temp_folder = QAction("Open Current Temp Folder…", self); self.act_open_temp_folder.setToolTip("Open the active processing output folder, or the session temp folder if no output folder is active"); self.act_open_temp_folder.triggered.connect(self.open_current_temp_folder); file_menu.addAction(self.act_open_temp_folder)
        file_menu.addSeparator()
        self.act_show_results = QAction("Show Results…", self); self.act_show_results.setShortcut(QKeySequence("Ctrl+Shift+R")); self.act_show_results.triggered.connect(self.metrics_store.show_results_dock); file_menu.addAction(self.act_show_results)
        self.act_save = QAction("Save View As…", self); self.act_save.setShortcut(QKeySequence("Ctrl+S")); self.act_save.triggered.connect(self.file_mgr.save_view); file_menu.addAction(self.act_save)
        self.act_save_data = QAction("Save Data As…", self); self.act_save_data.setShortcut(QKeySequence.SaveAs); self.act_save_data.triggered.connect(self.file_mgr.save_data_as); file_menu.addAction(self.act_save_data)
        self.act_export_metrics = QAction("Export Metrics to Excel…", self); self.act_export_metrics.setShortcut(QKeySequence("Ctrl+E")); self.act_export_metrics.triggered.connect(self.metrics_store.export_metrics_excel); file_menu.addAction(self.act_export_metrics)
        self.act_Reset= QAction("Reset view…", self); self.act_Reset.setShortcut(QKeySequence("Ctrl+R")); self.act_Reset.setToolTip("Return to original view and clear on-screen annotations"); self.act_Reset.triggered.connect(self.reset_view); file_menu.addAction(self.act_Reset)
        self.act_close = QAction("Close", self); self.act_close.setShortcut(QKeySequence.Close); self.act_close.triggered.connect(self.file_mgr.close_current); file_menu.addAction(self.act_close)

        file_menu.addSeparator()
        self.act_quit = QAction("Quit", self); self.act_quit.setShortcut(QKeySequence.Quit); self.act_quit.setMenuRole(QAction.MenuRole.QuitRole); self.act_quit.triggered.connect(self.quit_app); file_menu.addAction(self.act_quit)

        # Process menu (auto-enabled by file type)
        process_menu = self.menuBar().addMenu("Process"); self.process_menu = process_menu
        measures_menu = process_menu.addMenu("Measure")
        self.act_meas_allmarks = QAction("All hallmarks", self); self.act_meas_allmarks.triggered.connect(self.dispatcher.on_measure_allmarks); measures_menu.addAction(self.act_meas_allmarks)
        self.act_meas_volumes = QAction("Volumes", self); self.act_meas_volumes.triggered.connect(self.dispatcher.on_measure_volumes); measures_menu.addAction(self.act_meas_volumes)
        self.act_meas_area = QAction("Area", self); self.act_meas_area.triggered.connect(self.dispatcher.on_measure_area); measures_menu.addAction(self.act_meas_area)
        self.act_meas_perimeter = QAction("Perimeter", self); self.act_meas_perimeter.triggered.connect(self.dispatcher.on_measure_perimeter); measures_menu.addAction(self.act_meas_perimeter)
        self.act_meas_curve = QAction("Curve Length", self); self.act_meas_curve.triggered.connect(self.dispatcher.on_measure_curve_length); measures_menu.addAction(self.act_meas_curve)
        self.act_meas_stright = QAction("Straight", self); self.act_meas_stright.triggered.connect(self.dispatcher.on_measure_straight); measures_menu.addAction(self.act_meas_stright)
        self.act_meas_sulci = QAction("Sulci Depth", self); self.act_meas_sulci.triggered.connect(self.dispatcher.on_measure_sulci_depth); measures_menu.addAction(self.act_meas_sulci)
        
        analysis_menu = process_menu.addMenu("Analysis")
        self.act_meas_lgi = QAction("LGI", self); self.act_meas_lgi.triggered.connect(self.dispatcher.on_measure_lgi); analysis_menu.addAction(self.act_meas_lgi); self.act_meas_lgi.setToolTip("Compute Local Gyrification Index")
        self.act_meas_curvature = QAction("Curvature", self); self.act_meas_curvature.triggered.connect(self.dispatcher.on_measure_curvature); analysis_menu.addAction(self.act_meas_curvature)
        self.act_meas_compactness = QAction("Compactness", self); self.act_meas_compactness.triggered.connect(self.dispatcher.on_measure_compactness); analysis_menu.addAction(self.act_meas_compactness); self.act_meas_compactness.setToolTip("Measure of how closely a shape approaches the most space-efficient form")
        self.act_hausdorf = QAction("Hausdorff distance", self); self.act_hausdorf.triggered.connect(self.dispatcher.on_measure_hausdorff); analysis_menu.addAction(self.act_hausdorf)
        self.act_similarity_profile = QAction("Similarity Profile", self); self.act_similarity_profile.triggered.connect(self.dispatcher.on_measure_similarity_profile); analysis_menu.addAction(self.act_similarity_profile); self.act_similarity_profile.setToolTip("Gestational Age Similarity Profile (GASP) compares the current brain's morphometrics to reference profiles for each gestational week, returning a similarity score that may help estimate the brain's developmental age.")
        process_menu.addSeparator()
        
        self.act_img_batch = QAction("Process images batch", self); self.act_img_batch.triggered.connect(self.dispatcher.on_process_batch); process_menu.addAction(self.act_img_batch)
        self.act_optimization = QAction("Optimization", self); self.act_optimization.triggered.connect(self.dispatcher.on_optimization); process_menu.addAction(self.act_optimization)
        process_menu.addSeparator()
        self.act_nitfi2png = QAction("Nifti masking…", self); self.act_nitfi2png.triggered.connect(self.Nifti_to_png); process_menu.addAction(self.act_nitfi2png)
        self.act_niftiextractor = QAction("Nifti extract regions…", self); self.act_niftiextractor.triggered.connect(self.Nifti_extractor); process_menu.addAction(self.act_niftiextractor)

        # Adjustments menu
        Adjustments_menu = self.menuBar().addMenu("Adjustments"); self.Adjustments_menu = Adjustments_menu
        self.act_set_custom_label = QAction("Custom label…", self); self.act_set_custom_label.triggered.connect(self.settings.set_custom_label); Adjustments_menu.addAction(self.act_set_custom_label)
        self.act_set_image_scale = QAction("Set Image Scale…", self); self.act_set_image_scale.triggered.connect(self.settings.set_image_scale); Adjustments_menu.addAction(self.act_set_image_scale)
        self.act_set_scale = QAction("Set Scale From Scalebar…", self);self.act_set_scale.triggered.connect(self.settings.set_scale_from_scalebar);
        Adjustments_menu.addAction(self.act_set_scale)
        self.act_kernel_size = QAction("Set Kernel Size…", self); self.act_kernel_size.triggered.connect(self.settings.set_kernel_dialog); Adjustments_menu.addAction(self.act_kernel_size)
        self.act_perimeter_options = QAction("Perimeter Method…", self); self.act_perimeter_options.triggered.connect(self.settings.set_perimeter_options_dialog); Adjustments_menu.addAction(self.act_perimeter_options)
        self.act_slice_thickness = QAction("Set Slice Thickness…", self); self.act_slice_thickness.triggered.connect(self.settings.set_slice_thickness_dialog); Adjustments_menu.addAction(self.act_slice_thickness); self.act_slice_thickness.setToolTip("Set the distance between slices")
        self.act_cnt_threshold = QAction("Set filtered Threshold…", self); self.act_cnt_threshold.setShortcut(QKeySequence("Ctrl+T")); self.act_cnt_threshold.triggered.connect(self.settings.set_cnt_threshold_dialog); Adjustments_menu.addAction(self.act_cnt_threshold)
        self.act_sulcus_depth_threshold = QAction("Sulcus Depth Threshold…", self); self.act_sulcus_depth_threshold.setToolTip("Minimum sulcus depth (mm) counted as a sulcus across all measurements"); self.act_sulcus_depth_threshold.triggered.connect(self.settings.set_sulcus_depth_threshold_dialog); Adjustments_menu.addAction(self.act_sulcus_depth_threshold)
        self.act_slice_kind_override = QAction("Slice Kind Override…", self); self.act_slice_kind_override.setToolTip("Manually set the slice kind (axial/coronal/sagittal/cropped) instead of auto-detection"); self.act_slice_kind_override.triggered.connect(self.settings.set_slice_kind_override_dialog); Adjustments_menu.addAction(self.act_slice_kind_override)
        self.act_gasp_options = QAction("GASP Options", self); self.act_gasp_options.triggered.connect(self._open_gasp_options); Adjustments_menu.addAction(self.act_gasp_options)
        # Contour-accounting mode: 3-way exclusive submenu under Adjustments.
        from PySide6.QtGui import QActionGroup
        contour_mode_menu = Adjustments_menu.addMenu("Contour Accounting")
        self.contour_mode_group = QActionGroup(self)
        self.contour_mode_group.setExclusive(True)

        self.act_contour_outer = QAction("Outer contours only", self)
        self.act_contour_outer.setCheckable(True)
        self.act_contour_outer.setToolTip(
            "Measure the outer brain contour only. Internal contours (e.g. ventricles) "
            "are ignored. This is the default behaviour."
        )
        self.act_contour_subtract = QAction("Subtract internal contours", self)
        self.act_contour_subtract.setCheckable(True)
        self.act_contour_subtract.setToolTip(
            "Subtract the area of contours nested inside the outer brain outline "
            "(e.g. ventricles) from the cross-section area. Internal contours must "
            "still pass the filtered-area threshold."
        )
        self.act_contour_internal_only = QAction("Internal contours only", self)
        self.act_contour_internal_only.setCheckable(True)
        self.act_contour_internal_only.setToolTip(
            "Measure only the internal contour areas (e.g. ventricles). The outer "
            "brain contour is ignored. Internal contours must still pass the "
            "filtered-area threshold."
        )
        for act in (self.act_contour_outer, self.act_contour_subtract, self.act_contour_internal_only):
            self.contour_mode_group.addAction(act)
            contour_mode_menu.addAction(act)
        self.act_contour_outer.setChecked(True)

        def _on_contour_mode_changed(action):
            mode = {
                self.act_contour_outer: "outer",
                self.act_contour_subtract: "subtract",
                self.act_contour_internal_only: "internal_only",
            }.get(action)
            if mode:
                self.contour_mode = mode
        self.contour_mode_group.triggered.connect(_on_contour_mode_changed)

        # Surface-connected cavity correction (volume / surface area) — enable
        # toggle + area threshold combined in one dialog.
        self.act_cavity_options = QAction("Surface-Connected Cavities…", self)
        self.act_cavity_options.setToolTip(
            "Enable/disable the surface-connected cavity correction and set its area threshold.")
        self.act_cavity_options.triggered.connect(self.settings.set_cavity_options_dialog)
        Adjustments_menu.addAction(self.act_cavity_options)

        self.act_annotate_square = QAction("Annotation…", self); self.act_annotate_square.setShortcut(QKeySequence("Ctrl+Shift+A"));self.act_annotate_square.setToolTip("Drag a square on the image and save the crop to the temp folder"); self.act_annotate_square.triggered.connect(self.annotate_square); Adjustments_menu.addAction(self.act_annotate_square)
        self.act_upscale_image = QAction("Upscale Image…", self); self.act_upscale_image.setShortcut(QKeySequence("Ctrl+Shift+U")); self.act_upscale_image.setToolTip("Smooth LANCZOS upscale + sharpen the current image and reload it"); self.act_upscale_image.triggered.connect(self.upscale_current_image); Adjustments_menu.addAction(self.act_upscale_image)
        self.act_choose_regions = QAction("ROI selection…", self); self.act_choose_regions.setShortcut(QKeySequence("Ctrl+Shift+R"));self.act_choose_regions.setToolTip("Pick label IDs to include when processing NIfTI Hallmarks"); self.act_choose_regions.triggered.connect(self.choose_regions_dock);Adjustments_menu.addAction(self.act_choose_regions)
        self.act_set_physical_dim = QAction("Mesh dimensions…", self);self.act_set_physical_dim.setToolTip("Define the physical dimensions of the VTK mesh."); self.act_set_physical_dim.triggered.connect(self.settings.load_mesh_and_ask_geometry);Adjustments_menu.addAction(self.act_set_physical_dim)
    
        # Freesurfer menu
        Freesurfer_menu = self.menuBar().addMenu("Freesurfer Viewer")
        self.act_view_surfacses = QAction("Surfaces…", self); self.act_view_surfacses.setToolTip("Display the brain surface reconstructed with FreeSurfer (e.g. pial, white)."); self.act_view_surfacses.triggered.connect(self.view_freesurfer_surfaces); Freesurfer_menu.addAction(self.act_view_surfacses)
        self.act_view_morph_map = QAction("Morph maps…", self); self.act_view_morph_map.setToolTip("Display the morph map of a brain surface reconstructed with FreeSurfer (e.g. slucs, thickness, curve)."); self.act_view_morph_map.triggered.connect(self.view_morph_map); Freesurfer_menu.addAction(self.act_view_morph_map)
        self.act_pial_to_stl = QAction("Pial → STL…", self); self.act_pial_to_stl.triggered.connect(self.dispatcher.on_pial_to_stl); Freesurfer_menu.addAction(self.act_pial_to_stl)
        self.act_pial_merge = QAction("Combined STL…", self); self.act_pial_merge.triggered.connect(self.dispatcher.on_combined_stl); Freesurfer_menu.addAction(self.act_pial_merge)

        # Examples menu
        Examples_menu = self.menuBar().addMenu("Examples")
        Fetal_brain_2D_sections = Examples_menu.addMenu("Fetal brain 2D sections")
        Fetal_brain_3D = Examples_menu.addMenu("Fetal brain 3D")
        fill_2D_sections = Fetal_brain_2D_sections.addAction("Filled 2D sections"); fill_2D_sections.setShortcut(QKeySequence("Ctrl+Alt+F")); fill_2D_sections.setToolTip("Open example filled 2D fetal brain sections by gestational week"); fill_2D_sections.triggered.connect(self.choose_gestational_week_2D_fill)
        cropped_2D_sections = Fetal_brain_2D_sections.addAction("Cropped 2D sections"); cropped_2D_sections.setShortcut(QKeySequence("Ctrl+Alt+C")); cropped_2D_sections.setToolTip("Open example cropped 2D fetal brain sections by gestational week"); cropped_2D_sections.triggered.connect(self.choose_gestational_week_2D_cropped)
        surface_mri_nifti = Fetal_brain_3D.addAction("Fetal surface MRI"); surface_mri_nifti.setToolTip("Open an example 3D fetal brain surface MRI segmentation (NIfTI) by gestational week"); surface_mri_nifti.triggered.connect(self.choose_gestational_week_3D_surface_mri)
        brain_stl_3D = Fetal_brain_3D.addAction("Fetal brain STL"); brain_stl_3D.setToolTip("Open an example 3D fetal brain surface mesh (STL) by gestational week"); brain_stl_3D.triggered.connect(self.choose_gestational_week_3D_brain_stl)

        # Settings menu (visualization options)
        settings_menu = self.menuBar().addMenu("Settings"); self.settings_menu = settings_menu
        self.act_preferences = QAction("Preferences…", self)
        self.act_preferences.setMenuRole(QAction.MenuRole.PreferencesRole)
        self.act_preferences.setShortcut(QKeySequence.Preferences)
        self.act_preferences.triggered.connect(self._open_preferences)
        settings_menu.addAction(self.act_preferences)

        # About menu
        about_menu = self.menuBar().addMenu("About")
        act_about_info = QAction("About FetoMorph", self, triggered=self.show_about_info)
        act_about_info.setMenuRole(QAction.MenuRole.NoRole)  # keep it in this menu (macOS moves "About…" to the app menu otherwise)
        about_menu.addAction(act_about_info)
        self.act_user_guide = QAction("User Guide", self); self.act_user_guide.setShortcut(QKeySequence.HelpContents); self.act_user_guide.setToolTip("How to use FetoMorph"); self.act_user_guide.triggered.connect(self.show_user_guide)
        about_menu.addAction(self.act_user_guide)
        about_menu.addSeparator()
        about_menu.addAction(QAction("Contributors", self, triggered=self.show_about_contributors))
        about_menu.addAction(QAction("Acknowledgements", self, triggered=self.show_about_acknowledgements))
        about_menu.addAction(QAction("Copyright and License", self, triggered=self.show_about_copyright))
        about_menu.addAction(QAction("Citing FetoMorph", self, triggered=self.show_about_citation))
        about_menu.addAction(QAction("Declaration", self, triggered=self.show_about_declaration))
        about_menu.addSeparator()
        about_menu.addAction(QAction("ReadMe", self, triggered=self.open_readme))

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
            self.act_perimeter_options,
            self.act_cnt_threshold,
            self.act_sulcus_depth_threshold,
            self.act_set_custom_label,
            self.act_meas_allmarks,
            self.act_meas_volumes,
            self.act_meas_area,
            self.act_meas_perimeter,
            self.act_meas_curve,
            self.act_meas_lgi,
            self.act_meas_stright,
            self.act_meas_compactness,
            self.act_meas_sulci,
        ]:
            action.setEnabled(False)

        # drag & drop
        self.setAcceptDrops(True)

        # Console hooking
        self._orig_stdout = sys.stdout; self._orig_stderr = sys.stderr
        self._qt_console = QtConsole(); self._qt_console.text.connect(self.view.append_progress)
        sys.stdout = TeeStream(self._orig_stdout, self._qt_console); sys.stderr = TeeStream(self._orig_stderr, self._qt_console)
        vtk_output = QtVTKOutputWindow(self._qt_console); vtkOutputWindow.SetInstance(vtk_output)
        print("Application started. Progress output will appear here.")

        self.all_actions = {self.act_show_results, self.act_Reset, self.act_close, self.act_quit, self.act_imp_img, self.act_imp_vtk, self.act_imp_stl, self.act_imp_nii, self.act_save, self.act_save_data, self.act_export_metrics, self.act_meas_allmarks, self.act_meas_perimeter, self.act_meas_area, self.act_meas_volumes, self.act_meas_lgi, self.act_meas_sulci, self.act_meas_curvature, self.act_meas_compactness, self.act_hausdorf, self.act_set_custom_label,  self.act_set_image_scale, self.act_set_scale,  self.act_kernel_size, self.act_perimeter_options, self.act_slice_thickness,  self.act_cnt_threshold, self.act_sulcus_depth_threshold, self.act_contour_outer, self.act_contour_subtract, self.act_contour_internal_only, self.act_cavity_options, self.act_annotate_square, self.act_choose_regions, self.act_optimization, self.act_nitfi2png, self.act_niftiextractor, self.act_pial_to_stl, self.act_pial_merge, self.act_img_batch, self.act_set_physical_dim}
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
        self.ribbon.add_action("Measure", self.act_meas_curve)
        self.ribbon.add_action("Measure", self.act_meas_stright)
        self.ribbon.add_action("Measure", self.act_meas_sulci)
        
        self.ribbon.add_action("Analysis", self.act_meas_lgi)
        self.ribbon.add_action("Analysis", self.act_meas_curvature)
        self.ribbon.add_action("Analysis", self.act_meas_compactness)
        self.ribbon.add_action("Analysis", self.act_hausdorf)
        self.ribbon.add_action("Analysis", self.act_similarity_profile)
        
        self.ribbon.add_action("Process", self.act_img_batch)
        self.ribbon.add_action("Process", self.act_optimization)
        self.ribbon.add_action("Process", self.act_nitfi2png)
        self.ribbon.add_action("Process", self.act_niftiextractor)


        self.ribbon.add_action("Adjustments", self.act_set_custom_label)
        self.ribbon.add_action("Adjustments", self.act_set_image_scale)
        self.ribbon.add_action("Adjustments", self.act_set_scale)
        self.ribbon.add_action("Adjustments", self.act_kernel_size)
        self.ribbon.add_action("Adjustments", self.act_perimeter_options)
        self.ribbon.add_action("Adjustments", self.act_slice_thickness)
        self.ribbon.add_action("Adjustments", self.act_cnt_threshold)
        self.ribbon.add_action("Adjustments", self.act_sulcus_depth_threshold)
        self.ribbon.add_action("Adjustments", self.act_slice_kind_override)
        self.ribbon.add_action("Adjustments", self.act_cavity_options)
        self.ribbon.add_action("Adjustments", self.act_annotate_square)
        self.ribbon.add_action("Adjustments", self.act_upscale_image)
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
        self.orient_combo.currentTextChanged.connect(self.view.on_orientation_changed)
        self.nav_tb.addWidget(self.orient_combo)
        
        self.view_mode = QComboBox()
        self.view_mode.addItems(["2D", "3D"])
        self.view_mode.setCurrentText("2D")
        self.view_mode.currentTextChanged.connect(self.view.on_view_changed)
        self.nav_tb.addWidget(self.view_mode)
        
        self.nav_tb.addSeparator()

        self.slice_caption = QLabel("Section:")
        self.nav_tb.addWidget(self.slice_caption)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setPageStep(5)
        self.slice_slider.valueChanged.connect(self.view.on_slice_slider_changed)
        self.nav_tb.addWidget(self.slice_slider)

        self.slice_value_label = QLabel("—")
        self.nav_tb.addWidget(self.slice_value_label)

        self.nav_tb.addSeparator()

        self.zoom_controls = ZoomControlsWidget(self)
        self.zoom_controls.bind_image_label(self.image_label)
        self.nav_tb.addWidget(self.zoom_controls)

        self.view.set_zoom_controls_visible(False)

    @property
    def is_vtk(self) -> bool:
        return self.current_kind is not None and self.current_kind.startswith("vtk")


    def _open_preferences(self):
        """Show the visualization preferences dialog."""
        dlg = PreferencesDialog(self.viz, self)
        dlg.exec()

    def _open_gasp_options(self):
        """Show Gestational Age Similarity Profile options."""
        dlg = GASPOptionsDialog(self.viz, self)
        dlg.exec()

    def _on_viz_settings_changed(self):
        """Apply live visualization changes (VTK colors, view toggles)."""
        vs = self.viz
        if hasattr(self, "vtk_view"):
            self.vtk_view.renderer.SetBackground(*vs.vtk_background_rgbf)
            self.vtk_view.vtkWidget.GetRenderWindow().Render()
        if hasattr(self, "view"):
            self.view.set_zoom_controls_visible(vs.show_zoom_controls)
            self.view.label_overlay_enabled = bool(vs.show_label_overlay)
            if self.view.slice_nav_mode == "nifti" and hasattr(self, "slice_slider"):
                self.view.show_nifti_slice(self.slice_slider.value())

    def quit_app(self):
        """Gracefully shut down the application, stopping workers and timers first."""
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
        
    def closeEvent(self, e):
        """Restore original stdout/stderr and clean up the temp directory on exit.

        Args:
            e: The QCloseEvent.
        """
        sys.stdout = self._orig_stdout; sys.stderr = self._orig_stderr
        try: shutil.rmtree(self.temp_dir, ignore_errors=True); print(f"[Temp] Cleaned: {self.temp_dir}")
        except Exception as ex: print(f"[Temp] Cleanup error: {ex}")
        super().closeEvent(e)
        self.statusBar().clearMessage()


        

                
                    
                
            
            



    # ----- DnD -----
    def dragEnterEvent(self, e):
        """Accept drag events that carry file URLs.

        Args:
            e: The QDragEnterEvent.
        """
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        """Handle a file drop by routing the first URL to the appropriate loader.

        Args:
            e: The QDropEvent containing one or more file URLs.
        """
        urls = e.mimeData().urls()
        if not urls: return
        local = urls[0].toLocalFile()
        if not local: return
        print(f"Dropped: {local}")
        eext = ext(local)
        if eext in IMAGE_EXTS: self.file_mgr.load_image(local)
        elif eext in NIFTI_EXTS: self.file_mgr.load_nifti(local)
        elif eext == ".stl": self.file_mgr.load_stl(local)
        elif eext == ".vtk": self.file_mgr.load_vtk(local)
        elif eext == ".vti":
            rdr = vtkXMLImageDataReader(); rdr.SetFileName(local); rdr.Update(); img = rdr.GetOutput()
            self.vtk_view.show_image2d(img); self.view.show_widget(self.vtk_view); self.view.sync_slice_controls()
            print(f"VTI loaded (drop). Extent={img.GetExtent()} Spacing={img.GetSpacing()} Range={img.GetScalarRange()}"); self._set_current("vtk_image", local)
        else:
            QMessageBox.information(self, "Unsupported", f"Unsupported file: {local}")

    # ----- Action state mgmt -----
    def _set_current(self, kind: str | None, path: str | None):
        """Update the active file identity and enable/disable actions accordingly.

        Args:
            kind: File type string (e.g. "image", "nifti", "vtk_image",
                "vtk_poly", "vtk_surface", "stl", "Freesurfer",
                "Optimization") or None when closing.
            path: Absolute filesystem path of the active file, or None.
        """
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
            self.view.reset_png_navigation()
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
                self.act_perimeter_options,
                self.act_cnt_threshold,
                self.act_set_custom_label,
                self.act_set_image_scale,
                self.act_set_scale,
                self.act_meas_allmarks,
                self.act_meas_volumes,
                self.act_meas_area,
                self.act_meas_perimeter,
                self.act_meas_curve,
                self.act_meas_compactness,
                self.act_meas_lgi,
                self.act_meas_sulci,
                self.act_optimization,
                self.act_slice_thickness,
            ]:
                action.setEnabled(has_file)

        self.view.reset_png_navigation()
        self._update_process_actions()


    def _set_contour_accounting_enabled(self, flag: bool) -> None:
        """Enable/disable the Contour Accounting radio actions together."""
        for a in (self.act_contour_outer, self.act_contour_subtract,
                  self.act_contour_internal_only):
            a.setEnabled(flag)

    def _update_process_actions(self):
        """Enable or disable Process/Analysis menu actions based on the current file type.

        Each file kind (image, nifti, stl, vtk) supports a different
        subset of measurements; this method keeps the UI consistent.
        """
        kind = self.current_kind
        image_loaded = kind == "image"
        imported_item_loaded = kind is not None and kind != "Optimization"
        self.act_slice_kind_override.setVisible(image_loaded)
        self.act_slice_kind_override.setEnabled(image_loaded)
        self.act_sulcus_depth_threshold.setEnabled(imported_item_loaded)

        # Contour Accounting applies only to the 2-D image pipeline — single
        # images and planar meshes (which route through it); the image batch
        # loads its first image first, so that case is covered by the "image"
        # branch. Off by default (nothing applicable loaded); enabled per-kind
        # below. True 3-D STL/VTK and NIfTI keep it disabled.
        self._set_contour_accounting_enabled(False)

        # Surface-connected cavity correction applies to 3-D geometry only
        # (STL/VTK volumetric slicing and NIfTI); off by default and for 2-D
        # images / planar meshes.
        self.act_cavity_options.setEnabled(False)

        # Upscaling is a 2-D raster operation — only enabled when an image is
        # loaded (re-enabled in the "image" branch below).
        self.act_upscale_image.setEnabled(False)

        if kind == "stl" or (kind is not None and kind.startswith("vtk")):
            is_planar = self._flat_axis is not None
            self.act_meas_area.setEnabled(True)
            self.act_meas_perimeter.setEnabled(is_planar)
            self.act_meas_compactness.setEnabled(True)
            self.act_meas_lgi.setEnabled(True)
            self.act_meas_sulci.setEnabled(True)
            self.act_meas_volumes.setEnabled(not is_planar)
            self.act_meas_allmarks.setEnabled(True)
            self.act_nitfi2png.setEnabled(False)
            self.act_meas_curvature.setEnabled(False)
            self.act_meas_curve.setEnabled(False)
            self.act_slice_thickness.setEnabled(True)
            self.act_cavity_options.setEnabled(not is_planar)
            # 3-D slicing ignores contour_mode; planar meshes use the image path.
            self._set_contour_accounting_enabled(is_planar)
            self.act_set_image_scale.setEnabled(False)
            self.act_niftiextractor.setEnabled(False)
            self.act_set_scale.setEnabled(False)
            self.slice_slider.setEnabled(False)
            self.orient_combo.setEnabled(False)
            self.view_mode.setEnabled(False)
            self.act_set_image_scale.setEnabled(False)
            self.act_set_scale.setEnabled(False)
            self.act_perimeter_options.setEnabled(False)
            self.nav_tb.hide()
            self.view.set_zoom_controls_visible(False)


        if kind == "nifti":
            self.act_meas_area.setEnabled(True)
            self.act_meas_perimeter.setEnabled(False)
            self.act_meas_compactness.setEnabled(False)
            self.act_meas_lgi.setEnabled(True)
            self.act_meas_sulci.setEnabled(True)
            self.act_meas_volumes.setEnabled(True)
            self.act_meas_allmarks.setEnabled(True)
            self.act_choose_regions.setEnabled(True)
            self.label_overlay_enabled = True
            self.act_cavity_options.setEnabled(True)
            self._set_contour_accounting_enabled(False)
            self.act_nitfi2png.setEnabled(True)
            self.act_meas_curvature.setEnabled(False)
            self.act_meas_curve.setEnabled(False)
            self.act_set_image_scale.setEnabled(False)
            self.act_set_scale.setEnabled(False)
            self.act_slice_thickness.setEnabled(False)
            self.act_niftiextractor.setEnabled(True)
            self.slice_slider.setEnabled(True)
            self.orient_combo.setEnabled(True)
            self.view_mode.setEnabled(True)
            self.act_set_image_scale.setEnabled(False)
            self.act_set_scale.setEnabled(False)
            self.act_perimeter_options.setEnabled(True)
            self.nav_tb.show()
            self.view.set_zoom_controls_visible(False)
            for w in (self.orient_combo, self.view_mode, self.slice_caption, self.slice_slider, self.slice_value_label):
                w.setVisible(True)

        elif kind == "image":
            self._set_contour_accounting_enabled(True)
            self.act_meas_area.setEnabled(True)
            self.act_meas_perimeter.setEnabled(True)
            self.act_meas_compactness.setEnabled(True)
            self.act_meas_lgi.setEnabled(True)
            self.act_meas_volumes.setEnabled(False)
            self.act_meas_sulci.setEnabled(True)
            self.act_optimization.setEnabled(True)
            self.act_meas_allmarks.setEnabled(True)
            self.act_annotate_square.setEnabled(True)
            self.act_upscale_image.setEnabled(True)
            self.act_nitfi2png.setEnabled(False)
            self.act_hausdorf.setEnabled(True)
            self.act_meas_curvature.setEnabled(True)
            self.act_meas_stright.setEnabled(True)
            self.act_meas_curve.setEnabled(True)
            self.act_slice_thickness.setEnabled(False)
            self.act_niftiextractor.setEnabled(False)
            self.slice_slider.setEnabled(False)
            self.orient_combo.setEnabled(False)
            self.view_mode.setEnabled(False)
            self.act_set_image_scale.setEnabled(True)
            self.act_set_scale.setEnabled(True)
            self.act_perimeter_options.setEnabled(True)
            self.nav_tb.show()
            for w in (self.orient_combo, self.view_mode, self.slice_caption, self.slice_slider, self.slice_value_label):
                w.setVisible(False)
            self.view.set_zoom_controls_visible(True)

        if kind is not None and kind.startswith("vtk"):
            self.act_set_physical_dim.setEnabled(True)
        
        else:
            self.act_set_physical_dim.setEnabled(False)


            




    def _dir_has_files(self, d: str) -> bool:
        """Return True if directory *d* contains at least one file (recursive).

        Args:
            d: Directory path to inspect.

        Returns:
            True if any file exists under *d*, False otherwise.
        """
        if not d or not os.path.isdir(d):
            return False
        for _, _, files in os.walk(d):
            if files:
                return True
        return False
        
        
      
      
        
    
    def reset_view(self):
        """Reload the original file, clear all on-screen annotations, and reset navigation."""
        for meth in ("cancel_square_selection", "cancel_scalebar_measure", "clear_annotations", "clear_line_annotations"):
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
                self.view.show_widget(self.image_label)
                self.image_label.setImage(pm)
                self._active_view = "image"
                _hide_image_widgets()  # hides slider etc., label shown above via _show_widget

            elif kind == "nifti":
                # reuse your existing loader to reset orientation/slider
                self._set_current("nifti", path)
                self.view.on_view_changed(self.view_mode.currentText())

            elif kind == "stl" or (kind is not None and kind.startswith("vtk")):
                poly = _read_mesh(path)
                # show in your VTK view
                self.view.show_widget(self.vtk_view)
                self.vtk_view.show_polydata(poly)
                # hide image UI
                _hide_image_widgets()
                self._active_view = "vtk"

            # turn off any PNG navigation
            if hasattr(self, "disable_png_navigation"):
                self.view.disable_png_navigation()

            self.view.append_progress("\n[View] Reloaded and cleared annotations.\n")
            self.statusBar().showMessage("Reloaded; annotations cleared.", 3000)

        except Exception as ex:
            QMessageBox.warning(self, "Reset View", f"Could not reload:\n{ex}")

    def open_current_temp_folder(self):
        """Open the active output folder, falling back to the session temp dir."""
        folder = self.current_output_dir if self.current_output_dir else self.temp_dir
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self,
                "Open Temp Folder",
                f"Temp folder not found:\n{folder or '(none)'}",
            )
            return
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        if ok:
            print(f"[Temp] Opened folder: {folder}")
        else:
            QMessageBox.warning(
                self,
                "Open Temp Folder",
                f"Could not open folder:\n{folder}",
            )

    def open_home_folder(self):
        """Open the current user's home folder in the system file browser."""
        folder = str(Path.home())
        if not os.path.isdir(folder):
            QMessageBox.warning(
                self,
                "Open Home Folder",
                f"Home folder not found:\n{folder}",
            )
            return
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        if ok:
            print(f"[File] Opened home folder: {folder}")
        else:
            QMessageBox.warning(
                self,
                "Open Home Folder",
                f"Could not open folder:\n{folder}",
            )

    def show_user_guide(self):
        """Open the scrollable in-app user guide."""
        from widgets.guide_dialog import GuideDialog
        GuideDialog(self).exec()

    def show_about_info(self):
        QMessageBox.information(
            self,
            "About FetoMorph",
            "FetoMorph is a desktop platform that standardizes the measurement and "
            "analysis of fetal brain morphology, enabling consistent validation across "
            "different computational models of fetal brain development. It extracts "
            "quantitative descriptors—including surface area, perimeter, volume, "
            "gyrification, compactness, curvature, and sulcal profiles—from 2D image "
            "slices, 3D NIfTI volumes, and STL/VTK surface meshes.\n\n"
            "By applying the same analysis pipeline to simulated and real fetal brain "
            "data, FetoMorph provides a unified framework for comparing model outputs "
            "with age-specific reference statistics. Its Gestational Age Similarity "
            "Profile (GASP) objectively evaluates how closely each simulation reproduces "
            "realistic developmental morphology and timing. \n\n"

            "GitHub repository:\n" "https://github.com/SaeedZarzor/FetoMorph \n\n"
            "DOI:\n" "https://doi.org/10.5281/zenodo.21353636\n\n"          
        )

    def show_about_contributors(self):
        QMessageBox.information(
            self,
            "About FetoMorph - Contributors",
            "FetoMorph was developed at Institute of Continuum Mechanics and Biomechanics," 
            "Friedrich-Alexander-Universität Erlangen-Nürnberg, 91058, Erlangen, Germany.\n\n"
            "The project was carried out in collaboration with Department of Computing, " 
            "Imperial College London, London, SW7 2AZ, UK. \n\n"
            "Contributors\n\n"
            "Mohammad Saeed Zarzor — Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, and coding \n"
            "Qiang Ma — Conceptualization, Data curation, Data analysis \n"
            "Bernhard Kainz — Conceptualization, Resources \n"
            "Silvia Budday — Conceptualization, Supervision, Project administration, Funding acquisition \n\n"
        )

    def show_about_copyright(self):
        QMessageBox.information(
            self,
            "About FetoMorph - Copyright and License",
            "Copyright (c) 2026 Mohammad Saeed Zarzor.\n\n"
            "FetoMorph is licensed under the MIT License. See the LICENSE file "
            "for the complete license terms."
        )
    
    def show_about_citation(self):
        QMessageBox.information(
            self,
            "About FetoMorph - Citing",
            "When using FetoMorph in research, please cite:\n"
            "ZARZOR, M. S., Ma, Q., Kainz, B.& Budday, S. (2026)."
            "FetoMorph: A Unified Toolkit for Validating Computational"
            "Models of Human Brain Development (Version v1) [Computer software]."
            "Zenodo. European Congress on Computational Methods in Applied Sciences"
            "and Engineering (ECCOMAS), Munich, Germany."
            "  \n\n"
        )

    def show_about_acknowledgements(self):
        QMessageBox.information(
            self,
            "About FetoMorph - Acknowledgements",
            "Acknowledgement is given to the fetal brain imaging datasets provided "
            "by the developing Human Connectome Project (dHCP), undertaken by the "
            "KCL–Imperial–Oxford Consortium and funded by the European Research "
            "Council under the European Union’s Seventh Framework Programme "
            "(FP7/2007–2013), ERC Grant Agreement No. 319456, and ERC Consolidator "
            "Grant No. 101083647.\n\n"
            "FetoMorph was developed as part of the BRAINIACS project, funded by "
            "the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) "
            "through grant BU 3728/1-1, awarded to SB.\n\n"
            "We also sincerely thank Median Almurey, Ahmad Baradiei, Yanal Moulla, "
            "Rami Musleh, and Divyashree Doddbele for their valuable contributions, "
            "to the design implementation, testing, validation, and continued "
            "improvement of the software. \n\n"
            "FetoMorph is built with Python, PySide6/Qt, VTK, OpenCV, NumPy, "
            "pandas, openpyxl, Matplotlib, PyVista, and ONNX Runtime.\n\n"

        )
    
    def show_about_declaration(self):
        QMessageBox.information(
            self,
            "About FetoMorph - Declaration",
            "AI-Assisted Development\n\n"
            "Parts of FetoMorph were developed with the assistance of AI coding "
            "tools, including Claude Code (Anthropic, Claude Opus 4.8) and "
            "Codex 5.5 (OpenAI). These tools were used to support "
            "implementation, refactoring, testing, and documentation. All "
            "AI-assisted output was reviewed, tested, and validated by the "
            "authors, who take full responsibility for the correctness and "
            "scientific integrity of the software.\n\n"
            "Icon Credits\n\n"
            "Icons made by kliwir art, Freepik, Three musketeers, FetchLab, "
            "juicy_fish, Us and Up, Pixel perfect, Fathema Khanom, meaicon, "
            "Karyative, Iconic Panda, JessHG, FACH, Anggara, samlakodad, and Infinite "
            "Dendrogram from Flaticon."
        )

    def open_readme(self):
        readme_path = Path(__file__).resolve().parent / "README.md"
        if not readme_path.is_file():
            QMessageBox.warning(
                self,
                "ReadMe",
                f"README.md not found:\n{readme_path}",
            )
            return
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(readme_path)))
        if not ok:
            QMessageBox.warning(
                self,
                "ReadMe",
                f"Could not open README.md:\n{readme_path}",
            )


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
            default_n = self.metrics_store._roi_counter_by_source.get(base_name, 0) + 1
            default_label = f"ROI-{default_n}"
            label_text, ok = QInputDialog.getText(self, "Annotation Label", f"Label for {base_name}:", text=default_label)
            if not ok or not str(label_text).strip():
                text = default_label  # keep going even if dialog canceled

            self.image_label.add_square_annotation(img_rect, color=Qt.yellow, pen_width=2, fill_alpha=0)
                        
            out_dir = os.path.join(self.temp_dir, "roi_crops"); os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}_roi_{ts}.png")
            ok_save = cropped_qimg.save(out_path)
            self.metrics_store.annotation_records.append(str(label_text))
            self.metrics_store.annotations_by_source.setdefault(base_name, []).append(str(label_text))
            self.metrics_store._roi_counter_by_source[base_name] = default_n

            if ok_save:
                # keep the path(s) for later use
                self.last_annotated_path = out_path
                self.metrics_store.annotation_labels_by_path[out_path] = str(label_text or "")

                print(f"[Annotate] Saved ROI → {out_path}")
                self.view.append_progress(f"[Annotate] ROI {img_rect.getRect()} → {out_path}")
                self.current_output_dir = out_dir
                try:
                    self.statusBar().showMessage(f"Saved ROI: {out_path}", 5000)
                except Exception:
                    logger.debug("statusBar showMessage failed", exc_info=True)
            else:
                QMessageBox.warning(self, "Annotate", "Failed to save cropped image.")

        # show hint in the status bar (this line needs to be in a method -> self is defined)
        try:
            self.statusBar().showMessage("Drag to select a square region… (Esc to cancel)")
        except Exception:
            logger.debug("statusBar showMessage failed", exc_info=True)
        
        # start selection on the image widget

        self.image_label.start_square_selection(_on_done)
        self.image_label.setFocus()

    def upscale_current_image(self):
        """Upscale the current 2-D image (LANCZOS + sharpen) and reload it.

        Writes the result to ``<temp>/upscaled`` and loads it as the current
        image so subsequent measurements run on the sharper, higher-resolution
        frame. The burned-in scale bar is enlarged with the frame, so re-detect
        the scale on the upscaled image (do not divide the old mm/pixel by the
        factor).
        """
        from helpers import Upscaling

        if self.current_kind != "image" or not self.current_path:
            QMessageBox.warning(self, "Upscale Image", "Load a 2-D image first.")
            return

        scale, ok = QInputDialog.getInt(
            self, "Upscale Image", "Scale factor (×):",
            Upscaling.DEFAULT_SCALE, 2, 16, 1)
        if not ok:
            return

        src = self.last_annotated_path or self.current_path
        out_dir = os.path.join(self.temp_dir, "upscaled")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        out_path = os.path.join(out_dir, f"{base}_{scale}x.png")
        try:
            Upscaling.upscale_image_file(src, out_path, scale=scale)
        except Exception as ex:
            logger.error("Upscale failed: %s", ex)
            QMessageBox.critical(self, "Upscale Failed", f"{type(ex).__name__}: {ex}")
            return

        self.file_mgr.load_image(out_path)
        self.current_output_dir = out_dir
        print(f"[Upscale] {src} → {out_path} ({scale}x)")
        try:
            self.statusBar().showMessage(f"Upscaled {scale}× → {out_path}", 5000)
        except Exception:
            logger.debug("statusBar showMessage failed", exc_info=True)


    def get_label_for_cropped_path(self, path: str) -> str | None:
        """Return the annotation label for a saved cropped image path, or None."""
        if not path:
            return None
        # normalize to avoid mismatches
        key = os.path.abspath(os.path.expanduser(path))
        # fast path: direct dict
        if key in self.metrics_store.annotation_labels_by_path:
            return self.metrics_store.annotation_labels_by_path[key]
        # fallback: try raw key and search records if you keep them
        else:
            return None

#--------------- select and show labels -------------------------------


        
    def choose_regions_dock(self):
        """Show or create the ROI-selection dock for NIfTI label regions.

        Populates the dock with all labels present in the loaded volume
        and lets the user toggle which labels to include in measurements.
        The selection is stored in ``self.nifti_selected_regions`` when
        the user presses Apply.
        """
        # Setup context same as before
        self.slice_nav_mode = "nifti"
        idx = int(self.slice_slider.value()) if hasattr(self, "slice_slider") else 0
        self.view_mode.setCurrentText("2D")
        self.view_mode.setEnabled(False)
        self.view.nifti_set_orientation(self.orient_combo.currentText())
        self.view.on_slice_slider_changed(idx)

        # Determine labels
        labels_available = sorted(set(int(x) for x in self.labels_available))
        if not labels_available:
            QMessageBox.warning(self, "Regions", "No discrete labels detected in this NIfTI.")
            return

        # Prepare colors LUT
        if not hasattr(self, "nifti_label_lut"):
            self.nifti_label_lut = {}
        for lab in labels_available:
            self.nifti_label_lut.setdefault(lab, self.view._color_for_label(lab))

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
                    self.view.append_progress(f"[Regions] Selected labels: {sorted(selected)} \n")
                    self.statusBar().showMessage(f"Regions set: {sorted(selected)}", 3000)
                except Exception:
                    print("[Regions] Selected:", sorted(selected))
                # Optional: refresh display
                if hasattr(self, "view") and hasattr(self.view, "show_nifti_slice"):
                    idx2 = int(self.slice_slider.value()) if hasattr(self, "slice_slider") else 0
                    self.view.show_nifti_slice(idx2)
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



    def ask_processing_options(self):
        """Show a dialog for optional image-processing tweaks (colour, smoothing, scalebar).

        Returns:
            A dict with keys ``unify_color``, ``add_scale_bar``,
            ``smooth_kind``, and ``smooth_strength``, or None if the
            user cancels.
        """
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
        """Convert the currently displayed NIfTI slice to a standalone PNG image.

        Grabs the on-screen slice, applies user-chosen processing options
        (colour unification, smoothing, scalebar), saves the result to a
        temp folder, and loads it as a regular image for further 2-D
        measurements.
        """
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
        self.file_mgr.load_image(out_path)

# -------------------------------------------------------

    def Nifti_extractor(self):
        """Extract selected label regions from the current NIfTI volume into a new file.

        Creates a masked NIfTI containing only the voxels whose labels
        are in ``nifti_selected_regions`` (or all available labels if
        none are selected), saves it to a temp directory, and loads the
        result.
        """
        uid = uuid.uuid4().hex[:8]
        out_dir = os.path.join(self.temp_dir, f"nifti_allmarks_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        self.current_output_dir = out_dir
    
        labels = self.nifti_selected_regions if self.nifti_selected_regions else self.labels_available

        nii_output = nifti_extractor (self, self.current_path, out_dir, valid_labels = labels)
                
        self.file_mgr.load_nifti(nii_output)
# ---- Metrics Dock (per-path, reads from self.metrics) -----------------------


# ------------------ hausdorff helpers ------------------

    def wait_for_resume(self):
        """Block the current call stack until the user presses Shift+Alt+E.

        Uses a nested QEventLoop so the GUI remains responsive while
        waiting for the resume shortcut.
        """
        loop = QEventLoop(self)
        # unique connect; if already connected, reconnect
        try:
            self._resume_sc.activated.connect(loop.quit, Qt.UniqueConnection)
        except TypeError:
            try: self._resume_sc.activated.disconnect(loop.quit)
            except Exception: logger.debug("disconnect before reconnect", exc_info=True)
            self._resume_sc.activated.connect(loop.quit)
        loop.exec()
        # cleanup for next time
        try: self._resume_sc.activated.disconnect(loop.quit)
        except Exception: logger.debug("disconnect after exec", exc_info=True)

    def _enter_adjustment_mode(self):
        """Restrict the UI to annotation/scale/kernel actions only.

        Used during batch processing and Hausdorff workflows so the user
        can adjust the image before confirming with Shift+Alt+E.
        """
        allow = { self.act_annotate_square, self.act_cnt_threshold, self.act_set_scale, self.act_set_image_scale, self.act_set_custom_label, self.act_kernel_size, self.act_perimeter_options}

        for a in self.all_actions:
            if a in allow:
                a.setEnabled(True)
            else:
                a.setEnabled(False)

    def _exit_adjustment_mode(self):
        """Re-enable all actions after leaving the restricted adjustment mode."""
        for a in self.all_actions:
            a.setEnabled(True)
            self._update_process_actions()


    def annotation_con(self, out_dir):
        """Convert the current (possibly annotated) image and record metrics.

        Used as a helper by the Hausdorff workflow to prepare each image
        before distance computation.

        Args:
            out_dir: Directory to save the converted image artefacts.

        Returns:
            Tuple of (annotated_bgr, basename, contour_array, custom_label).
        """
        image_path = self.current_path
        if self.last_annotated_path is not None:
            image_path = self.last_annotated_path

        annotated, basename, array = convert_image(image_path, out_dir, pixel_spacing= self.pixel_size, min_contour_area=self.cnt_threshold)

        label_text = self.get_label_for_cropped_path(image_path)
        if label_text and annotated is not None:
            annotated = put_label_on_bgr(annotated, label_text, pos="topleft")
    
        self.metrics_store.record_metric_for(self.current_path, label=label_text ,
                pixel_size_units = f"{self.units_length}/pixel",
                pixel_size = self.pixel_size,
                unit = self.units_length)
        label = getattr(self, "custom_label", None)
        return annotated, basename, array, label
        
    def ask_align_direction(self):
        """Prompt the user to choose an alignment direction for Hausdorff comparison.

        Returns:
            One of "right_bottom", "left_top", "left", "up", "none",
            or None if the user cancels.
        """
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
        """Load and display one or both FreeSurfer surface hemispheres in the VTK viewer.

        Prompts the user to choose between loading a single hemisphere
        or both (lh + rh).  The surfaces are rendered via the VTK viewer
        and the application state is set to "Freesurfer" mode.
        """
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
            self.view.show_widget(self.vtk_view)
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
            self.view.show_widget(self.vtk_view)
            self.vtk_view.show_pial_both(lh_file, rh_file)

        self._set_current("Freesurfer", self.Freesurfer_record[0]['path'])
        
        
    def view_morph_map(self):
        """Overlay a FreeSurfer scalar morph map (sulc, thickness, curv) on loaded surfaces.

        Requires that FreeSurfer surfaces have already been loaded via
        ``view_freesurfer_surfaces``.  Prompts the user to select
        matching morph files for each loaded hemisphere.
        """
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
            self.view.show_widget(self.vtk_view)
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
                
            self.view.show_widget(self.vtk_view)
            self.vtk_view.show_freesurfer_morph_both(lh_file,lh_morph, rh_file,rh_morph)

        else:
            return
# ================= ِExamples ============================

    def choose_gestational_week_2D_fill(self):
        """Open a filled 2D fetal brain section for a user-chosen gestational week."""
        dlg = GestationalWeeksDialog(self, initial=24)
        if dlg.exec() != QDialog.Accepted:
            return
        week = dlg.value()
        axis = dlg.axis().lower()
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Examples", "full_slices", str(week), axis)
        if not os.path.isdir(base):
            QMessageBox.warning(self, "Not Found", f"No data folder for week {week} / {axis}.\n{base}")
            return
        browser = ImageBrowserDialog(self, folder=base, title=f"Week {week} — {axis.capitalize()} — Select Image")
        if browser.exec() != QDialog.Accepted:
            return
        path = browser.selected_path()
        if not path:
            return
        self.file_mgr.load_image(path)

    def choose_gestational_week_2D_cropped(self):
        """Open a cropped 2D fetal brain section for a user-chosen gestational week."""
        dlg = GestationalWeeksDialog(self, initial=24)
        if dlg.exec() != QDialog.Accepted:
            return
        week = dlg.value()
        axis = dlg.axis().lower()
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Examples", "cropped_slices", str(week), axis)
        if not os.path.isdir(base):
            QMessageBox.warning(self, "Not Found", f"No data folder for week {week} / {axis}.\n{base}")
            return
        browser = ImageBrowserDialog(self, folder=base, title=f"Week {week} — {axis.capitalize()} — Select Image")
        if browser.exec() != QDialog.Accepted:
            return
        path = browser.selected_path()
        if not path:
            return
        self.file_mgr.load_image(path)

    def choose_gestational_week_3D_surface_mri(self):
        """Open an example 3D fetal surface-MRI NIfTI for a user-chosen week."""
        dlg = GestationalWeeksDialog(self, initial=24, show_axis=False)
        if dlg.exec() != QDialog.Accepted:
            return
        week = dlg.value()
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Examples", "fetal_surface_MRI", f"{week}week")
        path = os.path.join(base, "seg.nii")
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Not Found",
                                f"No surface-MRI NIfTI for week {week}.\n{path}")
            return
        self.file_mgr.load_nifti(path)

    def choose_gestational_week_3D_brain_stl(self):
        """Open an example 3D fetal brain surface mesh (STL) for a chosen week."""
        dlg = GestationalWeeksDialog(self, initial=24, show_axis=False)
        if dlg.exec() != QDialog.Accepted:
            return
        week = dlg.value()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Examples", "fetal_brain_stl", f"{week}.stl")
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Not Found",
                                f"No brain STL for week {week}.\n{path}")
            return
        self.file_mgr.load_stl(path)


# ---------------------------
# Entry point
# ---------------------------
def main():
    """Launch the FetoMorph application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("FetoMorph")
    app.setApplicationDisplayName("FetoMorph")

    # One consistent slate/teal look across every window and dialog.
    from theme import apply_theme
    apply_theme(app)

    win = MainWindow();
    win.setWindowTitle("FetoMorph")
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
