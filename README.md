# FetoMorph

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

A desktop application for morphometric analysis of fetal brain data. FetoMorph supports multiple imaging modalities — 2D histological slices, 3D volumetric scans (NIfTI), and surface meshes (STL/VTK) — and provides a comprehensive set of measurement, visualization, and optimization tools.

Built with PySide6 (Qt6), VTK, and OpenCV.

---

## Table of Contents

- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Supported File Formats](#supported-file-formats)
- [Major Features](#major-features)
  - [2D Image Measurements](#2d-image-measurements)
  - [3D Volumetric Measurements (NIfTI)](#3d-volumetric-measurements-nifti)
  - [Surface Mesh Measurements (STL/VTK)](#surface-mesh-measurements-stlvtk)
  - [Curvature Analysis](#curvature-analysis)
  - [Hausdorff Distance](#hausdorff-distance)
  - [Batch Processing](#batch-processing)
  - [Batch Measurement Reports](#batch-measurement-reports)
  - [Multi-Objective Optimization](#multi-objective-optimization)
  - [FreeSurfer Integration](#freesurfer-integration)
  - [Visualization](#visualization)
  - [Scale and Unit Configuration](#scale-and-unit-configuration)
  - [Export](#export)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Project Structure](#project-structure)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- On macOS, GTK+ libraries may be required:
  ```bash
  brew install gtk+ glib gobject-introspection cairo
  ```
- On Linux (Debian/Ubuntu):
  ```bash
  sudo apt-get install libgtk-3-dev libglib2.0-dev libcairo2-dev
  ```

### Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd FetoMorph
   ```

2. Create and activate a virtual environment:

   macOS / Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   py -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

### Core Dependencies

| Package | Purpose |
|---------|---------|
| PySide6 | Qt6 GUI framework |
| vtk | 3D rendering and visualization |
| pyvista / pyvistaqt | Pythonic VTK interface with Qt embedding |
| numpy | Numerical computing |
| scipy | Scientific computing and optimization |
| opencv-python | Image processing and contour analysis |
| nibabel | NIfTI and FreeSurfer file I/O |
| trimesh | Triangle mesh manipulation |
| matplotlib | Plotting (curvature profiles, optimization results) |
| pandas | Tabular data handling |
| openpyxl | Excel export |
| scikit-image | Image processing algorithms (marching cubes) |
| pillow | Additional image format support |

---

## Running the Application

```bash
python FetoMorph.py
```

The main window opens at 1200x900 with a ribbon toolbar, menu bar, 2D/3D viewer, and a results dock panel.

---

## Supported File Formats

### Input

| Format | Extensions |
|--------|-----------|
| 2D Images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.gif` |
| NIfTI Volumes | `.nii`, `.nii.gz` |
| VTK Legacy | `.vtk` |
| STL Meshes | `.stl` |
| FreeSurfer Surfaces | `.pial`, `.white`, `.inflated` |

### Output

| Format | Content |
|--------|---------|
| Excel (`.xlsx`) | Measurement metrics |
| PNG / JPEG | Annotated images, screenshots, plots |
| NIfTI (`.nii.gz`) | Extracted label regions |
| STL | Converted pial surfaces |

---

## Major Features

### 2D Image Measurements

Compute morphometric hallmarks from brain slice images:

- **Area** — cross-sectional area in physical units
- **Perimeter** — boundary length of the brain contour
- **Convex Perimeter** — outer perimeter after morphological closing
- **LGI (Local Gyrification Index)** — ratio of inner to outer perimeter
- **Sulci Depth** — min, max, and mean depth of convexity defects
- **Sulci Count** — number of detected sulcal folds

The processing pipeline uses binary thresholding, OpenCV contour detection, morphological closing with a configurable elliptical kernel, and convexity defect analysis.

### 3D Volumetric Measurements (NIfTI)

Load NIfTI segmentation volumes and compute:

- **Volume** (cm^3) — integrated across all slices
- **Surface Area** (cm^2) — sum of per-slice areas
- **Dimensions** — physical lengths along PA, LR, and IS axes
- **LGI** — global gyrification index
- **Sulci Depth Statistics** — aggregated across the volume

Supports FreeSurfer label filtering with a default set of cortical regions. Voxel size is extracted automatically from the NIfTI header.

### Surface Mesh Measurements (STL/VTK)

Slice through surface meshes along configurable axes and compute:

- **Area** and **Volume** from rendered cross-sections
- **LGI** and **Sulci Depth** via per-slice contour analysis
- **Physical Dimension Scaling** — define real-world X/Y/Z dimensions
- **Automatic Scale Calibration** — red reference cube detection for mm/px conversion

VTK meshes support configurable slice direction (X, Y, or Z axis).

### Curvature Analysis

- Compute local curvature along contours using polynomial fitting
- Classify regions as convex (positive) or concave (negative)
- Configurable neighbourhood window size
- Output: curvature profile plot saved as PNG

### Hausdorff Distance

Compare two contours with:

- **Directed Hausdorff** — max distance from set A to nearest point in set B
- **Symmetric Hausdorff** — max of both directions
- **Alignment modes**: right-bottom, left-top, or centroid
- Output: annotated comparison image

### Batch Processing

- Select a folder of 2D slices for automated processing
- All hallmark measurements computed per image
- Dynamic threshold adjustment when LGI falls below 1
- Annotated output images and Excel summary generated automatically

### Batch Measurement Reports

This branch includes a reproducible command-line workflow that runs the image
measurement pipeline across gestational-week and anatomical-axis folders. It
writes one Excel report per folder and can add statistical summaries and
boxplots to each workbook.

Run the supplied full-slice example configuration:

```powershell
python scripts\run_master_measurement_reports.py --config scripts\master_measurement_reports_config.example.json
```

Run the cropped-slice configuration:

```powershell
python scripts\run_master_measurement_reports.py --config scripts\master_measurement_reports_cropped_config.example.json
```

Analyze the generated workbooks:

```powershell
python scripts\analyze_master_measurement_reports.py --input-root measurements
```

See [scripts/run_master_measurement_reports.md](scripts/run_master_measurement_reports.md)
for configuration fields, calibration choices, review images, output layout,
and additional examples.

### Multi-Objective Optimization

NSGA-II / NSGA-III optimization over slice measurement data:

- **Objectives**: LGI, max/min/mean sulci depth, area, cell density
- **Per-objective direction**: maximize or minimize
- **Constraints**: upper bounds on cell density, sulci count, max depth
- **Configurable generations** (default: 200)
- **Output**: Pareto front visualization and optimal slice indices exported to Excel

### FreeSurfer Integration

- Load and render FreeSurfer surfaces (pial, white, inflated, sphere)
- Display morph map overlays (sulcal depth, thickness, curvature)
- Convert pial surfaces to STL format (single or combined hemispheres)
- Label-based coloring with lookup table support

### Visualization

#### 2D Viewer
- Aspect-preserving image display with zoom
- Contour overlays (red: brain boundary, green: convex hull)
- Sulci depth markers with color-coded indicators
- Interactive line drawing for scale-bar calibration
- Rectangle ROI selection with auto-save

#### 3D VTK Viewer
- Surface mesh rendering with lighting
- GPU-accelerated volume rendering
- Axis-aligned slice navigation (sagittal, coronal, axial)
- Color window/level adjustment
- Orientation marker widget

#### NIfTI Slice Viewer
- Browse slices along any anatomical axis
- Segmentation label overlay with toggle
- Label-to-color mapping from FreeSurfer LUT

#### Result Viewer
- Cycle through annotated result images (Ctrl+M / Ctrl+Shift+M)
- Scatter plots for optimization Pareto fronts
- Curvature profile line charts

### Scale and Unit Configuration

- **Manual entry**: set unit (mm, um, cm, m, in, or custom) and pixel size
- **Scale-bar calibration**: draw a line on the image and enter the physical length
- **Mesh dimensions dialog**: define real-world X/Y/Z sizes for VTK meshes with live 3D preview
- Per-file scale tracking

### Export

- **Excel**: all metrics in a multi-row spreadsheet with metadata columns (file path, parameters, unit, annotation)
- **View screenshot**: PNG or JPEG of the current display (2D or 3D)
- **Data export**: copy result folders or individual files to a chosen destination
- **NIfTI region extraction**: save filtered label masks as compressed `.nii.gz`

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Import image |
| Ctrl+Shift+V | Import VTK file |
| Ctrl+Shift+L | Import STL file |
| Ctrl+Shift+N | Import NIfTI file |
| Ctrl+V | Save view as image |
| Ctrl+Shift+S | Save data |
| Ctrl+E | Export metrics to Excel |
| Ctrl+R | Reset view |
| Ctrl+W | Close current file |
| Ctrl+Q | Quit |
| Ctrl+T | Set contour threshold |
| Ctrl+Shift+A | Annotation (rectangle ROI) |
| Ctrl+Shift+R | ROI region selection |
| Ctrl+M | Next result image |
| Ctrl+Shift+M | Previous result image |
| Ctrl+Alt+F | Example: filled 2D sections |
| Ctrl+Alt+C | Example: cropped 2D sections |

---

## Project Structure

```
FetoMorph/
├── FetoMorph.py              # Main application entry point
├── deps.py                   # Centralized dependency imports
├── constants.py              # Application constants and defaults
├── icons.py                  # Icon loader
├── ribbon.py                 # Office-style ribbon toolbar
├── requirements.txt          # Pinned Python dependencies
│
├── widgets/                  # Custom UI dialogs and components
│   ├── scaled_image_label.py       # 2D image viewer with measurements
│   ├── vtk_viewer.py               # 3D VTK rendering widget
│   ├── contour_threshold.py        # Contour threshold dialog
│   ├── kernel_size.py              # Morphology kernel size dialog
│   ├── slice_thickness.py          # Slice thickness dialog
│   ├── unit_scale.py               # Unit and pixel size dialog
│   ├── scalebar_set_scale.py       # Scale-bar calibration dialog
│   ├── options_dialog.py           # Processing options dialog
│   ├── geometry_dialog.py          # 3D mesh dimension editor
│   ├── region_dock.py              # NIfTI region selection dock
│   ├── gestational_weeks_dialog.py # Gestational week and axis selector
│   ├── image_browser_dialog.py     # Thumbnail image browser
│   ├── optimization_widgets.py     # Optimization configuration dialog
│   └── recent_paths.py             # Recent file management
│
├── managers/                 # Application state, files, views, and dispatch
│   ├── file_manager.py
│   ├── measurement_dispatcher.py
│   ├── metrics_store.py
│   ├── settings_manager.py
│   └── view_manager.py
│
├── functions/                # Measurement and processing algorithms
│   ├── measurements_image.py       # 2D image morphometrics
│   ├── measurements_nifti.py       # NIfTI volumetric analysis
│   ├── measurements_stl.py         # STL mesh measurements
│   ├── measurements_vtk.py         # VTK mesh measurements
│   ├── measurement_batch.py        # Batch image processing
│   ├── curvature.py                # Curvature profiling
│   ├── hausdorff.py                # Hausdorff distance computation
│   ├── nifti_to_image.py           # NIfTI to PNG slice extraction
│   ├── nifti_to_stl.py             # NIfTI to STL conversion
│   ├── pial_to_stl.py              # FreeSurfer pial to STL
│   ├── nii_extractor.py            # FreeSurfer region extraction
│   └── optimization.py             # NSGA-II/III optimization
│
├── helpers/                  # Utility modules
│   ├── helpers.py                  # Common helper functions
│   ├── read_excel.py               # Excel file reading
│   ├── check_mesh.py               # Mesh validation
│   └── slice_kind_classifier.py    # ONNX slice classifier
│
├── scripts/                  # Batch reporting and model utilities
│   ├── run_master_measurement_reports.py
│   ├── analyze_master_measurement_reports.py
│   ├── master_measurement_reports_config.example.json
│   ├── master_measurement_reports_cropped_config.example.json
│   └── run_master_measurement_reports.md
│
├── models/
│   └── slice_kind_cnn.onnx         # Slice-orientation classifier
│
├── assets/
│   └── icons/                # UI icons (PNG)
│
└── Examples/                 # Sample fetal brain data
    ├── full_slices/                # Full anatomical sections by week
    └── cropped_slices/             # Cropped sections by week
```
