# FetoMorph

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

FetoMorph is a desktop platform that standardizes the measurement and analysis of fetal brain morphology, enabling consistent validation across different computational models of fetal brain development. It extracts quantitative descriptors including surface area, perimeter, volume, gyrification, compactness, curvature, and sulcal profiles from 2D image slices, 3D NIfTI volumes, and STL/VTK surface meshes.

By applying the same analysis pipeline to simulated and real fetal brain data, FetoMorph provides a unified framework for comparing model outputs with age-specific reference statistics. Its Gestational Age Similarity Profile (GASP) evaluates how closely each simulation reproduces realistic developmental morphology and timing.

Built with PySide6 (Qt6), VTK, OpenCV, and an ONNX deep-learning model for slice-kind classification.

- GitHub repository: <https://github.com/SaeedZarzor/FetoMorph>
- DOI: <https://doi.org/10.5281/zenodo.21353636>
---

## Table of Contents

- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Supported File Formats](#supported-file-formats)
- [Major Features](#major-features)
  - [2D Image Measurements](#2d-image-measurements)
  - [Sulci Classification](#sulci-classification)
  - [Slice-Kind Classifier (Deep Learning)](#slice-kind-classifier-deep-learning)
  - [3D Volumetric Measurements (NIfTI)](#3d-volumetric-measurements-nifti)
  - [Surface Mesh Measurements (STL/VTK)](#surface-mesh-measurements-stlvtk)
  - [Cavity Correction](#cavity-correction)
  - [Perimeter Estimation Methods](#perimeter-estimation-methods)
  - [Compactness](#compactness)
  - [Curvature Analysis](#curvature-analysis)
  - [Hausdorff Distance](#hausdorff-distance)
  - [GASP — Gestational Age Similarity Profile](#gasp--gestational-age-similarity-profile)
  - [Batch Processing](#batch-processing)
  - [Multi-Objective Optimization](#multi-objective-optimization)
  - [FreeSurfer Integration](#freesurfer-integration)
  - [Visualization](#visualization)
  - [Preferences](#preferences)
  - [Scale and Unit Configuration](#scale-and-unit-configuration)
  - [Export](#export)
- [Contributors](#contributors)
- [Copyright and License](#copyright-and-license)
- [Citing FetoMorph](#citing-fetoMorph)
- [Acknowledgements](#acknowledgements)
- [Icon Credits](#icon-credits)
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
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
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
| trimesh / numpy-stl | Triangle / STL mesh manipulation |
| matplotlib | Plotting (curvature profiles, optimization results) |
| pandas | Tabular data and reference-profile handling |
| openpyxl | Excel export |
| scikit-image | Image processing algorithms (marching cubes) |
| scikit-learn | Reference-profile statistics and ML utilities |
| pymoo | NSGA-II / NSGA-III multi-objective optimization |
| onnxruntime / onnx | Slice-kind CNN inference |
| torch | Slice-kind CNN training (offline scripts) |
| pillow | Additional image format support |

> **Note:** `onnxruntime` is required only for slice-kind classification and is loaded lazily. If it (or the model file) is missing, the app still runs — image slices simply fall back to the fixed sulci-depth rule. `torch` is needed only for re-training the model under `scripts/`.

---

## Running the Application

```bash
python FetoMorph.py
```

The main window opens at 1200×900 with a ribbon toolbar, a menu bar (File, Measures, Analysis, Process, Adjustments, FreeSurfer, Examples, Settings, About), a 2D/3D viewer, an embedded output console, and a results dock panel.

---

## Supported File Formats

### Input

| Format | Extensions |
|--------|-----------|
| 2D Images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.gif` |
| NIfTI Volumes | `.nii`, `.nii.gz` |
| VTK Legacy | `.vtk` |
| STL Meshes | `.stl` |
| FreeSurfer Surfaces | `.pial`, `.white`, `.inflated`, `.sphere` |

### Output

| Format | Content |
|--------|---------|
| Excel (`.xlsx`) | Measurement metrics, per-class sulci tables, GASP reports |
| PNG / JPEG | Annotated images, screenshots, plots |
| NIfTI (`.nii.gz`) | Extracted label regions |
| STL | Converted pial surfaces |

---

## Major Features

### 2D Image Measurements

Compute morphometric hallmarks from brain slice images:

- **Area** — cross-sectional area in physical units
- **Perimeter** — boundary length of the brain contour (arc-length or Crofton)
- **Convex Perimeter** — outer perimeter after morphological closing
- **Curve Length** / **Straight** — measured along or across the contour
- **LGI (Local Gyrification Index)** — ratio of inner to outer perimeter
- **Compactness** — how closely the shape approaches the most space-efficient form
- **Sulci Depth** — min, max, and mean depth of convexity defects
- **Sulci Count** — number of detected sulcal folds, split per class

The pipeline binarises each mask with **Otsu's method** (automatic per-image threshold), detects contours with OpenCV, applies morphological closing with a configurable elliptical kernel (default 5 mm diameter), and analyses convexity defects. Kernel size, contour-area threshold, and perimeter method are all user-configurable from the **Adjustments** menu.

For 2D images, **Adjustments → Slice Kind Override…** can force the automatic slice-kind classifier to treat the current image as sagittal, coronal, axial, or cropped. This override is shown only while an image is loaded. **Adjustments → Sulcus Depth Threshold…** becomes active after an item is imported and sets the minimum sulcus depth counted across measurements.

### Sulci Classification

Each detected sulcus is binned by its depth as a fraction of the brain's slice length into four colour-coded classes:

| Class | Depth (% of slice length) | Default colour |
|-------|---------------------------|----------------|
| Primary | 15–50% | red |
| Secondary | 5–15% | gold/orange |
| Tertiary | 1.5–5% | cyan |
| Unclassified | outside all ranges | light gray |

Per-class counts and depth statistics (count, raw values, min/max/mean) are written to the Excel export, and markers are drawn on the annotated image in the class colour (colours are user-customisable in **Preferences**).

### Slice-Kind Classifier (Deep Learning)

A tiny ONNX convolutional neural network (`models/slice_kind_cnn.onnx`) labels each 2D image as a full MRI slice — **sagittal**, **coronal**, or **axial** — or as a cropped sub-slice band (`not_full_slice`). The classifier:

- reframes each input (tight-crops the brain, pads to a centred square) to match the training layout, so wide letterboxed renders are not distorted;
- decides whether the sulci-depth filter uses the fixed 0.5 mm rule (cropped bands) or a percent-of-slice-length rule (full slices);
- is loaded lazily and degrades gracefully — if `onnxruntime` or the model file is unavailable, images are treated as `not_full_slice`.

Training, evaluation, and cross-validation scripts live in `scripts/` (`train_slice_kind_cnn.py`, `eval_slice_kind_cnn.py`, `cv_slice_kind_cnn.py`).

### 3D Volumetric Measurements (NIfTI)

Load NIfTI segmentation volumes and compute:

- **Volume** (cm³) — integrated across all slices
- **Surface Area** (cm²) — sum of per-slice areas
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

VTK meshes support a configurable slice direction (X, Y, or Z axis). Thin surface meshes (e.g. FreeSurfer pial) can be **filled at render time** (`vtkStripper` + `vtkContourTriangulator`) so each cross-section reads as a solid region rather than a hollow boundary curve, while concavities and genuine enclosed voids are preserved.

### Cavity Correction

A surface-connected cavity (a hole that opens onto the outside of the brain) is corrected during slice-by-slice 3D integration:

- its area is **subtracted** from the cross-section before the volume integral, and
- its wall perimeter is **added** to the 3D surface area.

Fully-enclosed internal voids are left untouched (treated as solid). Surface-connectivity is recovered exactly for NIfTI (`scipy.ndimage.binary_fill_holes`) and by cross-slice cavity tracking in physical-mm coordinates for STL/VTK. GI/LGI is deliberately unchanged — cavity walls never enter the gyrification perimeter sums. Enable/disable and set an area threshold from **Adjustments → Cavity correction options**.

### Perimeter Estimation Methods

Perimeter defaults to a 4-direction **Crofton** estimator (`Adjustments → Perimeter Method…`), which measures the filled binary mask after local 2D isotropic resampling, reducing curvature and pixel-grid bias for noisy rasterized boundaries. The legacy OpenCV `cv2.arcLength` polygonal length is also available. Crofton can under-estimate straight axis-aligned edges, so it is applied to both LGI perimeter legs; the GI ratio stays similar, while absolute surface area and compactness benefit most. STL/VTK workflows keep `arcLength` because their dominant error comes from rendered screenshot resolution.

### Compactness

A 2D shape-compactness metric (area vs. perimeter) quantifying how closely a cross-section approaches the most space-efficient form. Available from the **Analysis** menu and included in the all-hallmarks export.

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

### GASP — Gestational Age Similarity Profile

GASP compares a brain's morphometrics (area, perimeter, LGI, compactness, sulci counts, and per-class sulcus values) against reference statistics for each gestational week (24–38) and axis (axial / coronal / sagittal), returning a per-week similarity score that may help estimate developmental age.

- **Reference profiles** are loaded from `Examples/gestational_week_reference.csv` (one row per week × axis, with n / mean / std / min / max per metric).
- **Cropped images** use `Examples/gestational_week_cropped_reference.csv` and normalized comparison metrics by default, avoiding raw unit-based area, perimeter, and raw sulcal-depth comparisons.
- **Normalized comparison** uses LGI, compactness, total sulcus count, and normalized total sulcal depth when those columns are available in the reference CSV. Full-section images can opt into normalized comparison from **Adjustments → GASP Options**.
- **Scoring** supports a Gaussian (weighted-mean of per-metric similarities) or Mahalanobis (weighted squared z-scores) model, with an optional out-of-range penalty.
- **Configurable** per-metric weights and penalty behaviour via **Adjustments → GASP Options**.
- **Manual entry** dialog lets you run GASP from hand-entered hallmark values without re-measuring.
- **Output**: a per-run results folder with a similarity-score-per-week Excel report and the source image.

### Batch Processing

- Select a folder of 2D slices for automated processing
- All hallmark measurements (including per-class sulci) computed per image
- Dynamic threshold adjustment when LGI falls below 1
- Annotated output images and an Excel summary generated automatically

### Multi-Objective Optimization

NSGA-II / NSGA-III optimization (via **pymoo**) over slice measurement data.
Select one or more measurement Excel files and the optimizer finds the
Pareto-optimal slices.

- **Objectives**: built from the columns of the selected files, so any metric
  the measurement pipeline exports can be optimized — LGI, area, perimeter,
  compactness, max/min/mean sulci depth, normalized depth, sulci counts (total
  and per class), cell density. Add an objective per row, each with its own
  **maximize / minimize** direction. At least two are required: the optimizer
  trades objectives off against each other, so a single objective is just a
  sort.
- **Constraints**: bound any column from either side — `CellDensity ≤ 2500`,
  `SulciCount ≥ 2`. Add as many as needed; only slices satisfying all of them
  are considered. Values are limited to the range present in the data.
- **Algorithm**: NSGA-II (up to 3 objectives) or NSGA-III (any number).
  NSGA-II is disabled automatically beyond 3 objectives.
- **Configurable generations** (default: 200)
- **Output**: `Pareto-optimal solutions.xlsx`, a scatter plot per objective
  pair, and `optimization_parameters.txt` recording the exact configuration
  for reproducibility

Per-sulcus columns (`Primary_depth_1`, `Unclassified_depth_2_norm`, …) are not
offered — they describe individual sulci rather than the slice, and how many
exist varies per file. The aggregates derived from them (max/min/mean depth)
are offered instead.

### FreeSurfer Integration

- Load and render FreeSurfer surfaces (pial, white, inflated, sphere)
- Display morph map overlays (sulcal depth, thickness, curvature)
- Convert pial surfaces to STL format (single or combined hemispheres)
- Label-based coloring with lookup table support

### Visualization

#### 2D Viewer
- Aspect-preserving image display with zoom controls
- Contour overlays (red: brain boundary, green: convex hull)
- Class-coloured sulci depth markers
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

### Preferences

A tabbed **Preferences** dialog (Settings menu) centralises visualization settings, applied live and persisted between sessions:

- **Text and Sizes** — contour thickness, text scale, and marker-radius multipliers
- **Colors** — boundary, convex-hull, and per-class sulcus marker colours (named or custom)
- **View** — 3-D viewer toggles and display options

### Scale and Unit Configuration

- **Manual entry**: set unit (mm, µm, cm, m, in, or custom) and pixel size
- **Scale-bar calibration**: draw a line on the image and enter the physical length
- **Mesh dimensions dialog**: define real-world X/Y/Z sizes for VTK meshes with live 3D preview
- Per-file scale tracking

### Export

- **Excel**: all metrics in a multi-row spreadsheet with metadata columns (file path, parameters, unit, annotation) plus per-class sulci tables, shared across every exporter
- **View screenshot**: PNG or JPEG of the current display (2D or 3D)
- **Data export**: copy result folders or individual files to a chosen destination
- **NIfTI region extraction**: save filtered label masks as compressed `.nii.gz`
- **GASP report**: per-week similarity scores and source image in a dedicated results folder

### File and About Menus

- **File → Open Home Folder…** opens the current user's home directory.
- **File → Open Current Temp Folder…** opens the active processing output folder, or the session temp folder if no output folder is active.
- **About** contains About FetoMorph, User Guide, Contributors, Acknowledgements, Copyright and License, Citing FetoMorph, Icon Credits, and ReadMe entries.

---


## Contributors

FetoMorph was developed at the Institute of Continuum Mechanics and Biomechanics, Friedrich-Alexander-Universität Erlangen-Nürnberg, 91058 Erlangen, Germany.

The project was carried out in collaboration with the Department of Computing, Imperial College London, London, SW7 2AZ, UK.

| Contributor | Contribution |
|-------------|--------------|
| Mohammad Saeed Zarzor | Conceptualization, methodology, software, validation, formal analysis, investigation, data curation, and coding |
| Qiang Ma | Conceptualization, data curation, data analysis |
| Bernhard Kainz | Conceptualization, resources |
| Silvia Budday | Conceptualization, supervision, project administration, funding acquisition |

## Copyright and License

Copyright (c) 2026 Mohammad Saeed Zarzor.

FetoMorph is licensed under the MIT License. See the `LICENSE` file for the complete license terms.

## Citing FetoMorph

When using FetoMorph in research, please cite:

```text
ZARZOR, M. S., Ma, Q., Kainz, B.& Budday, S. (2026). FetoMorph: A Unified Toolkit for Validating Computational Models of Human Brain Development (Version v1) [Computer software]. Zenodo. European Congress on Computational Methods in Applied Sciences and Engineering (ECCOMAS), Munich, Germany. 
```

## Acknowledgements

Acknowledgement is given to the fetal brain imaging datasets provided by the developing Human Connectome Project (dHCP), undertaken by the KCL-Imperial-Oxford Consortium and funded by the European Research Council under the European Union's Seventh Framework Programme (FP7/2007-2013), ERC Grant Agreement No. 319456, and ERC Consolidator Grant No. 101083647.

FetoMorph was developed as part of the BRAINIACS project, funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through grant BU 3728/1-1, awarded to SB.

We also sincerely thank Median Almurey, Ahmad Baradiei, Yanal Moulla, Rami Musleh, and Divyashree Doddbele for their valuable contributions to the design implementation, testing, validation, and continued improvement of the software.

## Icon Credits

Icons made by kliwir art, Freepik, Three musketeers, FetchLab, juicy_fish, Us and Up, Pixel perfect, Fathema Khanom, meaicon, Karyative, Iconic Panda, JessHG, FACH, Anggara, samlakodad, and Infinite Dendrogram from Flaticon.

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
| Ctrl+T | Set filtered threshold |
| Ctrl+Shift+A | Annotation (rectangle ROI) |
| Ctrl+Shift+R | ROI region selection |
| Ctrl+M | Next result image |
| Ctrl+Shift+M | Previous result image |
| Shift+Alt+E | Resume / re-run last action |
| Ctrl+, (⌘+,) | Open Preferences |
| Ctrl+Alt+F | Example: filled 2D sections |
| Ctrl+Alt+C | Example: cropped 2D sections |

---

## Project Structure

```
FetoMorph/
├── FetoMorph.py              # Main application window and menu wiring
├── deps.py                   # Centralized dependency imports
├── constants.py              # Application constants and defaults
├── icons.py                  # Icon loader
├── ribbon.py                 # Office-style ribbon toolbar
├── requirements.txt          # Pinned Python dependencies
│
├── managers/                 # Application controllers (separation of concerns)
│   ├── file_manager.py             # Import, load, save, close operations
│   ├── measurement_dispatcher.py   # All measurement / processing operations
│   ├── metrics_store.py            # Metrics storage, table display, Excel export
│   ├── settings_manager.py         # Calibration, units, processing parameters
│   ├── view_manager.py             # Display, slice navigation, pixmap carousel
│   └── visualization_settings.py   # User-tunable text/colors/sizes/view toggles
│
├── widgets/                  # Custom UI dialogs and components
│   ├── scaled_image_label.py       # 2D image viewer with measurement modes
│   ├── vtk_viewer.py               # Embeddable 3D VTK rendering widget
│   ├── zoom_controls.py            # Reusable zoom controls
│   ├── contour_threshold.py        # Contour area threshold dialog
│   ├── kernel_size.py              # Morphology kernel size dialog
│   ├── perimeter_options.py        # Perimeter method dialog
│   ├── cavity_options.py           # Cavity-correction options dialog
│   ├── slice_thickness.py          # Slice thickness dialog
│   ├── unit_scale.py               # Unit and pixel size dialog
│   ├── scalebar_set_scale.py       # Scale-bar calibration dialog
│   ├── options_dialog.py           # Image processing options dialog
│   ├── geometry_dialog.py          # 3D mesh dimension editor (live preview)
│   ├── region_dock.py              # NIfTI region selection dock
│   ├── gestational_weeks_dialog.py # Gestational week and axis selector
│   ├── image_browser_dialog.py     # Thumbnail image browser
│   ├── optimization_widgets.py     # Optimization configuration dialog
│   ├── manual_gasp_dialog.py       # Manual GASP data-entry dialog
│   ├── preferences_dialog.py       # Tabbed Preferences + GASP Options dialogs
│   └── recent_paths.py             # Recent file management
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
│   ├── validation.py               # QC plotting for NIfTI results
│   └── optimization.py             # NSGA-II/III optimization
│
├── helpers/                  # Utility modules
│   ├── helpers.py                  # Shared measurement / annotation helpers
│   ├── cavities.py                 # Surface-connected cavity correction
│   ├── check_mesh.py               # Heuristic brain-mesh classifier
│   ├── slice_kind_classifier.py    # ONNX slice-kind inference wrapper
│   ├── slice_state.py              # Slice geometry state along an axis
│   ├── gestational_week_profile.py # Reference-statistics registry + GASP scoring
│   ├── gasp_export.py              # GASP results-folder builder
│   ├── results_excel_format.py     # Shared result Excel reader/writer
│   └── read_excel.py               # Excel ingestion for optimization
│
├── models/                   # Trained models
│   ├── slice_kind_cnn.onnx         # Slice-kind CNN (ONNX)
│   └── slice_kind_cnn_report.md    # Training report
│
├── scripts/                  # Offline ML / data scripts
│   ├── train_slice_kind_cnn.py     # Train the slice-kind CNN
│   ├── eval_slice_kind_cnn.py      # Evaluate the trained model
│   ├── cv_slice_kind_cnn.py        # Leave-weeks-out cross-validation
│   └── migrate_reference_kernel.py # Add kernel_size_mm to the reference CSV
│
├── tests/
│   └── test_mask_perimeter.py      # Crofton vs. arc-length perimeter tests
│
├── assets/
│   └── icons/                # UI icons (PNG)
│
└── Examples/                 # Sample fetal brain data
    ├── gestational_week_reference.csv  # GASP per-week reference statistics
    ├── full_slices/                    # Full anatomical sections by week
    └── cropped_slices/                 # Cropped sections by week
```
