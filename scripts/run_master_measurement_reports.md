# Master Measurement Report Runner

`run_master_measurement_reports.py` runs the `master` branch full-slice batch measurement pipeline using `functions/measurement_batch.py`.

It generates:

- annotated PNGs
- one Excel report per `week/axis`
- a mean table on the right side of the report sheet

Default settings:

- input root: `Examples\full_slices`
- output root: `measurements`
- output section: `Filled_2D_sections`
- weeks: `24` to `38`
- axes: `axial`, `coronal`, `sagittal`
- default calibration uses the measured scalebar `42 px = 20 mm`
- effective `pixel_size = 20 / 42 mm/pixel`
- when both `scalebar_measured_pixels` and `scalebar_real_world_length` are provided, `pixel_size` is computed as `real_world_length / measured_pixels`
- if either scalebar value is provided, both must be provided
- `--pixel-size` is only used directly when scalebar calibration is not active
- `kernel_size = 25`
- `cnt_threshold = 2000`
- `unit = mm`
- `auto_scalebar = false` (when true, `pixel_size` is measured per folder from the embedded scalebar; see Cropped slices)
- `single_pass_metrics = false` (when true, core metrics are recomputed single-pass after the batch run; see Cropped slices)

Both are `false` for full slices: their scale is already a constant `42 px = 20 mm` every week, and their large brains aren't affected by the LGI retry. These are crop-specific fixes.

The generated workbook also appends metadata rows:

- `PixelSizeUnits`
- `KernelSize`
- `ScalebarMeasuredPixels:`
- `ScalebarRealWorldLength:`
- `ScalebarRealWorldUnit:`

Supported CLI options:

- `--config`
- `--input-root`
- `--output-root`
- `--section-label`
- `--weeks`
- `--axes`
- `--pixel-size`
- `--scalebar-measured-pixels`
- `--scalebar-real-world-length`
- `--kernel-size`
- `--cnt-threshold`
- `--unit`
- `--auto-scalebar`
- `--single-pass-metrics`
- `--area-close-mm` (cropped slices; see below)
- `--review`
- `--review-upscale-min`
- `--log-level {DEBUG,INFO,WARNING,ERROR}`

Run all weeks and all axes:

```powershell
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py
```

Run with config:

```powershell
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py --config scripts\master_measurement_reports_config.example.json
```

Run one week and one axis:

```powershell
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py --weeks 24 --axes axial
```

Override the scalebar calibration explicitly:

```powershell
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py --weeks 24 --axes axial --scalebar-measured-pixels 42 --scalebar-real-world-length 20
```

Run with direct pixel size instead of scalebar calibration:

```powershell
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py --config scripts\master_measurement_reports_config.example.json --pixel-size 0.47619047619047616
```

Note: if the config still contains `scalebar_measured_pixels` and `scalebar_real_world_length`, those values take precedence and `pixel_size` will be recomputed from the scalebar. To use direct `pixel_size`, remove or null out both scalebar fields in the config.

Clean rerun:

```powershell
Get-ChildItem .\measurements -Directory | ForEach-Object {
    $target = Join-Path $_.FullName "Filled_2D_sections"
    if (Test-Path $target) {
        Remove-Item -Recurse -Force $target
    }
}
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py
```

Post-process the generated Excel reports:

`analyze_master_measurement_reports.py` reads each `week*_Batch_Allmarks.xlsx` workbook under `measurements`, analyzes one workbook per `week/axis`, and writes the results back into the same Excel file on a new `Analysis` sheet.

The `Analysis` sheet includes:

- mean and standard deviation for `area`, `perimeter`, `LGI`, and `Compactness`
- rounded `Sulci_count` values and rounded per-class sulcus counts
- mean and standard deviation for sulcus categories such as `primary`, `secondary`, `tertiary`, and `unclassified`
- embedded boxplots for the core metrics, grouped sulcus-count comparison across primary/secondary/tertiary/unclassified classes, and sulcus categories

Run for all weeks and all axes:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_master_measurement_reports.py
```

Run one week and one axis:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_master_measurement_reports.py --weeks 24 --axes axial
```

---

## Cropped slices

`Examples\cropped_slices` contains axial and coronal bands only (no sagittal). Outputs go inside the shared `measurements` folder, under a `cropped_slices` subfolder per week:

```
measurements\
└── 24\
    ├── Filled_2D_sections\   ← full slices (existing)
    └── cropped_slices\       ← cropped slices (new)
        ├── axial\
        └── coronal\
```

### Cropped-slice calibration and metrics (`auto_scalebar`, `single_pass_metrics`)

Cropped bands need two extra fixes that the example config (`master_measurement_reports_cropped_config.example.json`) turns on:

- `auto_scalebar: true` — derive `pixel_size` per folder from each image's **own embedded scalebar bar** instead of the single configured value. Cropped bands are rendered at a different zoom per week and per axis, so a fixed `42 px = 20 mm` is wrong for almost all of them. The detected bar width ranges roughly `32–57 px` (e.g. wk24 axial `35 px → 0.571 mm/px`, wk24 coronal `32 px → 0.625 mm/px`, wk38 axial `57 px → 0.351 mm/px`). The detected bar width is written to the `ScalebarMeasuredPixels:` metadata row, and the per-folder `pixel_size` to `PixelSize:`. If no bar is detected (e.g. full slices), it falls back to the configured `pixel_size`.
- `single_pass_metrics: true` — after the batch run, recompute `area`, `perimeter`, `perimeter_convex`, `LGI`, and `Compactness` with a single pass (reusing `compute_kernel_convex` / `compactness_2D`) and overwrite those columns. This bypasses the in-core LGI auto-retry, which bumps `cnt_threshold` by `+500` whenever `LGI < 1` and, on a small cropped brain (a few hundred to ~2000 px), marches the threshold past the whole brain and zeroes the row out. The recompute reports `LGI < 1` honestly (a real artifact of heavily clipped edge bands) instead of destroying the measurement.
- `kernel_size: 9` — smaller than the full-slice default of `25`, which is oversized for these tiny crops.

Both flags are off by default; only the cropped config enables them. They can also be forced on the command line with `--auto-scalebar` and `--single-pass-metrics`.

Note: cropped bands clip the brain at the image edge, so `perimeter`, `LGI`, and `Compactness` include the straight crop-edge cuts and are best read as band-relative rather than whole-structure measures. `area` is the cleanest physical quantity. (These reports are intended for comparison against partial-domain simulation cases, so the per-band — partial-domain — granularity is by design.)

### Segmentation and review fixes (the three suggestions)

The cropped-slice review showed three problems with the original brightness-threshold segmentation; all three are fixed inside the script (no changes to `functions/`, `helpers/`, or `constants.py`):

1. **Bright labels (e.g. cyan) were dropped.** The original binarisation kept a pixel only when its grayscale value was `≤ BINARY_THRESHOLD_DEFAULT (200)`. That is right for a grayscale render but wrong for a colour label-map: cyan sits at gray `≈ 212` and was excluded, and pure-white interior cortex (`255`) was excluded entirely. `single_pass_metrics` and the review renderer now segment with **`segment_foreground`**: on a white-background image (auto-detected from the border) the foreground is **every non-white pixel** plus any white **fully enclosed** by tissue (interior cortex), so all label colours are kept regardless of brightness. Dark-background renders fall back to the original threshold, so full-slice behaviour is unchanged.

2. **The inner white cortex was not counted in the area (`area_close_mm`).** With `area_close_mm > 0` the reported `area` is measured on the brain after a **morphological close** that fills the white cortex ring between the outer rim and the core, so it counts as ROI. The close distance is physical (`area_close_mm`) and scaled per folder via the detected `pixel_size`, so it is consistent across the differently-zoomed weeks. `perimeter` and `LGI` are still taken from the **folded** boundary, so the gyrification signal is unchanged. The cropped config sets `area_close_mm: 5.0`; `0` disables it (area = labelled tissue only). Force it on the command line with `--area-close-mm`.

3. **The review image was tiny and missing annotations (`review`).** With `review: true` (or `--review`), an enlarged annotated PNG is written per image to a `review/` sub-folder beside the workbook. The whole frame — **scalebar included** — is upscaled with nearest-neighbour to a minimum side of `review_upscale_min` (default `600`, `--review-upscale-min`); measurement still runs at native resolution, so calibration is unaffected. The annotation matches the full-slice report: **red** inner (folded) contour, **green** convex/close envelope, **blue** convexity-defect chords (the outer line wrapping the region), and **gray** sulci-depth markers.

In addition, when `single_pass_metrics` is on, the **sulci columns** (`Sulci_count`, `min/max/mean/total_depth`) are also recomputed on the new segmentation (matching the fixed-mm depth rule used for the review markers), so the workbook counts agree with the markers drawn on the review image.

Config keys for the above (all in `master_measurement_reports_cropped_config.example.json`): `area_close_mm`, `review`, `review_upscale_min`.

Run all weeks, axial and coronal:

```powershell
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py --config scripts\master_measurement_reports_cropped_config.example.json
```

Run one week and one axis:

```powershell
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py --input-root Examples\cropped_slices --output-root measurements --section-label cropped_slices --weeks 24 --axes axial --kernel-size 9 --auto-scalebar --single-pass-metrics --area-close-mm 5 --review
```

Clean rerun:

```powershell
Get-ChildItem .\measurements -Directory | ForEach-Object {
    $target = Join-Path $_.FullName "cropped_slices"
    if (Test-Path $target) {
        Remove-Item -Recurse -Force $target
    }
}
.\.venv\Scripts\python.exe scripts\run_master_measurement_reports.py --config scripts\master_measurement_reports_cropped_config.example.json
```

Post-process cropped reports:

All sulci in cropped slices are classified as `unclassified` (fixed-mm depth rule). Because the pipeline writes no values for the per-class sulcus cells, those columns are dropped from the workbook entirely, so cropped reports have no `Primary_*`/`Secondary_*`/`Tertiary_*`/`Unclassified_*` columns — only `Sulci_count` and the `min/max/total/mean_depth` columns.

`analyze_master_measurement_reports.py` handles this gracefully: it still writes the core-metric summary (`area`, `perimeter`, `LGI`, `Compactness`), the rounded `Sulci_count` tables, the depth-metric summary (`min/max/total/mean_depth`), the core/depth boxplots, and the count boxplot, but **skips the per-class Sulcus Value Summary table and the per-class/grouped sulcus-value plots** (and records why in the Analysis sheet's `Note`). It does not error on the missing columns.

The `--report-glob` flag scopes discovery to the `cropped_slices` subfolders only, so full-slice workbooks are not affected.

Run for all weeks, axial and coronal:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_master_measurement_reports.py --input-root measurements --axes axial coronal --report-glob "cropped_slices/*/week*_Batch_Allmarks.xlsx"
```

Run one week and one axis:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_master_measurement_reports.py --input-root measurements --axes axial coronal --weeks 24 --report-glob "cropped_slices/*/week*_Batch_Allmarks.xlsx"
```
