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
- mean and standard deviation for sulcus categories such as `primary`, `secondary`, and `tertiary`
- embedded boxplots for the core metrics and sulcus categories

Run for all weeks and all axes:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_master_measurement_reports.py
```

Run one week and one axis:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_master_measurement_reports.py --weeks 24 --axes axial
```
