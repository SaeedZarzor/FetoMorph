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
- `pixel_size = 20 / 42 mm/pixel`
- when both `scalebar_measured_pixels` and `scalebar_real_world_length` are provided, `pixel_size` is computed as `real_world_length / measured_pixels`
- `kernel_size = 25`
- `cnt_threshold = 2000`
- `unit = mm`

The generated workbook also appends metadata rows:

- `ScalebarMeasuredPixels:`
- `ScalebarRealWorldLength:`
- `ScalebarRealWorldUnit:`

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
