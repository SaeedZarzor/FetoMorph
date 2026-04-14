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
- `pixel_size = 1 / 41 mm/pixel`
- `kernel_size = 25`
- `cnt_threshold = 2000`
- `unit = mm`

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
