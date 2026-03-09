# Area Band CLI Batch Run Guide

This guide describes how to run `scripts/area_band_cli.py` for all subjects and all axes, including pial overlays.

## Prerequisites

- Python virtual environment already created at `.venv` in this repo.
- Each subject folder contains `seg.nii.gz`.
- Each subject folder also contains `lh.pial` and `rh.pial` (per-subject pials).

## Activate the virtual environment

PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

## Batch run for a single axis

Replace the paths if your folder layout differs.

```powershell
python scripts\area_band_cli.py `
  --batch-dir "C:\Users\Divya\git\FetoMorph\assets\data\fetal_surface" `
  --batch-out "C:\Users\Divya\git\FetoMorph\area_band_output_multi" `
  --axis x `
  --n 20 `
  --p 0.9 `
  --use-default-labels `
  --use-pial-overlay `
  --pial-space scanner `
  --axis-subdir `
  --no-crosshair `
  --pial-line-thickness 1
```

Key inputs:
- `--batch-dir`: Folder with subject subfolders; each must contain `seg.nii.gz`.
- `--batch-out`: Output base folder.
- `--axis`: One of `x`, `y`, or `z`.
- `--n`: Number of slices to sample (default 10).
- `--p`: Top-p fraction for band (default 0.8).
- `--use-default-labels`: Uses built-in label list from the CLI.
- `--use-pial-overlay`: Enable pial overlay and auto-detect `lh.pial`/`rh.pial` per subject.
- `--pial-space`: Use `scanner` for pials in scanner RAS.
- `--axis-subdir`: Writes to `axis_x`, `axis_y`, or `axis_z` under each subject output.
- `--no-crosshair`: Disables the crosshair overlay.
- `--pial-line-thickness`: Pial overlay line thickness in pixels.
- `--no-pial-overlay`: Disable pial overlay (overrides config).

## Batch run for each axis

Run all three axes with separate commands:

```powershell
python scripts\area_band_cli.py `
  --batch-dir "C:\Users\Divya\git\FetoMorph\assets\data\fetal_surface" `
  --batch-out "C:\Users\Divya\git\FetoMorph\area_band_output_multi" `
  --axis x --n 20 --p 0.9 --use-default-labels --use-pial-overlay --pial-space scanner --axis-subdir --no-crosshair --pial-line-thickness 1

python scripts\area_band_cli.py `
  --batch-dir "C:\Users\Divya\git\FetoMorph\assets\data\fetal_surface" `
  --batch-out "C:\Users\Divya\git\FetoMorph\area_band_output_multi" `
  --axis y --n 20 --p 0.9 --use-default-labels --use-pial-overlay --pial-space scanner --axis-subdir --no-crosshair --pial-line-thickness 1

python scripts\area_band_cli.py `
  --batch-dir "C:\Users\Divya\git\FetoMorph\assets\data\fetal_surface" `
  --batch-out "C:\Users\Divya\git\FetoMorph\area_band_output_multi" `
  --axis z --n 20 --p 0.9 --use-default-labels --use-pial-overlay --pial-space scanner --axis-subdir --no-crosshair --pial-line-thickness 1

## Run from a config file (with optional CLI overrides)

The CLI can load a JSON config (matching `AreaBandConfig` fields) and also accepts
extra keys for batch runs: `batch_dir`, `batch_out`, `axis_subdir`, `all_axes`.
Any CLI flags you pass will override the config values.

Example config (batch, all axes) — `configs/area_band_config_batch.json`:
```json
{
  "file_path": "",
  "out_dir": "",
  "axis": "z",
  "n": 20,
  "p": 0.9,
  "save_png": true,
  "profile_plot": true,
  "show_crosshair": false,
  "area_labels": [2,3,4,5,6,11,12,13,14,15,17],
  "use_pial_overlay": true,
  "pial_space": "scanner",
  "pial_line_thickness": 1,
  "batch_dir": "C:\Users\Divya\git\FetoMorph\assets\data\fetal_surface",
  "batch_out": "C:\Users\Divya\git\FetoMorph\area_band_output_multi",
  "axis_subdir": true,
  "all_axes": true
}
```

Run it:
```powershell
python scripts\area_band_cli.py --config "C:\Users\Divya\git\FetoMorph\configs\area_band_config_batch.json"
```

Override example (use different smoothing):
```powershell
python scripts\area_band_cli.py `
  --config "C:\Users\Divya\git\FetoMorph\configs\area_band_config_batch.json" `
  --pial-line-thickness 1
```

## Validation errors

If an invalid value is provided, the CLI prints a concise error and exits:

```
area_band_cli.py: error: p must be in (0, 1]
```

## Single run (config + CLI override)

Use `configs/area_band_config_single.json` and pass the case-specific inputs:

```powershell
python scripts\area_band_cli.py `
  --config "C:\Users\Divya\git\FetoMorph\configs\area_band_config_single.json" `
  --file "C:\path\to\seg.nii.gz" `
  --out "C:\path\to\out"
```

Tip: if you must use the batch config for a single case, clear batch mode by passing
`--batch-dir "none"` and `--batch-out "none"` so the CLI treats them as empty.
```

## Output layout

Outputs are written under:
```
C:\Users\Divya\git\FetoMorph\area_band_output_multi\<subject>\axis_x
C:\Users\Divya\git\FetoMorph\area_band_output_multi\<subject>\axis_y
C:\Users\Divya\git\FetoMorph\area_band_output_multi\<subject>\axis_z
```

Each axis folder includes:
- `brain_slices\` PNGs with label and pial overlays.
- `area_band_summary.json` and `area_band_run_summary.json` with metrics.
- CSV and Excel summaries.
