# Area Band CLI Batch Run Guide

This guide describes how to run `scripts/area_band_cli.py` for all subjects and all axes, including pial overlays.

Imaging data is not part of this repository, so every path in the examples below
is written as `<add your ... path>`. Replace those with your own locations.

## Prerequisites

- Python virtual environment already created at `.venv` in this repo.
- Each subject folder contains a label volume. The default name is `seg.nii.gz`;
  datasets that name it differently set `--seg-glob` / `"seg_pattern"` (the dHCP
  atlas ships `tissue-t30.00_dhcp-19.nii.gz`, so its config uses
  `"seg_pattern": "tissue-*_dhcp-19.nii.gz"`).
- Pial overlays are optional. If `lh.pial` / `rh.pial` are present and
  `--use-pial-overlay` is set they are drawn; datasets without surfaces simply
  run without them.

## Labels

There is no built-in default label list, because label ids mean different
tissues in different segmentation schemes. Every run states its labels
explicitly, via `--labels` or the config's `area_labels`.

Point `--label-legend` / `"label_legend"` at the dataset's legend so labels are
reported by name. Supported formats: two-column `.csv`, ITK-SnAP `.txt` label
description, and `.xlsx`. The legend usually ships with the dataset — for the
dHCP atlas it is `info/dhcp-atlas-summary-info-19-labels.csv` inside the download
(an equivalent ITK-SnAP `.txt` sits beside it). If a dataset has no legend file,
labels are reported as bare numbers and the legend checks below are skipped.

Before sampling, the CLI checks each volume and **fails** when:

- a requested label is not present in that volume, or
- the volume contains labels the legend does not describe (the legend does not
  belong to this dataset).

It **warns** when the legend describes labels the volume lacks. It also prints
the resolved selection by name, for example:

```
area_labels: 5 (Fetal WM Left), 6 (Fetal WM Right), 7 (Lateral Ventricle Left), ...
```

Read that line. When two schemes share an id range a list written for one stays
valid against the other and silently selects different tissue — no automatic
check can catch that, but the printed names make it obvious.

## Activate the virtual environment

PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

## Batch run for a single axis

Replace the paths if your folder layout differs.

```powershell
python scripts\area_band_cli.py `
  --batch-dir "<add your dataset path>" `
  --batch-out "<add your output path>" `
  --axis x `
  --n 20 `
  --p 0.9 `
  --labels 3 4 5 6 11 12 13 14 15 17 `
  --label-legend "<add your label legend path>" `
  --use-pial-overlay `
  --pial-space scanner `
  --axis-subdir `
  --no-crosshair `
  --pial-line-thickness 1
```

Key inputs:
- `--batch-dir`: Folder with subject subfolders; each must contain a label volume.
- `--batch-out`: Output base folder.
- `--axis`: One of `x`, `y`, or `z`.
- `--n`: Number of slices to sample (default 10).
- `--p`: Top-p fraction for band (default 0.8).
- `--labels`: Label ids to include (required; no built-in default).
- `--label-legend`: Legend file used to name labels in messages and errors.
- `--seg-glob`: Glob for the label volume in each case folder (default `seg.nii.gz`).
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
  --batch-dir "<add your dataset path>" `
  --batch-out "<add your output path>" `
  --axis x --n 20 --p 0.9 --labels 3 4 5 6 11 12 13 14 15 17 --label-legend "<add your label legend path>" --use-pial-overlay --pial-space scanner --axis-subdir --no-crosshair --pial-line-thickness 1

python scripts\area_band_cli.py `
  --batch-dir "<add your dataset path>" `
  --batch-out "<add your output path>" `
  --axis y --n 20 --p 0.9 --labels 3 4 5 6 11 12 13 14 15 17 --label-legend "<add your label legend path>" --use-pial-overlay --pial-space scanner --axis-subdir --no-crosshair --pial-line-thickness 1

python scripts\area_band_cli.py `
  --batch-dir "<add your dataset path>" `
  --batch-out "<add your output path>" `
  --axis z --n 20 --p 0.9 --labels 3 4 5 6 11 12 13 14 15 17 --label-legend "<add your label legend path>" --use-pial-overlay --pial-space scanner --axis-subdir --no-crosshair --pial-line-thickness 1

## Run from a config file (with optional CLI overrides)

The CLI can load a JSON config (matching `AreaBandConfig` fields) and also accepts
extra keys for batch runs: `batch_dir`, `batch_out`, `axis_subdir`, `all_axes`,
`seg_pattern`, `label_legend`. Any CLI flags you pass will override the config
values. Keys the CLI does not recognise are ignored, so a config may carry notes
(`dataset`, `label_scheme`, `label_selection_note`) documenting its own choices.

Prefer one config per dataset, holding its `area_labels` next to the
`label_legend` those ids came from. A relative `label_legend` is resolved first
as given, then relative to the config file.

### Example configs vs local configs

Input and output paths are machine-specific, so they are kept out of the repo:

- `configs/*.example.json` are committed. They carry the dataset's labels,
  legend, seg pattern and sampling settings, with `batch_dir` / `batch_out`
  (or `file_path` / `out_dir`) left empty.
- `configs/*.json` are gitignored local run configs.

To run a dataset, copy its example and fill in the two paths:

```powershell
Copy-Item configs\area_band_config_dhcp_atlas.example.json configs\area_band_config_dhcp_atlas.json
# edit batch_dir / batch_out, then:
python scripts\area_band_cli.py --config "configs\area_band_config_dhcp_atlas.json"
```

Or keep the example untouched and pass the paths on the command line, since CLI
flags override config values:

```powershell
python scripts\area_band_cli.py `
  --config "configs\area_band_config_dhcp_atlas.example.json" `
  --batch-dir "<add your dataset path>" `
  --batch-out "<add your output path>"
```

Available examples:

| example | dataset | labels |
| --- | --- | --- |
| `area_band_config_batch.example.json` | `fetal_surface` subjects, batch | `3,4,5,6,11,12,13,14,15,17` |
| `area_band_config_single.example.json` | one case | same as above |
| `area_band_config_dhcp_atlas.example.json` | dHCP atlas GW21-GW36 | `5,6,7,8,9,14,15,16,17,18` |

Both label lists are the cerebrum excluding the cortical ribbon, so the two
datasets are measured on the same boundary (the `fetal_surface` runs omitted
their cortex label, giving a white-matter boundary rather than a pial one). The
lists are **not** interchangeable: ids mean different tissues in the two schemes.

Override example (use different smoothing):
```powershell
python scripts\area_band_cli.py `
  --config "configs\area_band_config_batch.json" `
  --pial-line-thickness 1
```

## Validation errors

If an invalid value is provided, the CLI prints a concise error and exits:

```
area_band_cli.py: error: p must be in (0, 1]
```

Label problems are reported the same way, naming the labels where a legend is
available:

```
area_band_cli.py: error: seg.nii.gz: requested label(s) not in this volume: 18 (Third Ventricle)
  volume contains: 1 (eCSF Left), 2 (eCSF Right), ...
  Label ids are scheme-specific - check area_labels against this dataset's legend.
```

```
area_band_cli.py: error: tissue-t30.00_dhcp-19.nii.gz: volume contains label(s) the legend does not describe: 18, 19
  The legend does not match this volume - wrong label_legend for this dataset?
```

A batch run whose `seg_pattern` matches nothing also fails rather than reporting
an empty result:

```
area_band_cli.py: error: no case folder under ... contained a file matching seg pattern 'seg.nii.gz'
```

## Single run (config + CLI override)

Use `configs/area_band_config_single.json` and pass the case-specific inputs:

```powershell
python scripts\area_band_cli.py `
  --config "configs\area_band_config_single.json" `
  --file "<add your label volume path>" `
  --out "<add your output path>"
```

Tip: if you must use the batch config for a single case, clear batch mode by passing
`--batch-dir "none"` and `--batch-out "none"` so the CLI treats them as empty.
```

## Output layout

Outputs are written under the `--batch-out` folder, one subfolder per case:
```
<your output path>\<case>\axis_x
<your output path>\<case>\axis_y
<your output path>\<case>\axis_z
```

Each axis folder includes:
- `brain_slices\` PNGs with label and pial overlays.
- `area_band_summary.json` and `area_band_run_summary.json` with metrics.
- CSV and Excel summaries.

## Scale bar

Each slice PNG carries a bar of exactly 20 mm, measured from the column spacing
of the slicing axis (a shorter round length is substituted, and stated in the
label, if 20 mm will not fit the frame). Never calibrate a pixel size from the
drawn bar — take it from the voxel grid, which is what the bar itself is built
from.

The bar sits in a blank strip **below** the section, not in a corner of it. On
these volumes the brain reaches the lower-right corner in most slices, so a bar
drawn there lay across tissue and interfered with measuring it; it was also being
erased out of the section by the background pass that runs between the two
scale-bar passes, taking a bar-shaped bite out of the brain with it.

That makes the file taller than the section, so `brain_slices\` also holds:

- `scale_bar_strip.json` — `scale_bar_strip_px`, `frame_height`, `frame_width`,
  `image_height`.

**Anything measuring these PNGs must resolve normalised coordinates against
`frame_height`, not the file height.** `crop_band_roi` reads this sidecar and
does exactly that, which is why its bounding boxes are unchanged by the strip; a
tool that ignores it slides every box down by a fraction of the strip. A missing
sidecar means a run from before the strip existed, where the bar was inside the
frame.
