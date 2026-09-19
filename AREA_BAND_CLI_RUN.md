# Area Band CLI Batch Run Guide

This guide describes how to run `scripts/area_band_cli.py` for all subjects and all axes, including pial overlays.

Imaging data is not part of this repository, so every path in the examples below
is written as `<add your ... path>`. Replace those with your own locations.

## Prerequisites

- Python 3.12 (the version used by the current project environment).
- Each subject folder contains a label volume. The default name is `seg.nii.gz`;
  datasets that name it differently set `--seg-glob` / `"seg_pattern"` (the dHCP
  atlas ships `tissue-t30.00_dhcp-19.nii.gz`, so its config uses
  `"seg_pattern": "tissue-*_dhcp-19.nii.gz"`).
- Pial overlays are optional. If `lh.pial` / `rh.pial` are present and
  `--use-pial-overlay` is set they are drawn; datasets without surfaces simply
  run without them.

For a first-time Windows setup from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.win.txt
```

If `.venv` already exists, only activate it.

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
- `--all-axes`: Run `x`, `y`, and `z` in one command; axis subdirectories are
  enabled automatically.
- `--n`: Number of slices to sample (default 10).
- `--p`: Top-p fraction for band (default 0.8).
- `--labels`: Label ids to include (required; no built-in default).
- `--label-legend`: Legend file used to name labels in messages and errors.
- `--seg-glob`: Glob for the label volume in each case folder (default `seg.nii.gz`).
- `--use-pial-overlay`: Enable pial overlay and auto-detect `lh.pial`/`rh.pial` per subject.
- `--pial-lh` / `--pial-rh`: Explicit surface paths for a single case or for a
  nonstandard layout.
- `--pial-space`: Use `scanner` for pials in scanner RAS.
- `--axis-subdir`: Writes to `axis_x`, `axis_y`, or `axis_z` under each subject output.
- `--no-crosshair`: Disables the crosshair overlay.
- `--pial-line-thickness`: Pial overlay line thickness in pixels.
- `--no-pial-overlay`: Disable pial overlay (overrides config).
- `--no-png` / `--no-profile-plot`: Disable slice PNGs or the profile plot.

For `scanner` space, vertices are mapped through the matching NIfTI affine. Use
`--pial-space tkr` only for surfaces in FreeSurfer tkregister RAS. The loader
first looks for `lh.pial` and `rh.pial` beside the segmentation; explicit
`--pial-lh` and `--pial-rh` paths override that lookup.

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
```

The same run can be launched once with `--all-axes` instead of issuing three
commands. It writes separate `axis_x`, `axis_y`, and `axis_z` directories.

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
| `area_band_config_dhcp_atlas.example.json` | dHCP atlas GW21-GW36, `parcellations_scaled` | `5,6,7,8,9,14,15,16,17,18` |
| `area_band_config_dhcp_atlas_cortex.example.json` | same, cortical GM included | `3,4,5,6,7,8,9,14,15,16,17,18` |
| `area_band_config_dhcp_atlas_truesize.example.json` | dHCP atlas GW21-GW36, clean labels at true size (**needs a rescale step**) | `5,6,7,8,9,14,15,16,17,18` |
| `area_band_config_dhcp_atlas_truesize_cortex.example.json` | same, cortical GM included | `3,4,5,6,7,8,9,14,15,16,17,18` |

For size-aware work use the `truesize` pair. They cannot be pointed at an atlas
folder directly - see
[True-size dHCP atlas](#true-size-dhcp-atlas-the-root-to-use-for-size-aware-work)
below.

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

## True-size dHCP atlas (the root to use for size-aware work)

The atlas ships two parcellation roots and **neither is usable as-is** for
size-aware slice measurement:

- `parcellations_scaled` is clean - one connected component per structure - but
  has brain size normalised out (bbox flat at about 81x102x81 mm on every week),
  so absolute area and perimeter carry no growth signal.
- `parcellations_orig_size` preserves true size but is a badly
  nearest-neighbour-warped parcellation. **This is what produces visible dots and
  white lines in the renders.** At GW36 it holds **66,413 connected components
  across 12 labels**, 63,305 of them 8 voxels or smaller against only 12 larger
  than 1000. Its isolated-voxel rate is **35-65 per mille on all 16 weeks versus
  0.03-0.10 in the scaled root**, and it carries 3,265 interior background hole
  components against 22. The small midline structures are shattered and
  volume-inflated 1.7-2.7x: cavum septum pellucidum arrives as 2,833 pieces whose
  largest holds just **35.7%** of its volume, third ventricle as 1,032 pieces /
  37.1%.

**Do not try to filter the speckle out.** A 3x3x3 majority vote costs CSP 45.7%
of its volume, third ventricle 34.7% and lateral ventricle L 20.5% - they are
thin sheets, outvoted by the surrounding white matter - and drags GW36 coronal
LGI from 1.749 to 1.581. Surgical speck removal (dissolve components under N
voxels, refill from the nearest label) is no better: CSP still drifts 41-53% at
every threshold tried, because roughly **64% of its voxels genuinely are specks**.
There is nothing to recover locally.

### The fix: rescale the clean labels to true size

The two roots turn out to be related by a near-pure per-axis scale about the
brain centre, so the clean labels can be resampled to the true size:

```powershell
python scripts\rescale_atlas_to_true_size.py `
  --scaled-root "<atlas>\parcellations_scaled" `
  --orig-root   "<atlas>\parcellations_orig_size" `
  --dst         "<atlas>\parcellations_truesize" `
  --label-legend "<atlas>\info\dhcp-atlas-summary-info-19-labels.csv"
```

The scale is measured from a **debris-free** mask (largest connected component,
holes filled) - on the raw mask the floating specks around the orig_size brain
inflate its bounding box and the scale comes out wrong. Resampling is
nearest-neighbour, because any smooth interpolation would average label ids,
which is the very defect that ruins GW27/GW28 in `parcellations_orig_size`.

Result: Dice against the orig_size brain 0.9898 (GW21) to 0.9965 (GW32), extents
matched exactly, and at GW36 **92 components instead of 66,413** with area within
1.3% and LGI moving only 1.749 -> 1.735. At GW21 LGI goes 1.382 -> 1.299, that
drop being the speckle-inflated perimeter coming off. All 16 weeks are present.

Then point the `truesize` config's `batch_dir` at that root and run the CLI
normally.

**Shape is the scaled atlas's; size is the original-size atlas's.** State both
wherever the numbers are reported - it is a methodological choice, not a repair
of the original-size volumes. Never mix this root with a raw
`parcellations_orig_size` run in one workbook.
### Reading the results against the scaled run

- **Dimensionless ratios (LGI) are comparable. Anything with units is not.**
  The scaled atlas holds the bbox flat at about 81x102x81 mm across all weeks;
  this root grows 51x62x48 mm at GW21 to 89x105x88 mm at GW36.
- **Both roots carry all 16 weeks.** GW22's scaled file is named
  `tissue-t22_dhcp-19.nii.gz`, without the `.00` every other week has, so a
  glob finds it but a hardcoded `tissue-t22.00_...` name does not.
- **GW27 and GW28 are not degraded here.** Their interpolated-label defect
  belongs to `parcellations_orig_size` (see the warning below); the labels in
  this root come from the clean scaled volumes, and the original-size data is
  only ever read to measure a bounding box.
- **The crop ROIs from `crop_band_config_dhcp_atlas*.json` do not transfer.**
  Brain centring in the frame is identical between the roots, so the midline
  constraint survives, but the fixed normalised box no longer lands on the same
  anatomy. Tissue fill inside the coronal box is flat at 58-61% across the
  scaled weeks and ramps 16% -> 67% at true size, so at GW21 the box is 84%
  background. A true-size crop needs a brain-bbox-relative ROI, not a
  frame-relative one.

### Never point the sampler at `parcellations_orig_size` directly

Besides the speckle, three separate things break, and the second one breaks
silently:

1. Every volume is `(180, 221, 180, 1)`. That trailing axis is a degenerate
   singleton, not a time series and not per-label probability maps, but left in
   place it makes `NiftiAreaSampler.mask` 4D, so `self.shape` gains a fourth
   entry and each "2D" slice comes out as `(221, 180, 1)`.
2. **GW27 and GW28 are float64 with continuously interpolated label values**
   (~2.7M distinct values, and a background of denormal noise around `1e-81`
   rather than exact zero). The sampler masks with `np.isin` against integer
   ids, which is an exact float comparison, so those two weeks resolve to **5
   and 13 mask voxels** - blank slices. Nothing catches it: `_check_labels`
   applies `np.rint` before testing membership, so it reports labels 1-19
   present and passes.
3. Case folders are named `21`..`36`, where `parcellations_scaled` uses
   `GW21`..`GW36`, so output folders would not line up between runs.

Use `scripts/rescale_atlas_to_true_size.py` instead. It reads the original-size
volumes only to measure a debris-free bounding box, so none of the above applies.

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
- `area_band_slices.csv` and `.xlsx` with run metadata and one row per sampled
  slice.
- `area_profile_axis0.csv`, `area_profile_axis1.csv`, and
  `area_profile_axis2.csv`, plus `area_profiles.png` when profile plotting is
  enabled.

Important result fields:

| field | meaning |
| --- | --- |
| `x_max`, `left`, `right` | Normalised positions along the selected axis. `left` and `right` bound the region whose area is at least `p * f_max`. |
| `x_max_idx`, `left_idx`, `right_idx` | Corresponding voxel indices. |
| `x_max_mm`, `left_mm`, `right_mm` | Corresponding positions in millimetres from index zero. |
| `f_max` | Maximum selected-label cross-sectional area in cm². |
| `band_length_mm` | Physical distance between the left and right band boundaries. |
| `area_cm2` | Selected-label area of one sampled slice in cm². |
| `perimeter_mm` | Sum of the external selected-mask contour lengths in millimetres. |
| `perimeter_convex_mm` | External perimeter after morphological closing; this is the LGI denominator despite the historical field name. |
| `lgi` | `perimeter_mm / perimeter_convex_mm` for that slice. |
| `pos`, `idx`, `pos_mm` | Normalised slice position, voxel index, and distance from index zero in millimetres. |

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
