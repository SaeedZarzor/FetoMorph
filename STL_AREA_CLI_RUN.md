# STL Area Band CLI Guide

This guide describes how to run `scripts/stl_area_cli.py`, which samples 2D
slices from surface meshes (STL/VTK/PLY/OBJ) using the same maximum-area rule
`scripts/area_band_cli.py` applies to NIfTI label volumes.

Mesh data is not part of this repository, so every path below is written as
`<add your ... path>`. Replace those with your own locations.

## What this does differently from the NIfTI CLI

A surface mesh has **no labels**, so there is nothing to select: the mesh *is*
the mask. `--labels`, `--label-legend` and the pial overlay options have no
counterpart here. Whichever regions you want measured must be chosen upstream,
when the surface is generated.

Axis names are the **mesh's own x/y/z** and carry no anatomical meaning. A
surface has no orientation convention we can trust, so do not read them as
axial/coronal/sagittal without checking the dataset yourself.

## Why the mesh is solidified first

Cutting a surface yields open polylines wherever the mesh is not watertight, and
brain surfaces rarely are. Closing each polyline independently produces
overlapping polygons whose summed area can exceed the section's own bounding
box, so a naive maximum lands on an artefact rather than on the widest slice.

The pipeline therefore recovers a solid before measuring anything:

```
surface-voxelise                 -> hollow voxel shell
3D binary closing (--seal)       -> seal the holes
flood-fill background from the border
interior = whatever the outside could not reach
  -> written as a NIfTI carrying the mesh's index-to-mm transform
  -> handed to the existing NiftiAreaSampler, unmodified
```

Everything downstream of the maximum-area rule is shared with the NIfTI path, so
the outputs are directly comparable.

## Prerequisites

- Python virtual environment already created at `.venv` in this repo.
- A mesh, or a folder of them. Files starting with `._` are skipped: those are
  macOS AppleDouble sidecars, which carry the `.stl` suffix but hold no geometry.

## Running

Single mesh:

```bash
.venv/Scripts/python.exe scripts/stl_area_cli.py \
  --stl "<add your mesh path>/29.stl" \
  --out "<add your output path>" \
  --axis z --n 20 --p 0.9
```

Every mesh in a folder:

```bash
.venv/Scripts/python.exe scripts/stl_area_cli.py \
  --stl-dir "<add your mesh folder>" \
  --out "<add your output path>" \
  --axis z --n 20 --p 0.9
```

All three axes:

```bash
.venv/Scripts/python.exe scripts/stl_area_cli.py \
  --stl-dir "<add your mesh folder>" \
  --out "<add your output path>" \
  --all-axes --n 20 --p 0.9
```

From a config: copy `configs/stl_area_band_config.example.json` to
`configs/stl_area_band_config.json` (gitignored), fill in `stl_dir`/`out_dir`,
then `--config configs/stl_area_band_config.json`. Command-line flags override
the config file.

## Settings that matter

### `--pitch` (default 0.25 mm)

Voxel pitch of the intermediate solid. Finer keeps more sulcal detail and costs
time and memory; 0.5 mm roughly quarters both.

### `--seal` (default 1)

Radius in voxels of the 3D closing that seals the mesh's holes; the kernel is
`(2*seal+1)^3`. **This is the only step that erodes real sulcal detail, so keep
it as small as will still seal.** On the fetal surfaces tested, `seal=1`
sealed every mesh while keeping 85–96% of the raw cross-section perimeter;
`seal=2` cost a further 3–7% for no benefit.

Raise it only when `solid_volume_cm3` comes out far *below* `mesh_volume_cm3`,
which means the flood-fill leaked through a hole the closing failed to bridge.
A few percent *above* the mesh volume is normal and expected — see below.

### The area bias, and why it is not corrected

The voxel shell sits astride the surface rather than inside it, so the solid
carries about half a voxel of margin all round. That makes areas read slightly
high, by a known and predictable amount. Against an analytic sphere of radius
10 mm:

| pitch | volume vs exact | max cross-section vs exact |
|---|---|---|
| 0.5 mm | +9.0% | +5.1% |
| 0.25 mm | +4.1% | +2.7% |

Both match `(1 + pitch/2R)` raised to the third and second power respectively, so
the bias is purely that half-voxel margin — it shrinks with finer pitch and with
larger objects. On the fetal surfaces it shows up as the +3–4% volume difference
in the run log; it is larger there than a sphere would predict because a
gyrified surface carries far more area per unit volume for the margin to act on.

It is left uncorrected because the obvious fix — eroding a voxel back —
overshoots into a −2.6% error, trading a known bias for a different one. Since it
is systematic and near-constant across a cohort, it cancels in any comparison
between meshes; only treat the absolute cm² as approximate.

### `--kernel-mm` (default 5.0)

Size of the morphological closing that defines the outer perimeter, in
millimetres. It is specified in mm and converted to pixels per dataset, because
`AreaBandConfig.kernel_size` is in **pixels** and a fixed pixel count is a
different physical operation at every pitch. At 0.25 mm/voxel the sampler's own
default of 5 px is a 1.25 mm closing, which finds no sulci at all and reports
LGI ≈ 1.00 across the board — which looks like a bug rather than a setting.

Measured LGI against kernel size at pitch 0.25 mm:

| kernel px | 5 | 9 | 15 | 21 | 31 | 41 |
|---|---|---|---|---|---|---|
| kernel mm | 1.25 | 2.25 | 3.75 | 5.25 | 7.75 | 10.25 |
| mean LGI | 1.003 | 1.004 | 1.199 | **1.328** | 1.319 | 1.323 |

It plateaus around 5 mm, which is where the default sits.

### `--keep-nifti`

Off by default: the intermediate solid is written to a temporary directory and
deleted once sampling finishes. Keeping it costs roughly 25 MB per mesh at pitch
0.25, and lets the solid feed any other NIfTI tool in the repo.

## Outputs

Per mesh and axis, in `<out>/<stem>/axis_<a>/`:

- `brain_slices_outline/` — **the deliverable**: each slice drawn straight from
  the mesh cross-section at full detail, with a scale bar and its sidecar
- `brain_slices/` — the sampler's filled silhouettes. **Off by default**;
  add `--fill-png` to write them. Filling a section closes the fissure and every
  narrow sulcus, so these read as a blob rather than a brain — they exist only
  because `crop_band_roi` locates a subject by `axis_<a>/brain_slices`
- `area_band_slices.csv` / `.xlsx`, `area_band_summary.json` — written by
  `NiftiAreaSampler`, identical in form to the NIfTI pipeline's
- `area_profile_axis{0,1,2}.csv`, `area_profiles.png` — the area profiles
- `stl_area_band_slices.csv` / `.xlsx` — the above plus the mesh-measured
  columns below
- `stl_area_band_summary.json` — everything, plus the mesh probe and the
  solidification settings actually used

The `<stem>` level is dropped for a single mesh; `--no-axis-subdir` drops the
axis level, which then puts the output beyond `crop_band_roi`'s reach — it
identifies a subject by the presence of `axis_<a>/brain_slices`.

## Perimeter and LGI: which column to use

Sealing the mesh rounds off fine sulcal detail, which biases any perimeter
measured off the solid. So the inner perimeter is also measured on the **raw
mesh cross-section** — exact polyline length, full detail — while the outer
perimeter comes from the sampler's morphological closing, a ~5 mm operation a
sub-millimetre seal cannot meaningfully disturb.

| column | source | use |
|---|---|---|
| `area_cm2` | solid raster | **yes** — robust, this is what the band is built on |
| `perimeter_mm` | solid raster | loses ~15% of the boundary on average |
| `perimeter_mesh_mm` | raw mesh section | **prefer for perimeter** — exact, full detail |
| `perimeter_convex_mm` | solid raster + closing | the LGI denominator |
| `lgi` | raster / raster | lower scatter (CV 10.8%) — one basis, errors partly cancel |
| `lgi_mesh` | mesh / raster | faithful numerator, smoothed denominator (CV 13.6%) |

### Which LGI

Neither dominates, so both are emitted. `perimeter_mesh_mm` runs **+15% on
average** above `perimeter_mm`, but the gap ranges **−4% to +42%** across slices
— it is not a constant offset that cancels in a trend.

There is also a definitional difference: `perimeter_mesh_mm` counts **every**
boundary in the section, internal ones included, while `perimeter_mm` uses
`cv2.RETR_EXTERNAL` and counts only the outermost. Separating an internal
boundary from the silhouette is **not reliably decidable from the section
geometry** — the surface is open, so the silhouette itself arrives as several
disconnected arcs, and polygon containment, union containment and a 2D
flood-fill were each tried and each failed on real slices. The difference is
therefore documented rather than guessed at.

Treat both LGI columns as indicative only. `lgi` at least keeps numerator and
denominator on one measurement basis. The denominator they share is the weak
link either way (see below), and per-band LGI is already known to be dominated
by geometry rather than gyrification.

## Warnings the CLI emits

- **`no band found`** — the solid came out empty. The mesh is probably not a
  closed surface at all.
- **`band collapsed`** — the n sampled positions map onto only a handful of
  distinct slices, meaning the band is thinner than the voxel pitch. The area
  profile spiked rather than rising to a plateau, which is what a mesh that is
  not a brain, or one the sealing failed on, looks like. Treat those slices as
  suspect.

`--require-brain` additionally aborts on meshes that `helpers/check_mesh.py`
classifies as `not_brain`. Note its scoring is size- and shape-based and can be
fooled: a flat slab was classified `fetal_brain`, and only the band-collapse
warning caught it.
