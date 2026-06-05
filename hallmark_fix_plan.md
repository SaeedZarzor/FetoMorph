# Hallmark Measurement — Detailed Fix Plan

Sequenced, dependency-aware roadmap for resolving the issues identified
in [`hallmark_measurement_review.md`](./hallmark_measurement_review.md).

> **Prerequisite assumption (not a fix item).**
> Every input image is treated as **isotropic** — i.e. the same
> `pixel_size` along the x and y axes. The codebase makes this
> assumption today (`compute_image_allmarks` takes a single
> `pixel_size`, the morphological kernel is a square,
> `cv2.contourArea` and `cv2.arcLength` are scaled by `pixel_size`
> and `pixel_size²`). NIfTI handles voxel anisotropy internally and
> is unaffected. Users feeding non-isotropic image stacks
> (e.g. ultrasound exports with rectangular pixels) **must
> resample to square pixels first** — surface a one-line warning
> in the import dialog and in the Excel header, but do not write a
> code path for anisotropic in-plane handling. This is the
> "anisotropy hint" referenced in the review.

Items below are grouped into four phases. Each item lists the
**file(s) and lines**, the **change** (code-level), the **expected
numerical effect**, and a **verification step**. Run the phases in
order: later phases assume the constants and helpers introduced in
earlier ones. After each item, re-run an existing measurement
(`Examples/cropped_slices`) and diff against the prior run — anything
more than ~3 % per-slice shift outside what the item description
predicts is a red flag.

Legend:
- **Effort:** S (≤1 hr), M (1–4 hr), L (>4 hr)
- **Risk:** how likely the change breaks downstream / GASP reference

---

## Phase 1 — Foundations & correctness blockers

These fix silent wrong-answer bugs and put the kernel onto a
resolution-independent footing **before** any other numbers change.
Doing kernel-size-in-mm first means the corrected totals from items
3–4 are already self-consistent, and the GASP reference only needs to
be regenerated once at the end of the phase.

### 1. Make `kernel_size` a millimetre quantity end-to-end *(do first)*
**Effort:** M · **Risk:** medium (user-facing default changes; GASP
reference profile will need to record the mm value)

The closing kernel is currently sized in pixels (`kernel_size = 25`
default), so the same brain rendered at different resolutions yields
different LGIs. Move it to mm and convert per-call. Because we're
assuming isotropic input images (prerequisite at top), the conversion
is the one-line `K_px = round(K_mm / pixel_size)`.

**Settings layer**

| File | Change |
|---|---|
| `constants.py` | Add `DEFAULT_KERNEL_SIZE_MM = 5.0`; keep `DEFAULT_KERNEL_SIZE` as a derived legacy value (`max(3, round(5.0 / 0.2))`) for any code that hasn't migrated yet |
| `managers/settings_manager.py` | Replace `self.kernel_size: int = DEFAULT_KERNEL_SIZE` with `self.kernel_size_mm: float = DEFAULT_KERNEL_SIZE_MM`; add a `kernel_size_px(pixel_size_mm)` accessor that returns `max(3, int(round(self.kernel_size_mm / pixel_size_mm)))` (odd if morphology requires it) |
| `widgets/kernel_size.py` | Dialog now reads/writes mm. Title: "Morphology kernel diameter (mm)"; range 0.5–25 mm, step 0.5 |

**Measurement-function layer**

For every `compute_*_allmarks` function that takes `kernel_size`,
change the parameter to `kernel_size_mm: float` and convert at the
top of the function:

```python
# Insert near the top of compute_image_allmarks / compute_stl_allmarks /
# compute_vtk_allmarks / compute_nifti_allmarks / process_on_images_batch
def _to_kernel_px(kernel_size_mm: float, pixel_size_mm: float) -> int:
    px = max(3, int(round(kernel_size_mm / max(pixel_size_mm, 1e-9))))
    return px if px % 2 == 1 else px + 1   # odd diameter for symmetric structuring element

kernel_size = _to_kernel_px(kernel_size_mm, pixel_size)
```

For STL/VTK where pixel_size varies per slice (it's recomputed from
the calibration cube), do the conversion *inside* the loop right
after `mm_per_px = calc_scale(...)`:

```python
kernel_size_px = _to_kernel_px(kernel_size_mm, mm_per_px)
kernel = compute_kernel_convex(kernel_size_px)
```

For NIfTI use `pixel_size_x` (in-plane spacing) since the morphology
is 2-D in the slice plane.

**Dispatcher layer**

| File | Change |
|---|---|
| `managers/measurement_dispatcher.py` | Every call site that passes `kernel_size=self.mw.kernel_size` switches to `kernel_size_mm=self.mw.settings.kernel_size_mm` |

**Excel header layer**

Every Excel exporter that previously logged `Kernel size: 25` now
records two columns: `Kernel size (mm)` and `Kernel size (px)` (the
in-loop value). For STL/VTK whose `mm_per_px` varies per slice, the
per-slice table also gets a `Kernel_px` column so reviewers can see
exactly what filter was applied to each slice.

**GASP reference**

`Examples/gestational_week_reference.csv` was collected at
`kernel_size = 25 px` with a known `pixel_size_units` and
`kernel_size` column. Convert that table once:

```python
# scripts/migrate_reference_kernel.py — one-shot migration
import pandas as pd
df = pd.read_csv("Examples/gestational_week_reference.csv")
df["kernel_size_mm"] = df["kernel_size"] * df["pixel_size_x"]  # whatever col holds px
df.to_csv("Examples/gestational_week_reference.csv", index=False)
```

Then update `helpers/gestational_week_profile.py` to read the new
column.

**Verify.** Render the same example brain at two zoom levels (resample
a slice to 0.5× and 2× resolution) and check that LGI matches within
2 %. Before the fix: spread can hit ~20 %.

---

### 2. Replace silent-fallback magic values with `None` / NaN
**Effort:** S · **Risk:** low

| File | Line(s) | Current | Replace with |
|---|---|---|---|
| `helpers/helpers.py` | 515-533 | `return 1.0` when no red contour found | `return None` |
| `functions/measurements_image.py` | 131-132 | `perimeter_convex_sum = 1` | `perimeter_convex_sum = None` |

Then audit callers for `None`-propagation.

```python
# functions/measurements_stl.py around line 196 — already calls calc_scale
mm_per_px = calc_scale(img_rgb, cube_len)
if mm_per_px is None:
    print(f"[STL allmarks] slice {idx} skipped — no calibration cube found")
    continue   # do NOT append the slice to rows / saved_pngs / valid_slices
```

```python
# functions/measurements_image.py around line 135
if perimeter_convex_sum is None:
    perimeter_Rate = None
    comp = None
else:
    perimeter_convex = perimeter_convex_sum * pixel_size
    perimeter_Rate = perimeter / perimeter_convex
    comp = compactness_2D(area, perimeter)
```

Wire `None` through the downstream Excel writers (already supported by
`helpers/results_excel_format.py` — `drop_empty_columns=True` will
hide a column that's all-None).

**Verify.** Crop the red cube out of one slice in a debug copy of an
example STL render and re-run. Before the fix: the row stays with
`mm_per_px = 1.0` and reports area/perim in raw pixels labelled "mm".
After the fix: the row is dropped from `Mesh_Allmarks.xlsx`, log emits
one warning, the totals still sum from the surviving slices.

---

### 3. Use exact mesh `volume` and `area` (PyVista) for STL/VTK totals
**Effort:** S · **Risk:** medium (changes returned totals → GASP
reference values move)

| File | Function | Lines | Change |
|---|---|---|---|
| `functions/measurements_stl.py` | `compute_stl_allmarks` | 307-311 | Replace stack-of-slabs `brain_volume`/`Area` with PyVista's exact totals |
| `functions/measurements_vtk.py` | `compute_vtk_allmarks` | 290-294 | Same; multiply by `prod(mesh_dim_scaled)` for the scaled-physical case |

```python
# functions/measurements_stl.py, replacing lines 307-311
# `mesh` is already in scope from `pv.read(file_path)` at the top
brain_volume = float(mesh.volume) / 1000.0   # mm^3 → cm^3
Area         = float(mesh.area)   / 100.0    # mm^2 → cm^2
GI_total     = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
comp_3D      = compactness_3D(brain_volume, Area)
```

For VTK, the mesh is in *model* units, not mm. Scale into physical mm:

```python
# functions/measurements_vtk.py, replacing lines 290-294
scale_factor = float(np.prod(mesh_dim_scaled))           # volume scale
brain_volume = float(mesh.volume) * scale_factor / 1000  # → cm^3

# Surface area scales as the product of the two in-plane scales
in_plane_scale = float(np.prod(mesh_dim_scaled) / mesh_dim_scaled[axis_index])
Area = float(mesh.area) * in_plane_scale / 100           # → cm^2

GI_total = (sum_inner_mm / sum_outer_mm) if sum_outer_mm > 0 else 0.0
comp     = compactness_3D(brain_volume, Area)
```

Keep the per-slice rows (`rows.append([... area_mm, inner_perim_mm,
outer_perim_mm, ...])`) intact — those still come from the
slice-integration loop and feed the per-slice Excel table.

**Quick sanity unit-test (run once):**

```python
import pyvista as pv, math
s = pv.Sphere(radius=10.0)
assert abs(s.volume - (4/3) * math.pi * 10**3) / s.volume < 1e-3
assert abs(s.area   - 4 * math.pi * 10**2)     / s.area   < 1e-2
```

**Verify.** On the existing example STL the new compactness should
land in `(0.3, 0.7)` and **never exceed 1.0**. The
`if compactness > 1.0` warning block at
`functions/measurements_stl.py:362-365` should become unreachable
for valid closed meshes. Delete it once you've confirmed.

---

### 4. NIfTI surface area via marching cubes
**Effort:** M · **Risk:** medium

| File | Function | Lines | Change |
|---|---|---|---|
| `functions/measurements_nifti.py` | `compute_nifti_allmarks` | 252-258 | Replace `Area = sum_inner · pixel_size_y / 100` with a marching-cubes surface-area calculation on the full 3-D `brain_mask` |

```python
# functions/measurements_nifti.py, replacing line 257
from skimage.measure import marching_cubes

verts, faces, _, _ = marching_cubes(
    brain_mask.astype(np.uint8),
    level=0.5,
    spacing=(pixel_size_x, pixel_size_y, pixel_size_z),
)
v0 = verts[faces[:, 0]]
v1 = verts[faces[:, 1]]
v2 = verts[faces[:, 2]]
tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
Area = float(tri_areas.sum()) / 100.0   # mm^2 → cm^2
```

Volume stays as the voxel-sum (already exact at line 254). Add
`scikit-image` to `requirements.txt` if not present (it's a common
dependency — likely already in the lockfile via other deps).

**Verify.** Build a synthetic binary sphere and check S matches
analytically:

```python
import numpy as np, math
from skimage.measure import marching_cubes

N, R = 128, 40.0
xs = np.arange(N) - N/2
X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
mask = (X**2 + Y**2 + Z**2 <= R**2).astype(np.uint8)
v, f, *_ = marching_cubes(mask, 0.5, spacing=(1.0, 1.0, 1.0))
S = 0.5 * np.linalg.norm(
    np.cross(v[f[:,1]]-v[f[:,0]], v[f[:,2]]-v[f[:,0]]), axis=1
).sum()
expected = 4 * math.pi * R**2
print(f"S = {S:.1f}  expected = {expected:.1f}  err = {abs(S-expected)/expected*100:.2f}%")
# Expect < 2% error
```

---

## Phase 2 — Calibration bias

### 5. Force orthographic projection on every PyVista screenshot
**Effort:** S · **Risk:** low

| File | Lines (`pv.Plotter` instantiations) | Change |
|---|---|---|
| `functions/measurements_stl.py` | 171-172 (`compute_stl_allmarks`) plus any other `pv.Plotter(...)` calls in the file | After `p = pv.Plotter(...)` add `p.parallel_projection = True` |
| `functions/measurements_vtk.py` | 140-141 (`compute_vtk_allmarks`) plus other plotters | Same |

Quick grep to find every Plotter:

```
grep -n "pv\.Plotter\b" functions/measurements_*.py
```

**Effect.** Removes per-slice mm/px drift from perspective
foreshortening of the calibration cube. Slices near the camera will
no longer report a slightly smaller `mm_per_px` than slices far from
the camera.

**Verify.** Add a temporary log line printing `mm_per_px` for every
slice in a single run. Before: it can drift 1–3 % across the stack.
After: variation < 0.1 %.

---

## Phase 3 — Refinements

### 6. Frustum integration for the slice-area sanity check (NIfTI fallback)
**Effort:** S · **Risk:** low

Optional — only needed if you decide to skip the marching-cubes
implementation in item 4. The frustum approximation halves the bias
of the stack-of-slabs estimate.

```python
import math
S = 0.0
for i in range(len(slice_areas) - 1):
    P_i,  P_i1  = inner_perim[i], inner_perim[i + 1]
    r_i         = math.sqrt(slice_areas[i]      / math.pi)
    r_i1        = math.sqrt(slice_areas[i + 1]  / math.pi)
    slant       = math.hypot(slice_spacing, r_i1 - r_i)
    S += 0.5 * (P_i + P_i1) * slant
```

Drop in `functions/measurements_nifti.py` around line 257. Skip
entirely if item 4 is done.

**Verify.** Same synthetic-sphere test as item 4; expect ~5 % residual
under-estimate instead of ~21 %.

---

### 7. Simpson's rule for slice-integrated volume
**Effort:** S · **Risk:** low

Applies only if you still need the slice-integrated volume as a
sanity check (after item 3, the authoritative volume is
`mesh.volume`). Use Simpson's 1/3 rule for the secondary check.

```python
import numpy as np

def simpson(slice_areas: list[float], dz: float) -> float:
    n = len(slice_areas)
    if n < 3:
        return float(sum(slice_areas) * dz)
    if n % 2 == 0:
        # Simpson's 1/3 needs an odd number of points; trim last
        n -= 1
    a = np.asarray(slice_areas[:n], dtype=float)
    w = np.ones(n)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    return float((dz / 3.0) * np.sum(a * w))
```

Drop wherever the existing `sum_area * slice_thickness_eff`
appears for a non-authoritative secondary volume report.

**Verify.** Synthetic sphere — Riemann error scales like 1/N, Simpson
like 1/N⁴ at the same N. At N = 50 slices for an R = 40 sphere, expect
Riemann ≈ 1.5 % error, Simpson ≈ < 0.01 %.

---

### 8. Reduce contour-polygon perimeter bias
**Effort:** S · **Risk:** low–medium (slightly shrinks perimeter →
slightly bumps compactness, which is the intended correction)

Two implementation paths — pick one per measurement modality.

**Option (a) — pre-smooth contours (cheap).** Insert right after each
`findContours → filter` block:

```python
filtered_contours = [
    cv2.approxPolyDP(c, epsilon=0.5, closed=True)
    for c in filtered_contours
]
```

ε = 0.5 px keeps the contour faithful to the mask while removing
single-pixel zigzags that inflate perimeter. Apply in
`measurements_image.py`, `measurements_stl.py`, `measurements_vtk.py`,
and `measurement_batch.py`.

**Option (b) — Crofton perimeter (more accurate).** Use for NIfTI
since it's already a clean binary mask:

```python
from skimage.measure import perimeter_crofton

# Replace per-slice inner perimeter computation in measurements_nifti.py
inner_perimeter = float(perimeter_crofton(inner_mask_only, directions=4)) * pixel_size_x
```

Use (b) for NIfTI (binary mask), (a) for STL/VTK/image-batch
(rendered grayscales with anti-aliasing).

**Verify.** Rasterise a perfect disk of radius 1024 px and measure.

| Estimator | Reported perimeter | Error |
|---|---|---|
| True | 2π·1024 ≈ 6434 px | — |
| Polygonal (today) | ~6700 px | +4 % |
| `approxPolyDP(ε=0.5)` | ~6470 px | +0.6 % |
| Crofton (4 directions) | ~6460 px | +0.4 % |

---

## Phase 4 — Reproducibility & polish

### 9. Method-signature hash in every results Excel
**Effort:** S · **Risk:** low

Each results Excel grows one new metadata field, "Method signature".
GASP refuses to compare measurements against the reference profile
unless the signatures match (or warns loudly).

**Formatter**

```python
# helpers/results_excel_format.py — add to ResultsSheet
@dataclass
class ResultsSheet:
    ...
    method_signature: str | None = None
```

Render under the row-3 metadata line (`File name / Folder / User /
Date`):

```python
# _render_sheet, after the row-3 pair block
if sheet.method_signature:
    ws.cell(row=4, column=2, value="Method signature").font = _LABEL_FONT
    ws.cell(row=4, column=3, value=sheet.method_signature)
```

**Builder**

```python
# helpers/method_signature.py (new)
import hashlib, json
from constants import (
    SULCUS_TERTIARY_MIN_FRACTION, SULCUS_TERTIARY_MAX_FRACTION,
    SULCUS_SECONDARY_MIN_FRACTION, SULCUS_SECONDARY_MAX_FRACTION,
    SULCUS_PRIMARY_MIN_FRACTION,   SULCUS_PRIMARY_MAX_FRACTION,
)

FETOMORPH_METHOD_VERSION = "2026.06"

def method_signature(*, kernel_size_mm, pixel_size, cnt_threshold,
                     contour_mode) -> str:
    payload = {
        "v":              FETOMORPH_METHOD_VERSION,
        "kernel_mm":      round(float(kernel_size_mm), 4),
        "pixel_size":     round(float(pixel_size),    6),
        "cnt_threshold":  round(float(cnt_threshold), 2),
        "contour_mode":   contour_mode,
        "sulcus_frac":    [
            SULCUS_TERTIARY_MIN_FRACTION, SULCUS_TERTIARY_MAX_FRACTION,
            SULCUS_SECONDARY_MIN_FRACTION, SULCUS_SECONDARY_MAX_FRACTION,
            SULCUS_PRIMARY_MIN_FRACTION,   SULCUS_PRIMARY_MAX_FRACTION,
        ],
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:10]
```

Plumb into every Excel caller (`measurements_*.py`, `metrics_store.py`,
`gasp_export.py`). The GASP registry CSV gets a `method_signature`
column too; bump `FETOMORPH_METHOD_VERSION` whenever any algorithm
change in Phases 1–3 lands.

**Verify.** Same settings → identical signatures; change any one of
the inputs → signature changes. Run the GASP compare with mismatched
signatures and confirm the warning fires.

---

### 10. Otsu-adaptive binary thresholds
**Effort:** S · **Risk:** medium (changes contours on
unusually-illuminated renders)

| File | Current | Change |
|---|---|---|
| `functions/measurements_image.py:74`, `functions/measurements_stl.py:207`, etc. | `cv2.threshold(im_bw, BINARY_THRESHOLD_DEFAULT, 255, 1)` | `cv2.threshold(im_bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)` |
| `functions/measurements_vtk.py:180` | `cv2.threshold(gray, BINARY_THRESHOLD_VTK, 255, 0)` | `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)` |
| `constants.py:10-11` | `BINARY_THRESHOLD_VTK = 150`, `BINARY_THRESHOLD_DEFAULT = 200` | Delete |

Otsu picks the bimodal cut per image, so the per-render constants
become unnecessary.

**Verify.** Generate a synthetic render with a mid-grey background
(`fill=128`) and a slightly-brighter brain (`fill=180`). Fixed
threshold misclassifies the brain; Otsu finds the right cut.

---

### 11. Rename `perimeter_convex` → `perimeter_outer_envelope`
**Effort:** S · **Risk:** zero

The "convex" name is misleading: it's the morphological-closing
envelope, **not** the convex hull.

| File | Symbol or column |
|---|---|
| `functions/measurements_image.py` | local variable `perimeter_convex`, `perimeter_convex_sum` |
| `helpers/gasp_export.py` | header keys if any |
| `managers/metrics_store.py` | `ensure_metric_row`'s `"Perimeter_convex"` key |
| `Examples/gestational_week_reference.csv` | column header (keep `perimeter_convex` as a deprecated alias for one release) |

Do this **last** — pure rename, would otherwise churn diffs across
the math fixes.

**Verify.** `grep -rn "perimeter_convex" .` returns only the renamed
sites and the backward-compatibility alias.

---

## Suggested execution checklist

```
Phase 1 — Foundations & correctness blockers
[ ] 1.  kernel_size in millimetres (end-to-end, with px conversion)  ← do first
[ ] 2.  Hard-fail fallbacks (None / NaN propagation)
[ ] 3.  Mesh.volume / mesh.area for STL & VTK totals
[ ] 4.  Marching cubes for NIfTI surface area

Phase 2 — Calibration bias
[ ] 5.  parallel_projection = True on every PyVista plotter

Phase 3 — Refinements
[ ] 6.  Frustum integration (skip if item 4 is done)
[ ] 7.  Simpson's rule for the slice-volume sanity check
[ ] 8.  Contour smoothing (approxPolyDP) or Crofton perimeter

Phase 4 — Reproducibility & polish
[ ] 9.  Method-signature hash in every results Excel
[ ] 10. Otsu-adaptive binary thresholds
[ ] 11. Rename perimeter_convex → perimeter_outer_envelope
```

After each item, re-run the example measurement
(`Examples/cropped_slices`) and diff the new per-slice Excel against
the previous one — anything more than ~3 % unexpected shift is a sign
the change had unintended scope and should be reviewed before moving
to the next item.

Once Phase 1 is complete, regenerate
`Examples/gestational_week_reference.csv` from a known good cohort so
downstream GASP comparisons reflect the corrected formulas (corrected
totals from items 3–4 AND the millimetre kernel from item 1). Bump
`FETOMORPH_METHOD_VERSION` (item 9) at the same time so the new
reference can't be cross-matched against any historical run.

---

## Out of scope (handled by the isotropy hint, not by code)

- **Per-axis anisotropic mm/px for convexity defect depths.** The
  helper `defect_mm_per_px_and_fixed` (`helpers/helpers.py:421-452`)
  exists and is correct, but wiring it into image / STL / VTK is
  unnecessary as long as we ship the isotropy assumption in the
  import dialog and Excel header. NIfTI already routes through the
  helper for voxel anisotropy; that stays.

Show an **isotropy warning** in two places — covered by a single
small change, not counted in the checklist above:

```python
# managers/file_manager.py — inside load_image / import_image
pm = QPixmap(path)
# When user calls set_image_scale, set both pixel_size_x and pixel_size_y;
# the dialog already accepts a single number, so just emit a one-liner:
print("[Calibration] FetoMorph assumes square pixels (px_x == px_y). "
      "If your image has non-square pixels, resample before measuring.")
```

```python
# helpers/results_excel_format.py — add to the footer rendering
ws.cell(row=row, column=2,
        value="Assumes isotropic pixels (px_x == px_y). "
              "Non-isotropic input must be resampled before measuring."
       ).font = Font(italic=True, color="888888")
```

That's the only "anisotropy" touch needed — a documentation hint, not
a code path.
