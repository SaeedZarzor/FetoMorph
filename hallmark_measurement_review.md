# Hallmark Measurement Methods — Mathematical Review

Scope: the morphometric kernels in `helpers/helpers.py`,
`functions/measurements_image.py`, `functions/measurements_nifti.py`,
`functions/measurements_stl.py`, and `functions/measurements_vtk.py`.
Constants live in `constants.py`. Below I work through each hallmark's
formula, list the assumptions baked in, flag where the math is biased
or fragile, and recommend concrete improvements. File:line references
point to the implementations.

---

## 1. Area (2D) — `Σ cv2.contourArea(c) · pxs²`

Implementation: `functions/measurements_image.py:103-108`,
`functions/measurements_stl.py:229,234`,
`functions/measurements_vtk.py:218`.

**Math.** `cv2.contourArea` evaluates the shoelace integral for a
closed polygon. For a polygon with vertices `(x_i, y_i)`:

```
A = ½ |Σ (x_i · y_{i+1} − x_{i+1} · y_i)|
```

Multiplying by `pixel_size²` converts pixel² → physical². Summing
across multiple connected components yields total foreground area.

**Correctness.** The shoelace formula is exact for polygons. The only
error is the polygonal approximation of the true contour — and since
the contour is rasterized to begin with, the approximation is
essentially perfect: `cv2.contourArea` returns the polygonal area,
not the rasterized pixel count. (Note: this can differ from
`np.sum(mask)·pxs²` by up to half a pixel-perimeter — usually
negligible.)

**Assumptions / limitations.**

- Isotropic pixel size — `area = px_count · pxs²` only when x and y
  have the same spacing. NIfTI handles this correctly via
  `pixel_area_mm2 = pixel_size_x × pixel_size_z`
  (`measurements_nifti.py:118-163`), but `compute_image_allmarks`
  assumes a single `pixel_size` value; if applied to anisotropic
  images, area is wrong by a factor `pxs_y / pxs_x`.
- Holes are not subtracted in the image/STL/batch paths (they use
  `RETR_EXTERNAL`). VTK uses `RETR_CCOMP` plus
  `split_inner_and_internal_contours`
  (`helpers/helpers.py:468-513`) but the user picks
  "outer / subtract / internal_only" — the `subtract` mode is the
  only one that does outer − internal, and even there the holes are
  filtered by `min_contour_area`, so small ventricles may slip past
  either filter.
- `cv2.contourArea` returns absolute area, so it ignores winding
  direction. A hole drawn as a separate outer contour would be added
  rather than subtracted, biasing area high.

**Recommendation.** Pick the contour-accounting mode explicitly
per-modality (most fetal brain workflows want "outer minus
internal"), and document the per-mode formula in the user dialog. If
the codebase is willing to deviate from OpenCV, replace the
polygon-area sum with the signed shoelace sum and use the hierarchy
to subtract holes automatically.

---

## 2. Perimeter (2D) — `Σ cv2.arcLength(c, True) · pxs`

Implementation: `functions/measurements_image.py:104,109`,
`functions/measurements_stl.py:230,232`, etc.

**Math.** `cv2.arcLength` sums Euclidean distances between
consecutive contour vertices. For a closed contour
`{p_0, p_1, …, p_{n-1}}`:

```
P = Σ_{i=0}^{n-1} ‖p_{(i+1) mod n} − p_i‖
```

Multiplying by `pixel_size` converts to physical units.

**Correctness.** This computes the **polygonal perimeter** of the
contour, which systematically **overestimates the true smooth
boundary length** for sharply digitized contours. The Crofton /
pixel-grid bias is well known: for a straight 45° edge of a binary
mask, the polygonal perimeter is `n · √2` whereas the true
(anti-aliased) length is also `n · √2` — they agree on diagonals.
For staircase boundaries (axis-aligned discrete steps), the
polygonal length is `n` while the true length is `n` only if the
boundary is genuinely axis-aligned; if it's a smooth curve quantised
to a stair, the polygonal length over-estimates by up to ~4/π ≈ 1.27
on average.

In OpenCV, contours from `CHAIN_APPROX_SIMPLE` already collapse
axis-aligned runs into endpoints, so the staircase bias is largely
avoided. The remaining error is the discretization of curvature, and
is small (≤ few %) for smooth biological boundaries.

**Assumptions / limitations.**

- Isotropic pixel size again (same as Area).
- `arcLength(..., True)` closes the polygon — for `RETR_EXTERNAL`
  from a foreground mask this is always correct.
- Smoothing of the contour (e.g., `approxPolyDP`) is not applied, so
  noise in the rasterized boundary adds to perimeter. This biases
  LGI (next section) **high**.

**Recommendation.** For perimeter-based statistics (LGI,
compactness), apply a light Douglas–Peucker simplification
(`cv2.approxPolyDP(c, epsilon=0.5, closed=True)`) before measuring.
Or use the **Crofton perimeter** (sum of 4 chord lengths at
0°/45°/90°/135°) — this is the standard stereological estimator and
has much lower bias on noisy masks.

---

## 3. Compactness — 2D `4πA/P²` and 3D `36πV²/S³`

Implementation: `helpers/helpers.py:750-758`.

**Math.** Both are the standard **isoperimetric quotient**:

- 2D: `ψ₂ = 4πA / P² ∈ (0, 1]`, with `ψ₂ = 1` iff the shape is a disk.
- 3D: `ψ₃ = 36πV² / S³ ∈ (0, 1]`, with `ψ₃ = 1` iff the shape is a
  ball.

Both follow directly from the isoperimetric inequalities
`P² ≥ 4πA` and `S³ ≥ 36πV²`.

**Correctness.** The formulas are mathematically exact.

**Assumptions / limitations.**

- `ψ ∈ (0, 1]` holds for *true* P, A, V, S. Because P is biased
  **high** (Section 2) and S is biased **low** (Section 8), ψ₂ tends
  to be **underestimated** and ψ₃ tends to be **overestimated** in
  this codebase. The latter is why `compute_stl_allmarks` warns when
  `compactness > 1.0` (`measurements_stl.py:362-365`) — that's a
  smoking gun for the systematic bias.
- Division by zero is guarded for P=0 and S=0 in `compactness_2D` /
  `compactness_3D`, but **not** when A=0 with P>0:
  `compactness_2D(0, 100) = 0` — fine. The 3D version returns 0 on
  S=0; for V=0, S>0 it also returns 0 — fine.

**Recommendation.** Document the bias direction in the GUI/help:
"Compactness > 1 may occur because surface area is computed as a
stack-of-slabs lateral integral, which under-estimates the true
surface; expect ψ₃ ∈ [0, ~1.3] in practice." Even better — fix the
surface area formula (Section 8), then ψ₃ becomes a real geometric
quantity.

---

## 4. Gyrification Index (LGI) — `inner_perimeter / outer_perimeter`

Implementation: `functions/measurements_image.py:121-135`,
`functions/measurements_stl.py:223-233,310`, NIfTI/VTK analogues.

**Math.** The outer envelope is computed as

```
outer_mask = (inner_mask) ● K   (morphological closing, K = disk of diameter `kernel_size`)
```

Then `outer_perim = arcLength(boundary(outer_mask))`. The reported
LGI is

```
LGI = inner_perim / outer_perim
```

**Correctness vs. canonical definition.** Zilles' original 2D
Gyrification Index is `P_pial / P_outline`, where the outline is a
smooth envelope drawn by an expert tracer. The morphological-closing
surrogate is a reasonable *approximation* of that envelope, but it
is **kernel-dependent**:

- Sulci whose mouth opening is **wider** than K survive the close →
  they're counted as "convex" boundary, not as gyrification. They
  depress LGI.
- Sulci whose mouth opening is **narrower** than K get filled →
  counted as gyrification. They lift LGI.

Mathematically, for a structuring element of radius r = K/2:

```
closing fills any sulcus whose opening width < 2r and depth < r
```

so the operator acts like a low-pass filter on the boundary with a
cutoff wavelength ~ K pixels. The chosen `kernel_size = 25` (used in
the GASP reference) corresponds to ~25 pixels of fill, which is
reasonable for histological slices but completely scale-dependent.

**Assumptions / limitations.**

- LGI is unitless and scale-invariant under uniform isotropic
  scaling **only if the kernel size scales with the image**.
  Currently `kernel_size` is set in pixels, so a brain rendered at
  higher resolution will yield a different LGI than the same brain
  at lower resolution — a serious confound.
- The "outer rebuilt from inner only" trick
  (`measurements_image.py:121-122`) is good — it prevents noise
  blobs from creating spurious outer components. Confirmed correct.
- `perimeter_convex_sum = 1` fallback
  (`measurements_image.py:131-132`) when no outer contour is found
  means `LGI = perimeter / pxs · 1 / pxs = perimeter / pxs²` — a
  meaningless huge number. The fallback should be `None` or skip the
  slice.
- Misleading name: the variable is `perimeter_convex`, but it's the
  **closed-envelope perimeter**, not the **convex hull perimeter**.
  Real Zilles GI usually uses convex hull or smoothed envelope, but
  not morphological close. The naming hides this design choice from
  reviewers.

**Recommendation.**

1. **Scale the kernel** with the image: define `kernel_size_mm`
   (user-facing) and convert
   `kernel_size_px = round(kernel_size_mm / pixel_size_x)`. Document
   the millimeter value used in the Excel header so different runs
   are comparable.
2. **Replace the fallback** `perimeter_convex_sum = 1` with
   `None`/`NaN` and propagate downstream.
3. **Offer the true convex hull** as an alternative outer boundary:
   `outer_perim = arcLength(cv2.convexHull(np.vstack(inner_contours)))`.
   The convex hull is parameter-free, mathematically clean, and
   corresponds to a different (also valid) definition of GI. Let the
   user pick.
4. **Rename** `perimeter_convex` → `perimeter_outer_envelope` and
   update headers.

---

## 5. Sulcus depth via `cv2.convexityDefects` — `depth_mm = d · pxs / 256`

Implementation: image (`measurements_image.py:144-175`), STL
(`measurements_stl.py:243-269`), VTK, NIfTI
(`measurements_nifti.py:200-231`).

**Math.** For a contour with computed convex hull, each defect is
the triple `(start, end, far)` where the far point is the contour
vertex with maximum perpendicular distance to the chord
`start → end`. OpenCV returns this distance as an integer **in 8.8
fixed-point** units of *pixels*:

```
depth_pixel = d / 256
```

Multiplied by mm/pixel (`pixel_size` or `mm_per_px`), this gives the
perpendicular distance from the deepest sulcus point to the local
convex envelope chord.

**Correctness.** The conversion `d / DEFECT_FIXED_POINT = d / 256`
matches OpenCV's documented behavior. The isotropic
`depth_value = d × pixel_size / 256` is correct **only when
pixel_size_x = pixel_size_y** (the defect's normal direction is
unknown a priori).

NIfTI handles anisotropic voxels properly via
`defect_mm_per_px_and_fixed` (`helpers/helpers.py:421-452`): it
reconstructs the unit normal to the chord `(start → end)` in pixel
space, then scales each component by the anisotropic pixel size:

```
unit_normal û = (u_x, u_y) in pixel space
mm_per_px_in_normal_direction = √((s_x · u_x)² + (s_z · u_y)²)
mm_per_fixed = mm_per_px / 256
```

This is **mathematically correct** for the perpendicular-distance
interpretation. STL/VTK don't go through this path because their
`mm_per_px` is rasterized from a render (single value, treated as
isotropic), which is a reasonable assumption for orthographic
renders of a calibration cube.

**Assumptions / limitations.**

- `convexityDefects` requires the input contour to be **simple,
  closed, oriented clockwise** with monotonic hull indices. The
  guard `np.all(np.diff(hull.ravel()) > 0)` catches non-monotonic
  hulls but `cv2.convexHull(..., clockwise=True)` is supposed to
  enforce orientation. Self-intersecting contours from segmentation
  noise can still slip through; OpenCV may produce nonsensical
  defects (negative depths after fixed-point conversion). The
  current code uses `if d > DEFECT_FIXED_POINT` (i.e. depth > 1 px)
  as a guard, which is a reasonable but ad-hoc threshold.
- The defect depth is **perpendicular to the chord**, not the true
  geodesic depth into the sulcus from the cortical surface. For
  complex, branching sulci this under-estimates the true depth.
  It's the right quantity for a "gyrification proxy" but it's *not*
  a true geodesic sulcal depth (the literature uses Eikonal solvers
  on a closed-skull mask for that).
- The depth is sensitive to where convex hull vertices land. A
  sulcus with two nearby gyri on either side may collapse into one
  defect with a longer chord, biasing depth high; a sulcus next to a
  single tall gyrus may be split into two defects.

**Recommendation.**

1. Add the **anisotropic correction path** to image / STL / VTK
   (not just NIfTI) — it's the only mathematically defensible option
   when calling `convexityDefects`.
2. For a more robust sulcal depth field, consider an
   **Eikonal-based depth**: solve `∇d = 1` with `d = 0` on the
   outer-envelope boundary, then read the value at each sulcus
   extremum. This is the canonical "sulcal depth" used in CIVET /
   FreeSurfer. Heavyweight but the math is well-defined.
3. Guard `d > DEFECT_FIXED_POINT` (currently `> 256` ≡ depth > 1
   pixel) is OK as noise-floor but should be exposed as a setting,
   not a magic constant.

---

## 6. Sulci classification — `depth / slice_length` → primary / secondary / tertiary

Implementation: `helpers/helpers.py:118-135`, constants
`SULCUS_*_FRACTION`.

**Math.** Given a defect of depth `d` (mm) and a slice
characteristic length `L` (mm), the depth fraction `frac = d / L` is
compared to three intervals:

- primary: 15–50%
- secondary: 5–15%
- tertiary: 1.5–5%
- else: unclassified

`L` is set to `max(W_px, H_px) · pixel_size` (the longest in-plane
bbox side of the brain).

**Correctness / assumptions.**

- This is a **heuristic taxonomy** with no anatomical ground truth.
  It is *consistent* across runs as long as `L` is consistent, but
  the thresholds are not derived from anatomy — they're best treated
  as analyst-defined bins. Document this clearly in user-facing docs
  so the GASP downstream isn't read as "primary sulci count" in the
  clinical sense.
- The intervals `[15, 50]`, `[5, 15]`, `[1.5, 5]` are
  **non-overlapping at the boundaries** (15% goes to primary because
  the test uses `≤`; 5% goes to secondary; 1.5% to tertiary). One
  issue: with `frac == 0.15`, both primary's `>=` and secondary's
  `<` boundary logic give "primary" — which is fine. But
  `frac == 0.05` triggers secondary (because `<= 0.05` for tertiary
  is false at the upper end and `<= 0.50` for primary is false at
  the lower end; secondary's `<= 0.15` with `>= 0.05` matches). The
  thresholds are mutually exclusive — good.
- `slice_length` is the **longest** side of the brain bbox, but the
  user can slice along any axis. A brain that's 100mm AP × 80mm LR
  × 75mm IS, sliced coronally, gives `L = max(100, 80) = 100` (AP ×
  LR plane). Sliced axially, gives `L = max(100, 80) = 100` again
  (different plane). This is bias-free across slice orientations
  only because of the `max` — but a sulcus of depth 20mm gets
  labelled "primary" at L=100mm and "secondary" at L=150mm. So the
  **same anatomical sulcus can change class as the brain grows**
  (GA increases) — a real problem for the GASP comparison if
  reference profiles were collected at different brain sizes.

**Recommendation.**

1. Anchor classification to an **absolute mm scale** instead of a
   fraction of slice length, OR normalize by an age-corrected length
   (e.g., the mean fetal AP length at the slice's gestational
   week). The current normalization makes the bins age-dependent.
2. Move the thresholds to the `VisualizationSettings` / preferences
   so analysts can tune them per cohort.

---

## 7. Volume (3D) — stack-of-slabs integration

Implementation: STL (`measurements_stl.py:308`), NIfTI
(`measurements_nifti.py:254`), VTK (`measurements_vtk.py:291`).

**Math.** All three implement the **Riemann (midpoint) rule**:

```
V ≈ Σ_i A_i · Δh_i
```

For STL: `brain_volume = sum_area · slice_thickness_eff` (then
/1000 for mm³→cm³), with
`slice_thickness_eff = brain_dim[axis] / N` where
`N = len(slice_positions)` from `np.arange(low, high, slice_thickness)`
(`measurements_stl.py:146`). For NIfTI:
`sum_area · pixel_size_y / 1000`, where `pixel_size_y` is the
voxel-spacing along the chosen axis. For VTK: same as STL.

**Correctness.** The Riemann sum converges to V at rate O(Δh) for
smooth surfaces. Simpson's rule would converge at O(Δh⁴), but
Riemann is fine for the resolutions typically used (slice spacing <
1 mm).

**Assumptions / limitations.**

- **STL `slice_thickness_eff` ≠ what the user asked for.** The user
  passes `slice_thickness`, and the code uses
  `slice_thickness_eff = brain_dim[axis] / N`. Because
  `N = len(np.arange(low, high, slice_thickness))`, the effective
  spacing is `(high - low) / N` ≈ `slice_thickness` but not exactly
  when the brain extent isn't a multiple of `slice_thickness`. Worst
  case `slice_thickness_eff ≠ slice_thickness` by up to
  `slice_thickness / N`. Minor.
- **Skipped slices break the sum.** Slices with
  `section.n_points == 0` are skipped via `continue`. `sum_area`
  only accumulates kept slices, but `slice_thickness_eff` is
  computed from `N` *before* skipping. If many slices are skipped
  at the brain extremes (PyVista returns empty intersections just
  outside the brain), the volume estimate is slightly off —
  `slice_thickness_eff` is biased low. In practice the brain
  extends across all of its bbox so this rarely matters.
- **NIfTI volume = voxel sum (exact).** Because
  `slice_area = np.sum(slice_mask) × pixel_size_x × pixel_size_z`
  and
  `volume = Σ slice_area × pixel_size_y = (Σ N_voxel) × pixel_size_x × pixel_size_z × pixel_size_y`,
  the formula is exactly the voxel-volume sum. No discretization
  error beyond the segmentation itself.
- The `/ 1000` factor (mm³ → cm³) and `/ 100` (mm² → cm²) are
  correct.

**Recommendation.**

1. STL/VTK: replace `slice_thickness_eff = brain_dim/N` with
   `slice_thickness_eff = slice_thickness` directly. The current
   formula was probably intended as a sanity check but introduces a
   small bias.
2. STL/VTK: upgrade to **Simpson's rule** for the Riemann sum —
   same cost, ~3× lower error:

   ```
   V ≈ (Δh/3) [A_0 + 4A_1 + 2A_2 + 4A_3 + … + A_{N-1}]
   ```

3. STL has a `mesh.volume` PyVista property that returns the
   **exact closed-surface volume** via the divergence theorem. Use
   it as the authoritative value; the slice integration is then
   just for the per-slice table. Same for surface area below.

---

## 8. Surface area (3D) — `Σ inner_perim · slice_thickness`

Implementation: STL (`measurements_stl.py:309`), NIfTI
(`measurements_nifti.py:257`), VTK (`measurements_vtk.py:292`).

**Math.** `Area = sum_inner_perim · slice_thickness_eff`. This is
the **lateral surface area** of a stack of right prisms — i.e. it
integrates `∫ P(z) dz` where P(z) is the slice perimeter.

**Correctness.** For a closed smooth surface with cross-sectional
perimeter P(z), the integral `∫ P(z) dz` gives the **lateral**
surface area — i.e. the sum of perimeters × heights. The **true**
surface area is

```
S_true = ∫ P(z) · √(1 + (dr/dz)²) dz
```

where the `√(…)` factor accounts for the slope of the surface as z
varies. The codebase omits this factor.

**Consequences.**

- For a perfect sphere: `P(z) = 2π√(R²−z²)`. Lateral integral
  `∫_{-R}^R 2π√(R²−z²) dz = π²R² ≈ 9.87 R²`. True surface area =
  `4πR² ≈ 12.57 R²`. So **lateral approximation under-estimates by
  ~21.5%**.
- This is exactly why STL/VTK compactness occasionally exceeds 1
  (Section 3) — the compactness formula divides by a too-small
  surface area.

**Assumptions / limitations.**

- Top and bottom caps are not added. For a sphere they're 0 (single
  point), but for a fetal brain the IS-extreme slices have non-zero
  area that should also contribute to S.
- The "stack of slabs" model treats each slab as a right prism —
  physically wrong for any non-flat brain region.

**Recommendation.**

1. Use the **PyVista `mesh.area` property** as the authoritative
   STL/VTK surface area. It computes the exact triangle-sum surface
   area via `Σ ½ |(v_1 − v_0) × (v_2 − v_0)|`. Same complexity as a
   slice integral, mathematically exact for the input mesh.
2. For NIfTI (where no mesh exists), apply a **marching cubes
   pass** (`skimage.measure.marching_cubes`) to extract a surface
   from the binary mask, then sum triangle areas. The lateral
   integral is fast but biased; marching cubes is the correct
   approach.
3. Failing those, the lateral integral can be corrected by the
   slope factor:

   ```
   S ≈ Σ ((P_i + P_{i+1}) / 2) · √(Δh² + ((r_{i+1} − r_i))²)
   ```

   where `r_i` is an equivalent radius `√(A_i / π)`. This is the
   frustum approximation and removes most of the bias.

---

## 9. Scale calibration via red reference cube

Implementation: `helpers/helpers.py:515-533`, called from
`measurements_stl.py:196` and `measurements_vtk.py:171`.

**Math.** A small thin red cube of known side length `cube_len`
(mm) is added to each PyVista render. The code finds the largest
red blob, takes its bounding-box width `w` in pixels, and returns

```
mm_per_px = cube_len / w
```

**Correctness.** This is the right idea (a known physical reference
detected in pixel space). However:

- **Orientation sensitivity.** The cube is positioned to show its
  face perpendicular to the camera (`make_scale_cube` sets the
  cube's thin axis along the slice direction). The bbox width then
  ≈ the projected face width in pixels. Under an **orthographic**
  camera, this is exact. Under a **perspective** camera, the bbox
  width depends on the cube's distance from the camera, which
  differs slightly between slice positions. PyVista's
  `view_xy / view_xz / view_yz` set parallel projection in most
  versions, but this should be made explicit
  (`p.parallel_projection = True`) to avoid the per-slice drift.
- **Color detection robustness.** The mask `(R > 150) & (G < 50)`
  is a coarse colour gate. Anti-aliasing at the cube edge may bleed
  into the brain rendering (which is mostly grey-white for STL or
  black-white for VTK), but in practice this is reliable.
- **Bounding box uses width only.** `w` is the bounding-box `x`
  extent. If the cube is rotated in-plane (it shouldn't be —
  PyVista respects axis alignment), this would under-estimate. The
  cube being thin along the slice axis means its visible face is
  square, so `w ≈ h`; using `max(w, h)` would be slightly more
  robust.
- **Fallback `mm_per_px = 1.0`.** If no red contour is found,
  returns 1.0 with a warning. Downstream measurements become
  "perimeter in pixels" reported as millimetres — silently wrong.
  This should hard-fail or return `None`.

**Recommendation.**

1. Force orthographic projection in PyVista before each screenshot
   (`p.parallel_projection = True`).
2. Replace the bbox `w` with the **diagonal-corrected major edge**
   of the cube face: fit a `cv2.minAreaRect` to the largest red blob
   and use the longer side length. This is more robust to
   anti-aliasing and slight rotations.
3. Change the no-red fallback from `return 1.0` to `return None`,
   and propagate the error so the slice is skipped with a clear log
   message.
4. Add an internal sanity check: render a cube of known side,
   measure it, and assert `mm_per_px · w ≈ cube_len` within ±1%
   before the loop starts.

---

## 10. Cross-cutting concerns

### Isotropy assumption

`compute_image_allmarks` uses a single `pixel_size`. NIfTI handles
anisotropy correctly for area/depth, but the LGI computation (which
feeds the closing operation in *pixel space*) treats the kernel as
isotropic — a 25 × 25 pixel ellipse on a 1.0 × 0.5 mm/pixel grid is
a 25 × 12.5 mm physical kernel, not a 25 × 25 mm disk. Recommend
resampling to isotropic spacing before the morphology stage, or
replacing `getStructuringElement(MORPH_ELLIPSE, (K, K))` with an
anisotropic ellipse sized in physical units.

### Magic constants

- `DEFECT_FIXED_POINT = 256` — fundamental OpenCV constant, fine.
- `BINARY_THRESHOLD_DEFAULT = 200`, `BINARY_THRESHOLD_VTK = 150` —
  depend on render background. Make these per-render adaptive
  (Otsu's method on the gray image:
  `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`).
- `kernel_size = 25` default — pixel-space, not mm-space
  (Section 4).
- `if d > DEFECT_FIXED_POINT` in STL/VTK — effectively "ignore
  defects < 1 pixel deep". Reasonable but undocumented.

### Numerical robustness

- `perimeter_convex_sum = 1` and `mm_per_px = 1.0` fallbacks
  silently produce meaningless results. Always prefer `None` / NaN
  propagation.
- The "monotonic hull indices" check
  (`np.all(np.diff(hull.ravel()) > 0)`) is a guard against OpenCV's
  known bug with `convexHull(..., returnPoints=False)`. Acceptable
  workaround.

### Reproducibility

The GASP downstream compares measured hallmarks to a reference
profile (`Examples/gestational_week_reference.csv`). For this to be
meaningful, **every parameter affecting the formulas must be
identical between reference-collection and current-measurement
runs**: kernel size, pixel size, contour threshold, sulci
classification thresholds, slice orientation classification. Many of
these are currently global defaults that drift. The new per-row
"adjustment parameters" columns in the dock export are a good
start; consider also persisting a single "method signature" hash
with every Excel so users can verify their measurement is comparable
to the reference.

---

## 11. Summary of recommended improvements (prioritised)

| Priority | Change | Why |
|---|---|---|
| **P0** | Use `mesh.area` / `mesh.volume` in STL & VTK for the totals | Eliminates the ~20 % under-estimate of surface area and the resulting compactness > 1 |
| **P0** | NIfTI surface area via marching cubes | Same as above, no mesh available so MC is the only option |
| **P0** | Replace `perimeter_convex_sum = 1` and `mm_per_px = 1.0` fallbacks with `None` / NaN | Silent garbage today |
| **P1** | Define `kernel_size` in millimetres, convert to pixels per call | LGI currently varies with image resolution |
| **P1** | Force `parallel_projection = True` before each PyVista screenshot | Removes per-slice mm/px drift |
| **P1** | Propagate anisotropic mm-per-px (Equation in Section 5) to image / STL / VTK depth conversion | Currently only NIfTI is correct under anisotropy |
| **P2** | Replace stack-of-slabs S with frustum integration when a mesh isn't available | Roughly halves S bias for fetal-brain-like shapes |
| **P2** | Upgrade volume integration to Simpson's rule | Free accuracy improvement |
| **P2** | Apply `approxPolyDP(epsilon ≈ 0.5)` before perimeter calculation, OR switch to Crofton perimeter | Removes ~few % positive bias in P |
| **P3** | Persist a method-signature hash in every Excel header | Makes reference comparison reproducible |
| **P3** | Make `BINARY_THRESHOLD_*` Otsu-adaptive | Removes render-color brittleness |
| **P3** | Rename `perimeter_convex` → `Closed-envelope perimeter` | Code clarity |


## Changes include
- Replace `perimeter_convex_sum = 1` and `mm_per_px = 1.0` fallbacks with `None` / NaN 
- Define `kernel_size` in millimetres, convert to pixels per call 
- Force `parallel_projection = True` before each PyVista screenshot 
- Upgrade volume/ Surface integration to Simpson's rule
- Include surface cups (up and buttom surfaces) -> Eliminates the compactness >1
- Include cavity option with threshould on inner areas
- Make the filttered threshould in mm instade of pixels
- Apply `approxPolyDP(epsilon ≈ 0.5)` before perimeter calculation, OR switch to Crofton perimeter as an optional setting
- Make `BINARY_THRESHOLD_*` Otsu-adaptive
- Rename `perimeter_convex` → `perimeter_outer_envelope`

## Changes planned
- Persist a method-signature hash in every Excel header | Makes reference comparison reproducible


## Changes ignored
- Use `mesh.area` / `mesh.volume` in STL & VTK for the totals 
- NIfTI surface area via marching cubes
- Propagate anisotropic mm-per-px (Equation in Section 5) to image / STL / VTK depth conversion
- Replace stack-of-slabs S with frustum integration when a mesh isn't available -> The results genertaed not correct

