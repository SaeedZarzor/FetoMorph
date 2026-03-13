"""Heuristic brain-mesh classifier for FetoMorph.

Scores an STL / VTK mesh on several geometric criteria (size, aspect
ratios, bilateral symmetry, gyrification proxy, connectivity) and
classifies it as ``"adult_brain"``, ``"fetal_brain"``, or ``"not_brain"``.

Scoring bands and weight coefficients are based on published fetal /
adult brain morphometry ranges.  Each sub-score is normalised to [0, 1]
and combined in a weighted sum; the higher aggregate wins.

Weight rationale (see ``check_brain``):
    * **Size** (adult 0.26 / fetal 0.36) — strongest single signal; fetal
      brains are more size-variable, so the weight is higher there.
    * **Shape** (0.20 / 0.18) — aspect-ratio bands differ between age groups.
    * **Symmetry** (0.18 / 0.16) — bilateral symmetry along the mid-sagittal
      plane; less discriminative for fetuses because folding is still developing.
    * **Hemisphere balance** (0.12 / 0.10) — fraction of vertices in each
      hemisphere; healthy brains are roughly 50/50.
    * **Gyrification proxy** (0.19 / 0.15) — surface area / convex-hull area;
      adult cortex is more folded than fetal.
    * **Connectivity** (0.05 / 0.05) — number of significant connected
      components; a clean brain mesh is typically one component.
"""

import numpy as np
import trimesh
import pyvista as pv

from math import isfinite as _py_isfinite

# Optional SciPy for faster symmetry
try:
    from scipy.spatial import cKDTree  # noqa: F401
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _np_isfinite(x):
    """Return ``True`` if *x* is a finite number (handles non-numeric types)."""
    try:
        return np.isfinite(x)
    except Exception:
        return False

def _is_finite(x):
    """Return ``True`` if *x* is not ``None`` and is finite."""
    return x is not None and _np_isfinite(x)

def _bool_in_range(x, lo, hi):
    """Return ``True`` if *x* is finite and within ``[lo, hi]``."""
    return _is_finite(x) and lo <= x <= hi

def _safe_ratio(a, b):
    """Divide *a* by *b*, returning ``nan`` on zero / non-finite inputs."""
    if b == 0 or not _np_isfinite(a) or not _np_isfinite(b):
        return np.nan
    return float(a / b)

def _pca_axes(points: np.ndarray):
    """Return the centroid and PCA axes (rows of Vt) for a point cloud."""
    c = points.mean(axis=0)
    X = points - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return c, Vt  # axes rows

def _unit_scale_to_mm(extents):
    """Guess whether extents are in mm, cm, or m and convert to mm.

    Picks the unit that places the longest extent closest to known brain
    sizes (140–200 mm adult, 60–120 mm fetal).
    """
    L = float(np.max(extents)) if np.max(extents) != 0 else 0.0
    def score_mm(Lmm):
        if 140 <= Lmm <= 200: return 0
        if 60 <= Lmm <= 120:  return 1
        return 2 + min(abs(Lmm - 170), abs(Lmm - 90)) / 170.0
    candidates = {"mm": L, "cm": L*10.0, "m": L*1000.0}
    best = min(candidates, key=lambda k: score_mm(candidates[k]))
    scale = 1.0 if best == "mm" else 10.0 if best == "cm" else 1000.0
    return extents * scale  # now mm

def _symmetry_score(points, center, axes):
    """Compute bilateral symmetry as histogram overlap in the PCA frame.

    Projects vertices into PCA space, splits into positive/negative
    halves along the second axis, bins the XZ projection, and returns
    1 − (bin-wise difference / total).  Falls back to a pure-NumPy
    histogram when SciPy is unavailable.
    """
    # histogram fallback (no SciPy required)
    R = axes.T
    pts = (points - center) @ R
    xz = pts[:, [0, 2]]
    bins = 24

    def to_idx(v):
        vmin, vmax = v.min(), v.max()
        if vmax == vmin:
            return np.zeros_like(v, dtype=int)
        return np.clip(
            np.floor((v - vmin) / (vmax - vmin + 1e-12) * (bins - 1)).astype(int),
            0, bins - 1
        )

    xr = to_idx(xz[:, 0]); zr = to_idx(xz[:, 1])
    pos = (pts[:, 1] >= 0)
    Hpos = np.zeros((bins, bins), int); Hneg = np.zeros((bins, bins), int)
    np.add.at(Hpos, (xr[pos], zr[pos]), 1)
    np.add.at(Hneg, (xr[~pos], zr[~pos]), 1)
    denom = Hpos.sum() + Hneg.sum()
    if denom == 0: return 0.0
    sim = 1.0 - (np.abs(Hpos - Hneg).sum() / denom)
    return float(np.clip(sim, 0.0, 1.0))

def _hemisphere_balance(vertices, center, axes):
    """Fraction of vertices in the negative-Y hemisphere (PCA frame)."""
    R = axes.T
    pts = (vertices - center) @ R
    if len(pts) == 0: return np.nan
    return float((pts[:, 1] < 0).mean())

def _significant_components(mesh: trimesh.Trimesh, frac_threshold=0.01):
    """Count connected components that contribute at least *frac_threshold* of total area."""
    comps = mesh.split(only_watertight=False)
    if len(comps) == 0: return 0, 0.0
    areas = np.array([c.area for c in comps], float)
    total = areas.sum()
    if total <= 0: return len(comps), 0.0
    order = np.argsort(areas)[::-1]
    areas_sorted = areas[order]
    largest_share = float(areas_sorted[0] / total)
    # count comps contributing at least threshold of total area
    sig_count = int((areas_sorted / total >= frac_threshold).sum())
    if sig_count == 0: sig_count = 1
    return sig_count, largest_share

def load_vtk_as_trimesh(path):
    """Load a VTK file and convert it to a ``trimesh.Trimesh``."""
    mesh = pv.read(path)
    mesh_tri = mesh.triangulate()
    vertices = np.array(mesh_tri.points)
    faces = np.array(mesh_tri.faces)  # ensure numpy array
    faces = faces.reshape((-1, 4))[:, 1:4]  # drop first value in each row
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ---------- main ----------

def check_brain(path: str) -> dict:
    """Classify a mesh as adult brain, fetal brain, or not-brain.

    Computes geometry metrics (bounding-box extents, aspect ratios, PCA
    symmetry, hemisphere balance, surface-area / convex-hull ratio,
    connected-component count) and scores the mesh against known adult
    and fetal morphometry bands.  See module docstring for weight rationale.

    Args:
        path: Path to an ``.stl`` or ``.vtk`` mesh file.

    Returns:
        A dict with keys ``label``, ``adult_score``, ``fetal_score``, and
        various per-metric values for downstream reporting.
    """
    # Load mesh
    if path.lower().endswith(".vtk"):
        mesh = load_vtk_as_trimesh(path)
    else:
        mesh = trimesh.load(path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms)

    total_faces = int(mesh.faces.shape[0])
    n_comp_total = len(mesh.split(only_watertight=False))
    sig_comp, largest_share = _significant_components(mesh, frac_threshold=0.01)

    # geometry stats
    bbox = mesh.bounds
    extents = bbox[1] - bbox[0]
    extents_mm = _unit_scale_to_mm(extents)
    L, W, H = np.sort(extents_mm)[::-1]
    r_LW = _safe_ratio(L, W)
    r_WH = _safe_ratio(W, H)

    center, axes = _pca_axes(mesh.vertices)
    sym = _symmetry_score(mesh.vertices, center, axes)
    hemi_frac = _hemisphere_balance(mesh.vertices, center, axes)
    hemi_balance = 1.0 - abs(hemi_frac - 0.5) / 0.5 if _is_finite(hemi_frac) else 0.0
    hemi_balance = float(np.clip(hemi_balance, 0.0, 1.0))

    raw_area = float(mesh.area)
    hull_area = float(mesh.convex_hull.area) if mesh.convex_hull is not None else np.nan
    sa_ch_ratio = _safe_ratio(raw_area, hull_area)

    unit_scale = _safe_ratio(L, np.max(extents))  # mm per model unit
    watertight = bool(mesh.is_watertight)
    vol_mm3 = np.nan
    if watertight and _is_finite(unit_scale):
        vol_mm3 = float(mesh.volume) * (unit_scale ** 3)

    # ---- Scoring bands ----
    # Ranges are derived from published brain morphometry data.
    # L = longest bounding-box extent (mm after unit normalisation).
    adult_L_range = (150, 195)   # adult brain longest axis range (mm)
    fetal_L_range = (60, 130)    # fetal brain longest axis range (mm)
    adult_ratio_LW = (1.10, 1.55)   # length/width aspect ratio — adult
    adult_ratio_WH = (1.15, 1.95)   # width/height aspect ratio — adult
    fetal_ratio_LW = (1.00, 1.70)   # length/width — fetal (rounder)
    fetal_ratio_WH = (1.00, 2.10)   # width/height — fetal
    adult_sa_ch = (1.7, 3.9)        # surface-area / convex-hull — adult (highly folded)
    fetal_sa_ch  = (1.25, 3.0)      # surface-area / convex-hull — fetal (less folded)
    adult_vol_mm3 = (8.0e5, 1.7e6)  # brain volume (mm³) — adult
    fetal_vol_mm3 = (8.0e4, 7.0e5)  # brain volume (mm³) — fetal

    # Early rejection: meshes with many disconnected components whose
    # largest piece is < 90% of total area are unlikely to be a brain.
    early = (sig_comp > 3 and largest_share < 0.9)

    # size
    adult_size = 1.0 if _bool_in_range(L, *adult_L_range) else 0.0
    fetal_size = 1.0 if _bool_in_range(L, *fetal_L_range) else 0.0
    if _is_finite(vol_mm3):
        adult_size = max(adult_size, 1.0 if _bool_in_range(vol_mm3, *adult_vol_mm3) else 0.0)
        fetal_size = max(fetal_size, 1.0 if _bool_in_range(vol_mm3, *fetal_vol_mm3) else 0.0)

    # shape
    shape_adult = 1.0 if (_bool_in_range(r_LW, *adult_ratio_LW) and _bool_in_range(r_WH, *adult_ratio_WH)) else 0.0
    shape_fetal = 1.0 if (_bool_in_range(r_LW, *fetal_ratio_LW) and _bool_in_range(r_WH, *fetal_ratio_WH)) else 0.0

    # symmetry
    sym_score = float(np.clip((sym - 0.55) / 0.45, 0.0, 1.0))
    hemi_score = float(np.clip((hemi_balance - 0.55) / 0.45, 0.0, 1.0))

    # gyrification proxy
    adult_gyr = 1.0 if _bool_in_range(sa_ch_ratio, *adult_sa_ch) else 0.0
    fetal_gyr = 1.0 if _bool_in_range(sa_ch_ratio, *fetal_sa_ch) else 0.0

    # connectivity
    conn_score = 1.0 if sig_comp == 1 else 0.7 if sig_comp == 2 else 0.4 if sig_comp == 3 else 0.0

    # Weighted aggregate — coefficients sum to 1.0 for each age group.
    # See module docstring for weight rationale.
    adult_score = (
        0.26 * adult_size +    # size match
        0.20 * shape_adult +   # aspect-ratio match
        0.18 * sym_score +     # bilateral symmetry
        0.12 * hemi_score +    # hemisphere balance
        0.19 * adult_gyr +     # gyrification proxy
        0.05 * conn_score      # mesh connectivity
    )
    fetal_score = (
        0.36 * fetal_size +    # size match (higher weight — fetal size is key discriminator)
        0.18 * shape_fetal +   # aspect-ratio match
        0.16 * sym_score +     # bilateral symmetry
        0.10 * hemi_score +    # hemisphere balance
        0.15 * fetal_gyr +     # gyrification proxy
        0.05 * conn_score      # mesh connectivity
    )

    label = "not_brain" if early else ("adult_brain" if adult_score >= fetal_score else "fetal_brain")
    if not early and max(adult_score, fetal_score) < 0.55:
        label = "not_brain"

    return {
        "file": path.split("/")[-1],
        "label": label,
        "adult_score": round(float(adult_score), 3),
        "fetal_score": round(float(fetal_score), 3),
        "sig_components": int(sig_comp),
        "largest_component_area_share": round(largest_share, 3),
        "L_mm": float(L), "W_mm": float(W), "H_mm": float(H),
        "r_LW": r_LW, "r_WH": r_WH,
        "sym_0to1": sym, "hemi_balance_0to1": hemi_balance,
        "sa_over_convexhull": sa_ch_ratio,
        "watertight": bool(watertight),
        "volume_cm3": None if not _is_finite(vol_mm3) else float(vol_mm3)/1000.0,
        "components_total": int(n_comp_total),
        "faces": total_faces,
        "vertices": int(mesh.vertices.shape[0]),
        "early_reject": "not_brain" if early else None
    }
