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
    try:
        return np.isfinite(x)
    except Exception:
        return False

def _is_finite(x):
    return x is not None and _np_isfinite(x)

def _bool_in_range(x, lo, hi):
    return _is_finite(x) and lo <= x <= hi

def _safe_ratio(a, b):
    if b == 0 or not _np_isfinite(a) or not _np_isfinite(b):
        return np.nan
    return float(a / b)

def _pca_axes(points: np.ndarray):
    c = points.mean(axis=0)
    X = points - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return c, Vt  # axes rows

def _unit_scale_to_mm(extents):
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
    R = axes.T
    pts = (vertices - center) @ R
    if len(pts) == 0: return np.nan
    return float((pts[:, 1] < 0).mean())

def _significant_components(mesh: trimesh.Trimesh, frac_threshold=0.01):
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
    mesh = pv.read(path)
    mesh_tri = mesh.triangulate()
    vertices = np.array(mesh_tri.points)
    faces = mesh_tri.faces.reshape((-1, 4))[:, 1:4]  # drop count
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ---------- main ----------

def check_brain(path: str) -> dict:
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

    # bands
    adult_L_range = (150, 195)
    fetal_L_range = (60, 130)
    adult_ratio_LW = (1.10, 1.55)
    adult_ratio_WH = (1.15, 1.95)
    fetal_ratio_LW = (1.00, 1.70)
    fetal_ratio_WH = (1.00, 2.10)
    adult_sa_ch = (1.7, 3.9)
    fetal_sa_ch  = (1.25, 3.0)
    adult_vol_mm3 = (8.0e5, 1.7e6)
    fetal_vol_mm3 = (8.0e4, 7.0e5)

    # early non-brain
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

    # aggregate
    adult_score = (
        0.26 * adult_size +
        0.20 * shape_adult +
        0.18 * sym_score +
        0.12 * hemi_score +
        0.19 * adult_gyr +
        0.05 * conn_score
    )
    fetal_score = (
        0.36 * fetal_size +
        0.18 * shape_fetal +
        0.16 * sym_score +
        0.10 * hemi_score +
        0.15 * fetal_gyr +
        0.05 * conn_score
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
