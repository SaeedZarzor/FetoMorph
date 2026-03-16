from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import nibabel as nib

try:
    from nibabel.freesurfer.io import read_geometry as fs_read_geometry
except Exception:
    fs_read_geometry = None


# -------------------------
# Helpers: pial path + mapping
# -------------------------

def _autodetect_pial_paths(file_path: str, cfg) -> Tuple[Optional[str], Optional[str]]:
    base = os.path.dirname(os.path.abspath(file_path))
    lh = getattr(cfg, "pial_lh_path", None) or os.path.join(base, "lh.pial")
    rh = getattr(cfg, "pial_rh_path", None) or os.path.join(base, "rh.pial")
    lh = lh if os.path.isfile(lh) else None
    rh = rh if os.path.isfile(rh) else None
    return lh, rh


def _tkr_default_vox2ras() -> np.ndarray:
    # FreeSurfer default tkregister vox2ras (for 256³ conformed volumes).
    return np.array([
        [-1.0,  0.0,  0.0, 128.0],
        [ 0.0,  0.0,  1.0,-128.0],
        [ 0.0, -1.0,  0.0, 128.0],
        [ 0.0,  0.0,  0.0,   1.0],
    ], dtype=np.float64)


def _map_to_vox_scanner(verts_mm: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map RAS/world(mm) -> voxel using NIfTI affine."""
    ones = np.ones((verts_mm.shape[0], 1), dtype=np.float64)
    xyz1 = np.hstack([verts_mm.astype(np.float64), ones])
    ijk1 = (xyz1 @ np.linalg.inv(affine).T)
    return ijk1[:, :3]


def _map_to_vox_tkr_heuristic(verts_mm: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Heuristic mapping for tkregister RAS (tkr) -> voxel, then scale to target volume shape.
    Use only if scanner mapping fails (mostly outside volume).
    """
    vox2ras_tkr = _tkr_default_vox2ras()
    ras2vox_tkr = np.linalg.inv(vox2ras_tkr)

    ones = np.ones((verts_mm.shape[0], 1), dtype=np.float64)
    ras1 = np.hstack([verts_mm.astype(np.float64), ones])
    ijk_tkr = (ras1 @ ras2vox_tkr.T)[:, :3]  # in 0..255 style indices (roughly)

    nx, ny, nz = shape
    scale = np.array([
        (max(nx - 1, 1)) / 255.0,
        (max(ny - 1, 1)) / 255.0,
        (max(nz - 1, 1)) / 255.0
    ], dtype=np.float64)
    return ijk_tkr * scale


def _fraction_inside(arr: Optional[np.ndarray], shape: Tuple[int, int, int]) -> float:
    if arr is None or arr.size == 0:
        return 0.0
    nx, ny, nz = shape
    x, y, z = arr.T
    inside = (
        (x >= 0) & (x < nx - 1) &
        (y >= 0) & (y < ny - 1) &
        (z >= 0) & (z < nz - 1)
    )
    return float(inside.mean()) if inside.size else 0.0


def _find_ref_volume_for_pial(seg_path: str, seg_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """
    Look for a nearby T2w (preferred) or T1w volume and use its affine for pial mapping,
    but only if its grid matches seg_shape (canonical).
    """
    base = os.path.dirname(os.path.abspath(seg_path))
    candidates = []

    # Prefer T2w
    for name in os.listdir(base):
        lower = name.lower()
        if lower.endswith("_t2w.nii") or lower.endswith("_t2w.nii.gz"):
            candidates.append(os.path.join(base, name))

    # Else try T1w
    if not candidates:
        for name in os.listdir(base):
            lower = name.lower()
            if lower.endswith("_t1w.nii") or lower.endswith("_t1w.nii.gz"):
                candidates.append(os.path.join(base, name))

    for cand in candidates:
        try:
            ref = nib.as_closest_canonical(nib.load(cand))
            if ref.shape[:3] == seg_shape:
                return ref.affine
        except Exception:
            continue

    return None


# -------------------------
# Public: load pial mesh in voxel space
# -------------------------

def load_pial_vertices_to_vox(
    cfg,
    affine: np.ndarray,
    shape: Tuple[int, int, int],
    file_path: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Load FreeSurfer pial surfaces (lh/rh) and map vertices into *voxel coordinates*.

    Returns:
      {"lh": {"verts": (N,3) float64, "faces": (M,3) int32} or None,
       "rh": {"verts": (N,3) float64, "faces": (M,3) int32} or None}
    """
    out: Dict[str, Optional[Dict[str, Any]]] = {"lh": None, "rh": None}
    if fs_read_geometry is None:
        return out

    lh_path, rh_path = _autodetect_pial_paths(file_path, cfg)

    # Prefer a nearby T2/T1 affine if it matches this grid (mirrors your alignment check logic)
    ref_affine = affine
    ref_from_t2 = _find_ref_volume_for_pial(file_path, shape)
    if ref_from_t2 is not None:
        ref_affine = ref_from_t2

    def _map_with_fallback(verts_mm: np.ndarray) -> Optional[np.ndarray]:
        try:
            pial_space = str(getattr(cfg, "pial_space", "scanner")).lower()
            primary_tkr = pial_space.startswith("tkr")

            mapped_primary = (
                _map_to_vox_tkr_heuristic(verts_mm, shape)
                if primary_tkr else
                _map_to_vox_scanner(verts_mm, ref_affine)
            )

            if _fraction_inside(mapped_primary, shape) > 0.02:
                return mapped_primary

            # fallback to alternate
            mapped_alt = (
                _map_to_vox_scanner(verts_mm, ref_affine)
                if primary_tkr else
                _map_to_vox_tkr_heuristic(verts_mm, shape)
            )
            if _fraction_inside(mapped_alt, shape) > 0.02:
                print("[pial_overlay] Fallback to alternate pial_space for", file_path)
                return mapped_alt

            return mapped_primary
        except Exception:
            return None

    try:
        if lh_path:
            v_lh, f_lh = fs_read_geometry(lh_path)
            v_lh_vox = _map_with_fallback(v_lh)
            if v_lh_vox is not None:
                out["lh"] = {"verts": v_lh_vox.astype(np.float64, copy=False),
                             "faces": f_lh.astype(np.int32, copy=False)}

        if rh_path:
            v_rh, f_rh = fs_read_geometry(rh_path)
            v_rh_vox = _map_with_fallback(v_rh)
            if v_rh_vox is not None:
                out["rh"] = {"verts": v_rh_vox.astype(np.float64, copy=False),
                             "faces": f_rh.astype(np.int32, copy=False)}
    except Exception:
        return {"lh": None, "rh": None}

    return out


# -------------------------
# Triangle-plane intersection + projection
# -------------------------

def _project_vox_to_rc(pts_vox: np.ndarray, slice_axis: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D voxel coords -> (row, col) indices for the 2D slice image.
    IMPORTANT: This matches your existing orientation convention.
    """
    if slice_axis == 2:
        rr = pts_vox[:, 0]  # x -> rows
        cc = pts_vox[:, 1]  # y -> cols
    elif slice_axis == 1:
        rr = pts_vox[:, 0]  # x -> rows
        cc = pts_vox[:, 2]  # z -> cols
    else:
        rr = pts_vox[:, 1]  # y -> rows
        cc = pts_vox[:, 2]  # z -> cols
    return rr, cc


def _triangle_plane_segments(
    verts_vox: np.ndarray,
    faces: np.ndarray,
    slice_axis: int,
    slice_idx: float,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute intersection line segments between mesh triangles and plane:
      coord[slice_axis] == slice_idx  (in voxel space)

    Returns: (K, 2, 3) float64 segments.
    """
    if verts_vox is None or faces is None or verts_vox.size == 0 or faces.size == 0:
        return np.zeros((0, 2, 3), dtype=np.float64)

    tri = verts_vox[faces]  # (M,3,3)
    d = tri[:, :, slice_axis] - float(slice_idx)  # (M,3)

    # Candidate triangles: must have both signs OR touch plane
    pos = d > eps
    neg = d < -eps
    has_pos = pos.any(axis=1)
    has_neg = neg.any(axis=1)
    touch = (np.abs(d) <= eps).any(axis=1)

    use = (has_pos & has_neg) | touch
    if not np.any(use):
        return np.zeros((0, 2, 3), dtype=np.float64)

    tri = tri[use]
    d = d[use]

    segs = []

    for t, dv in zip(tri, d):
        pts = []

        def _add(p):
            # avoid duplicates (vertex-on-plane + edge intersection can coincide)
            for q in pts:
                if np.linalg.norm(p - q) < 1e-5:
                    return
            pts.append(p)

        # add vertices on plane
        for i in range(3):
            if abs(dv[i]) <= eps:
                _add(t[i])

        # edge crossings
        edges = ((0, 1), (1, 2), (2, 0))
        for i, j in edges:
            di, dj = dv[i], dv[j]
            if (di > eps and dj < -eps) or (di < -eps and dj > eps):
                # di + a*(dj-di) = 0  => a = di/(di-dj)
                a = di / (di - dj)
                p = t[i] + a * (t[j] - t[i])
                _add(p)

        # Typical case: exactly 2 points => segment
        if len(pts) == 2:
            segs.append([pts[0], pts[1]])
        # Degenerate (triangle lies on plane) or weird numeric cases are ignored.

    if not segs:
        return np.zeros((0, 2, 3), dtype=np.float64)

    return np.asarray(segs, dtype=np.float64)


# -------------------------
# Public: draw pial on slice (triangle-plane intersection)
# -------------------------

def draw_pial_on_slice(
    annotated: np.ndarray,
    slice_axis: int,
    slice_idx: int,
    pial_vox: Dict[str, Optional[Dict[str, Any]]],
    spacing: Tuple[float, float, float],
    cfg,
) -> np.ndarray:
    """
    Draw pial overlay using triangle-plane intersection.
    Produces sharp, continuous outlines.
    """
    H, W = annotated.shape[:2]

    # Use a small numeric eps in voxel units to avoid off-plane artifacts.
    tol_mm = float(getattr(cfg, "pial_tolerance_mm", 0.0) or 0.0)
    tol_vox = tol_mm / float(spacing[slice_axis])
    eps = float(min(max(1e-6, 0.02 * max(tol_vox, 1e-6)), 1e-2))  # safe cap

    # Line thickness control (optional).
    thickness = int(getattr(cfg, "pial_line_thickness", 1) or 1)
    thickness = max(1, thickness)

    target = annotated

    def _draw_mesh(mesh: Optional[Dict[str, Any]], color: Tuple[int, int, int]):
        if not mesh:
            return

        verts = mesh.get("verts", None)
        faces = mesh.get("faces", None)
        if verts is None or faces is None:
            return

        segs = _triangle_plane_segments(verts, faces, slice_axis, float(slice_idx), eps=eps)
        if segs.size == 0:
            return

        p0 = segs[:, 0, :]
        p1 = segs[:, 1, :]

        r0, c0 = _project_vox_to_rc(p0, slice_axis)
        r1, c1 = _project_vox_to_rc(p1, slice_axis)

        r0i = np.rint(r0).astype(int)
        c0i = np.rint(c0).astype(int)
        r1i = np.rint(r1).astype(int)
        c1i = np.rint(c1).astype(int)

        Ht, Wt = target.shape[:2]
        thickness_draw = thickness
        color_draw = tuple(int(x) for x in color)
        mask = np.zeros((Ht, Wt), dtype=np.uint8)

        for y0, x0, y1, x1 in zip(r0i, c0i, r1i, c1i):
            # quick reject if segment is completely outside same side
            if (x0 < 0 and x1 < 0) or (x0 >= Wt and x1 >= Wt) or (y0 < 0 and y1 < 0) or (y0 >= Ht and y1 >= Ht):
                continue
            cv2.line(
                mask,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                255,
                thickness=thickness_draw,
                lineType=cv2.LINE_AA,
            )

        target[mask > 0] = color_draw

    _draw_mesh(pial_vox.get("lh"), tuple(getattr(cfg, "pial_lh_color_bgr", (0, 255, 0))))
    _draw_mesh(pial_vox.get("rh"), tuple(getattr(cfg, "pial_rh_color_bgr", (0, 0, 255))))

    return annotated
