"""Surface-connected cavity correction for 3-D volume and surface area.

A sliced 3-D geometry is integrated slice-by-slice (volume = ``∫ A dh``,
surface = Simpson lateral + caps). That treats every cross-section as solid and
only integrates the *outer* contour, so a hole/cavity is mishandled:

- **Volume** counts an open void as tissue (the cross-section area includes it).
- **Surface area** omits the **cavity wall** (a hole's inner perimeter is never
  added to the lateral surface).

This module decides, per slice hole, whether the cavity is **surface-connected**
(opens onto the outside → correct it: subtract its area from the volume integral,
add its wall perimeter to the surface) or **fully enclosed** (sealed internal
void → leave it as solid). Surface-connectivity is a 3-D property, recovered two
ways:

- **STL / VTK** (screenshot-based slices, per-slice scale): track cavities across
  consecutive slices in physical-mm coordinates and classify each track by its
  ends (sealed into tissue vs. open to exterior). See :func:`cavity_correction_tracking`.
- **NIfTI** (native 3-D voxel mask): exact ``scipy.ndimage.binary_fill_holes``.
  See :func:`cavity_correction_nifti`.

GI/LGI must NOT change: the cavity-wall perimeter goes only into the 3-D surface
lateral, never into the GI inner/outer perimeter sums.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SliceCavity:
    """One hole detected in one slice, in physical (mm) units."""

    area_mm2: float
    perim_mm: float
    centroid_mm: tuple[float, float]
    bbox_mm: tuple[float, float, float, float]  # (x, y, w, h)
    # Original cv2 contour in pixel space (for drawing the annotated overlay).
    contour_px: np.ndarray | None = None


@dataclass
class SliceRecord:
    """Per-slice topology for the STL/VTK cavity tracker (physical mm)."""

    idx: int
    position_mm: float
    cavities: list[SliceCavity] = field(default_factory=list)
    # Outer tissue contours and hole contours in physical mm, shaped (N,1,2)
    # float32 for cv2.pointPolygonTest (used by the end-of-track sealed/open test).
    outer_polys_mm: list[np.ndarray] = field(default_factory=list)
    hole_polys_mm: list[np.ndarray] = field(default_factory=list)


@dataclass
class CavityCorrection:
    """Per-slice corrections and summary stats (all values in mm / mm²)."""

    area_subtract_by_idx: dict[int, float] = field(default_factory=dict)
    perim_add_by_idx: dict[int, float] = field(default_factory=dict)
    # slice idx -> list of pixel-space contours of its surface-connected cavities
    # (for drawing the annotated overlay).
    surface_connected_by_idx: dict[int, list] = field(default_factory=dict)
    n_surface_connected: int = 0
    n_enclosed: int = 0
    total_cavity_area_mm2: float = 0.0
    total_wall_perim_mm: float = 0.0

    @classmethod
    def empty(cls) -> "CavityCorrection":
        return cls()

    def area_subtract(self, idx: int) -> float:
        return float(self.area_subtract_by_idx.get(idx, 0.0))

    def perim_add(self, idx: int) -> float:
        return float(self.perim_add_by_idx.get(idx, 0.0))


# ---------------------------------------------------------------------------
# Union-find for grouping cavity tracks (handles split/merge branches)
# ---------------------------------------------------------------------------
class _DisjointSet:
    def __init__(self):
        self._parent: dict = {}

    def find(self, x):
        p = self._parent.setdefault(x, x)
        while p != x:
            self._parent[x] = self._parent.setdefault(p, p)
            x, p = p, self._parent[p]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def groups(self) -> dict:
        out: dict = {}
        for node in list(self._parent):
            out.setdefault(self.find(node), []).append(node)
        return out


# ---------------------------------------------------------------------------
# Geometry helpers (pure, unit-testable)
# ---------------------------------------------------------------------------
def _equiv_radius(area_mm2: float) -> float:
    return math.sqrt(max(area_mm2, 0.0) / math.pi)


def _bbox_overlap(a: tuple, b: tuple) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def _cavities_match(a: SliceCavity, b: SliceCavity, *, max_area_ratio: float = 6.0) -> bool:
    """Two cavities on neighbouring slices are the same column if they overlap
    or sit close (physical mm), with a bounded area ratio."""
    lo = max(min(a.area_mm2, b.area_mm2), 1e-9)
    if max(a.area_mm2, b.area_mm2) / lo > max_area_ratio:
        return False
    if _bbox_overlap(a.bbox_mm, b.bbox_mm):
        return True
    dx = a.centroid_mm[0] - b.centroid_mm[0]
    dy = a.centroid_mm[1] - b.centroid_mm[1]
    dist = math.hypot(dx, dy)
    return dist <= (_equiv_radius(a.area_mm2) + _equiv_radius(b.area_mm2))


def _point_in_solid_tissue(point_mm: tuple[float, float], record: SliceRecord) -> bool:
    """True iff *point_mm* lies inside an outer tissue contour and NOT inside any
    hole of *record* — i.e. solid tissue (the cavity sealed off at this slice)."""
    pt = (float(point_mm[0]), float(point_mm[1]))
    inside_outer = any(
        cv2.pointPolygonTest(poly, pt, False) >= 0 for poly in record.outer_polys_mm
    )
    if not inside_outer:
        return False
    inside_hole = any(
        cv2.pointPolygonTest(poly, pt, False) >= 0 for poly in record.hole_polys_mm
    )
    return not inside_hole


def _point_is_exterior(point_mm: tuple[float, float], record: SliceRecord) -> bool:
    """True iff *point_mm* is outside every outer tissue contour of *record*
    (exterior background) — the cavity opened to the outside at this slice."""
    pt = (float(point_mm[0]), float(point_mm[1]))
    return all(
        cv2.pointPolygonTest(poly, pt, False) < 0 for poly in record.outer_polys_mm
    )


# ---------------------------------------------------------------------------
# STL / VTK — track cavities across slices and classify
# ---------------------------------------------------------------------------
def cavity_correction_tracking(
    records: list[SliceRecord],
    *,
    area_threshold_mm2: float,
    link_window: int = 2,
) -> CavityCorrection:
    """Classify cavities for the screenshot-based (STL/VTK) workflows.

    Cavities are linked across ``±link_window`` slices in physical-mm space and
    grouped into tracks. A track is **surface-connected** if it reaches the first
    or last valid slice, or if at either end the cavity's footprint lies in
    exterior background on the adjacent slice; otherwise (both ends seal into
    tissue) it is a **fully-enclosed** void and left untouched.

    Only cavities with ``area_mm2 > area_threshold_mm2`` participate.
    """
    out = CavityCorrection.empty()
    if not records:
        return out

    records = sorted(records, key=lambda r: r.position_mm)
    by_pos_index = {r.idx: i for i, r in enumerate(records)}  # idx -> sorted position

    # node = (slice_idx, cavity_index); only above-threshold cavities are tracked.
    nodes: list[tuple[int, int]] = []
    cav_of: dict[tuple[int, int], SliceCavity] = {}
    for r in records:
        for ci, cav in enumerate(r.cavities):
            if cav.area_mm2 > area_threshold_mm2:
                node = (r.idx, ci)
                nodes.append(node)
                cav_of[node] = cav
    if not nodes:
        return out

    ds = _DisjointSet()
    for n in nodes:
        ds.find(n)

    # Link across the position window.
    for pi in range(len(records)):
        r = records[pi]
        for pj in range(pi + 1, min(pi + link_window + 1, len(records))):
            r2 = records[pj]
            for ci, ca in enumerate(r.cavities):
                if ca.area_mm2 <= area_threshold_mm2:
                    continue
                for cj, cb in enumerate(r2.cavities):
                    if cb.area_mm2 <= area_threshold_mm2:
                        continue
                    if _cavities_match(ca, cb):
                        ds.union((r.idx, ci), (r2.idx, cj))

    # Classify each track group by its end behaviour.
    surface_nodes: set[tuple[int, int]] = set()
    for root, members in ds.groups().items():
        members = [m for m in members if m in cav_of]
        if not members:
            continue
        members.sort(key=lambda m: by_pos_index[m[0]])
        first_node, last_node = members[0], members[-1]
        first_pos, last_pos = by_pos_index[first_node[0]], by_pos_index[last_node[0]]

        open_end = False
        # Reaching the field-of-view boundary counts as open (FOV-cut, not sealed).
        if first_pos == 0 or last_pos == len(records) - 1:
            open_end = True
        else:
            lo_neighbor = records[first_pos - 1]
            hi_neighbor = records[last_pos + 1]
            if _point_is_exterior(cav_of[first_node].centroid_mm, lo_neighbor):
                open_end = True
            elif _point_is_exterior(cav_of[last_node].centroid_mm, hi_neighbor):
                open_end = True

        if open_end:
            out.n_surface_connected += 1
            surface_nodes.update(members)
        else:
            out.n_enclosed += 1

    # Aggregate per-slice corrections for surface-connected cavities.
    for (idx, ci) in surface_nodes:
        cav = cav_of[(idx, ci)]
        out.area_subtract_by_idx[idx] = out.area_subtract_by_idx.get(idx, 0.0) + cav.area_mm2
        out.perim_add_by_idx[idx] = out.perim_add_by_idx.get(idx, 0.0) + cav.perim_mm
        out.total_cavity_area_mm2 += cav.area_mm2
        out.total_wall_perim_mm += cav.perim_mm
        if cav.contour_px is not None:
            out.surface_connected_by_idx.setdefault(idx, []).append(cav.contour_px)
    return out


# ---------------------------------------------------------------------------
# NIfTI — exact 3-D classification via binary_fill_holes
# ---------------------------------------------------------------------------
def cavity_correction_nifti(
    mask_3d: np.ndarray,
    slice_indices,
    *,
    axis: int,
    pixel_size_x: float,
    pixel_size_z: float,
    area_threshold_mm2: float,
):
    """Classify cavities for NIfTI using the exact 3-D voxel mask.

    ``binary_fill_holes`` fills only voids NOT connected to the volume border —
    i.e. the fully-enclosed voids. A slice hole overlapping the filled region is
    enclosed; otherwise it is surface-connected (a cross-section of an open
    cavity). ``filled`` doubles as the net "tissue + enclosed" mask the caller
    uses for the corrected cross-section area.

    Args:
        mask_3d: 3-D binary tissue mask (any nonzero = tissue).
        slice_indices: the slice indices (along ``axis``) that were measured.
        axis: slicing axis (0/1/2); slices are taken as ``mask[:, idx, :]`` etc.
        pixel_size_x, pixel_size_z: in-plane voxel sizes (mm) of a slice.
        area_threshold_mm2: minimum cavity area to count.

    Returns:
        ``(correction, filled_3d)`` — a :class:`CavityCorrection` carrying only
        ``perim_add_by_idx`` + stats (area is taken from ``filled_3d`` by the
        caller), and the filled 3-D mask.
    """
    from scipy.ndimage import binary_fill_holes

    out = CavityCorrection.empty()
    mask = np.asarray(mask_3d) > 0
    filled = binary_fill_holes(mask)
    enclosed = filled & ~mask  # fully-enclosed voids
    pixel_area = float(pixel_size_x) * float(pixel_size_z)

    def _slice(vol, idx):
        if axis == 0:
            return vol[idx, :, :]
        if axis == 1:
            return vol[:, idx, :]
        return vol[:, :, idx]

    for idx in slice_indices:
        mask_2d = _slice(mask, idx).astype(np.uint8)
        enclosed_2d = _slice(enclosed, idx)
        contours, hierarchy = cv2.findContours(
            mask_2d, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        hier = hierarchy[0]
        for ci, cnt in enumerate(contours):
            if hier[ci][3] == -1:  # outer contour, not a hole
                continue
            area_mm2 = float(cv2.contourArea(cnt)) * pixel_area
            if area_mm2 <= area_threshold_mm2:
                continue
            # Is this hole an enclosed void or a surface-connected cavity?
            # Restrict to background pixels inside the contour (drawContours fills
            # a boundary ring of tissue that would otherwise skew the fraction).
            hole_mask = np.zeros_like(mask_2d)
            cv2.drawContours(hole_mask, [cnt], -1, 1, thickness=cv2.FILLED)
            hole_bg = (hole_mask > 0) & (mask_2d == 0)
            hole_px = int(np.count_nonzero(hole_bg))
            if hole_px > 0:
                overlap = int(np.count_nonzero(hole_bg & enclosed_2d))
                is_enclosed = overlap > 0.5 * hole_px
            else:
                # Degenerate tiny contour the raster missed — sample its centroid.
                m = cv2.moments(cnt)
                if m["m00"] != 0:
                    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                else:
                    cx, cy = int(cnt[0][0][0]), int(cnt[0][0][1])
                cy = min(max(cy, 0), enclosed_2d.shape[0] - 1)
                cx = min(max(cx, 0), enclosed_2d.shape[1] - 1)
                is_enclosed = bool(enclosed_2d[cy, cx])
            if is_enclosed:
                out.n_enclosed += 1
                continue
            # Surface-connected: add the wall perimeter (scaled like inner_perim).
            cnt_mm = np.ascontiguousarray(
                cnt.astype(np.float32).reshape(-1, 1, 2)
                * np.array([pixel_size_x, pixel_size_z], dtype=np.float32))
            perim_mm = float(cv2.arcLength(cnt_mm, True))
            out.n_surface_connected += 1
            out.perim_add_by_idx[idx] = out.perim_add_by_idx.get(idx, 0.0) + perim_mm
            out.total_cavity_area_mm2 += area_mm2
            out.total_wall_perim_mm += perim_mm
            out.surface_connected_by_idx.setdefault(idx, []).append(np.asarray(cnt))

    return out, filled


# ---------------------------------------------------------------------------
# Contour → physical-mm helpers for building SliceRecords (STL/VTK callers)
# ---------------------------------------------------------------------------
def contour_to_mm(contour, mm_per_px: float, center_px) -> np.ndarray:
    """Convert a cv2 contour (px) to physical mm coords relative to *center_px*.

    Returns an ``(N,1,2)`` float32 array suitable for cv2.pointPolygonTest. The
    cube centre is a consistent in-plane physical origin across slices, so the
    result is comparable slice-to-slice."""
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    cx, cy = float(center_px[0]), float(center_px[1])
    pts = (pts - np.array([cx, cy], dtype=np.float32)) * np.float32(mm_per_px)
    return np.ascontiguousarray(pts.reshape(-1, 1, 2), dtype=np.float32)


def make_slice_cavity(contour, mm_per_px: float, center_px) -> SliceCavity:
    """Build a :class:`SliceCavity` (physical mm) from a hole contour (px)."""
    area_mm2 = float(cv2.contourArea(contour)) * (mm_per_px ** 2)
    perim_mm = float(cv2.arcLength(contour, True)) * float(mm_per_px)
    poly_mm = contour_to_mm(contour, mm_per_px, center_px).reshape(-1, 2)
    x = float(poly_mm[:, 0].min())
    y = float(poly_mm[:, 1].min())
    w = float(poly_mm[:, 0].max() - x)
    h = float(poly_mm[:, 1].max() - y)
    cxm = float(poly_mm[:, 0].mean())
    cym = float(poly_mm[:, 1].mean())
    return SliceCavity(area_mm2=area_mm2, perim_mm=perim_mm,
                       centroid_mm=(cxm, cym), bbox_mm=(x, y, w, h),
                       contour_px=np.asarray(contour))
