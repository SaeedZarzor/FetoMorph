"""
STL Area Band Sampler
---------------------

Sample n 2D slices from a triangulated surface (STL/VTK/PLY/OBJ) using the same
maximum-area rule the NIfTI pipeline uses, for meshes that carry no labels.

Why this is not just "slice the mesh and take the biggest cross-section"
-----------------------------------------------------------------------
Cutting a surface mesh yields a set of open polylines wherever the mesh is not
watertight, and brain surfaces rarely are. Closing each polyline independently
produces overlapping polygons whose summed area can exceed the section's own
bounding box, so the naive maximum lands on an artefact rather than on the
widest slice. The fix is to recover a solid before measuring anything:

    surface-voxelise -> 3D binary closing (seal the holes)
      -> flood-fill the background inward from the grid border
      -> interior is whatever the outside could not reach

The resulting boolean volume is written as a NIfTI carrying the mesh's own
index-to-millimetre transform as its affine, and handed to the existing
``NiftiAreaSampler`` unchanged. Everything downstream of the maximum-area rule
- golden-section search, top-p band bounds, per-slice PNGs, the scale bar and
its sidecar, the CSV/Excel/JSON logs - is therefore shared with the NIfTI path
and stays consistent with it.

Labels
------
A surface has none. ``area_labels``/``valid_labels`` are meaningless here: the
mesh *is* the mask, and the volume is written as a single label (1). Whatever
region selection you need has to happen upstream, when the surface is produced.

Axes
----
Axis names are the mesh's own x/y/z. No anatomical meaning is claimed or
inferred - a surface carries no orientation convention we can trust.

Pictures
--------
The outlines are the deliverable. The solid is for *measuring*, not for looking
at: filling a section closes the interhemispheric fissure and every sulcus
narrower than the voxel pitch, so the filled slice reads as a blob however fine
the pitch gets (318 / 336 / 378 mm of boundary at pitch 0.5 / 0.25 / 0.1 against
the mesh's own 400.6 mm). ``render_section_outlines`` therefore draws each slice
straight from the mesh cross-section into ``brain_slices_outline/``, at full
detail. The sampler's filled PNGs are off by default; ``save_png`` brings them
back for the one case that needs them, ``crop_band_roi``, which locates a
subject by ``axis_<a>/brain_slices``.

Perimeter and LGI
-----------------
``perimeter_mesh_mm`` is the exact cross-section polyline length and runs ~+15%
above the raster's ``perimeter_mm`` on average, ranging -4% to +42% between
slices - not a constant offset that cancels in a trend. It also counts every
boundary in the section, internal ones included, where ``perimeter_mm`` counts
only the outermost; that difference is documented in ``section_perimeter_mm``
rather than guessed at, because it is not reliably decidable from the section.

So treat both LGI columns as indicative only. ``lgi`` at least keeps numerator
and denominator on one measurement basis; the denominator they share wobbles
+/-15% between adjacent slices regardless (see the run guide).

Typical usage
>>> from functions.stl_area_sampler import StlAreaBandConfig, sample_stl_band
>>> cfg = StlAreaBandConfig(stl_path="29.stl", out_dir="out", axis="z", n=20, p=0.9)
>>> result = sample_stl_band(cfg)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore[import]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "OpenCV (cv2) is required but could not be imported. "
        "Please install 'opencv-python-headless' in your environment."
    )

try:
    import trimesh  # type: ignore[import]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "trimesh is required but could not be imported. "
        "Please install 'trimesh' in your environment."
    )

try:
    from scipy.ndimage import binary_closing, label as cc_label  # type: ignore[import]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "SciPy is required but could not be imported. "
        "Please install 'scipy' in your environment."
    )

import nibabel as nib

from functions.nifti_area_sampler import AreaBandConfig, NiftiAreaSampler
from helpers.check_mesh import check_brain


MESH_SUFFIXES = (".stl", ".vtk", ".ply", ".obj", ".off")

# The morphological closing that defines the outer perimeter is specified in
# millimetres here and converted to pixels per dataset, because AreaBandConfig
# takes it in pixels and a fixed pixel count means a different physical
# operation at every voxel pitch. At 0.25 mm/vox the AreaBandConfig default of 5
# is a 1.25 mm closing, which finds no sulci at all and reports LGI ~= 1.00.
DEFAULT_KERNEL_MM = 5.0


@dataclass
class StlAreaBandConfig:
    """Configuration for STL area band sampling.

    - stl_path: Path to the input surface mesh.
    - out_dir: Directory where logs and PNGs are written.
    - axis: One of 'x'|'y'|'z' or 0|1|2, in the mesh's own frame.
    - n: Number of slices to sample within the top-p band.
    - p: Fraction of the max area (0..1) to define the band.
    - pitch_mm: Voxel pitch of the intermediate solid, in mm.
    - seal: Radius (in voxels) of the 3D closing that seals mesh holes. The
      kernel is (2*seal+1)^3. Keep this as small as will still seal: it is the
      only step that erodes real sulcal detail.
    - kernel_mm: Morphological closing for the outer perimeter, in mm.
    - min_contour_area: Minimum 2D contour area (in pixels) to keep.
    - save_png / profile_plot / show_crosshair / draw_contours: passed through.
    - keep_nifti: Keep the intermediate volume. When False it is written to a
      temporary directory and deleted once sampling finishes.
    - nifti_dir: Where to keep it when keep_nifti is set. Defaults to out_dir.
    - scale_to_mm: Rescale the mesh to millimetres when its extents indicate it
      was authored in cm or m.
    - require_brain: Abort when check_brain() classifies the mesh as not_brain.
    """
    stl_path: str
    out_dir: str
    axis: str | int = "z"
    n: int = 20
    p: float = 0.9
    pitch_mm: float = 0.25
    seal: int = 1
    kernel_mm: float = DEFAULT_KERNEL_MM
    min_contour_area: float = 30.0
    # Off by default: the sampler's filled silhouettes are a by-product, not
    # the deliverable. Filling a section closes the fissure and every narrow
    # sulcus, so those PNGs read as a blob; the outlines below are the picture.
    # Turn back on with --fill-png if crop_band_roi needs a brain_slices/ dir.
    save_png: bool = False
    profile_plot: bool = True
    show_crosshair: bool = False
    draw_contours: bool = False
    draw_section_outline: bool = True
    outline_pitch_mm: Optional[float] = None
    outline_line_px: int = 1
    # Set to "brain_slices" (with save_png False) to make the outlines the set
    # crop_band_roi sees: it locates a subject by axis_<a>/brain_slices and
    # reads only from there, so outlines are otherwise beyond its reach.
    outline_dirname: str = "brain_slices_outline"
    keep_nifti: bool = False
    nifti_dir: Optional[str] = None
    scale_to_mm: bool = True
    require_brain: bool = False


# --------------------------------------------------------------------------
# Mesh loading and validation
# --------------------------------------------------------------------------

def probe_mesh(path: str) -> Dict[str, Any]:
    """Describe a mesh before committing to it.

    Wraps ``check_brain`` and adds the unit scale it implies. That scale is the
    reason this runs at all: a surface carries no units, so a mesh authored in
    centimetres would otherwise silently produce areas 100x too small.
    """
    info = dict(check_brain(path))
    mesh = _load_raw_mesh(path)
    raw_extent = float(np.max(mesh.extents)) if mesh.extents.size else 0.0
    # check_brain reports L_mm already converted, so their ratio is the scale.
    info["unit_scale_to_mm"] = (
        float(info["L_mm"]) / raw_extent if raw_extent > 0 else 1.0
    )
    info["raw_extents"] = [float(x) for x in mesh.extents]
    return info


def _load_raw_mesh(path: str) -> trimesh.Trimesh:
    """Load a surface as a single Trimesh, in whatever units it was authored."""
    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        geoms = [g for g in getattr(mesh, "geometry", {}).values()
                 if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No triangulated geometry found in {path}")
        mesh = trimesh.util.concatenate(geoms)
    return mesh


def load_mesh(path: str, *, scale_to_mm: bool = True,
              unit_scale: Optional[float] = None) -> trimesh.Trimesh:
    """Load a surface as a single Trimesh, optionally rescaled to millimetres.

    Pass ``unit_scale`` from an earlier ``probe_mesh`` result to avoid probing
    again: the probe runs ``check_brain``, which builds a convex hull, so it is
    much more expensive than the load itself.
    """
    mesh = _load_raw_mesh(path)
    if scale_to_mm:
        scale = probe_mesh(path)["unit_scale_to_mm"] if unit_scale is None else unit_scale
        if not np.isclose(scale, 1.0, rtol=1e-3):
            mesh.apply_scale(scale)
    return mesh


# --------------------------------------------------------------------------
# Solidification
# --------------------------------------------------------------------------

def solidify(mesh: trimesh.Trimesh, pitch_mm: float = 0.25,
             seal: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a surface into a filled boolean volume plus its index-to-mm affine.

    Returns ``(solid, affine)`` where ``solid[i, j, k]`` is True inside the
    mesh, and ``affine`` maps voxel indices to millimetres in the mesh frame.

    The flood-fill runs on the *closed* shell rather than the raw one so that
    holes in the surface cannot leak the fill into the interior; without the
    closing step a single unclosed cross-section collapses that slice's area to
    the shell alone.
    """
    if pitch_mm <= 0:
        raise ValueError(f"pitch_mm must be positive, got {pitch_mm}")
    if seal < 0:
        raise ValueError(f"seal must be non-negative, got {seal}")

    grid = mesh.voxelized(pitch=pitch_mm)
    shell = np.asarray(grid.matrix, dtype=bool)

    # Pad so the border is guaranteed to be background for the flood-fill, and
    # so the closing has room to work at the extremes of the mesh.
    pad = seal + 2
    shell = np.pad(shell, pad)

    sealed = binary_closing(shell, np.ones((2 * seal + 1,) * 3)) if seal > 0 else shell

    labels, _ = cc_label(~sealed)
    border_ids = np.unique(np.concatenate([
        labels[0].ravel(), labels[-1].ravel(),
        labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ]))
    outside = np.isin(labels, border_ids)
    solid = ~outside

    affine = _padded_affine(np.asarray(grid.transform, dtype=float), pad)
    return solid, affine


def _padded_affine(transform: np.ndarray, pad: int) -> np.ndarray:
    """Shift a voxel grid's index-to-mm transform to account for np.pad.

    Padding prepends `pad` voxels on every axis, so index i in the padded array
    is index i-pad in the original: the origin moves back by pad voxels.
    """
    affine = transform.copy()
    affine[:3, 3] -= affine[:3, :3] @ (np.ones(3) * float(pad))

    diag = np.diag(affine[:3, :3])
    if not np.allclose(affine[:3, :3], np.diag(diag)) or np.any(diag <= 0):
        raise ValueError(
            "Expected a positive diagonal voxel transform so the volume is "
            f"already in RAS-like order, got:\n{affine[:3, :3]}"
        )
    return affine


def write_solid_nifti(solid: np.ndarray, affine: np.ndarray, path: str) -> str:
    """Write the solid as a single-label NIfTI. Returns the path written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(solid.astype(np.uint8), affine), path)
    return path


def kernel_px_for(pitch_mm: float, kernel_mm: float = DEFAULT_KERNEL_MM) -> int:
    """Odd pixel width of a `kernel_mm` morphological kernel at `pitch_mm`."""
    if pitch_mm <= 0:
        raise ValueError(f"pitch_mm must be positive, got {pitch_mm}")
    px = max(3, int(round(float(kernel_mm) / float(pitch_mm))))
    return px if px % 2 == 1 else px + 1


# --------------------------------------------------------------------------
# Cross-section perimeter, measured on the mesh rather than on the solid
# --------------------------------------------------------------------------

def section_perimeter_mm(mesh: trimesh.Trimesh, axis: int,
                         coord_mm: float) -> Optional[float]:
    """Total boundary length of the mesh's cross-section at `coord_mm`.

    Exact polyline length, so it keeps the sulcal detail the sealing step
    rounds off. Open polylines contribute their arc length, which is legitimate
    boundary; only *area* is undefined for them, and area comes from the solid.

    Note this counts **every** boundary in the section, internal ones included,
    whereas the sampler's own ``perimeter_mm`` uses ``cv2.RETR_EXTERNAL`` and
    counts only the outermost. The two therefore do not measure the same thing,
    and ``lgi_mesh`` inherits the difference. Telling an internal boundary from
    the silhouette is not reliably decidable from the section alone - the
    surface is open, so the silhouette itself arrives as several disconnected
    arcs, and neither polygon containment nor a 2D flood-fill survives that.
    The difference is documented rather than guessed at.
    """
    normal = np.zeros(3)
    normal[axis] = 1.0
    origin = mesh.bounds.mean(axis=0).copy()
    origin[axis] = float(coord_mm)
    section = mesh.section(plane_origin=origin, plane_normal=normal)
    if section is None:
        return None
    return float(section.length)


def section_loops(section, axis: int) -> List[Tuple[np.ndarray, float]]:
    """The section's boundary loops as 2D points plus their length in mm.

    Points are projected onto the two in-plane axes, which is exact because the
    section lies in a plane normal to `axis`, and preserves length. Using this
    rather than trimesh's ``to_planar`` keeps every slice in one frame, so the
    rendered slices are registered with each other instead of each landing in
    its own arbitrary 2D basis.
    """
    col_ax, row_ax = in_plane_axes(axis)
    loops: List[Tuple[np.ndarray, float]] = []
    for entity in section.entities:
        pts3 = section.vertices[entity.points]
        if len(pts3) < 2:
            continue
        pts = np.column_stack([pts3[:, col_ax], pts3[:, row_ax]])
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if length <= 0:
            continue
        loops.append((pts, length))
    return loops


def in_plane_axes(axis: int) -> Tuple[int, int]:
    """The two mesh axes that span a slice normal to `axis`, as (column, row).

    Matches NiftiAreaSampler.inplane_spacings: axis 0 -> (y,z), 1 -> (x,z),
    2 -> (x,y), so an outline lines up with the sampler's own frame.
    """
    return {0: (1, 2), 1: (0, 2), 2: (0, 1)}[axis]


def render_section_outlines(mesh: trimesh.Trimesh, axis: int,
                            coords_mm: Sequence[float], out_dir: str,
                            *, pitch_mm: float = 0.25, margin_mm: float = 5.0,
                            bar_length_mm: float = 20.0,
                            line_px: int = 1,
                            names: Optional[Sequence[str]] = None) -> List[str]:
    """Draw each cross-section straight from the mesh, at full detail.

    The solidified raster cannot produce this picture: filling the section
    closes the interhemispheric fissure and every sulcus narrower than the
    voxel pitch, so its outline reads as a blob rather than a brain. These
    polylines are the mesh's own geometry, so nothing is lost.

    All slices share one frame, derived from the mesh bounds, so they are
    spatially registered with each other and can be cropped as a set.
    """
    os.makedirs(out_dir, exist_ok=True)
    col_ax, row_ax = in_plane_axes(axis)
    lo = mesh.bounds[0]
    hi = mesh.bounds[1]

    x0, x1 = lo[col_ax] - margin_mm, hi[col_ax] + margin_mm
    y0, y1 = lo[row_ax] - margin_mm, hi[row_ax] + margin_mm
    w = max(2, int(np.ceil((x1 - x0) / pitch_mm)))
    h = max(2, int(np.ceil((y1 - y0) / pitch_mm)))

    strip_px = _outline_strip_px(w, h)
    normal = np.zeros(3)
    normal[axis] = 1.0

    written: List[str] = []
    for i, coord in enumerate(coords_mm):
        canvas = np.full((h + strip_px, w, 3), 255, dtype=np.uint8)

        origin = mesh.bounds.mean(axis=0).copy()
        origin[axis] = float(coord)
        section = mesh.section(plane_origin=origin, plane_normal=normal)
        if section is not None:
            for pts, _ in section_loops(section, axis):
                px = np.empty((len(pts), 2), dtype=np.int32)
                px[:, 0] = np.round((pts[:, 0] - x0) / pitch_mm)
                # Flip the row axis so +mm points up, as the sampler renders it.
                px[:, 1] = np.round((y1 - pts[:, 1]) / pitch_mm)
                cv2.polylines(canvas, [px], isClosed=False, color=(0, 0, 255),
                              thickness=max(1, int(line_px)), lineType=cv2.LINE_AA)

        _draw_outline_scale_bar(canvas, h, strip_px, pitch_mm, bar_length_mm)

        name = names[i] if names and i < len(names) else f"outline_{i:03d}.png"
        path = os.path.join(out_dir, name)
        cv2.imwrite(path, canvas)
        written.append(path)

    _write_outline_sidecar(out_dir, h, w, strip_px, pitch_mm)
    return written


_BAR_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _outline_bar_style(frame_w: int, frame_h: int) -> Tuple[float, int, int, int]:
    """Font scale, label height, bar thickness and margin for a given frame."""
    span = min(frame_w, frame_h)
    scale = float(np.clip(span / 620.0, 0.40, 0.85))
    (_, text_h), _ = cv2.getTextSize("20 mm", _BAR_FONT, scale, 1)
    thickness = max(2, int(round(0.005 * span)))
    margin = max(10, int(round(0.03 * span)))
    return scale, int(text_h), thickness, margin


def _outline_strip_px(frame_w: int, frame_h: int) -> int:
    """Height of the bar strip below the section.

    Derived from what actually has to fit - margin, bar, gap, label, margin -
    rather than from a fraction of the frame. A fixed fraction leaves the label
    nowhere to go on a small frame, and it ends up clamped against the bottom
    edge and printed over the bar.
    """
    _, text_h, thickness, margin = _outline_bar_style(frame_w, frame_h)
    gap = max(6, int(round(0.5 * text_h)))
    return margin + thickness + gap + text_h + margin


def _draw_outline_scale_bar(canvas: np.ndarray, frame_h: int, strip_px: int,
                            pitch_mm: float, bar_length_mm: float) -> None:
    """Draw the bar and its label inside the strip beneath the section.

    The bar sits in its own strip rather than over the section, and the label is
    measured before the bar is sized, so a bar can never end up shorter than the
    text naming it.
    """
    h, w = canvas.shape[:2]
    scale, text_h, thickness, margin = _outline_bar_style(w, frame_h)

    bar_mm = float(bar_length_mm)
    for candidate in (bar_mm, 10.0, 5.0, 2.0, 1.0):
        bar_px = int(round(candidate / pitch_mm))
        if 1 <= bar_px <= w - 2 * margin:
            bar_mm = candidate
            break
    else:
        return

    label = f"{bar_mm:g} mm"
    (text_w, _), _ = cv2.getTextSize(label, _BAR_FONT, scale, 1)

    x1 = w - margin
    x0 = max(margin, x1 - bar_px)
    y = frame_h + margin
    cv2.line(canvas, (x0, y), (x1, y), (0, 0, 0), thickness)

    gap = max(6, int(round(0.5 * text_h)))
    baseline = y + thickness + gap + text_h
    tx = max(margin, min(w - text_w - margin, x0 + (bar_px - text_w) // 2))
    cv2.putText(canvas, label, (tx, baseline), _BAR_FONT, scale,
                (0, 0, 0), 1, cv2.LINE_AA)


def _write_outline_sidecar(out_dir: str, frame_h: int, frame_w: int,
                           strip_px: int, pitch_mm: float) -> None:
    """Record the bar strip so a cropper can exclude it, as the sampler does.

    The key names are not ours to choose: ``crop_band_roi`` reads
    ``scale_bar_strip_px`` and nothing else, and it silently treats a sidecar
    without that key as a run from before the strip existed - i.e. no strip at
    all. That is the one failure mode this file exists to prevent, and it is
    invisible in the output: the crop lands, just in the wrong place. Every
    normalised box then resolves against the full image height instead of the
    section frame, sliding down by ``y_norm * strip_px`` - on a 217px outline
    frame with its 37px strip, 18px at mid-height, which is how a coronal box
    crosses the interhemispheric midline into the other hemisphere.

    So this mirrors ``NiftiAreaSampler.write_scale_bar_sidecar`` field for
    field. Both are read by the same function, and outlines can be written
    straight into ``brain_slices/`` (``--outline-dirname``) where they are the
    set it reads. ``mm_per_px`` is the one addition - the mesh path knows its
    own pitch, and a cropped outline needs it for ``--pixel-size-mm``.
    """
    payload = {
        "scale_bar_strip_px": int(strip_px),
        "frame_height": int(frame_h),
        "frame_width": int(frame_w),
        "image_height": int(frame_h) + int(strip_px),
        "mm_per_px": float(pitch_mm),
        "note": (
            "The scale bar is drawn in a blank strip below the section. "
            "Resolve normalised boxes against frame_height, not image_height."
        ),
    }
    try:
        with open(os.path.join(out_dir, "scale_bar_strip.json"), "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as ex:
        print(f"[StlAreaBand] Failed to write outline sidecar: {ex}")


def _axis_index(axis: str | int) -> int:
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key not in ("x", "y", "z"):
            raise ValueError(f"axis must be x, y, z or 0, 1, 2 - got {axis!r}")
        return {"x": 0, "y": 1, "z": 2}[key]
    idx = int(axis)
    if idx not in (0, 1, 2):
        raise ValueError(f"axis must be x, y, z or 0, 1, 2 - got {axis!r}")
    return idx


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def sample_stl_band(cfg: StlAreaBandConfig) -> Dict[str, Any]:
    """Run the full STL -> solid -> NIfTI -> area band pipeline.

    Writes everything ``NiftiAreaSampler`` writes into ``cfg.out_dir``, plus a
    ``stl_area_band_slices.csv``/``.xlsx`` carrying the mesh-measured perimeter
    and LGI, and a ``stl_area_band_summary.json`` recording the mesh probe and
    the solidification settings used.
    """
    axis = _axis_index(cfg.axis)
    axis_name = "xyz"[axis]

    probe = probe_mesh(cfg.stl_path)
    if cfg.require_brain and probe.get("label") == "not_brain":
        raise ValueError(
            f"check_brain() classified {cfg.stl_path} as not_brain "
            f"(adult {probe.get('adult_score')}, fetal {probe.get('fetal_score')}). "
            "Pass require_brain=False to sample it anyway."
        )

    mesh = load_mesh(cfg.stl_path, scale_to_mm=cfg.scale_to_mm,
                     unit_scale=probe["unit_scale_to_mm"])
    solid, affine = solidify(mesh, pitch_mm=cfg.pitch_mm, seal=cfg.seal)

    solid_volume_cm3 = float(solid.sum()) * cfg.pitch_mm ** 3 / 1000.0
    mesh_volume_cm3 = float(abs(mesh.volume)) / 1000.0

    os.makedirs(cfg.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(cfg.stl_path))[0]

    tmp_dir: Optional[str] = None
    if cfg.keep_nifti:
        nifti_dir = cfg.nifti_dir or cfg.out_dir
        os.makedirs(nifti_dir, exist_ok=True)
        nifti_path = os.path.join(nifti_dir, f"{stem}_solid.nii.gz")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="stl_area_band_")
        nifti_path = os.path.join(tmp_dir, f"{stem}_solid.nii.gz")

    try:
        write_solid_nifti(solid, affine, nifti_path)

        band_cfg = AreaBandConfig(
            file_path=nifti_path,
            out_dir=cfg.out_dir,
            axis=axis_name,
            n=cfg.n,
            p=cfg.p,
            min_contour_area=cfg.min_contour_area,
            save_png=cfg.save_png,
            profile_plot=cfg.profile_plot,
            show_crosshair=cfg.show_crosshair,
            draw_contours=cfg.draw_contours,
            kernel_size=kernel_px_for(cfg.pitch_mm, cfg.kernel_mm),
            # Must stay on. _overlay_labels is the only thing that paints the
            # section: without it the frame renders as background and every PNG
            # comes out blank but for the crosshair and scale bar. A surface has
            # one label, so what it paints is a flat silhouette - which is
            # exactly the right picture for a mesh that carries no labels.
            overlay_labels=True,
        )
        result = NiftiAreaSampler(band_cfg).sample_band()
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    result.update(_mesh_section_metrics(mesh, axis, affine, result))

    if cfg.draw_section_outline and result.get("section_coord_mm"):
        # Deliberately independent of save_png: the outlines are the pictures
        # worth looking at, so turning off the sampler's filled set must not
        # take them with it. Names come from the slice indices rather than from
        # saved_pngs for the same reason - they stay stable and informative
        # when there are no filled PNGs to mirror.
        names = [
            f"band_axis{axis}_s{i:02d}_idx{idx:04d}_outline.png"
            for i, idx in enumerate(result.get("indices") or [])
        ]
        result["outline_pngs"] = render_section_outlines(
            mesh, axis, result["section_coord_mm"],
            os.path.join(cfg.out_dir, cfg.outline_dirname),
            pitch_mm=cfg.outline_pitch_mm or cfg.pitch_mm,
            line_px=cfg.outline_line_px,
            names=names or None,
        )

    result["stl"] = {
        "stl_path": os.path.abspath(cfg.stl_path),
        "axis_frame": "mesh",  # x/y/z are the mesh's own axes, not anatomical
        "pitch_mm": cfg.pitch_mm,
        "seal": cfg.seal,
        "seal_kernel_mm": (2 * cfg.seal + 1) * cfg.pitch_mm,
        "kernel_mm": cfg.kernel_mm,
        "kernel_px": band_cfg.kernel_size,
        "solid_shape": [int(x) for x in solid.shape],
        "solid_volume_cm3": round(solid_volume_cm3, 3),
        "mesh_volume_cm3": round(mesh_volume_cm3, 3),
        "solid_vs_mesh_volume_pct": (
            round(100.0 * (solid_volume_cm3 - mesh_volume_cm3) / mesh_volume_cm3, 2)
            if mesh_volume_cm3 > 0 else None
        ),
        "nifti_kept": bool(cfg.keep_nifti),
        "nifti_path": nifti_path if cfg.keep_nifti else None,
        "probe": probe,
        "config": asdict(cfg),
    }

    _write_stl_logs(cfg, result, stem)
    return result


def _mesh_section_metrics(mesh: trimesh.Trimesh, axis: int, affine: np.ndarray,
                          result: Dict[str, Any]) -> Dict[str, Any]:
    """Perimeter and LGI measured on the raw mesh at the sampled slice indices."""
    indices: Sequence[int] = result.get("indices") or []
    outer: Sequence[Optional[float]] = result.get("perimeter_convex_mm") or []

    origin_mm = float(affine[axis, 3])
    step_mm = float(affine[axis, axis])

    coords_mm: List[float] = []
    perims: List[Optional[float]] = []
    lgis: List[Optional[float]] = []

    for pos, idx in enumerate(indices):
        coord = origin_mm + float(idx) * step_mm
        coords_mm.append(coord)
        perim = section_perimeter_mm(mesh, axis, coord)
        perims.append(perim)

        out = outer[pos] if pos < len(outer) else None
        lgis.append(
            float(perim) / float(out) if perim and out and out > 0 else None
        )

    return {
        "section_coord_mm": coords_mm,
        "perimeter_mesh_mm": perims,
        "lgi_mesh": lgis,
    }


def _write_stl_logs(cfg: StlAreaBandConfig, result: Dict[str, Any], stem: str) -> None:
    """Write the STL-specific summary and the combined per-slice table."""
    try:
        with open(os.path.join(cfg.out_dir, "stl_area_band_summary.json"),
                  "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as ex:
        print(f"[StlAreaBand] Failed to write summary JSON: {ex}")

    rows: Dict[str, List[Any]] = {
        "pos": list(result.get("positions") or []),
        "idx": list(result.get("indices") or []),
        "pos_mm": list(result.get("positions_mm") or []),
        "section_coord_mm": list(result.get("section_coord_mm") or []),
        "area_cm2": list(result.get("areas_cm2") or []),
        "perimeter_mm": list(result.get("perimeter_mm") or []),
        "perimeter_mesh_mm": list(result.get("perimeter_mesh_mm") or []),
        "perimeter_convex_mm": list(result.get("perimeter_convex_mm") or []),
        "lgi": list(result.get("lgi") or []),
        "lgi_mesh": list(result.get("lgi_mesh") or []),
        "png_path": list(result.get("saved_pngs") or []),
    }
    width = max((len(v) for v in rows.values()), default=0)
    if width == 0:
        return
    for key, val in rows.items():
        if len(val) < width:
            rows[key] = val + [None] * (width - len(val))

    try:
        import csv
        csv_path = os.path.join(cfg.out_dir, "stl_area_band_slices.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(rows.keys()))
            for i in range(width):
                writer.writerow([rows[k][i] for k in rows])
    except Exception as ex:
        print(f"[StlAreaBand] Failed to write CSV: {ex}")

    try:
        import pandas as pd
        stl_meta = dict(result.get("stl") or {})
        stl_meta.pop("probe", None)
        stl_meta.pop("config", None)
        xlsx_path = os.path.join(cfg.out_dir, "stl_area_band_slices.xlsx")
        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame([{
                "stl": stem,
                "axis_name": result.get("axis_name"),
                "n": cfg.n,
                "p": cfg.p,
                "x_max": result.get("x_max"),
                "f_max": result.get("f_max"),
                "left": result.get("left"),
                "right": result.get("right"),
                "band_length_mm": result.get("band_length_mm"),
                **stl_meta,
            }]).to_excel(writer, sheet_name="summary", index=False)
            pd.DataFrame(rows).to_excel(writer, sheet_name="slices", index=False)
    except Exception as ex:
        print(f"[StlAreaBand] Failed to write Excel: {ex}")


def load_config_from_json(path: str) -> StlAreaBandConfig:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return StlAreaBandConfig(**data)


def run_from_config(json_path: str) -> Dict[str, Any]:
    return sample_stl_band(load_config_from_json(json_path))
