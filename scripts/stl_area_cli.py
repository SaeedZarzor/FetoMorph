"""
Command-line runner for the STL area band sampler.

Mirrors scripts/area_band_cli.py, but takes surface meshes instead of NIfTI
label maps. Because a surface has no labels there is nothing to select, so the
label arguments of the NIfTI CLI have no counterpart here; what replaces them
are the solidification settings (--pitch, --seal) and the physical closing size
(--kernel-mm).

Axis names are the mesh's own x/y/z and carry no anatomical meaning.

Single mesh:
    python -m scripts.stl_area_cli --stl brain.stl --out results --axis z --n 20 --p 0.9

Every mesh in a folder, all three axes:
    python -m scripts.stl_area_cli --stl-dir meshes --out results --all-axes

From a JSON config (StlAreaBandConfig fields, plus stl_dir/all_axes/axes):
    python -m scripts.stl_area_cli --config configs/stl_area_band.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional, Sequence

# Allow running as a plain script (python scripts/stl_area_cli.py) as well as
# a module, matching how area_band_cli.py is used.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.stl_area_sampler import (  # noqa: E402
    MESH_SUFFIXES,
    StlAreaBandConfig,
    sample_stl_band,
)

AXES = ("x", "y", "z")


def _is_mesh(name: str) -> bool:
    """Whether a filename is a mesh we should sample.

    Skips the AppleDouble sidecars (``._24.stl``) that macOS leaves behind on
    non-HFS volumes: they carry the .stl suffix but are resource forks, not
    geometry, and trimesh fails on them.
    """
    if name.startswith("._"):
        return False
    return os.path.splitext(name)[1].lower() in MESH_SUFFIXES


def find_meshes(stl_dir: str) -> List[str]:
    if not os.path.isdir(stl_dir):
        raise NotADirectoryError(f"--stl-dir is not a directory: {stl_dir}")
    found = sorted(
        os.path.join(stl_dir, name)
        for name in os.listdir(stl_dir)
        if _is_mesh(name) and os.path.isfile(os.path.join(stl_dir, name))
    )
    if not found:
        raise FileNotFoundError(f"No mesh files ({', '.join(MESH_SUFFIXES)}) in {stl_dir}")
    return found


def _split_config(path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a JSON config into StlAreaBandConfig fields and batch extras."""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    known = {f.name for f in dataclass_fields(StlAreaBandConfig)}
    cfg_kwargs = {k: v for k, v in data.items() if k in known}
    extra = {k: v for k, v in data.items() if k not in known}
    return cfg_kwargs, extra


def _validate(cfg: StlAreaBandConfig) -> None:
    if not os.path.isfile(cfg.stl_path):
        raise FileNotFoundError(f"Mesh not found: {cfg.stl_path}")
    if not 0.0 < cfg.p <= 1.0:
        raise ValueError(f"--p must be in (0, 1], got {cfg.p}")
    if cfg.n < 1:
        raise ValueError(f"--n must be at least 1, got {cfg.n}")
    if cfg.pitch_mm <= 0:
        raise ValueError(f"--pitch must be positive, got {cfg.pitch_mm}")
    if cfg.seal < 0:
        raise ValueError(f"--seal must be non-negative, got {cfg.seal}")
    if cfg.kernel_mm <= 0:
        raise ValueError(f"--kernel-mm must be positive, got {cfg.kernel_mm}")


def _fmt(value: Any, spec: str = ".3f") -> str:
    """Format a summary number, tolerating the None a degenerate run reports.

    NiftiAreaSampler returns None for x_max/left/right when no slice holds any
    mask at all - a mesh that is not a closed volume, for instance - so the
    summary has to survive that rather than crash on the way to reporting it.
    """
    if value is None:
        return "n/a"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def run_one(cfg: StlAreaBandConfig, *, quiet: bool = False) -> Dict[str, Any]:
    _validate(cfg)
    if not quiet:
        print(f"[StlAreaBand] {os.path.basename(cfg.stl_path)} "
              f"axis={cfg.axis} n={cfg.n} p={cfg.p} "
              f"pitch={cfg.pitch_mm}mm seal={cfg.seal} -> {cfg.out_dir}")

    result = sample_stl_band(cfg)

    if not quiet:
        meta = result.get("stl", {})
        probe = meta.get("probe", {})
        print(f"    mesh      : {probe.get('label')} "
              f"{_fmt(probe.get('L_mm'), '.0f')}x{_fmt(probe.get('W_mm'), '.0f')}x"
              f"{_fmt(probe.get('H_mm'), '.0f')} mm, "
              f"watertight={probe.get('watertight')}, "
              f"unit_scale={_fmt(probe.get('unit_scale_to_mm'), '.3g')}")
        print(f"    solid     : {meta.get('solid_volume_cm3')} cm3 vs mesh "
              f"{meta.get('mesh_volume_cm3')} cm3 "
              f"({_fmt(meta.get('solid_vs_mesh_volume_pct'), '+.1f')}%), "
              f"closing {meta.get('kernel_px')} px = {meta.get('kernel_mm')} mm")
        n_fill = sum(1 for p in (result.get("saved_pngs") or []) if p)
        n_outline = len(result.get("outline_pngs") or [])
        print(f"    band      : max {_fmt(result.get('f_max'), '.2f')} cm2 at "
              f"{_fmt(result.get('x_max'))}, band "
              f"{_fmt(result.get('left'))}-{_fmt(result.get('right'))} "
              f"({_fmt(result.get('band_length_mm'), '.1f')} mm), "
              f"{len(result.get('indices') or [])} slices")
        print(f"    pngs      : {n_outline} outline, {n_fill} filled")
        lgi = [v for v in (result.get("lgi_mesh") or []) if v]
        if lgi:
            print(f"    lgi_mesh  : mean {sum(lgi) / len(lgi):.3f} "
                  f"range {min(lgi):.3f}-{max(lgi):.3f}")

    name = os.path.basename(cfg.stl_path)
    indices = result.get("indices") or []
    if not result.get("saved_pngs") and not indices:
        print(f"    WARNING   : no band found - {name} produced an empty solid. "
              "Check that it is a closed surface.")
    elif len(set(indices)) < max(2, len(indices) // 2):
        # n positions collapsing onto a handful of slices means the band is
        # thinner than the voxel pitch: the area profile spikes rather than
        # rising to a plateau, which is what a mesh that is not a brain, or one
        # the sealing failed on, looks like.
        print(f"    WARNING   : {name} band collapsed - {len(indices)} positions "
              f"map to only {len(set(indices))} distinct slices over "
              f"{_fmt(result.get('band_length_mm'), '.2f')} mm. "
              "Treat these slices as suspect.")
    return result


def run_many(base: StlAreaBandConfig, stl_paths: Sequence[str], out_base: str,
             axes: Sequence[str], *, subject_subdir: bool, axis_subdir: bool,
             quiet: bool = False) -> List[Dict[str, Any]]:
    """Sample every (mesh, axis) pair, laying out directories for crop_band_roi.

    The layout is <out>/<stem>/axis_<a>/, because crop_band_roi identifies a
    subject by the presence of axis_<a>/brain_slices beneath it. The <stem>
    level is dropped for a single mesh and the axis level under
    --no-axis-subdir, which then puts the output beyond crop_band_roi's reach.
    """
    results: List[Dict[str, Any]] = []
    for stl_path in stl_paths:
        stem = os.path.splitext(os.path.basename(stl_path))[0]
        for axis in axes:
            out_dir = out_base
            if subject_subdir:
                out_dir = os.path.join(out_dir, stem)
            if axis_subdir:
                out_dir = os.path.join(out_dir, f"axis_{axis}")

            cfg = StlAreaBandConfig(**{
                **{f.name: getattr(base, f.name) for f in dataclass_fields(base)},
                "stl_path": stl_path,
                "out_dir": out_dir,
                "axis": axis,
            })
            try:
                results.append(run_one(cfg, quiet=quiet))
            except Exception as ex:
                print(f"[StlAreaBand] FAILED {stem} axis {axis}: {ex}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sample max-area 2D slices from surface meshes.")
    ap.add_argument("--config", help="Path to a JSON config (StlAreaBandConfig "
                                     "fields + optional stl_dir/all_axes/axes)")
    ap.add_argument("--stl", help="Path to a single mesh")
    ap.add_argument("--stl-dir", help="Folder of meshes to process")
    ap.add_argument("--out", help="Output directory")
    ap.add_argument("--axis", default=None, choices=list(AXES),
                    help="Mesh axis to slice along (default z)")
    ap.add_argument("--all-axes", action="store_true", help="Run x, y and z")
    ap.add_argument("--n", type=int, default=None, help="Slices to sample (default 20)")
    ap.add_argument("--p", type=float, default=None,
                    help="Band threshold as a fraction of max area (default 0.9)")

    ap.add_argument("--pitch", type=float, default=None, dest="pitch_mm",
                    help="Voxel pitch of the intermediate solid in mm (default 0.25)")
    ap.add_argument("--seal", type=int, default=None,
                    help="Hole-sealing radius in voxels (default 1). Raise only if "
                         "the solid volume comes out far below the mesh volume")
    ap.add_argument("--kernel-mm", type=float, default=None, dest="kernel_mm",
                    help="Outer-perimeter closing size in mm (default 5.0)")

    ap.add_argument("--keep-nifti", action="store_true",
                    help="Keep the intermediate solid volume (deleted otherwise)")
    ap.add_argument("--nifti-dir", default=None,
                    help="Where to keep it when --keep-nifti is set")
    ap.add_argument("--require-brain", action="store_true",
                    help="Abort on meshes check_brain() calls not_brain")
    ap.add_argument("--no-scale-to-mm", action="store_true",
                    help="Trust the mesh units as-is instead of auto-converting")

    ap.add_argument("--fill-png", action="store_true",
                    help="Also write the sampler's filled silhouette PNGs into "
                         "brain_slices/. Off by default: they read as a blob "
                         "because filling closes the fissure and narrow sulci")
    ap.add_argument("--no-outline", action="store_true",
                    help="Do not draw the mesh cross-section outlines")
    ap.add_argument("--outline-dirname", default=None,
                    help="Folder for the outlines (default brain_slices_outline). "
                         "Pass 'brain_slices' together with --no-png to make the "
                         "outlines the set crop_band_roi reads")
    ap.add_argument("--outline-pitch", type=float, default=None, dest="outline_pitch_mm",
                    help="mm per pixel for the outlines (default: --pitch)")
    ap.add_argument("--outline-line-px", type=int, default=None,
                    help="Outline stroke width in pixels (default 1)")
    ap.add_argument("--no-profile-plot", action="store_true",
                    help="Do not plot the area profile")
    ap.add_argument("--no-crosshair", action="store_true", help="No crosshair overlay")
    ap.add_argument("--draw-contours", action="store_true", help="Draw contours on PNGs")
    ap.add_argument("--no-axis-subdir", action="store_true",
                    help="Write slices directly under <out>/<stem>/ instead of "
                         "<out>/<stem>/axis_<a>/. The axis level is on by default "
                         "because crop_band_roi identifies a subject by it")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg_kwargs: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    if args.config:
        cfg_kwargs, extra = _split_config(args.config)

    # An explicit --stl names one mesh and must win over a folder the config
    # happens to carry, otherwise --config plus --stl silently runs the folder.
    if args.stl:
        stl_dir: Optional[str] = args.stl_dir
        stl_path: Optional[str] = args.stl
    else:
        stl_dir = args.stl_dir or extra.get("stl_dir")
        stl_path = cfg_kwargs.get("stl_path") or None
    out_base: Optional[str] = args.out or cfg_kwargs.get("out_dir")
    if not out_base:
        ap.error("an output directory is required (--out or out_dir in --config)")
    if not stl_dir and not stl_path:
        ap.error("a mesh is required (--stl, --stl-dir, or stl_path/stl_dir in --config)")

    axes: List[str]
    if args.all_axes or extra.get("all_axes"):
        axes = list(AXES)
    elif extra.get("axes"):
        axes = [str(a).lower() for a in extra["axes"]]
    elif args.axis:
        axes = [args.axis]
    else:
        axes = [str(cfg_kwargs.get("axis", "z")).lower()]

    # CLI flags win over the config file.
    for key, val in (("n", args.n), ("p", args.p), ("pitch_mm", args.pitch_mm),
                     ("seal", args.seal), ("kernel_mm", args.kernel_mm),
                     ("nifti_dir", args.nifti_dir),
                     ("outline_dirname", args.outline_dirname),
                     ("outline_pitch_mm", args.outline_pitch_mm),
                     ("outline_line_px", args.outline_line_px)):
        if val is not None:
            cfg_kwargs[key] = val
    for key, flag in (("keep_nifti", args.keep_nifti),
                      ("require_brain", args.require_brain),
                      ("save_png", args.fill_png),
                      ("draw_contours", args.draw_contours)):
        if flag:
            cfg_kwargs[key] = True
    for key, flag in (("draw_section_outline", args.no_outline),
                      ("profile_plot", args.no_profile_plot),
                      ("show_crosshair", args.no_crosshair),
                      ("scale_to_mm", args.no_scale_to_mm)):
        if flag:
            cfg_kwargs[key] = False

    cfg_kwargs["out_dir"] = out_base
    cfg_kwargs["stl_path"] = stl_path or ""
    base = StlAreaBandConfig(**cfg_kwargs)

    stl_paths = find_meshes(stl_dir) if stl_dir else [stl_path]

    results = run_many(
        base, stl_paths, out_base, axes,
        subject_subdir=len(stl_paths) > 1,
        axis_subdir=not args.no_axis_subdir,
        quiet=args.quiet,
    )

    print(f"[StlAreaBand] {len(results)}/{len(stl_paths) * len(axes)} runs completed "
          f"-> {out_base}")
    return 0 if len(results) == len(stl_paths) * len(axes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
