import argparse
import json
import os
import sys
from dataclasses import fields
from typing import List, Tuple, Dict, Any

sys.path.insert(0, os.getcwd())

from functions.nifti_area_sampler import AreaBandConfig, NiftiAreaSampler


LABELS_DEFAULT: List[int] = [2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 17]


def _load_config(path: str) -> Tuple[AreaBandConfig, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    cfg_fields = {f.name for f in fields(AreaBandConfig)}
    cfg_kwargs = {k: v for k, v in data.items() if k in cfg_fields}
    if "file_path" not in cfg_kwargs:
        cfg_kwargs["file_path"] = ""
    if "out_dir" not in cfg_kwargs:
        cfg_kwargs["out_dir"] = ""
    extra = {k: v for k, v in data.items() if k not in cfg_fields}
    return AreaBandConfig(**cfg_kwargs), extra


def _cli_has(flag: str) -> bool:
    return flag in sys.argv


def _normalize_empty(value: str | None) -> str | None:
    if value is None:
        return None
    if not str(value).strip():
        return None
    if str(value).strip().lower() in {"none", "null"}:
        return None
    return value


def _validate_config(cfg: AreaBandConfig) -> None:
    if cfg.n < 1:
        raise ValueError("n must be >= 1")
    if not (0.0 < float(cfg.p) <= 1.0):
        raise ValueError("p must be in (0, 1]")
    NiftiAreaSampler._normalize_axis(cfg.axis)


def run_single(file_path: str, out_dir: str, axis: str, n: int, p: float,
               area_labels: List[int] | None, show_crosshair: bool, profile_plot: bool,
               save_png: bool = True, use_pial_overlay: bool = False,
               pial_lh_path: str | None = None, pial_rh_path: str | None = None,
               pial_space: str = "scanner", pial_line_thickness: float | None = None) -> dict:
    cfg = AreaBandConfig(
        file_path=file_path,
        out_dir=out_dir,
        axis=axis,
        n=n,
        p=p,
        save_png=save_png,
        profile_plot=profile_plot,
        show_crosshair=show_crosshair,
        area_labels=area_labels,
        use_pial_overlay=use_pial_overlay,
        pial_lh_path=pial_lh_path,
        pial_rh_path=pial_rh_path,
        pial_space=pial_space,
        pial_line_thickness=pial_line_thickness,
    )
    sampler = NiftiAreaSampler(cfg)
    return sampler.sample_band()


def run_batch(base_dir: str, out_base: str, axis: str, n: int, p: float,
              area_labels: List[int] | None, show_crosshair: bool, profile_plot: bool,
              save_png: bool = True, use_pial_overlay: bool = False,
              pial_lh_path: str | None = None, pial_rh_path: str | None = None,
              pial_space: str = "scanner", axis_subdir: bool = False,
              pial_line_thickness: float | None = None) -> list[dict]:
    results = []
    for name in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, name)
        if not os.path.isdir(d):
            continue
        seg = os.path.join(d, "seg.nii.gz")
        if not os.path.isfile(seg):
            continue
        out_dir = os.path.join(out_base, name)
        if axis_subdir:
            out_dir = os.path.join(out_dir, f"axis_{axis}")
        os.makedirs(out_dir, exist_ok=True)
        res = run_single(
            seg,
            out_dir,
            axis,
            n,
            p,
            area_labels,
            show_crosshair,
            profile_plot,
            save_png,
            use_pial_overlay,
            pial_lh_path,
            pial_rh_path,
            pial_space,
            pial_line_thickness,
        )
        results.append({
            "case": name,
            "out_dir": out_dir,
            "x_max": res.get("x_max"),
            "f_max": res.get("f_max"),
            "left": res.get("left"),
            "right": res.get("right"),
            "n": len(res.get("positions") or []),
        })
        # Write a compact summary per case
        with open(os.path.join(out_dir, "area_band_run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(results[-1], f, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser(description="NIfTI area-band sampler runner")
    ap.add_argument("--config", help="Path to JSON config (AreaBandConfig fields + optional batch fields)")
    ap.add_argument("--all-axes", action="store_true", help="Run for all axes (x,y,z)")
    ap.add_argument("--file", help="Path to seg.nii.gz (single run)")
    ap.add_argument("--out", help="Output directory (single run)")
    ap.add_argument("--axis", default="z", choices=["x", "y", "z", "0", "1", "2"], help="Slicing axis")
    ap.add_argument("--n", type=int, default=10, help="Number of slices to sample")
    ap.add_argument("--p", type=float, default=0.8, help="Top-p threshold (0..1)")
    ap.add_argument("--labels", nargs="*", type=int, default=None, help="Label IDs to include in area/legend")
    ap.add_argument("--no-crosshair", action="store_true", help="Disable crosshair overlay")
    ap.add_argument("--no-profile-plot", action="store_true", help="Disable profile plotting")
    ap.add_argument("--no-png", action="store_true", help="Disable saving PNG images")

    ap.add_argument("--batch-dir", help="Base folder with multiple cases (each containing seg.nii.gz)")
    ap.add_argument("--batch-out", default="area_band_output", help="Output base for batch mode")
    ap.add_argument("--use-default-labels", action="store_true", help="Use built-in LABELS_DEFAULT for --labels")
    ap.add_argument("--pial-lh", help="Left hemisphere pial surface path (Freesurfer .pial)")
    ap.add_argument("--pial-rh", help="Right hemisphere pial surface path (Freesurfer .pial)")
    ap.add_argument("--pial-space", default="scanner", choices=["scanner", "tkr"],
                    help="Pial coordinate space (scanner or tkr)")
    ap.add_argument("--pial-line-thickness", type=float, default=None,
                    help="Pial overlay line thickness in pixels (default 1)")
    ap.add_argument("--use-pial-overlay", action="store_true",
                    help="Enable pial overlay (auto-detect per case when --pial-lh/--pial-rh are not set)")
    ap.add_argument("--no-pial-overlay", action="store_true",
                    help="Disable pial overlay (overrides config)")
    ap.add_argument("--axis-subdir", action="store_true",
                    help="In batch mode, write results to per-axis subfolders (axis_x, axis_y, axis_z)")

    args = ap.parse_args()

    if args.config:
        cfg, extra = _load_config(args.config)

        axis = cfg.axis
        labels = list(cfg.area_labels) if cfg.area_labels is not None else None
        n = cfg.n
        p = cfg.p
        show_crosshair = cfg.show_crosshair
        profile_plot = cfg.profile_plot
        save_png = cfg.save_png
        use_pial_overlay = cfg.use_pial_overlay
        pial_lh_path = cfg.pial_lh_path
        pial_rh_path = cfg.pial_rh_path
        pial_space = cfg.pial_space
        pial_line_thickness = cfg.pial_line_thickness

        file_path = cfg.file_path
        out_dir = cfg.out_dir
        batch_dir = _normalize_empty(extra.get("batch_dir"))
        batch_out = _normalize_empty(extra.get("batch_out"))
        axis_subdir = bool(extra.get("axis_subdir", False))
        all_axes = bool(extra.get("all_axes", False))

        if _cli_has("--axis"):
            axis = args.axis
        if _cli_has("--n"):
            n = args.n
        if _cli_has("--p"):
            p = args.p
        if _cli_has("--labels"):
            labels = args.labels
        if args.use_default_labels:
            labels = LABELS_DEFAULT
        if _cli_has("--file"):
            file_path = args.file
        if _cli_has("--out"):
            out_dir = args.out
        if _cli_has("--batch-dir"):
            batch_dir = _normalize_empty(args.batch_dir)
        if _cli_has("--batch-out"):
            batch_out = _normalize_empty(args.batch_out)
        if args.axis_subdir:
            axis_subdir = True
        if args.all_axes:
            all_axes = True
        if args.no_crosshair:
            show_crosshair = False
        if args.no_profile_plot:
            profile_plot = False
        if args.no_png:
            save_png = False
        if _cli_has("--pial-space"):
            pial_space = args.pial_space
        if _cli_has("--pial-line-thickness"):
            pial_line_thickness = args.pial_line_thickness
        if _cli_has("--pial-lh"):
            pial_lh_path = args.pial_lh
        if _cli_has("--pial-rh"):
            pial_rh_path = args.pial_rh
        if args.use_pial_overlay:
            use_pial_overlay = True
        if args.no_pial_overlay:
            use_pial_overlay = False

        if pial_lh_path or pial_rh_path:
            use_pial_overlay = True

        if axis in ("0", "1", "2"):
            axis = {"0": "x", "1": "y", "2": "z"}[axis]

        axes = ["x", "y", "z"] if all_axes else [axis]
        if all_axes and not axis_subdir:
            axis_subdir = True

        try:
            _validate_config(AreaBandConfig(file_path=file_path or "", out_dir=out_dir or "", axis=axis, n=n, p=p))
        except ValueError as exc:
            ap.error(str(exc))

        if batch_dir:
            if not os.path.isdir(batch_dir):
                ap.error(f"--batch-dir not found: {batch_dir}")
            if not batch_out:
                ap.error("Batch config missing batch_out (or provide --batch-out)")
            all_results = []
            for ax in axes:
                res = run_batch(
                    base_dir=batch_dir,
                    out_base=batch_out,
                    axis=ax,
                    n=n,
                    p=p,
                    area_labels=labels,
                    show_crosshair=show_crosshair,
                    profile_plot=profile_plot,
                    save_png=save_png,
                    use_pial_overlay=use_pial_overlay,
                    pial_lh_path=pial_lh_path,
                    pial_rh_path=pial_rh_path,
                    pial_space=pial_space,
                    axis_subdir=axis_subdir,
                    pial_line_thickness=pial_line_thickness,
                )
                all_results.append({"axis": ax, "results": res})
            print(json.dumps(all_results, indent=2))
            return 0

        if not file_path or not out_dir:
            ap.error("Config missing file_path/out_dir (or provide --file and --out)")

        single_results = {}
        for ax in axes:
            run_out = out_dir
            if axis_subdir:
                run_out = os.path.join(out_dir, f"axis_{ax}")
                os.makedirs(run_out, exist_ok=True)
            res = run_single(
                file_path=file_path,
                out_dir=run_out,
                axis=ax,
                n=n,
                p=p,
                area_labels=labels,
                show_crosshair=show_crosshair,
                profile_plot=profile_plot,
                save_png=save_png,
                use_pial_overlay=use_pial_overlay,
                pial_lh_path=pial_lh_path,
                pial_rh_path=pial_rh_path,
                pial_space=pial_space,
                pial_line_thickness=pial_line_thickness,
            )
            single_results[ax] = {k: res.get(k) for k in ("x_max", "f_max", "left", "right")}
        print(json.dumps(single_results, indent=2))
        return 0

    axis = args.axis
    if axis in ("0", "1", "2"):
        axis = {"0": "x", "1": "y", "2": "z"}[axis]

    labels = args.labels
    if args.use_default_labels and labels is None:
        labels = LABELS_DEFAULT

    if args.batch_dir:
        if not os.path.isdir(args.batch_dir):
            ap.error(f"--batch-dir not found: {args.batch_dir}")
        axes = ["x", "y", "z"] if args.all_axes else [axis]
        try:
            _validate_config(AreaBandConfig(file_path=args.file or "", out_dir=args.out or "", axis=axis, n=args.n, p=args.p))
        except ValueError as exc:
            ap.error(str(exc))
        all_results = []
        for ax in axes:
            res = run_batch(
                base_dir=args.batch_dir,
                out_base=args.batch_out,
                axis=ax,
                n=args.n,
                p=args.p,
                area_labels=labels,
                show_crosshair=not args.no_crosshair,
                profile_plot=not args.no_profile_plot,
                save_png=not args.no_png,
                use_pial_overlay=bool(args.pial_lh or args.pial_rh or args.use_pial_overlay),
                pial_lh_path=args.pial_lh,
                pial_rh_path=args.pial_rh,
                pial_space=args.pial_space,
                axis_subdir=(args.axis_subdir or args.all_axes),
                pial_line_thickness=args.pial_line_thickness,
            )
            all_results.append({"axis": ax, "results": res})
        print(json.dumps(all_results, indent=2))
        return 0

    if not args.file or not args.out:
        ap.error("For single run, provide --file and --out (or use --batch-dir)")

    axes = ["x", "y", "z"] if args.all_axes else [axis]
    try:
        _validate_config(AreaBandConfig(file_path=args.file, out_dir=args.out, axis=axis, n=args.n, p=args.p))
    except ValueError as exc:
        ap.error(str(exc))
    if len(axes) > 1:
        results = {}
        for ax in axes:
            out_dir = os.path.join(args.out, f"axis_{ax}")
            os.makedirs(out_dir, exist_ok=True)
            res = run_single(
                file_path=args.file,
                out_dir=out_dir,
                axis=ax,
                n=args.n,
                p=args.p,
                area_labels=labels,
                show_crosshair=not args.no_crosshair,
                profile_plot=not args.no_profile_plot,
                save_png=not args.no_png,
                use_pial_overlay=bool(args.pial_lh or args.pial_rh or args.use_pial_overlay),
                pial_lh_path=args.pial_lh,
                pial_rh_path=args.pial_rh,
                pial_space=args.pial_space,
                pial_line_thickness=args.pial_line_thickness,
            )
            results[ax] = {k: res.get(k) for k in ("x_max", "f_max", "left", "right")}
        print(json.dumps(results, indent=2))
        return 0

    res = run_single(
        file_path=args.file,
        out_dir=args.out,
        axis=axis,
        n=args.n,
        p=args.p,
        area_labels=labels,
        show_crosshair=not args.no_crosshair,
        profile_plot=not args.no_profile_plot,
        save_png=not args.no_png,
        use_pial_overlay=bool(args.pial_lh or args.pial_rh or args.use_pial_overlay),
        pial_lh_path=args.pial_lh,
        pial_rh_path=args.pial_rh,
        pial_space=args.pial_space,
        pial_line_thickness=args.pial_line_thickness,
    )
    print(json.dumps({k: res.get(k) for k in ("x_max", "f_max", "left", "right")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
