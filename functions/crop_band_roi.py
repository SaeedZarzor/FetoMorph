from __future__ import annotations
"""
Crop brain slice PNGs to a region of interest (ROI).

This utility is intended to post-process the PNG outputs produced by the
area-band sampler (see `area_band_output_multi/.../axis_*/brain_slices`). It
extracts a rectangular ROI similar to the red box shown in the provided
example image. By default the ROI is computed automatically from the
non-background pixels in the right half of each slice (skipping the legend),
with a small margin, and is written to a sibling directory named
`brain_slices_cropped` under each axis folder. A manual normalized bounding box
can also be provided when a fixed crop is preferred.

Typical usage
-------------
python -m functions.crop_band_roi \
    --band-root area_band_output_multi/24week_3day_sub-CC00862XX13_ses-41210 \
    --axes axis_x axis_y axis_z

Use --bbox 0.5625 0.1875 0.25 0.296875 to force a fixed normalized crop (x,y,w,h in [0,1]).
Axis-specific bboxes can be set in code defaults (see CropBandConfig.bbox_norm_by_axis).
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from functions.Nifti2image import draw_new_scale_bar

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1 in pixel coords


@dataclass
class CropBandConfig:
    band_root: str
    axes: Sequence[str] = ("axis_x", "axis_y", "axis_z")
    glob_pattern: str = "band_axis*.png"
    bbox_norm: Optional[Tuple[float, float, float, float]] = None  # x, y, w, h in [0,1]
    bbox_norm_by_axis: Optional[Dict[str, Tuple[float, float, float, float]]] = None
    row_fraction: Tuple[float, float] = (0.0, 1.0)  # region of rows to search for content
    col_fraction: Tuple[float, float] = (0.45, 1.0)  # skip legend (default: right half)
    margin_frac: float = 0.05  # extra border relative to min(image width, height)
    margin_px: Optional[int] = None  # explicit pixel margin (overrides margin_frac)
    output_dirname: str = "brain_slices_cropped"
    overwrite: bool = False
    verbose: bool = True
    manifest_name: str = "crop_manifest.json"
    min_foreground_pixels: int = 32  # ignore tiny specks
    validate_manual_bbox: bool = True  # require some foreground in manual bbox
    fallback_to_auto: bool = True  # if manual bbox fails validation, try auto bbox
    allow_auto_for_unset_axes: bool = False  # safer default: do not auto-crop unset axes when per-axis bboxes are provided
    add_scale_bar: bool = False  # redraw scale bar on cropped outputs
    scale_bar_label: str = "20 mm"
    scale_bar_fallback_frac: float = 0.10  # legacy; retained for backward compatibility
    scale_bar_top_pad_frac: float = 0.22  # extra white space above cropped image
    scale_bar_top_pad_px: Optional[int] = None  # explicit top pad (overrides fraction)
    scale_bar_length_scale: float = 0.30  # scale bar length as fraction of cropped width
    scale_bar_font_scale_ratio: float = 0.6  # smaller scale-bar text


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _bbox_from_mask(mask: np.ndarray, img_w: int, img_h: int, margin: int) -> Optional[BBox]:
    """Compute bounding box of a binary mask with padding."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = int(xs.min()) - margin
    x1 = int(xs.max()) + 1 + margin
    y0 = int(ys.min()) - margin
    y1 = int(ys.max()) + 1 + margin
    x0 = _clamp(x0, 0, img_w)
    x1 = _clamp(x1, x0 + 1, img_w)
    y0 = _clamp(y0, 0, img_h)
    y1 = _clamp(y1, y0 + 1, img_h)
    return x0, y0, x1, y1


def _norm_bbox_to_pixels(norm_bbox: Tuple[float, float, float, float], w: int, h: int) -> BBox:
    nx, ny, nw, nh = norm_bbox
    x0 = int(round(nx * w))
    y0 = int(round(ny * h))
    x1 = int(round((nx + nw) * w))
    y1 = int(round((ny + nh) * h))
    x0 = _clamp(x0, 0, w - 1)
    y0 = _clamp(y0, 0, h - 1)
    x1 = _clamp(x1, x0 + 1, w)
    y1 = _clamp(y1, y0 + 1, h)
    return x0, y0, x1, y1


def _bbox_has_foreground(img: np.ndarray, bbox: BBox, min_pixels: int) -> bool:
    x0, y0, x1, y1 = bbox
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    fg_mask = np.any(roi[:, :, :3] > 0, axis=2)
    return int(fg_mask.sum()) >= int(min_pixels)


def _compute_auto_bbox(img: np.ndarray, cfg: CropBandConfig) -> Optional[BBox]:
    h, w = img.shape[:2]
    # Auto-detect from foreground content within a column/row window.
    r0 = _clamp(int(cfg.row_fraction[0] * h), 0, h)
    r1 = _clamp(int(cfg.row_fraction[1] * h), r0 + 1, h)
    c0 = _clamp(int(cfg.col_fraction[0] * w), 0, w)
    c1 = _clamp(int(cfg.col_fraction[1] * w), c0 + 1, w)

    window = img[r0:r1, c0:c1, :3]
    fg_mask = np.any(window > 0, axis=2)
    if int(fg_mask.sum()) < cfg.min_foreground_pixels:
        return None

    margin = cfg.margin_px if cfg.margin_px is not None else int(round(cfg.margin_frac * min(w, h)))
    bbox = _bbox_from_mask(fg_mask, img_w=w, img_h=h, margin=margin)
    if bbox is None:
        return None
    # Re-add window offset
    x0, y0, x1, y1 = bbox
    return x0 + c0, y0 + r0, x1 + c0, y1 + r0


def _compute_bbox(img: np.ndarray, cfg: CropBandConfig, axis: str) -> Optional[BBox]:
    h, w = img.shape[:2]
    manual_norm: Optional[Tuple[float, float, float, float]] = None
    if cfg.bbox_norm is not None:
        manual_norm = cfg.bbox_norm
    elif cfg.bbox_norm_by_axis and axis in cfg.bbox_norm_by_axis:
        manual_norm = cfg.bbox_norm_by_axis[axis]

    if manual_norm is not None:
        manual_bbox = _norm_bbox_to_pixels(manual_norm, w, h)
        if (not cfg.validate_manual_bbox) or _bbox_has_foreground(img, manual_bbox, cfg.min_foreground_pixels):
            return manual_bbox
        if not cfg.fallback_to_auto:
            return None

    return _compute_auto_bbox(img, cfg)


def _draw_scale_bar_on_crop(cropped: np.ndarray, _source_img: np.ndarray, cfg: CropBandConfig) -> np.ndarray:
    if not cfg.add_scale_bar:
        return cropped

    base = cropped[:, :, :3].copy() if (cropped.ndim == 3 and cropped.shape[2] >= 3) else cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    h, w = base.shape[:2]
    # Crop-local scale bar logic:
    # length is a direct fraction of cropped width (no detection from source image).
    bar_len_px = int(round(float(cfg.scale_bar_length_scale) * max(1, w)))
    bar_len_px = max(3, min(max(3, w - 2), bar_len_px))

    pad = cfg.scale_bar_top_pad_px if cfg.scale_bar_top_pad_px is not None else int(round(float(cfg.scale_bar_top_pad_frac) * max(1, h)))
    pad = max(0, int(pad))

    # Estimate geometry first so we can enforce enough top white space for
    # both the bar and the label ("20 mm") above the cropped content.
    base2 = max(1, min(h + max(1, pad), w))
    thickness = max(1, int(0.007 * base2))
    margin = max(4, int(0.08 * base2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 1
    fscale = (thickness / 10.0) * float(cfg.scale_bar_font_scale_ratio) + 0.25
    text = str(cfg.scale_bar_label)
    (text_w, text_h), _ = cv2.getTextSize(text, font, fscale, font_thickness)
    gap = max(3, int(0.35 * text_h))
    y_default = margin + thickness
    # Minimum padding needed to keep label fully in the white strip.
    needed_pad = y_default + gap + text_h + 4
    if pad < needed_pad:
        pad = int(needed_pad)

    if pad > 0:
        canvas = np.full((h + pad, w, 3), 255, dtype=np.uint8)
        canvas[pad:pad + h, :, :] = base
    else:
        canvas = base

    # Keep bar+text fully inside the canvas (top-right in the top padding strip).
    h2, w2 = canvas.shape[:2]

    # Start from top-right, but move left if needed so label fits too.
    x_default = max(0, w2 - margin - int(bar_len_px))
    x_text_fit = max(0, w2 - margin - int(text_w))
    x_left = max(0, min(x_default, x_text_fit))
    y_bottom = y_default if pad <= 0 else max(1, min(max(1, pad - (gap + text_h + 2)), y_default))

    return draw_new_scale_bar(
        canvas,
        int(bar_len_px),
        where=(int(x_left), int(y_bottom)),
        text=text,
        font_scale_ratio=float(cfg.scale_bar_font_scale_ratio),
    )


def _iter_images(src_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(src_dir.glob(pattern))


def _is_subject_root(root: Path, axes: Sequence[str]) -> bool:
    return any((root / ax / "brain_slices").is_dir() for ax in axes)


def _subject_roots(cfg: CropBandConfig) -> List[Path]:
    root = Path(cfg.band_root)
    if _is_subject_root(root, cfg.axes):
        return [root]
    subjects: List[Path] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and _is_subject_root(d, cfg.axes):
            subjects.append(d)
    return subjects


def _axis_has_manual_bbox(cfg: CropBandConfig, axis: str) -> bool:
    if cfg.bbox_norm is not None:
        return True
    return bool(cfg.bbox_norm_by_axis and axis in cfg.bbox_norm_by_axis)


def _crop_single_subject(cfg: CropBandConfig, root: Path) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    for axis in cfg.axes:
        if cfg.bbox_norm is None and cfg.bbox_norm_by_axis and (not cfg.allow_auto_for_unset_axes) and (not _axis_has_manual_bbox(cfg, axis)):
            if cfg.verbose:
                print(f"[crop] skip axis '{axis}' in {root.name}: bbox_norm_by_axis provided but no bbox for this axis")
            continue
        src_dir = root / axis / "brain_slices"
        if not src_dir.is_dir():
            if cfg.verbose:
                print(f"[crop] skip {src_dir} (missing)")
            continue
        dst_dir = src_dir.parent / cfg.output_dirname
        dst_dir.mkdir(parents=True, exist_ok=True)

        for img_path in _iter_images(src_dir, cfg.glob_pattern):
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                if cfg.verbose:
                    print(f"[crop] failed to read {img_path}")
                continue

            bbox = _compute_bbox(img, cfg, axis)
            if bbox is None:
                if cfg.verbose:
                    print(f"[crop] no foreground found for {img_path}, skipping")
                continue
            x0, y0, x1, y1 = bbox
            cropped = img[y0:y1, x0:x1]
            cropped = _draw_scale_bar_on_crop(cropped, img, cfg)

            out_path = dst_dir / img_path.name
            if out_path.exists() and not cfg.overwrite:
                if cfg.verbose:
                    print(f"[crop] exists {out_path}, use --overwrite to replace")
                continue
            ok = cv2.imwrite(str(out_path), cropped)
            if not ok:
                if cfg.verbose:
                    print(f"[crop] failed to write {out_path}")
                continue
            results.append({"subject": root.name, "axis": axis, "src": str(img_path), "dst": str(out_path), "bbox": f"{x0},{y0},{x1},{y1}"})
            if cfg.verbose:
                print(f"[crop] {img_path.name} -> {out_path.name} bbox=({x0},{y0},{x1},{y1})")
    return results


def crop_band_slices(cfg: CropBandConfig) -> List[Dict[str, str]]:
    subject_roots = _subject_roots(cfg)
    if not subject_roots:
        raise FileNotFoundError(f"No subject roots with axis folders found under: {cfg.band_root}")

    all_results: List[Dict[str, str]] = []
    for root in subject_roots:
        results = _crop_single_subject(cfg, root)
        all_results.extend(results)
        # Write a manifest per subject root, even if empty (useful for debugging skips).
        if cfg.manifest_name:
            manifest_path = root / cfg.manifest_name
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump({"config": asdict(cfg), "subject_root": str(root), "outputs": results}, f, indent=2)
                if cfg.verbose:
                    print(f"[crop] wrote manifest {manifest_path}")
            except Exception as exc:  # pragma: no cover
                if cfg.verbose:
                    print(f"[crop] failed to write manifest: {exc}")

    return all_results


def _parse_bbox_map(raw: Any) -> Optional[Dict[str, Tuple[float, float, float, float]]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("bbox_norm_by_axis must be an object like {'axis_x': [x,y,w,h], ...}")
    parsed: Dict[str, Tuple[float, float, float, float]] = {}
    for axis, val in raw.items():
        if val is None:
            continue
        if not isinstance(val, (list, tuple)) or len(val) != 4:
            raise ValueError(f"bbox_norm_by_axis[{axis}] must be [x,y,w,h] or null")
        parsed[str(axis)] = tuple(float(v) for v in val)  # type: ignore[assignment]
    return parsed or None


def load_crop_config_from_json(path: str) -> CropBandConfig:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return CropBandConfig(
        band_root=data["band_root"],
        axes=data.get("axes", ("axis_x", "axis_y", "axis_z")),
        glob_pattern=data.get("glob_pattern", "band_axis*.png"),
        bbox_norm=tuple(data["bbox_norm"]) if data.get("bbox_norm") else None,
        bbox_norm_by_axis=_parse_bbox_map(data.get("bbox_norm_by_axis")),
        row_fraction=tuple(data.get("row_fraction", (0.0, 1.0))),
        col_fraction=tuple(data.get("col_fraction", (0.45, 1.0))),
        margin_frac=float(data.get("margin_frac", 0.05)),
        margin_px=data.get("margin_px"),
        output_dirname=data.get("output_dirname", "brain_slices_cropped"),
        overwrite=bool(data.get("overwrite", False)),
        verbose=bool(data.get("verbose", True)),
        manifest_name=data.get("manifest_name", "crop_manifest.json"),
        min_foreground_pixels=int(data.get("min_foreground_pixels", 32)),
        validate_manual_bbox=bool(data.get("validate_manual_bbox", True)),
        fallback_to_auto=bool(data.get("fallback_to_auto", True)),
        allow_auto_for_unset_axes=bool(data.get("allow_auto_for_unset_axes", False)),
        add_scale_bar=bool(data.get("add_scale_bar", False)),
        scale_bar_label=str(data.get("scale_bar_label", "20 mm")),
        scale_bar_fallback_frac=float(data.get("scale_bar_fallback_frac", 0.10)),
        scale_bar_top_pad_frac=float(data.get("scale_bar_top_pad_frac", 0.22)),
        scale_bar_top_pad_px=(int(data["scale_bar_top_pad_px"]) if data.get("scale_bar_top_pad_px") is not None else None),
        scale_bar_length_scale=float(data.get("scale_bar_length_scale", 0.30)),
        scale_bar_font_scale_ratio=float(data.get("scale_bar_font_scale_ratio", 0.6)),
    )


def _parse_args() -> CropBandConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Crop area-band PNG slices to a fixed or auto-detected ROI.")
    parser.add_argument("--config", default=None, help="Path to JSON config for crop settings.")
    parser.add_argument("--band-root", required=False, default=None, help="Root directory that contains axis_x/axis_y/axis_z folders.")
    parser.add_argument("--axes", nargs="*", default=None, help="Axis folders to process.")
    parser.add_argument("--glob", default=None, dest="glob_pattern", help="Glob pattern for slice PNGs.")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, help="Manual normalized bbox x y w h in [0,1].")
    parser.add_argument("--row-frac", nargs=2, type=float, default=None, help="Row fraction window (start end) for auto bbox.")
    parser.add_argument("--col-frac", nargs=2, type=float, default=None, help="Col fraction window (start end) for auto bbox.")
    parser.add_argument("--margin-frac", type=float, default=None, help="Margin as fraction of min(w,h) when auto bbox.")
    parser.add_argument("--margin-px", type=int, default=None, help="Explicit margin in pixels (overrides fraction).")
    parser.add_argument("--output-dirname", default=None, help="Name of output directory per axis.")
    parser.add_argument("--manifest-name", default=None, help="Manifest filename at band root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cropped files.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logs.")
    parser.add_argument("--min-foreground-pixels", type=int, default=None, help="Minimum non-zero pixels to accept a bbox.")
    parser.add_argument("--no-manual-bbox-validation", action="store_true", help="Do not validate manual bbox foreground.")
    parser.add_argument("--no-fallback-to-auto", action="store_true", help="Disable fallback to auto bbox if manual bbox is empty.")
    parser.add_argument("--allow-auto-for-unset-axes", action="store_true",
                        help="Allow auto-crop for axes without bbox when bbox_norm_by_axis is used.")
    parser.add_argument("--add-scale-bar", action="store_true",
                        help="Redraw a scale bar on cropped outputs.")
    parser.add_argument("--scale-bar-label", default=None,
                        help="Scale bar text label (default: '20 mm').")
    parser.add_argument("--scale-bar-fallback-frac", type=float, default=None,
                        help="Legacy option (currently not used by crop-local scale bar logic).")
    parser.add_argument("--scale-bar-top-pad-frac", type=float, default=None,
                        help="Top white padding as fraction of cropped height.")
    parser.add_argument("--scale-bar-top-pad-px", type=int, default=None,
                        help="Top white padding in pixels (overrides fraction).")
    parser.add_argument("--scale-bar-length-scale", type=float, default=None,
                        help="Scale bar length as fraction of cropped image width (e.g. 0.3).")
    parser.add_argument("--scale-bar-font-scale-ratio", type=float, default=None,
                        help="Scale bar text size ratio (smaller means smaller text).")

    args = parser.parse_args()
    if args.config:
        cfg = load_crop_config_from_json(args.config)
        if args.band_root:
            cfg.band_root = args.band_root
        if args.axes:
            cfg.axes = args.axes
        if args.glob_pattern:
            cfg.glob_pattern = args.glob_pattern
        if args.bbox:
            cfg.bbox_norm = tuple(args.bbox)
        if args.row_frac:
            cfg.row_fraction = tuple(args.row_frac)
        if args.col_frac:
            cfg.col_fraction = tuple(args.col_frac)
        if args.margin_frac is not None:
            cfg.margin_frac = args.margin_frac
        if args.margin_px is not None:
            cfg.margin_px = args.margin_px
        if args.output_dirname:
            cfg.output_dirname = args.output_dirname
        if args.manifest_name:
            cfg.manifest_name = args.manifest_name
        if args.overwrite:
            cfg.overwrite = True
        if args.quiet:
            cfg.verbose = False
        if args.min_foreground_pixels is not None:
            cfg.min_foreground_pixels = args.min_foreground_pixels
        if args.no_manual_bbox_validation:
            cfg.validate_manual_bbox = False
        if args.no_fallback_to_auto:
            cfg.fallback_to_auto = False
        if args.allow_auto_for_unset_axes:
            cfg.allow_auto_for_unset_axes = True
        if args.add_scale_bar:
            cfg.add_scale_bar = True
        if args.scale_bar_label is not None:
            cfg.scale_bar_label = args.scale_bar_label
        if args.scale_bar_fallback_frac is not None:
            cfg.scale_bar_fallback_frac = args.scale_bar_fallback_frac
        if args.scale_bar_top_pad_frac is not None:
            cfg.scale_bar_top_pad_frac = args.scale_bar_top_pad_frac
        if args.scale_bar_top_pad_px is not None:
            cfg.scale_bar_top_pad_px = args.scale_bar_top_pad_px
        if args.scale_bar_length_scale is not None:
            cfg.scale_bar_length_scale = args.scale_bar_length_scale
        if args.scale_bar_font_scale_ratio is not None:
            cfg.scale_bar_font_scale_ratio = args.scale_bar_font_scale_ratio
        if not cfg.band_root:
            parser.error("band_root is required in config or via --band-root")
        return cfg

    if not args.band_root:
        parser.error("--band-root is required when --config is not provided")

    return CropBandConfig(
        band_root=args.band_root,
        axes=args.axes if args.axes else ("axis_x", "axis_y", "axis_z"),
        glob_pattern=args.glob_pattern or "band_axis*.png",
        bbox_norm=tuple(args.bbox) if args.bbox else None,
        bbox_norm_by_axis=None,
        row_fraction=tuple(args.row_frac) if args.row_frac else (0.0, 1.0),
        col_fraction=tuple(args.col_frac) if args.col_frac else (0.45, 1.0),
        margin_frac=args.margin_frac if args.margin_frac is not None else 0.05,
        margin_px=args.margin_px,
        output_dirname=args.output_dirname or "brain_slices_cropped",
        overwrite=args.overwrite,
        verbose=not args.quiet,
        manifest_name=args.manifest_name or "crop_manifest.json",
        min_foreground_pixels=args.min_foreground_pixels if args.min_foreground_pixels is not None else 32,
        validate_manual_bbox=not args.no_manual_bbox_validation,
        fallback_to_auto=not args.no_fallback_to_auto,
        allow_auto_for_unset_axes=args.allow_auto_for_unset_axes,
        add_scale_bar=args.add_scale_bar,
        scale_bar_label=args.scale_bar_label if args.scale_bar_label is not None else "20 mm",
        scale_bar_fallback_frac=args.scale_bar_fallback_frac if args.scale_bar_fallback_frac is not None else 0.10,
        scale_bar_top_pad_frac=args.scale_bar_top_pad_frac if args.scale_bar_top_pad_frac is not None else 0.22,
        scale_bar_top_pad_px=args.scale_bar_top_pad_px,
        scale_bar_length_scale=args.scale_bar_length_scale if args.scale_bar_length_scale is not None else 0.30,
        scale_bar_font_scale_ratio=args.scale_bar_font_scale_ratio if args.scale_bar_font_scale_ratio is not None else 0.6,
    )


def main() -> None:
    cfg = _parse_args()
    crop_band_slices(cfg)


if __name__ == "__main__":
    main()
