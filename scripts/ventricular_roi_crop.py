import argparse
import json
import os
import sys
from typing import List

import numpy as np
import cv2

# Make sure "functions" package is importable, same trick as area_band_cli.py
sys.path.insert(0, os.getcwd())

from functions.nifti_area_sampler import (  # type: ignore
    NiftiAreaSampler,
    load_config_from_json,
)


def build_roi_mask_for_slice(
    sampler: NiftiAreaSampler,
    axis: int,
    idx: int,
    roi_labels: List[int],
) -> np.ndarray:
    """
    Return a 2D boolean mask for the given slice index that is True exactly
    where the segmentation label is in roi_labels.
    """
    if axis == 0:
        lab2d = sampler.image_data[idx, :, :]
    elif axis == 1:
        lab2d = sampler.image_data[:, idx, :]
    else:
        lab2d = sampler.image_data[:, :, idx]

    lab2d = np.rint(lab2d).astype(np.int32, copy=False)
    mask = np.isin(lab2d, roi_labels)
    return mask


def colorize_slice(
    sampler: NiftiAreaSampler,
    axis: int,
    idx: int,
    roi_only: bool,
    roi_labels: List[int],
) -> np.ndarray:
    """
    Create an RGB image for the given slice index.

    - If roi_only is True, only voxels with labels in roi_labels are colored.
    - Otherwise, all non-zero labels are colored, but the crop box will still
      be determined by roi_labels.
    """
    if axis == 0:
        lab2d = sampler.image_data[idx, :, :]
    elif axis == 1:
        lab2d = sampler.image_data[:, idx, :]
    else:
        lab2d = sampler.image_data[:, :, idx]

    lab2d = np.rint(lab2d).astype(np.int32, copy=False)
    H, W = lab2d.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    if roi_only:
        labels_to_color = roi_labels
    else:
        labels_to_color = [int(u) for u in np.unique(lab2d) if u != 0]

    if not labels_to_color:
        return rgb

    labels_to_color = sorted(set(labels_to_color))
    for lab in labels_to_color:
        mask = (lab2d == lab)
        if not np.any(mask):
            continue
        color = NiftiAreaSampler._label_color_bgr(int(lab))  # static method
        rgb[mask] = np.array(color, dtype=np.uint8)

    return rgb


def crop_with_margin(mask: np.ndarray, margin: int) -> tuple[int, int, int, int] | None:
    """
    Given a 2D boolean mask, return (row_min, row_max, col_min, col_max)
    defining the crop region with the specified pixel margin.

    Returns None if the mask is empty.
    """
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None

    r_min, r_max = int(ys.min()), int(ys.max())
    c_min, c_max = int(xs.min()), int(xs.max())

    if margin > 0:
        r_min -= margin
        r_max += margin
        c_min -= margin
        c_max += margin

    return r_min, r_max, c_min, c_max


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract ventricular ROI crops from area-band slices"
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to AreaBandConfig JSON (e.g. area_band_config.json)",
    )
    ap.add_argument(
        "--summary",
        help="Path to area_band_summary.json; "
             "default: <out_dir>/area_band_summary.json from config",
    )
    ap.add_argument(
        "--roi-labels",
        nargs="+",
        type=int,
        required=True,
        help="Label IDs that define the ventricular ROI (e.g. 4 14 15)",
    )
    ap.add_argument(
        "--margin",
        type=int,
        default=5,
        help="Pixel margin around the ROI bounding box (default: 5)",
    )
    ap.add_argument(
        "--roi-only",
        action="store_true",
        help="Color only roi-label voxels (otherwise color all labels)",
    )
    ap.add_argument(
        "--out-subdir",
        default="ventricular_crops",
        help="Subfolder under config.out_dir to store crops",
    )
    args = ap.parse_args()

    # Load the same config you used for the area-band sampler
    cfg = load_config_from_json(args.config)
    sampler = NiftiAreaSampler(cfg)  # loads NIfTI, normalizes axis, etc.

    axis = sampler.axis  # already normalized to 0/1/2
    seg_shape = sampler.shape

    # Locate the summary JSON from the area-band run
    summary_path = args.summary or os.path.join(cfg.out_dir, "area_band_summary.json")
    if not os.path.isfile(summary_path):
        ap.error(f"area_band_summary.json not found at: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    indices = summary.get("indices") or []
    if not indices:
        print("[ventricular_roi_crop] No indices found in summary; nothing to do.")
        return 0

    out_dir = os.path.join(cfg.out_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[ventricular_roi_crop] Using {len(indices)} slices; "
          f"axis={axis}, roi_labels={args.roi_labels}")
    print(f"[ventricular_roi_crop] Output directory: {out_dir}")

    H_full = seg_shape[(axis + 1) % 3]  # height in slice plane
    W_full = seg_shape[(axis + 2) % 3]  # width in slice plane

    for idx in indices:
        idx = int(idx)

        # 1) ROI mask for this slice
        roi_mask = build_roi_mask_for_slice(
            sampler=sampler,
            axis=axis,
            idx=idx,
            roi_labels=args.roi_labels,
        )

        if not np.any(roi_mask):
            # No ventricular labels on this slice; skip.
            continue

        # 2) Bounding box with margin
        bbox = crop_with_margin(roi_mask, margin=args.margin)
        if bbox is None:
            continue
        r_min, r_max, c_min, c_max = bbox

        # Clamp to image bounds
        r_min = max(0, r_min)
        c_min = max(0, c_min)
        r_max = min(H_full - 1, r_max)
        c_max = min(W_full - 1, c_max)

        if r_max <= r_min or c_max <= c_min:
            continue

        # 3) Colorized slice and crop
        full_rgb = colorize_slice(
            sampler=sampler,
            axis=axis,
            idx=idx,
            roi_only=args.roi_only,
            roi_labels=args.roi_labels,
        )
        crop_rgb = full_rgb[r_min : r_max + 1, c_min : c_max + 1, :]

        # 4) Save PNG
        out_name = f"ventricular_axis{axis}_{idx:04d}.png"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, crop_rgb)
        print(f"  saved {out_path}")

    print("[ventricular_roi_crop] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
