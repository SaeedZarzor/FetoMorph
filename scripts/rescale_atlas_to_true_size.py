"""Put the dHCP atlas's *clean* parcellation back at each week's true brain size.

Why this exists
---------------
The atlas ships two parcellation roots and neither is directly usable for
size-aware slice measurement:

- `parcellations_scaled` is clean - one connected component per structure - but
  has brain size normalised out (bbox flat at about 81x102x81 mm on every week),
  so absolute area and perimeter carry no growth signal.
- `parcellations_orig_size` preserves true size but is a badly
  nearest-neighbour-warped parcellation. At GW36 it holds **66,413 connected
  components across 12 labels**, 63,305 of them 8 voxels or smaller, against 12
  larger than 1000. Its isolated-voxel rate is 35-65 per mille on all 16 weeks
  versus 0.03-0.10 in the scaled root, and it carries 3,265 interior background
  hole components against 22. Those specks and holes are the visible dots and
  white lines, and they inflate the small midline structures 1.7-2.7x: cavum
  septum pellucidum arrives as 2,833 pieces whose largest holds just 35.7% of
  its volume, third ventricle as 1,032 pieces / 37.1%.

Filtering the speckle out does not work and should not be attempted. A 3x3x3
majority vote costs CSP 45.7% of its volume, third ventricle 34.7% and lateral
ventricle L 20.5% - they are thin sheets, outvoted by the surrounding white
matter - and drags GW36 coronal LGI from 1.749 to 1.581. Surgical speck removal
(dissolve components under N voxels, refill from the nearest label) is no better:
CSP still drifts 41-53% at every threshold tried, because roughly 64% of its
voxels genuinely *are* specks. There is nothing to recover locally.

What this script does instead
-----------------------------
The two roots turn out to be related by a near-pure per-axis scale about the
brain centre, so the clean labels can simply be resampled to the true size. Dice
against the orig_size brain runs 0.9898 at GW21 to 0.9953 at GW36. At GW36 the
result has 92 components instead of 66,413, matches the orig_size extent exactly,
lands within 1.3% on area, and moves LGI only 1.749 -> 1.735. At GW21 LGI goes
1.382 -> 1.299, that drop being the speckle-inflated perimeter coming off.

So shape comes from the scaled atlas and size from orig_size. State that
wherever the numbers are reported - it is a real methodological choice, not a
repair of the original-size volumes.

Two details that matter:

- The scale is measured from a **debris-free** mask (largest connected
  component, holes filled). Measured on the raw mask, the floating specks
  scattered around the orig_size brain inflate its bounding box and the scale
  comes out wrong.
- Resampling is nearest-neighbour (`order=0`). Any smooth interpolation would
  average label ids, which is exactly the defect that ruins GW27/GW28 in
  `parcellations_orig_size` - averaging cortical GM (3) with fetal WM (5) yields
  Cortical GM Right (4), a different tissue on the wrong side.

`parcellations_scaled` has no GW22, so no clean source for that week exists and
it is skipped rather than silently falling back to the speckled volume. This
yields 15 weeks against the raw orig_size root's 16.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

ORIG_GLOB = "transformed-t*.nii.gz"
SCALED_GLOB = "tissue-t*_dhcp-19.nii.gz"
OUT_SUFFIX = "_truesize"


def _parse_week(name: str) -> int | None:
    m = re.fullmatch(r"(?:GW)?(\d{2})", name.strip())
    return int(m.group(1)) if m else None


def _legend_ids(path: str | None) -> set[int]:
    if not path or not os.path.isfile(path):
        return set()
    ids: set[int] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                ids.add(int(str(row[0]).strip()))
            except ValueError:
                pass
    return {i for i in ids if i > 0}


def _load_labels(path: str) -> np.ndarray:
    """Read a label volume as integer ids, squeezing the orig_size singleton axis.

    Rounding matters for GW27/GW28 of `parcellations_orig_size`, which are
    float64 with interpolated values and a denormal-noise background; here the
    volume is only ever used to measure a bounding box, so the boundary
    ambiguity that makes those weeks unreliable for measurement is harmless.
    """
    img = nib.load(path)
    data = np.asanyarray(img.dataobj, dtype=np.float64)
    if data.ndim > 3:
        if any(d != 1 for d in data.shape[3:]):
            raise ValueError(f"{os.path.basename(path)}: genuinely 4D, shape {data.shape}")
        data = np.squeeze(data, axis=tuple(range(3, data.ndim)))
    out = np.rint(data).astype(np.int16)
    out[data < 0.5] = 0
    return out


def _brain_mask(vol: np.ndarray) -> np.ndarray:
    """Largest connected component with holes filled.

    The raw orig_size volumes carry floating debris around the brain. Measured
    on `vol > 0` the bounding box takes in that debris and the recovered scale is
    wrong, so the mask has to be reduced to the brain proper first.
    """
    m = vol > 0
    lb, n = ndimage.label(m)
    if n > 1:
        sizes = np.bincount(lb.ravel())
        sizes[0] = 0
        m = lb == int(sizes.argmax())
    return ndimage.binary_fill_holes(m)


def _bbox(mask: np.ndarray):
    nz = np.argwhere(mask)
    return nz.min(0), nz.max(0)


def rescale_one(orig_path: str, scaled_path: str, dst: str, legend: set[int]) -> dict:
    orig = _load_labels(orig_path)
    scaled = _load_labels(scaled_path)
    if orig.shape != scaled.shape:
        raise ValueError(
            f"shape mismatch: {orig.shape} (orig) vs {scaled.shape} (scaled); "
            f"this script assumes both roots share one voxel grid"
        )

    src_img = nib.load(scaled_path)
    orig_img = nib.load(orig_path)
    if not np.allclose(src_img.affine, orig_img.affine, atol=1e-4):
        raise ValueError(
            "affines differ between the two roots; the resample here is a pure "
            "voxel-grid operation and assumes they match"
        )

    m_orig = _brain_mask(orig)
    m_scaled = _brain_mask(scaled)
    lo_o, hi_o = _bbox(m_orig)
    lo_s, hi_s = _bbox(m_scaled)
    scale = (hi_o - lo_o + 1) / (hi_s - lo_s + 1)
    c_o = (lo_o + hi_o) / 2.0
    c_s = (lo_s + hi_s) / 2.0

    # Pull-resample: for each output (orig-space) voxel, read the scaled volume
    # at centre_scaled + (x - centre_orig) / scale. Nearest-neighbour only.
    mat = np.diag(1.0 / scale)
    offset = c_s - mat @ c_o
    out = ndimage.affine_transform(
        scaled, mat, offset=offset, order=0,
        output_shape=orig.shape, mode="constant", cval=0,
    ).astype(np.int16)

    present = {int(v) for v in np.unique(out) if v != 0}
    undescribed = sorted(present - legend) if legend else []
    if undescribed:
        raise ValueError(
            f"resampled volume carries label(s) the legend does not describe: "
            f"{undescribed}"
        )
    lost = sorted({int(v) for v in np.unique(scaled) if v != 0} - present)
    if lost:
        raise ValueError(
            f"label(s) {lost} present in the scaled source vanished from the "
            f"resampled volume - the downsample has erased a structure"
        )

    m_out = _brain_mask(out)
    dice = float(
        2 * np.logical_and(m_orig, m_out).sum() / (m_orig.sum() + m_out.sum())
    )

    if int(out.max()) > 255:
        raise ValueError(f"label id {int(out.max())} exceeds uint8")
    img = nib.Nifti1Image(out.astype(np.uint8), orig_img.affine)
    img.header.set_zooms(tuple(float(z) for z in orig_img.header.get_zooms()[:3]))
    # A residual scl_slope/scl_inter would rescale the integer ids back into
    # floats on read, reintroducing the interpolated-label bug by another route.
    img.header.set_slope_inter(None, None)
    img.header["descrip"] = b"dhcp-19 labels from scaled, rescaled to true size"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    nib.save(img, dst)

    def ncomp(vol):
        return int(sum(ndimage.label(vol == l)[1] for l in present))

    nz = np.argwhere(m_out)
    zooms = np.array(orig_img.header.get_zooms()[:3], dtype=float)

    return {
        "orig_source": os.path.basename(orig_path),
        "scaled_source": os.path.basename(scaled_path),
        "scale_factors": [round(float(v), 4) for v in scale],
        "dice_vs_origsize_brain": round(dice, 4),
        "labels_present": sorted(present),
        "components_origsize": ncomp(orig),
        "components_out": ncomp(out),
        "voxels": int((out > 0).sum()),
        "bbox_extent_mm": [round(float(v), 1) for v in ((nz.max(0) - nz.min(0) + 1) * zooms)],
        "bbox_extent_mm_origsize": [
            round(float(v), 1) for v in ((hi_o - lo_o + 1) * zooms)
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resample the clean parcellations_scaled labels to each "
                    "week's true brain size, measured from parcellations_orig_size."
    )
    ap.add_argument("--scaled-root", required=True,
                    help="parcellations_scaled folder (case folders GW21..GW36)")
    ap.add_argument("--orig-root", required=True,
                    help="parcellations_orig_size folder (case folders 21..36); "
                         "used only to measure true brain size")
    ap.add_argument("--dst", required=True, help="output root, case folders GW21..GW36")
    ap.add_argument("--label-legend", default=None,
                    help="dhcp-19 legend csv, used to verify the resampled label ids")
    ap.add_argument("--manifest-name", default="truesize_manifest.json")
    args = ap.parse_args()

    for tag, path in (("--scaled-root", args.scaled_root), ("--orig-root", args.orig_root)):
        if not os.path.isdir(path):
            ap.error(f"{tag} not found: {path}")

    legend = _legend_ids(args.label_legend)
    if args.label_legend and not legend:
        print(f"warning: no label ids read from {args.label_legend}; label check "
              f"skipped", file=sys.stderr)

    def index(root, pattern):
        found = {}
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            wk = _parse_week(name)
            if wk is None:
                continue
            matches = sorted(glob.glob(os.path.join(d, pattern)))
            # Several orig_size case folders also hold brain_extracted.nii.gz,
            # which is why the globs are name-specific rather than *.nii.gz.
            if len(matches) != 1:
                raise ValueError(
                    f"{d}: expected exactly 1 file matching '{pattern}', found "
                    f"{len(matches)}: {[os.path.basename(m) for m in matches]}"
                )
            found[wk] = matches[0]
        return found

    origs = index(args.orig_root, ORIG_GLOB)
    scaleds = index(args.scaled_root, SCALED_GLOB)

    weeks = sorted(set(origs) & set(scaleds))
    skipped = sorted(set(origs) - set(scaleds))
    if not weeks:
        ap.error("no week is present in both roots")

    print(f"{len(weeks)} weeks -> {args.dst}")
    if skipped:
        print(f"skipping GW{', GW'.join(str(w) for w in skipped)}: no clean source "
              f"in {os.path.basename(args.scaled_root)}, and falling back to the "
              f"speckled volume would mix two data qualities in one root\n")
    print(f"{'week':<6}{'scale (x,y,z)':<26}{'dice':>7}{'comps in':>10}"
          f"{'comps out':>11}  extent mm")
    entries = {}
    for wk in weeks:
        dst = os.path.join(
            args.dst, f"GW{wk}",
            os.path.basename(scaleds[wk]).replace(".nii.gz", f"{OUT_SUFFIX}.nii.gz"),
        )
        info = rescale_one(origs[wk], scaleds[wk], dst, legend)
        entries[f"GW{wk}"] = info
        print(f"GW{wk:<4}{str(info['scale_factors']):<26}"
              f"{info['dice_vs_origsize_brain']:>7.4f}"
              f"{info['components_origsize']:>10}{info['components_out']:>11}  "
              f"{info['bbox_extent_mm']}")

    manifest = {
        "scaled_root": os.path.abspath(args.scaled_root),
        "orig_root": os.path.abspath(args.orig_root),
        "output_root": os.path.abspath(args.dst),
        "seg_glob": f"tissue-*{OUT_SUFFIX}.nii.gz",
        "method": (
            "Labels come from parcellations_scaled, which is clean (one connected "
            "component per structure). Size comes from parcellations_orig_size, "
            "measured as the per-axis bounding-box ratio of a debris-free brain "
            "mask (largest connected component, holes filled). Resampling is "
            "nearest-neighbour about the brain centre, so no label id is ever "
            "averaged."
        ),
        "why_not_origsize_directly": (
            "parcellations_orig_size is nearest-neighbour-warped and speckled: "
            "66,413 connected components across 12 labels at GW36 against 12 "
            "larger than 1000 voxels, an isolated-voxel rate 1000x the scaled "
            "root on every week, and 3,265 interior hole components against 22. "
            "The small midline structures are shattered and volume-inflated "
            "1.7-2.7x (CSP's largest piece holds 35.7% of it). Majority-vote and "
            "speck-removal filtering both cost CSP 35-50% of its volume, because "
            "about 64% of its voxels are specks."
        ),
        "interpretation": (
            "Shape is the scaled atlas's; size is the original-size atlas's. "
            "Report both facts alongside any number taken from this root. Do not "
            "mix this root with a raw parcellations_orig_size run in one workbook."
        ),
        "skipped_weeks": [f"GW{w}" for w in skipped],
        "cases": entries,
    }
    os.makedirs(args.dst, exist_ok=True)
    mpath = os.path.join(args.dst, args.manifest_name)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest: {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
