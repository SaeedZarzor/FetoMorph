"""
NIfTI Area Band Sampler
-----------------------

This module provides tools to sample n 2D slices along a chosen axis of a
3D NIfTI label volume within the top p% region of the unimodal area profile.

Key features
- Single-slice 2D area evaluation (cm^2) at a relative position [0,1].
- Golden-section search to locate the maximum area slice.
- Threshold crossings (binary search) to get left/right bounds where
  area == p * max.
- Sampling of n equally spaced slices within [left, right].
- Optional label overlays with deterministic colors.
- Per-slice perimeter, convex/outer perimeter (via morphological closing),
  and LGI proxy (inner/outer).
- Rich logging: JSON summary, CSV/Excel tables, and a full area profile plot.

Typical usage
>>> from functions.nifti_area_sampler import AreaBandConfig, NiftiAreaSampler
>>> cfg = AreaBandConfig(file_path="seg.nii.gz", out_dir="out", axis="z", n=10, p=0.8)
>>> result = NiftiAreaSampler(cfg).sample_band()

See functions/nifti_area_sampler.md for a detailed guide.
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np

try:
    import cv2  # type: ignore[import]
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "OpenCV (cv2) is required but could not be imported. "
        "Please install 'opencv-python-headless' in your environment."
    )

import nibabel as nib
from functions.pial_overlay_tri import load_pial_vertices_to_vox, draw_pial_on_slice
from functions.Nifti2image import nifti_slice_to_image


AxisLike = int | str


@dataclass
class AreaBandConfig:
    """Configuration for area band sampling.

    - file_path: Path to the input NIfTI label map.
    - out_dir: Directory where logs and PNGs are written.
    - axis: One of 'x'|'y'|'z' or 0|1|2 (slice normal to that axis).
    - n: Number of slices to sample within the top-p band.
    - p: Fraction of the max area (0..1) to define the band [left,right].
    - min_contour_area: Minimum 2D contour area (in pixels) to keep.
    - tol_max: Golden-section search tolerance for locating max.
    - tol_thr: Threshold bisection tolerance for left/right crossings.
    - save_png: Save annotated slice images.
    - snap_to_valid: If selected slice is empty, snap to nearest containing mask.
    - valid_labels: Subset of labels to include (None means all nonzero labels).
    - overlay_labels: Whether to overlay colored label regions.
    - label_alpha: Alpha blending strength for label overlays (0..1).
    - kernel_size: Morph closing kernel size for convex perimeter.
    - profile_plot: Generate area-vs-index debug plot and CSV.
    """
    file_path: str
    out_dir: str
    axis: AxisLike = "z"              # 'x'|'y'|'z' or 0|1|2
    n: int = 10                        # number of slices to sample
    p: float = 0.8                     # fraction of max area (0..1)
    min_contour_area: float = 30.0     # min 2D area (px) to keep a contour
    tol_max: float = 1e-3              # tolerance for golden-section search
    tol_thr: float = 1e-4              # tolerance for threshold bisection
    save_png: bool = True              # save annotated slice images
    snap_to_valid: bool = False        # snap empty slice to nearest with mask
    valid_labels: Optional[Sequence[int]] = None  # labels to include (None -> use all nonzero)
    overlay_labels: bool = True        # overlay per-voxel labels with colors
    label_alpha: float = 1.0           # alpha for label overlay [0..1]; 1.0 matches Qt solid colors
    kernel_size: int = 5               # morphological closing kernel for outer perimeter
    profile_plot: bool = True          # generate area-vs-index debug plot
    show_crosshair: bool = True        # draw crosshair at other-axes maxima
    area_labels: Optional[Sequence[int]] = None  # labels used for area/overlay
    draw_contours: bool = False        # draw segmentation contours on PNGs
    # Pial overlay
    use_pial_overlay: bool = False
    pial_lh_path: Optional[str] = None
    pial_rh_path: Optional[str] = None
    pial_space: str = "scanner"       # 'scanner' or 'tkr' (tkReg RAS)
    pial_tolerance_mm: float = 0.2
    pial_line_thickness: Optional[float] = 1.0
    pial_lh_color_bgr: Tuple[int, int, int] = (0, 255, 0)  # green
    pial_rh_color_bgr: Tuple[int, int, int] = (0, 165, 255)  # orange


class NiftiAreaSampler:
    """Compute and sample a top-p% area band in a NIfTI label volume.

    Loads the NIfTI, builds a boolean mask from valid labels, and exposes
    methods to compute the area at arbitrary relative positions, find the
    maximum area location, determine top-p% band bounds, and sample slices.

    Returns a rich result dict and writes out logs/PNGs as requested.
    """
    def __init__(self, cfg: AreaBandConfig):
        self.cfg = cfg

        # Normalize axis
        self.axis = self._normalize_axis(cfg.axis)

        # Load NIfTI
        nii = nib.load(cfg.file_path)
        nii = nib.as_closest_canonical(nii)
        self.image_data = nii.get_fdata()
        self.affine = nii.affine
        self.header = nii.header
        self.spacing = tuple(float(x) for x in self.header.get_zooms()[:3])  # (sx, sy, sz) mm

        # Build mask from labels selection preference
        # Priority: area_labels > valid_labels > any non-zero
        labels_sel = None
        if cfg.area_labels is not None and len(cfg.area_labels) > 0:
            labels_sel = list({int(x) for x in cfg.area_labels})
        elif cfg.valid_labels is not None and len(cfg.valid_labels) > 0:
            labels_sel = list({int(x) for x in cfg.valid_labels})

        if labels_sel:
            self.mask = np.isin(self.image_data, labels_sel)
        else:
            # Any nonzero voxel counts as mask
            self.mask = (self.image_data != 0)

        self.mask = self.mask.astype(bool)
        self.shape = self.mask.shape  # (nx, ny, nz)

        try:
            arr_i = np.rint(self.image_data).astype(np.int32, copy=False)
            uniq = np.unique(arr_i)
            uniq = uniq[uniq != 0]
            # Constrain to selected labels if provided
            if cfg.area_labels:
                sel = set(int(x) for x in cfg.area_labels)
                uniq = np.array([u for u in uniq if int(u) in sel], dtype=np.int32)
            elif cfg.valid_labels:
                sel = set(int(x) for x in cfg.valid_labels)
                uniq = np.array([u for u in uniq if int(u) in sel], dtype=np.int32)
            self.global_labels: List[int] = sorted(set(int(u) for u in uniq))
        except Exception:
            self.global_labels = []

        # Precompute in-plane pixel area (mm^2) for each axis-normal slice
        sx, sy, sz = self.spacing
        # axis 0 => plane (y,z) => area per pixel = sy*sz
        # axis 1 => plane (x,z) => area per pixel = sx*sz
        # axis 2 => plane (x,y) => area per pixel = sx*sy
        self.plane_area_mm2 = {0: sy * sz, 1: sx * sz, 2: sx * sy}
        # In-plane spacings (mm) per axis for perimeter scaling
        self.inplane_spacings: Dict[int, Tuple[float, float]] = {
            0: (sy, sz),  # axis 0 slice → plane (y,z)
            1: (sx, sz),  # axis 1 slice → plane (x,z)
            2: (sx, sy),  # axis 2 slice → plane (x,y)
        }

        # Cache for slice areas keyed by (axis, idx)
        self._area_cache: Dict[Tuple[int, int], float] = {}

        # Output dirs
        if cfg.save_png:
            os.makedirs(cfg.out_dir, exist_ok=True)
            os.makedirs(os.path.join(cfg.out_dir, "brain_slices"), exist_ok=True)

        # Optional: load pial surfaces and map to voxel indices
        self._pial_vox: Dict[str, Optional[np.ndarray]] = {"lh": None, "rh": None}
        if cfg.use_pial_overlay:
            self._pial_vox = load_pial_vertices_to_vox(cfg=self.cfg, affine=self.affine, shape=self.shape, file_path=self.cfg.file_path)

    def _add_scale_bar(self, img: np.ndarray, bar_length_mm: float = 20.0) -> np.ndarray:
        """Draw a metric scale bar (e.g. 20 mm) in the lower-right corner.

        The bar length in pixels is derived from the in-plane voxel spacing
        for the current slicing axis.
        """
        if bar_length_mm <= 0:
            return img

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return img

        sx, sy = self.inplane_spacings[self.axis]
        # Use the first non-zero spacing as reference; fall back to a
        # reasonable fixed pixel length if spacing metadata is odd.
        ref = sx if sx > 0 else sy
        if ref > 0:
            px_per_mm = 1.0 / float(ref)
            bar_px = int(round(bar_length_mm * px_per_mm))
        else:
            bar_px = 0

        # Clamp bar length so it is always clearly visible
        min_bar = int(0.1 * w)
        max_bar = int(0.4 * w)
        if bar_px <= 0:
            bar_px = min_bar
        bar_px = max(min_bar, min(max_bar, bar_px))

        # Larger margin to move the bar/text up, closer to the main content
        margin = int(max(10, 0.12 * min(h, w)))
        thickness = max(2, int(0.006 * min(h, w)))
        y = h - margin
        x2 = w - margin
        x1 = max(margin, x2 - bar_px)

        color = (255, 255, 255)
        cv2.line(img, (x1, y), (x2, y), color, thickness)

        label = f"{int(bar_length_mm)} mm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 if min(h, w) < 256 else 0.6
        text_thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        text_x = max(margin, x2 - text_w)
        text_y = max(text_h + margin, y - int(1.5 * thickness))
        cv2.putText(img, label, (text_x, text_y), font, font_scale, color, text_thickness, cv2.LINE_AA)

        return img

    # ----- Axis-generic helpers -----
    def _area_at_index_axis(self, ax: int, idx: int) -> float:
        """Area (cm^2) for a given axis and integer slice index."""
        orig = self.axis
        try:
            self.axis = ax
            mask = self._extract_slice(idx)
            if not np.any(mask):
                return 0.0
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = [c for c in contours if cv2.contourArea(c) > float(self.cfg.min_contour_area)]
            px_area_sum = float(sum(cv2.contourArea(c) for c in filtered))
            return (px_area_sum * self.plane_area_mm2[ax]) / 100.0
        finally:
            self.axis = orig

    def _area_profile_for_axis(self, ax: int) -> Tuple[List[int], List[float]]:
        n = self.shape[ax]
        idxs = list(range(n))
        vals = [self._area_at_index_axis(ax, i) for i in idxs]
        return idxs, vals

    def _golden_section_max_for_axis(self, ax: int, tol: float) -> Tuple[float, float]:
        orig = self.axis
        try:
            self.axis = ax
            return self.golden_section_max(0.0, 1.0, tol)
        finally:
            self.axis = orig

    @staticmethod
    def _label_color_bgr(lab: int) -> Tuple[int, int, int]:
        # Match FetoMorph._color_for_label (HSV with golden ratio), but return BGR for OpenCV.
        import colorsys
        hue = (lab * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        return int(b * 255), int(g * 255), int(r * 255)

    def _overlay_labels(self, base_bgr: np.ndarray, idx: int) -> np.ndarray:
        """Overlay colored labels for the current slice index onto base_bgr.
        Uses self.image_data to extract label IDs per voxel.
        """
        if self.axis == 0:
            slice_labels = self.image_data[idx, :, :]
        elif self.axis == 1:
            slice_labels = self.image_data[:, idx, :]
        else:
            slice_labels = self.image_data[:, :, idx]

        # Round to nearest int labels
        lab2d = np.rint(slice_labels).astype(np.int32, copy=False)
        # Exclude background 0
        uniques_slice = np.unique(lab2d)
        uniques_slice = uniques_slice[uniques_slice != 0]
        # Constrain overlays to selected labels if provided
        if self.cfg.area_labels:
            valid = set(int(x) for x in self.cfg.area_labels)
            uniques_slice = np.array([u for u in uniques_slice if int(u) in valid], dtype=np.int32)
        elif self.cfg.valid_labels:
            valid = set(int(x) for x in self.cfg.valid_labels)
            uniques_slice = np.array([u for u in uniques_slice if int(u) in valid], dtype=np.int32)

        if uniques_slice.size == 0:
            return base_bgr

        overlay = base_bgr.copy()
        alpha = float(np.clip(self.cfg.label_alpha, 0.0, 1.0))
        # Color only labels present in this slice
        for lab in uniques_slice.tolist():
            color = self._label_color_bgr(int(lab))
            mask = (lab2d == lab)
            if not np.any(mask):
                continue
            if alpha >= 0.999:
                # Match Qt solid fill: assign exact color (BGR)
                overlay[mask] = np.array(color, dtype=np.uint8)
            else:
                # Alpha blend when explicitly requested
                overlay[mask] = (
                    (1 - alpha) * overlay[mask].astype(np.float32)
                    + alpha * np.array(color, dtype=np.float32)
                ).astype(np.uint8)

        return overlay


    @staticmethod
    def _normalize_axis(axis: AxisLike) -> int:
        if isinstance(axis, str):
            m = {"x": 0, "y": 1, "z": 2}
            ax = m.get(axis.lower())
            if ax is None:
                raise ValueError("axis must be one of 'x','y','z' or 0,1,2")
            return ax
        ax = int(axis)
        if ax not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        return ax

    def _slice_has_mask(self, idx: int) -> bool:
        if self.axis == 0:
            return np.any(self.mask[idx, :, :])
        if self.axis == 1:
            return np.any(self.mask[:, idx, :])
        return np.any(self.mask[:, :, idx])

    def _extract_slice(self, idx: int) -> np.ndarray:
        if self.axis == 0:
            return self.mask[idx, :, :].astype(np.uint8)
        if self.axis == 1:
            return self.mask[:, idx, :].astype(np.uint8)
        return self.mask[:, :, idx].astype(np.uint8)

    def _idx_from_pos(self, pos: float) -> int:
        pos = float(np.clip(pos, 0.0, 1.0))
        n = self.shape[self.axis]
        return int(round(pos * (n - 1)))

    def area_at(self, pos: float, *, return_idx=False) -> float | Tuple[int, float]:
        """
        Compute 2D area (cm^2) at relative position along the chosen axis.

        - Maps pos in [0,1] to a slice index along the configured axis.
        - Computes a binary contour area (in pixels), then scales by the
          in-plane mm^2 per pixel to get mm^2, and converts to cm^2.
        - If the slice is empty: returns 0 (or snaps to nearest valid if enabled).

        Returns either the area (float) or (index, area) if return_idx=True.
        """
        idx = self._idx_from_pos(pos)
        key = (self.axis, idx)

        if key in self._area_cache:
            area_cm2 = self._area_cache[key]
            return (idx, area_cm2) if return_idx else area_cm2

        if not self._slice_has_mask(idx) and self.cfg.snap_to_valid:
            # snap to nearest slice that has any mask
            n = self.shape[self.axis]
            found = None
            for d in range(1, n):
                if idx - d >= 0 and self._slice_has_mask(idx - d):
                    found = idx - d
                    break
                if idx + d < n and self._slice_has_mask(idx + d):
                    found = idx + d
                    break
            if found is not None:
                idx = found
                key = (self.axis, idx)

        slice_u8 = self._extract_slice(idx)
        if not np.any(slice_u8):
            area_cm2 = 0.0
            self._area_cache[key] = area_cm2
            return (idx, area_cm2) if return_idx else area_cm2

        contours, _ = cv2.findContours(slice_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.contourArea(c) > float(self.cfg.min_contour_area)]

        px_area_sum = float(sum(cv2.contourArea(c) for c in filtered))
        area_mm2 = px_area_sum * self.plane_area_mm2[self.axis]
        area_cm2 = area_mm2 / 100.0

        self._area_cache[key] = area_cm2
        return (idx, area_cm2) if return_idx else area_cm2

    # ----- Optimization helpers -----
    def golden_section_max(self, a: float = 0.0, b: float = 1.0, tol: float = 1e-3) -> Tuple[float, float]:
        """Locate the maximum of area_at(pos) over [a,b] via golden-section.

        Returns (x_max, f_max) where f_max = area_at(x_max).
        """
        phi = (5 ** 0.5 - 1) / 2.0
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = self.area_at(c)
        fd = self.area_at(d)
        while (b - a) > tol:
            if fc < fd:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = self.area_at(d)
            else:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = self.area_at(c)
        x_max = 0.5 * (a + b)
        f_max = self.area_at(x_max)
        return x_max, f_max

    def find_left_threshold(self, x_max: float, f_max: float, p: float = 0.8, tol: float = 1e-4) -> float:
        """Monotone bisection on [0,x_max] to solve area_at(x) = p * f_max."""
        target = p * f_max
        a, b = 0.0, float(x_max)
        if self.area_at(a) >= target:
            return a
        while (b - a) > tol:
            m = 0.5 * (a + b)
            if self.area_at(m) < target:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    def find_right_threshold(self, x_max: float, f_max: float, p: float = 0.8, tol: float = 1e-4) -> float:
        """Monotone bisection on [x_max,1] to solve area_at(x) = p * f_max."""
        target = p * f_max
        a, b = float(x_max), 1.0
        if self.area_at(b) >= target:
            return b
        while (b - a) > tol:
            m = 0.5 * (a + b)
            if self.area_at(m) < target:
                b = m
            else:
                a = m
        return 0.5 * (a + b)

    # ----- Main pipeline -----
    def sample_band(self) -> Dict[str, Any]:
        """Run the full pipeline and return a rich result dict.

        Steps
        - Optional: compute full area profile along the axis for debugging.
        - Golden-section search for the maximum area location and value.
        - Binary searches for left/right p% crossings.
        - Sample n positions uniformly in [left,right] and compute per-slice
          area, perimeter, convex perimeter (via closing), LGI, and save
          annotated PNGs with optional label overlays.
        - Write JSON (summary), CSV (slices), Excel (summary+slices), and
          area profile plot+CSV.
        """
        cfg = self.cfg
        # Optional: compute and save full area profile (area per slice along axis)
        def _area_at_index(idx: int) -> float:
            mask = self._extract_slice(idx)
            if not np.any(mask):
                return 0.0
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = [c for c in contours if cv2.contourArea(c) > float(cfg.min_contour_area)]
            px_area_sum = float(sum(cv2.contourArea(c) for c in filtered))
            return (px_area_sum * self.plane_area_mm2[self.axis]) / 100.0

        # If no mask anywhere, short-circuit
        if not np.any(self.mask):
            return {
                "axis": self.axis,
                "axis_name": {0: "x", 1: "y", 2: "z"}[self.axis],
                "axis_size_vox": self.shape[self.axis],
                "axis_spacing_mm": self.spacing[self.axis],
                "axis_length_mm": self.shape[self.axis] * self.spacing[self.axis],
                "x_max": None,
                "f_max": 0.0,
                "left": None,
                "right": None,
                "positions": [],
                "indices": [],
                "areas_cm2": [],
                "saved_pngs": [],
                "config": asdict(cfg),
            }

        # Pre-compute area profiles and maxima
        area_profiles: Dict[int, Dict[str, List[float] | List[int]]] = {}
        x_max_axes: Dict[int, float] = {}
        f_max_axes: Dict[int, float] = {}
        if cfg.profile_plot:
            for ax in (0, 1, 2):
                idxs, vals = self._area_profile_for_axis(ax)
                area_profiles[ax] = {"indices": idxs, "areas_cm2": vals}
                xm, fm = self._golden_section_max_for_axis(ax, cfg.tol_max)
                x_max_axes[ax], f_max_axes[ax] = xm, fm
        else:
            xm, fm = self._golden_section_max_for_axis(self.axis, cfg.tol_max)
            x_max_axes[self.axis], f_max_axes[self.axis] = xm, fm
        x_max = x_max_axes[self.axis]
        f_max = f_max_axes[self.axis]
        if f_max <= 0:
            return {
                "axis": self.axis,
                "axis_name": {0: "x", 1: "y", 2: "z"}[self.axis],
                "axis_size_vox": self.shape[self.axis],
                "axis_spacing_mm": self.spacing[self.axis],
                "axis_length_mm": self.shape[self.axis] * self.spacing[self.axis],
                "x_max": x_max,
                "f_max": f_max,
                "left": None,
                "right": None,
                "positions": [],
                "indices": [],
                "areas_cm2": [],
                "saved_pngs": [],
                "config": asdict(cfg),
            }

        left = self.find_left_threshold(x_max, f_max, p=cfg.p, tol=cfg.tol_thr)
        right = self.find_right_threshold(x_max, f_max, p=cfg.p, tol=cfg.tol_thr)

        # Sample N positions in [left, right]
        if cfg.n <= 1 or right <= left:
            positions = [x_max]
        else:
            positions = list(np.linspace(left, right, cfg.n))

        indices: List[int] = []
        areas_cm2: List[float] = []
        saved_pngs: List[Optional[str]] = []

        for sample_idx, pos in enumerate(positions):
            idx, area = self.area_at(pos, return_idx=True)
            indices.append(idx)
            areas_cm2.append(area)

            if cfg.save_png:
                slice_u8 = self._extract_slice(idx)
                annotated = np.stack([slice_u8 * 255] * 3, axis=-1)
                if cfg.overlay_labels:
                    annotated = self._overlay_labels(annotated, idx)
                annotated = np.ascontiguousarray(annotated)
                # Draw crosshair at maxima along other axes (optional)
                H, W = annotated.shape[:2]
                # Determine row/col indices for crosshair based on current axis
                # Max indices from other axes
                x_idx = self._idx_from_pos(x_max_axes.get(0, 0.5)) if (self.cfg.show_crosshair and 0 in x_max_axes) else None
                y_idx = self._idx_from_pos(x_max_axes.get(1, 0.5)) if (self.cfg.show_crosshair and 1 in x_max_axes) else None
                z_idx = self._idx_from_pos(x_max_axes.get(2, 0.5)) if (self.cfg.show_crosshair and 2 in x_max_axes) else None
                row_idx = None; col_idx = None
                if self.axis == 2:  # plane (x,y) ~ shape (nx, ny) -> rows=x, cols=y
                    row_idx = x_idx; col_idx = y_idx
                elif self.axis == 1:  # plane (x,z) -> rows=x, cols=z
                    row_idx = x_idx; col_idx = z_idx
                else:  # axis == 0, plane (y,z) -> rows=y, cols=z
                    row_idx = y_idx; col_idx = z_idx
                cross_col = (255, 255, 0)
                if self.cfg.show_crosshair and row_idx is not None:
                    r = max(0, min(H - 1, int(row_idx)))
                    cv2.line(annotated, (0, r), (W - 1, r), cross_col, 1)
                if self.cfg.show_crosshair and col_idx is not None:
                    c = max(0, min(W - 1, int(col_idx)))
                    cv2.line(annotated, (c, 0), (c, H - 1), cross_col, 1)
                contours, _ = cv2.findContours(slice_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filtered = [c for c in contours if cv2.contourArea(c) > float(cfg.min_contour_area)]
                if cfg.draw_contours and filtered:
                    cv2.drawContours(annotated, filtered, -1, (0, 0, 255), 1)
                # Optional: pial overlay lines for the current slice.
                if self.cfg.use_pial_overlay:
                    annotated = draw_pial_on_slice(annotated, self.axis, idx, self._pial_vox, self.spacing, self.cfg)

                annotated = self._add_scale_bar(annotated, bar_length_mm=20.0)
                # Include sample order to avoid filename collisions when multiple
                # sampled positions round to the same slice index.
                out_path = os.path.join(
                    cfg.out_dir,
                    "brain_slices",
                    f"band_axis{self.axis}_s{sample_idx:02d}_idx{idx:04d}.png",
                )
                cv2.imwrite(out_path, annotated)
                # Post-process the saved PNG using Nifti2image to smooth and reapply the scale bar.
                nifti_slice_to_image(
                    in_path=out_path,
                    out_path=out_path,
                    label_text="20 mm",
                    scale_bar=True,
                    smooth=None,
                    smooth_strength=None,
                )
                saved_pngs.append(out_path)
            else:
                saved_pngs.append(None)

        # ---- Per-slice perimeter and LGI (inner/outer) ----
        perimeters_mm: List[float] = []
        perimeters_convex_mm: List[float] = []
        lgi_vals: List[float] = []
        k = max(1, int(self.cfg.kernel_size))
        k = k + 1 if (k % 2 == 0) else k  # prefer odd-ish size, but cv2 accepts any
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        sx_in, sy_in = self.inplane_spacings[self.axis]
        for idx in indices:
            mask = self._extract_slice(idx)
            # inner contours
            c_in, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c_in = [c for c in c_in if (cv2.contourArea(c) > float(cfg.min_contour_area) and len(c) >= 2)]
            c_in_mm = []
            for c in c_in:
                pts = (c.reshape(-1, 2).astype(np.float32) * np.array([sx_in, sy_in], dtype=np.float32))
                c_in_mm.append(np.ascontiguousarray(pts))
            p_in = float(sum(cv2.arcLength(pts, True) for pts in c_in_mm))
            # outer via closing
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            c_out, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c_out = [c for c in c_out if (cv2.contourArea(c) > float(cfg.min_contour_area) and len(c) >= 2)]
            c_out_mm = []
            for c in c_out:
                pts = (c.reshape(-1, 2).astype(np.float32) * np.array([sx_in, sy_in], dtype=np.float32))
                c_out_mm.append(np.ascontiguousarray(pts))
            p_out = float(sum(cv2.arcLength(pts, True) for pts in c_out_mm))
            perimeters_mm.append(p_in)
            perimeters_convex_mm.append(p_out)
            lgi_vals.append((p_in / p_out) if p_out > 0 else 0.0)

        # ---- Compute extended logging/metrics ----
        axis_name = {0: "x", 1: "y", 2: "z"}[self.axis]
        n_vox = self.shape[self.axis]
        d_mm = self.spacing[self.axis]
        axis_len_mm = n_vox * d_mm
        x_max_idx = self._idx_from_pos(x_max)
        left_idx = self._idx_from_pos(left)
        right_idx = self._idx_from_pos(right)
        x_max_mm = x_max_idx * d_mm
        left_mm = left_idx * d_mm
        right_mm = right_idx * d_mm
        band_len_mm = max(0.0, (right_idx - left_idx)) * d_mm
        positions_mm = [i * d_mm for i in indices]

        result: Dict[str, Any] = {
            "axis": self.axis,
            "axis_name": axis_name,
            "axis_size_vox": n_vox,
            "axis_spacing_mm": d_mm,
            "axis_length_mm": axis_len_mm,
            "x_max": x_max,
            "x_max_idx": x_max_idx,
            "x_max_mm": x_max_mm,
            "f_max": f_max,
            "left": left,
            "left_idx": left_idx,
            "left_mm": left_mm,
            "right": right,
            "right_idx": right_idx,
            "right_mm": right_mm,
            "band_length_mm": band_len_mm,
            "positions": positions,
            "indices": indices,
            "positions_mm": positions_mm,
            "areas_cm2": areas_cm2,
            "perimeter_mm": perimeters_mm,
            "perimeter_convex_mm": perimeters_convex_mm,
            "lgi": lgi_vals,
            "saved_pngs": saved_pngs,
            "config": asdict(cfg),
            "area_profile": area_profiles.get(self.axis, {"indices": [], "areas_cm2": []}),
            "maxima": {
                "x": {
                    "x_max": (x_max_axes.get(0) if 'x_max_axes' in locals() else None),
                    "x_max_idx": (self._idx_from_pos(x_max_axes.get(0)) if ('x_max_axes' in locals() and 0 in x_max_axes) else None),
                    "f_max": (f_max_axes.get(0) if 'f_max_axes' in locals() else None),
                },
                "y": {
                    "y_max": (x_max_axes.get(1) if 'x_max_axes' in locals() else None),
                    "y_max_idx": (self._idx_from_pos(x_max_axes.get(1)) if ('x_max_axes' in locals() and 1 in x_max_axes) else None),
                    "f_max": (f_max_axes.get(1) if 'f_max_axes' in locals() else None),
                },
                "z": {
                    "z_max": (x_max_axes.get(2) if 'x_max_axes' in locals() else None),
                    "z_max_idx": (self._idx_from_pos(x_max_axes.get(2)) if ('x_max_axes' in locals() and 2 in x_max_axes) else None),
                    "f_max": (f_max_axes.get(2) if 'f_max_axes' in locals() else None),
                },
            },
        }

        # ---- Persist logs to disk ----
        try:
            import json, csv
            import pandas as pd
            os.makedirs(cfg.out_dir, exist_ok=True)
            # JSON summary
            with open(os.path.join(cfg.out_dir, "area_band_summary.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            # CSV per-slice rows
            csv_path = os.path.join(cfg.out_dir, "area_band_slices.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["axis", "axis_name", "axis_spacing_mm", "axis_length_mm", "left", "right", "left_idx", "right_idx", "p", "n", "x_max", "x_max_idx", "x_max_mm", "f_max"])
                w.writerow([self.axis, axis_name, d_mm, axis_len_mm, left, right, left_idx, right_idx, cfg.p, cfg.n, x_max, x_max_idx, x_max_mm, f_max])
                w.writerow([])
                w.writerow(["pos", "idx", "pos_mm", "area_cm2", "perimeter_mm", "perimeter_convex_mm", "lgi", "png_path"])
                for pos, idx, pos_mm, area, pin, pconv, lgi, png in zip(positions, indices, positions_mm, areas_cm2, perimeters_mm, perimeters_convex_mm, lgi_vals, saved_pngs):
                    w.writerow([pos, idx, pos_mm, area, pin, pconv, lgi, png or ""])
            # Area profile CSV and plot
            try:
                # Write per-axis profiles CSV
                for ax in (0, 1, 2):
                    prof = area_profiles.get(ax)
                    if not prof:
                        continue
                    idxs = prof.get("indices", [])
                    vals = prof.get("areas_cm2", [])
                    vox = self.shape[ax]
                    spc = self.spacing[ax]
                    csv_path = os.path.join(cfg.out_dir, f"area_profile_axis{ax}.csv")
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["idx", "pos", "pos_mm", "area_cm2"])
                        for i, a in zip(idxs, vals):
                            w.writerow([i, (i / max(1, vox - 1)), i * spc, a])
                if cfg.profile_plot:
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
                    for ax in (0, 1, 2):
                        prof = area_profiles.get(ax)
                        if not prof:
                            continue
                        ixs = prof.get("indices", [])
                        vls = prof.get("areas_cm2", [])
                        axplt = axes[ax]
                        axplt.plot(ixs, vls, "-", linewidth=1.5)
                        axplt.set_title(f"Axis {['x','y','z'][ax]}")
                        axplt.set_xlabel("index")
                        axplt.set_ylabel("area (cm^2)")
                        xm = x_max_axes.get(ax)
                        if xm is not None:
                            axplt.axvline(self._idx_from_pos(xm), color="red", linestyle=":", linewidth=1)
                        if ax == self.axis:
                            axplt.axvline(left_idx, color="green", linestyle="--", linewidth=1)
                            axplt.axvline(right_idx, color="green", linestyle="--", linewidth=1)
                    fig.tight_layout()
                    plt.savefig(os.path.join(cfg.out_dir, "area_profiles.png"), dpi=150)
                    plt.close(fig)
            except Exception as ex:
                print(f"[AreaBand] Failed to write profile: {ex}")
            # Excel export mirroring CSV
            try:
                df_meta = pd.DataFrame([
                    {
                        "axis": self.axis,
                        "axis_name": axis_name,
                        "axis_spacing_mm": d_mm,
                        "axis_length_mm": axis_len_mm,
                        "left": left,
                        "right": right,
                        "left_idx": left_idx,
                        "right_idx": right_idx,
                        "p": cfg.p,
                        "n": cfg.n,
                        "x_max": x_max,
                        "x_max_idx": x_max_idx,
                        "x_max_mm": x_max_mm,
                        "f_max": f_max,
                    }
                ])
                df_rows = pd.DataFrame({
                    "pos": positions,
                    "idx": indices,
                    "pos_mm": positions_mm,
                    "area_cm2": areas_cm2,
                    "perimeter_mm": perimeters_mm,
                    "perimeter_convex_mm": perimeters_convex_mm,
                    "lgi": lgi_vals,
                    "png_path": saved_pngs,
                })
                xlsx_path = os.path.join(cfg.out_dir, "area_band_slices.xlsx")
                # Try default engine first (openpyxl if available), then fall back to xlsxwriter
                try:
                    with pd.ExcelWriter(xlsx_path) as writer:
                        df_meta.to_excel(writer, sheet_name="summary", index=False)
                        df_rows.to_excel(writer, sheet_name="slices", index=False)
                except Exception:
                    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                        df_meta.to_excel(writer, sheet_name="summary", index=False)
                        df_rows.to_excel(writer, sheet_name="slices", index=False)
            except Exception as ex:
                print(f"[AreaBand] Failed to write Excel: {ex}")
        except Exception as ex:
            # Non-fatal logging error
            print(f"[AreaBand] Failed to write logs: {ex}")

        return result


def load_config_from_json(path: str) -> AreaBandConfig:
    import json
    # Accept files with or without BOM
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return AreaBandConfig(**data)


def run_from_config(json_path: str) -> Dict[str, Any]:
    cfg = load_config_from_json(json_path)
    sampler = NiftiAreaSampler(cfg)
    return sampler.sample_band()







