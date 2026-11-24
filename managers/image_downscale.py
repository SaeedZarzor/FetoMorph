"""Background downscaling of very large imported 2-D images.

Measurements (segmentation, contour extraction, morphology, curvature) scale
with the pixel count, so a very high-resolution import makes every step slow
without improving the result. When an imported image exceeds a pixel budget it
is resized (``INTER_AREA``, aspect ratio preserved) into a temp copy; the
original file is never modified.

``plan_target_size`` / ``downscale_image_file`` are pure (no Qt) so they can be
tested on their own. :class:`ImageDownscaler` runs them in a worker thread and
hands the result back to the GUI thread.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from typing import Callable

import cv2
from PySide6.QtCore import QObject, QThread, Signal, Slot

logger = logging.getLogger("fetomorph")


@dataclass(frozen=True)
class DownscaleResult:
    """Outcome of one downscale: where the copy is and how it was scaled."""
    src_path: str
    out_path: str
    orig_size: tuple[int, int]   # (width, height) in px
    new_size: tuple[int, int]

    @property
    def pixel_size_factor(self) -> float:
        """Multiplier for a mm/px value: original-pixel scale -> downscaled-pixel scale."""
        return self.orig_size[0] / self.new_size[0]


def plan_target_size(width: int, height: int, max_pixels: float) -> tuple[int, int] | None:
    """Return the downscaled ``(w, h)``, or ``None`` if no downscale is needed."""
    if width <= 0 or height <= 0 or width * height <= max_pixels:
        return None
    s = math.sqrt(max_pixels / float(width * height))
    return max(1, int(width * s)), max(1, int(height * s))


def _output_path(src: str, out_dir: str) -> str:
    # Same stem as the source so result/export names stay recognisable; a short
    # hash of the full path keeps two same-named files from colliding.
    tag = hashlib.sha1(os.path.abspath(src).encode("utf-8")).hexdigest()[:8]
    stem = os.path.splitext(os.path.basename(src))[0]
    return os.path.join(out_dir, tag, f"{stem}.png")


def downscale_image_file(src: str, out_dir: str, max_pixels: float) -> DownscaleResult:
    """Write a downscaled lossless PNG copy of *src* under *out_dir*.

    Raises ``ValueError`` if the image can't be decoded or needs no downscale.
    """
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not decode image: {src}")
    h, w = img.shape[:2]
    target = plan_target_size(w, h, max_pixels)
    if target is None:
        raise ValueError(f"Image is within the pixel budget: {w}x{h}")
    resized = cv2.resize(img, target, interpolation=cv2.INTER_AREA)
    out = _output_path(src, out_dir)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not cv2.imwrite(out, resized):
        raise ValueError(f"Could not write downscaled image: {out}")
    return DownscaleResult(src, out, (w, h), target)


class _Job(QThread):
    done = Signal(int, object)  # token, DownscaleResult | Exception

    def __init__(self, token: int, src: str, out_dir: str, max_pixels: float, parent=None):
        super().__init__(parent)
        self._token, self._src, self._out_dir, self._max = token, src, out_dir, max_pixels

    def run(self) -> None:
        try:
            payload = downscale_image_file(self._src, self._out_dir, self._max)
        except Exception as ex:  # reported to the GUI thread, never raised here
            payload = ex
        self.done.emit(self._token, payload)


class ImageDownscaler(QObject):
    """Runs downscales off the GUI thread; only the latest request is delivered."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._token = 0
        self._pending_src: str | None = None
        self._callback: Callable[[str, DownscaleResult | Exception], None] | None = None
        self._jobs: set[_Job] = set()

    def is_pending(self, src: str) -> bool:
        return self._pending_src == src

    def submit(self, src: str, out_dir: str, max_pixels: float,
               callback: Callable[[str, DownscaleResult | Exception], None]) -> None:
        """Start a background downscale; a newer submit supersedes older ones."""
        self._token += 1
        self._pending_src = src
        self._callback = callback
        job = _Job(self._token, src, out_dir, max_pixels, parent=self)
        job.done.connect(self._on_done)          # slot lives on the GUI thread -> queued
        job.finished.connect(lambda j=job: self._jobs.discard(j))
        self._jobs.add(job)
        job.start()

    @Slot(int, object)
    def _on_done(self, token: int, payload) -> None:
        if token != self._token:   # a newer import replaced this one
            return
        src, cb = self._pending_src, self._callback
        self._pending_src = self._callback = None
        if cb is not None and src is not None:
            cb(src, payload)

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """Wait for running jobs (called on app quit so no thread outlives Qt)."""
        self._callback = None
        for job in list(self._jobs):
            job.wait(timeout_ms)
