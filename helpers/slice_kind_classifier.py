"""Tiny ONNX-based classifier that labels a 2-D image as a full MRI slice
(sagittal / coronal / axial) or as a cropped sub-slice band.

Used by ``functions/measurements_image.py`` to decide whether the sulci
depth filter should switch from the fixed 0.5 mm rule to a percent-of-
slice-length rule. The model file lives at ``models/slice_kind_cnn.onnx``
and is produced by ``scripts/train_slice_kind_cnn.py``.

The ``onnxruntime`` import and the model itself are loaded lazily on the
first call so that:
  - application startup is unaffected,
  - the missing-model case degrades gracefully (returns ``not_full_slice``).
"""

from __future__ import annotations

from typing import Literal

from deps import *

SliceKind = Literal["sagittal", "coronal", "axial", "not_full_slice"]
_LABELS: tuple[SliceKind, ...] = ("sagittal", "coronal", "axial", "not_full_slice")
_INPUT_SIZE = 128
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "slice_kind_cnn.onnx"

_session = None  # cached ort.InferenceSession
_input_name: str | None = None
_load_failed = False


def _get_session():
    """Lazily build (and cache) the ONNX Runtime session."""
    global _session, _input_name, _load_failed
    if _session is not None:
        return _session
    if _load_failed:
        return None
    try:
        import onnxruntime as ort  # local import: optional runtime dep
    except ImportError:
        _load_failed = True
        return None
    if not _MODEL_PATH.is_file():
        _load_failed = True
        return None
    try:
        _session = ort.InferenceSession(str(_MODEL_PATH), providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
    except Exception:
        _load_failed = True
        _session = None
        return None
    return _session


def _reframe_to_training_layout(gray: np.ndarray) -> np.ndarray:
    """Tight-crop to the brain and pad to a centered square.

    Training images are white-background, ~square frames with the brain filling
    most of the canvas. Exported renders can instead be very wide, with black
    letterbox bars and a small brain inside large white margins. Resizing such a
    frame straight to 128x128 squishes the brain and feeds the model an
    out-of-distribution layout. This reframes any input to match training:

      1. isolate the brain as pixels that are neither near-white (background)
         nor near-black (letterbox bars / scale bar / text),
      2. crop to that bounding box,
      3. pad to a centered square with the white background value.

    Falls back to the original image when no clear foreground is found.
    """
    fg = (gray > 12) & (gray < 244)
    if int(fg.sum()) < int(0.001 * gray.size):
        return gray  # no distinct brain region; classify the frame as-is
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = gray[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    # Pad with white (255) to match the white-background training frames.
    return cv2.copyMakeBorder(
        crop, top, side - h - top, left, side - w - left,
        cv2.BORDER_CONSTANT, value=255,
    )


def classify_slice_kind(image_bgr: np.ndarray) -> tuple[SliceKind, float]:
    """Classify a single 2-D image.

    Args:
        image_bgr: BGR or grayscale ``np.ndarray`` (as returned by ``cv2.imread``).

    Returns:
        ``(label, confidence)`` where ``label`` is one of
        ``"sagittal" | "coronal" | "axial" | "not_full_slice"``.
        Falls back to ``("not_full_slice", 0.0)`` if the model file or
        onnxruntime is unavailable, or if ``image_bgr`` is invalid.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return ("not_full_slice", 0.0)

    sess = _get_session()
    if sess is None:
        return ("not_full_slice", 0.0)

    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    gray = _reframe_to_training_layout(gray)
    resized = cv2.resize(gray, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = (resized.astype(np.float32) / 255.0)[None, None, :, :]  # 1x1xHxW

    logits = sess.run(None, {_input_name: x})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    idx = int(probs.argmax())
    return (_LABELS[idx], float(probs[idx]))


def is_full_mri_slice(image_bgr: np.ndarray, min_confidence: float = 0.7) -> bool:
    """True iff the image is confidently classified as a full MRI slice."""
    label, conf = classify_slice_kind(image_bgr)
    return label != "not_full_slice" and conf >= min_confidence
