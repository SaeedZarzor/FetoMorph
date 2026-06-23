"""Evaluate the trained slice-kind ONNX model on the same train/val/test splits.

Companion to ``train_slice_kind_cnn.py``. Reuses that module's
``collect_samples`` / ``stratified_split`` (and identical ``SEED``) so the
splits match exactly, then scores ``models/slice_kind_cnn.onnx`` with
onnxruntime — the same runtime path the app uses.

Reports per-split accuracy (the train-vs-val/test gap is the overfitting
signal), per-class precision/recall/F1, and confusion matrices. Prints a
machine-readable JSON blob (between BEGIN_JSON / END_JSON markers) so the
numbers can be lifted straight into a report.

    python scripts/eval_slice_kind_cnn.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from train_slice_kind_cnn import (  # type: ignore
    LABELS,
    IMG_SIZE,
    MODEL_OUT,
    collect_samples,
    stratified_split,
)


def _preprocess(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)[None, None, :, :]


def _eval_split(sess: ort.InferenceSession, in_name: str, samples: list) -> dict:
    n = len(LABELS)
    cm = np.zeros((n, n), dtype=np.int64)
    for path, label in samples:
        logits = sess.run(None, {in_name: _preprocess(path)})[0]
        pred = int(np.asarray(logits).reshape(-1, n).argmax(axis=1)[0])
        cm[label, pred] += 1

    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = correct / max(1, total)

    per_class = {}
    for i, name in enumerate(LABELS):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {
            "support": support, "precision": prec, "recall": rec, "f1": f1,
        }

    return {"n": total, "accuracy": acc, "confusion": cm.tolist(),
            "per_class": per_class}


def main() -> None:
    if not Path(MODEL_OUT).exists():
        raise SystemExit(f"Model not found: {MODEL_OUT}. Train it first.")

    samples = collect_samples()
    train, val, test = stratified_split(samples)

    sess = ort.InferenceSession(str(MODEL_OUT), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    results = {
        "labels": list(LABELS),
        "model": str(MODEL_OUT),
        "splits": {
            "train": _eval_split(sess, in_name, train),
            "val": _eval_split(sess, in_name, val),
            "test": _eval_split(sess, in_name, test),
        },
    }

    for split in ("train", "val", "test"):
        r = results["splits"][split]
        print(f"{split:5s} n={r['n']:5d}  acc={r['accuracy']*100:6.2f}%")
    gap = results["splits"]["train"]["accuracy"] - results["splits"]["test"]["accuracy"]
    print(f"train-test gap: {gap*100:.2f} pp")

    print("BEGIN_JSON")
    print(json.dumps(results))
    print("END_JSON")


if __name__ == "__main__":
    main()
