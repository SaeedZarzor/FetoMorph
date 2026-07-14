"""Offline trainer for the FetoMorph slice-kind CNN.

Trains a tiny 4-class image classifier that distinguishes full MRI slices
(sagittal / coronal / axial) from cropped sub-slice bands. The trained model
is exported to ``models/slice_kind_cnn.onnx`` and consumed at runtime by
``helpers/slice_kind_classifier.py``.

Run manually (this script is NOT imported by the application):

    python scripts/train_slice_kind_cnn.py

Requires PyTorch. PyTorch is not in requirements.txt because it is only
needed offline; the runtime path uses onnxruntime instead.

Dataset is built from the ``traning_data/`` tree:
    traning_data/full_slices/<week>/sagittal/*.png  -> label 0
    traning_data/full_slices/<week>/coronal/*.png   -> label 1
    traning_data/full_slices/<week>/axial/*.png     -> label 2
    traning_data/cropped_slices/<week>/<orient>/*.png  -> label 3
    traning_data/cropped_slices/<flat_dir>/*.png       -> label 3
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as e:
    sys.exit(
        "PyTorch is required to run this script. Install it with:\n"
        "    pip install torch\n"
        f"(import error: {e})"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "traning_data"
MODEL_OUT = REPO_ROOT / "models" / "slice_kind_cnn.onnx"

# Train on the exact same preprocessing the runtime classifier applies, so the
# model never sees a train/serve skew. classify_slice_kind() reframes every
# image (tight-crop to the brain + pad to a centered square) BEFORE resizing;
# training must do the same or full-slice accuracy collapses at inference.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from helpers.slice_kind_classifier import _reframe_to_training_layout

LABELS = ("sagittal", "coronal", "axial", "not_full_slice")
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 1337

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def collect_samples() -> list[tuple[Path, int]]:
    """Walk Examples/ and return ``(path, label)`` tuples."""
    samples: list[tuple[Path, int]] = []

    full_root = EXAMPLES_DIR / "full_slices"
    for week_dir in sorted(full_root.glob("*")):
        if not week_dir.is_dir():
            continue
        for orient, label in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
            for p in sorted((week_dir / orient).glob("*.png")):
                samples.append((p, label))

    cropped_root = EXAMPLES_DIR / "cropped_slices"
    for week_dir in sorted(cropped_root.glob("*")):
        if not week_dir.is_dir():
            continue
        # Pick up PNGs in sub-directories (e.g. <week>/axial/*.png)
        for orient_dir in sorted(week_dir.glob("*")):
            if not orient_dir.is_dir():
                continue
            for p in sorted(orient_dir.glob("*.png")):
                samples.append((p, 3))
        # Pick up PNGs directly in the folder (flat dirs like
        # With_scale_bar/, first_batch/, second_batch/)
        for p in sorted(week_dir.glob("*.png")):
            samples.append((p, 3))

    return samples


def stratified_split(
    samples: list[tuple[Path, int]],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list, list, list]:
    """Stratified split by (week, label) so subjects don't leak across splits."""
    rng = random.Random(SEED)
    by_key: dict[tuple[str, int], list[tuple[Path, int]]] = {}
    for p, lbl in samples:
        # week dir is two levels up: Examples/<root>/<week>/<orient>/file.png
        week = p.parent.parent.name
        by_key.setdefault((week, lbl), []).append((p, lbl))

    train, val, test = [], [], []
    for key, items in by_key.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_frac))) if n >= 5 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 5 else 0
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    rng.shuffle(train)
    return train, val, test


class SliceKindDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # horizontal flip (safe: doesn't change orientation class)
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
        # small rotation up to +/- 5 degrees (no 90-degree rotations)
        if random.random() < 0.5:
            angle = random.uniform(-5.0, 5.0)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # brightness / contrast jitter
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-0.2, 0.2)  # contrast
            beta = random.uniform(-0.1, 0.1)  # brightness (already in [0,1] later)
            img = np.clip(img.astype(np.float32) * alpha + beta * 255.0, 0, 255).astype(np.uint8)
        # random crop with padding
        if random.random() < 0.5:
            pad = 8
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
            h, w = img.shape
            y0 = random.randint(0, 2 * pad)
            x0 = random.randint(0, 2 * pad)
            img = padded[y0 : y0 + h, x0 : x0 + w]
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        # Match the runtime classifier's preprocessing exactly (see note above).
        img = _reframe_to_training_layout(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        if self.augment:
            img = self._augment(img)
        x = img.astype(np.float32) / 255.0
        return torch.from_numpy(x).unsqueeze(0), label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TinySliceCNN(nn.Module):
    """5-conv block, ~150K params, grayscale 128x128 -> 4 logits."""

    def __init__(self, num_classes: int = 4):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, 16),   # 64
            block(16, 32),  # 32
            block(32, 64),  # 16
            block(64, 96),  # 8
            block(96, 128), # 4
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, np.ndarray]:
    model.eval()
    correct = total = 0
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
            for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                cm[t, p] += 1
    return (correct / max(1, total)), cm


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    samples = collect_samples()
    if not samples:
        sys.exit(f"No images found under {EXAMPLES_DIR}. Check the dataset layout.")

    counts = [0] * len(LABELS)
    for _, lbl in samples:
        counts[lbl] += 1
    print(f"Total samples: {len(samples)}")
    for i, name in enumerate(LABELS):
        print(f"  {name:16s} {counts[i]}")

    train, val, test = stratified_split(samples)
    print(f"Split sizes: train={len(train)} val={len(val)} test={len(test)}")

    # Class-balanced sampler for the training set (cropped class is ~2x bigger)
    train_labels = np.array([lbl for _, lbl in train])
    class_weights = 1.0 / np.bincount(train_labels, minlength=len(LABELS)).clip(min=1)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train),
        replacement=True,
    )

    train_loader = DataLoader(
        SliceKindDataset(train, augment=True),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        SliceKindDataset(val, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        SliceKindDataset(test, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = TinySliceCNN(num_classes=len(LABELS)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state: dict | None = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            running += loss.item() * y.numel()
            seen += y.numel()
        sched.step()
        train_loss = running / max(1, seen)
        val_acc, _ = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, cm = evaluate(model, test_loader)
    print(f"\nBest val acc: {best_val*100:.2f}%")
    print(f"Test acc:     {test_acc*100:.2f}%")
    print("Confusion matrix (rows=true, cols=pred):")
    header = "              " + " ".join(f"{n[:8]:>9s}" for n in LABELS)
    print(header)
    for i, name in enumerate(LABELS):
        row = " ".join(f"{cm[i, j]:9d}" for j in range(len(LABELS)))
        print(f"  {name:12s} {row}")

    # ---- Export ONNX ------------------------------------------------------
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=device)
    # Use the legacy TorchScript-based exporter so the model is written as a
    # single self-contained .onnx file (the dynamo exporter spills weights
    # into a sibling .onnx.data file, which would force us to ship two files).
    torch.onnx.export(
        model,
        dummy,
        str(MODEL_OUT),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"\nExported ONNX model to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
