"""Subject-level (leave-weeks-out) cross-validation for the slice-kind CNN.

Each gestational-week folder is one fetal subject (confirmed). The default
stratified split in ``train_slice_kind_cnn.py`` mixes slices of the SAME subject
across train/test, which overestimates generalization to UNSEEN subjects.

This script runs **5-fold GroupKFold grouped by week/subject**: the 15 week
subjects are round-robin assigned to 5 folds (3 unseen weeks per fold), so no
slice from a held-out subject is ever in training. ``full_slices/<week>`` and
``cropped_slices/<week>`` share a group (same subject); the three cropped batch
dirs (With_scale_bar / first_batch / second_batch) are their own groups.

A fresh model is trained per fold (same architecture/hyperparams as the real
trainer) and scored on its held-out subjects. Reports per-fold accuracy, the
pooled confusion matrix, per-class recall, and mean +/- std across folds.

This does NOT produce the shipped model (that stays the all-data ONNX); it is an
honest generalization estimate.

    python scripts/cv_slice_kind_cnn.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_slice_kind_cnn import (  # type: ignore
    LABELS, IMG_SIZE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, SEED,
    EXAMPLES_DIR, collect_samples, SliceKindDataset, TinySliceCNN,
    evaluate, device,
)

K_FOLDS = 5


def group_of(path: Path) -> str:
    """Subject group: shared week for full+cropped, else the cropped batch dir."""
    parts = path.relative_to(EXAMPLES_DIR).parts  # <root>/<second>/.../file.png
    second = parts[1] if len(parts) > 1 else parts[0]
    if second.isdigit():
        return f"week{int(second):02d}"
    return f"batch_{second}"


def assign_folds(groups: list[str]) -> dict[str, int]:
    weeks = sorted(g for g in groups if g.startswith("week"))
    batches = sorted(g for g in groups if not g.startswith("week"))
    fold_of: dict[str, int] = {}
    for i, g in enumerate(weeks):
        fold_of[g] = i % K_FOLDS
    for i, g in enumerate(batches):
        fold_of[g] = i % K_FOLDS
    return fold_of


def train_one(train_samples: list) -> nn.Module:
    labels = np.array([lbl for _, lbl in train_samples])
    class_w = 1.0 / np.bincount(labels, minlength=len(LABELS)).clip(min=1)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(class_w[labels]).double(),
        num_samples=len(train_samples), replacement=True,
    )
    loader = DataLoader(
        SliceKindDataset(train_samples, augment=True),
        batch_size=BATCH_SIZE, sampler=sampler, num_workers=0,
    )
    model = TinySliceCNN(num_classes=len(LABELS)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss_fn(model(x), y).backward()
            optim.step()
        sched.step()
    return model


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    samples = collect_samples()
    groups = sorted({group_of(p) for p, _ in samples})
    fold_of = assign_folds(groups)
    print(f"Total samples: {len(samples)}  | groups (subjects): {len(groups)}")
    print(f"Folds (K={K_FOLDS}):")
    for f in range(K_FOLDS):
        held = sorted(g for g, ff in fold_of.items() if ff == f)
        print(f"  fold {f}: held-out = {', '.join(held)}")

    fold_accs: list[float] = []
    pooled_cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    per_fold = []

    for f in range(K_FOLDS):
        test = [(p, l) for p, l in samples if fold_of[group_of(p)] == f]
        train = [(p, l) for p, l in samples if fold_of[group_of(p)] != f]
        test_loader = DataLoader(
            SliceKindDataset(test, augment=False), batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0,
        )
        model = train_one(train)
        acc, cm = evaluate(model, test_loader)
        fold_accs.append(acc)
        pooled_cm += cm
        # per-class recall on this fold
        recalls = {}
        for i, name in enumerate(LABELS):
            support = int(cm[i].sum())
            recalls[name] = (int(cm[i, i]) / support) if support else None
        held = sorted(g for g, ff in fold_of.items() if ff == f)
        per_fold.append({"fold": f, "held_out": held, "n_test": len(test),
                         "n_train": len(train), "accuracy": acc,
                         "recall": recalls})
        print(f"fold {f}: train={len(train)} test={len(test)} "
              f"acc={acc*100:.2f}%")

    accs = np.array(fold_accs)
    print(f"\nUNSEEN-SUBJECT (leave-weeks-out) accuracy: "
          f"mean={accs.mean()*100:.2f}%  std={accs.std()*100:.2f}%  "
          f"min={accs.min()*100:.2f}%  max={accs.max()*100:.2f}%")
    print("Pooled confusion matrix (rows=true, cols=pred):")
    header = "              " + " ".join(f"{n[:8]:>9s}" for n in LABELS)
    print(header)
    for i, name in enumerate(LABELS):
        row = " ".join(f"{pooled_cm[i, j]:9d}" for j in range(len(LABELS)))
        print(f"  {name:12s} {row}")

    out = {
        "k_folds": K_FOLDS,
        "mean_acc": float(accs.mean()), "std_acc": float(accs.std()),
        "min_acc": float(accs.min()), "max_acc": float(accs.max()),
        "pooled_confusion": pooled_cm.tolist(),
        "labels": list(LABELS), "folds": per_fold,
    }
    print("BEGIN_JSON")
    print(json.dumps(out))
    print("END_JSON")


if __name__ == "__main__":
    main()
