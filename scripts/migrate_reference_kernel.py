"""Add kernel_size_mm to Examples/gestational_week_reference.csv.

The shipped reference table was produced with kernel_size in pixels.  It does
not contain a per-row pixel-size column, so the default migration uses the
legacy reference scale of 0.2 mm/px: 25 px -> 5.0 mm.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="Examples/gestational_week_reference.csv",
    )
    parser.add_argument("--pixel-size-column", default=None)
    parser.add_argument("--default-pixel-size-mm", type=float, default=0.2)
    args = parser.parse_args()

    path = Path(args.csv_path)
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    fieldnames = list(rows[0].keys())
    if "kernel_size_mm" not in fieldnames:
        insert_at = fieldnames.index("kernel_size") + 1 if "kernel_size" in fieldnames else len(fieldnames)
        fieldnames.insert(insert_at, "kernel_size_mm")

    for row in rows:
        if row.get("kernel_size_mm"):
            continue
        kernel_px = float(row["kernel_size"])
        if args.pixel_size_column and row.get(args.pixel_size_column):
            pixel_size_mm = float(row[args.pixel_size_column])
        else:
            pixel_size_mm = args.default_pixel_size_mm
        row["kernel_size_mm"] = f"{kernel_px * pixel_size_mm:g}"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
