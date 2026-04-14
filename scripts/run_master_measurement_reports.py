from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.measurement_batch import process_on_images_batch


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_AXES = ("axial", "coronal", "sagittal")
DEFAULT_WEEKS = tuple(range(24, 39))
DEFAULT_CONFIG = {
    "input_root": str(REPO_ROOT / "Examples" / "full_slices"),
    "output_root": str(REPO_ROOT / "measurements"),
    "section_label": "Filled_2D_sections",
    "weeks": list(DEFAULT_WEEKS),
    "axes": list(DEFAULT_AXES),
    "pixel_size": 1.0 / 41.0,
    "kernel_size": 25,
    "cnt_threshold": 2000,
    "unit": "mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the master-branch FetoMorph batch measurement pipeline across "
            "week/axis folders and write one report workbook per folder."
        )
    )
    parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    parser.add_argument("--input-root", type=Path, help="Root folder containing week/axis image folders.")
    parser.add_argument("--output-root", type=Path, help="Root folder where reports will be written.")
    parser.add_argument(
        "--section-label",
        type=str,
        help='Section folder placed under each week, for example "Filled_2D_sections".',
    )
    parser.add_argument("--weeks", nargs="+", type=int, help="Weeks to process, for example: --weeks 24 25")
    parser.add_argument(
        "--axes",
        nargs="+",
        help="Axes to process, for example: --axes axial coronal sagittal",
    )
    parser.add_argument("--pixel-size", type=float, help="Physical pixel size in unit/pixel.")
    parser.add_argument("--kernel-size", type=int, help="Morphology kernel size.")
    parser.add_argument("--cnt-threshold", type=float, help="Minimum contour area threshold in pixels.")
    parser.add_argument("--unit", type=str, help='Length unit label, for example "mm".')
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console logging verbosity.",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(load_config(args.config))

    overrides = {
        "input_root": args.input_root,
        "output_root": args.output_root,
        "section_label": args.section_label,
        "weeks": args.weeks,
        "axes": args.axes,
        "pixel_size": args.pixel_size,
        "kernel_size": args.kernel_size,
        "cnt_threshold": args.cnt_threshold,
        "unit": args.unit,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    cfg["input_root"] = Path(cfg["input_root"]).resolve()
    cfg["output_root"] = Path(cfg["output_root"]).resolve()
    cfg["section_label"] = str(cfg["section_label"]).strip()
    cfg["weeks"] = [int(w) for w in cfg["weeks"]]
    cfg["axes"] = [str(a).strip().lower() for a in cfg["axes"]]
    cfg["pixel_size"] = float(cfg["pixel_size"])
    cfg["kernel_size"] = int(cfg["kernel_size"])
    cfg["cnt_threshold"] = float(cfg["cnt_threshold"])
    cfg["unit"] = str(cfg["unit"]).strip()
    return cfg


def find_image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def is_metadata_row(first_cell: Any) -> bool:
    if not isinstance(first_cell, str):
        return False
    value = first_cell.strip()
    return value.endswith(":") or value in {"PixelSizeUnits", "KernelSize"}


def add_summary_table(xlsx_path: Path) -> None:
    wb = load_workbook(xlsx_path)
    ws = wb.active

    headers = [ws.cell(row=1, column=col).value for col in range(1, ws.max_column + 1)]
    summary_start_col = ws.max_column + 3
    ws.cell(row=1, column=summary_start_col, value="Metric")
    ws.cell(row=1, column=summary_start_col + 1, value="Mean")

    data_rows: list[int] = []
    for row_idx in range(2, ws.max_row + 1):
        first_cell = ws.cell(row=row_idx, column=1).value
        if first_cell in (None, ""):
            continue
        if is_metadata_row(first_cell):
            continue
        data_rows.append(row_idx)

    skip_headers = {"File", "SliceKind"}
    summary_row = 2
    for col_idx, header in enumerate(headers, start=1):
        if header is None:
            continue
        header_text = str(header).strip()
        if header_text in skip_headers:
            continue

        values = []
        for row_idx in data_rows:
            value = ws.cell(row=row_idx, column=col_idx).value
            if value in (None, ""):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue

        if not values:
            continue

        ws.cell(row=summary_row, column=summary_start_col, value=header_text)
        mean_cell = ws.cell(
            row=summary_row,
            column=summary_start_col + 1,
            value=(sum(values) / len(values)),
        )
        mean_cell.number_format = "0.000000"
        summary_row += 1

    wb.save(xlsx_path)


def run_folder(
    input_dir: Path,
    output_dir: Path,
    week: int,
    axis: str,
    *,
    pixel_size: float,
    kernel_size: int,
    cnt_threshold: float,
    unit: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    process_on_images_batch(
        str(input_dir),
        str(output_dir),
        pixel_size=pixel_size,
        kernel_size=kernel_size,
        cnt_threshold=cnt_threshold,
        unit=unit,
    )

    default_xlsx = output_dir / "Batch_Allmarks.xlsx"
    final_xlsx = output_dir / f"week{week}_{axis}_Batch_Allmarks.xlsx"
    if final_xlsx.exists():
        final_xlsx.unlink()
    default_xlsx.replace(final_xlsx)
    add_summary_table(final_xlsx)
    return final_xlsx


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    settings = resolve_settings(args)

    input_root: Path = settings["input_root"]
    output_root: Path = settings["output_root"]
    section_label: str = settings["section_label"]
    weeks: list[int] = settings["weeks"]
    axes: list[str] = settings["axes"]

    if not input_root.exists():
        logging.error("Input root does not exist: %s", input_root)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    for week in weeks:
        for axis in axes:
            input_dir = input_root / str(week) / axis
            output_dir = output_root / str(week) / section_label / axis

            if not input_dir.exists():
                logging.warning("Missing folder, skipping: %s", input_dir)
                skipped += 1
                continue

            if not find_image_files(input_dir):
                logging.warning("No images found, skipping: %s", input_dir)
                skipped += 1
                continue

            logging.info("Processing week=%s axis=%s from %s", week, axis, input_dir)
            try:
                xlsx_path = run_folder(
                    input_dir,
                    output_dir,
                    week,
                    axis,
                    pixel_size=settings["pixel_size"],
                    kernel_size=settings["kernel_size"],
                    cnt_threshold=settings["cnt_threshold"],
                    unit=settings["unit"],
                )
                logging.info("Wrote workbook: %s", xlsx_path)
                processed += 1
            except Exception as exc:
                logging.exception("Failed week=%s axis=%s: %s", week, axis, exc)
                failed += 1

    logging.info("Done. processed=%s skipped=%s failed=%s", processed, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
