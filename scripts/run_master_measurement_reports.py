from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openpyxl import load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.measurement_batch import process_on_images_batch
from helpers.helpers import compute_kernel_convex, compactness_2D
from constants import BINARY_THRESHOLD_DEFAULT


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_AXES = ("axial", "coronal", "sagittal")
DEFAULT_WEEKS = tuple(range(24, 39))
DEFAULT_CONFIG = {
    "input_root": str(REPO_ROOT / "Examples" / "full_slices"),
    "output_root": str(REPO_ROOT / "measurements"),
    "section_label": "Filled_2D_sections",
    "weeks": list(DEFAULT_WEEKS),
    "axes": list(DEFAULT_AXES),
    "pixel_size": 20.0 / 42.0,
    "scalebar_measured_pixels": 42.0,
    "scalebar_real_world_length": 20.0,
    "kernel_size": 25,
    "cnt_threshold": 2000,
    "unit": "mm",
    "auto_scalebar": False,
    "single_pass_metrics": False,
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
    parser.add_argument(
        "--scalebar-measured-pixels",
        type=float,
        help="Measured scalebar length in pixels (for scale-from-scalebar calibration).",
    )
    parser.add_argument(
        "--scalebar-real-world-length",
        type=float,
        help="Real-world scalebar length in current unit (for scale-from-scalebar calibration).",
    )
    parser.add_argument("--kernel-size", type=int, help="Morphology kernel size.")
    parser.add_argument("--cnt-threshold", type=float, help="Minimum contour area threshold in pixels.")
    parser.add_argument("--unit", type=str, help='Length unit label, for example "mm".')
    parser.add_argument(
        "--auto-scalebar",
        action="store_true",
        default=None,
        help="Derive pixel_size per folder from each image's embedded scalebar bar "
             "(falls back to the configured pixel_size when no bar is detected).",
    )
    parser.add_argument(
        "--single-pass-metrics",
        action="store_true",
        default=None,
        help="After the batch run, recompute area/perimeter/perimeter_convex/LGI/"
             "Compactness with a single pass (no LGI threshold-climbing retry), so "
             "small cropped brains are never erased. Overwrites those columns.",
    )
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
        "scalebar_measured_pixels": args.scalebar_measured_pixels,
        "scalebar_real_world_length": args.scalebar_real_world_length,
        "kernel_size": args.kernel_size,
        "cnt_threshold": args.cnt_threshold,
        "unit": args.unit,
        "auto_scalebar": args.auto_scalebar,
        "single_pass_metrics": args.single_pass_metrics,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    cfg["input_root"] = Path(cfg["input_root"]).resolve()
    cfg["output_root"] = Path(cfg["output_root"]).resolve()
    cfg["section_label"] = str(cfg["section_label"]).strip()
    cfg["weeks"] = [int(w) for w in cfg["weeks"]]
    cfg["axes"] = [str(a).strip().lower() for a in cfg["axes"]]
    scalebar_px = cfg.get("scalebar_measured_pixels")
    scalebar_real = cfg.get("scalebar_real_world_length")
    if scalebar_px is not None or scalebar_real is not None:
        if scalebar_px is None or scalebar_real is None:
            raise ValueError(
                "Both scalebar_measured_pixels and scalebar_real_world_length must be provided together."
            )
        scalebar_px = float(scalebar_px)
        scalebar_real = float(scalebar_real)
        if scalebar_px <= 0 or scalebar_real <= 0:
            raise ValueError("Scalebar calibration values must be positive numbers.")
        cfg["scalebar_measured_pixels"] = scalebar_px
        cfg["scalebar_real_world_length"] = scalebar_real
        cfg["pixel_size"] = scalebar_real / scalebar_px
    else:
        cfg["scalebar_measured_pixels"] = None
        cfg["scalebar_real_world_length"] = None
        cfg["pixel_size"] = float(cfg["pixel_size"])

    cfg["kernel_size"] = int(cfg["kernel_size"])
    cfg["cnt_threshold"] = float(cfg["cnt_threshold"])
    cfg["unit"] = str(cfg["unit"]).strip()
    cfg["auto_scalebar"] = bool(cfg.get("auto_scalebar", False))
    cfg["single_pass_metrics"] = bool(cfg.get("single_pass_metrics", False))
    return cfg


def find_image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def measure_scalebar_pixels(
    image_path: Path,
    *,
    top_fraction: float = 0.35,
    max_bar_height: int = 6,
    min_aspect: float = 4.0,
) -> int | None:
    """Measure the embedded scalebar bar width (px) near the top of an image.

    The exported brain crops carry a thin horizontal scalebar (e.g. a 35-px bar
    labelled "20 mm") near the top. Only the top ``top_fraction`` of rows is
    searched, so the brain (which sits lower and is tall) can never be mistaken
    for the bar. Among components there, the bar is the widest one that is thin
    (height <= ``max_bar_height``) and elongated (width/height >= ``min_aspect``).
    Uses the same inverted threshold the measurement pipeline binarises with.

    Returns the bar width in pixels, or ``None`` when no bar is found
    (e.g. full slices, which carry no embedded scalebar).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # match measurement_batch binarisation
    _, im_bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
    cutoff = max(1, int(round(im_bw.shape[0] * top_fraction)))
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(im_bw[:cutoff, :], 8)
    best_width: int | None = None
    for i in range(1, num):  # label 0 is background
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if h <= max_bar_height and w >= min_aspect * max(h, 1):
            if best_width is None or w > best_width:
                best_width = w
    return best_width


def detect_folder_pixel_size(
    image_files: list[Path],
    real_world_length: float,
    fallback_pixel_size: float,
) -> tuple[float, int | None]:
    """Pick a per-folder pixel size from the median embedded scalebar width.

    Each slice in a folder is rendered at the same zoom, so the bar width is
    constant within a folder; the median guards against the odd missed bar.
    Returns ``(pixel_size, median_bar_px)``, falling back to
    ``fallback_pixel_size`` (and ``None``) when no bar is detected anywhere.
    """
    widths = [w for w in (measure_scalebar_pixels(p) for p in image_files) if w]
    if not widths:
        return fallback_pixel_size, None
    median_px = int(round(float(np.median(widths))))
    return real_world_length / median_px, median_px


def single_pass_metrics(
    image_path: Path,
    pixel_size: float,
    kernel_size: int,
    cnt_threshold: float,
) -> tuple[float, float, float, float, float] | None:
    """Faithful single-pass recompute of the core morphometrics for one image.

    Mirrors the per-image math in ``measurement_batch.process_on_images_batch``
    (same threshold, contour filter, and ``compute_kernel_convex`` /
    ``compactness_2D`` helpers) but WITHOUT the LGI threshold-climbing retry,
    which marches ``cnt_threshold`` past a small cropped brain and zeroes it out.

    Returns ``(area, perimeter, perimeter_convex, lgi, compactness)`` in physical
    units, or ``None`` when no contour survives ``cnt_threshold``.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, im_bw = cv2.threshold(gray, BINARY_THRESHOLD_DEFAULT, 255, 1)
    contours, _ = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) > cnt_threshold]
    if not kept:
        return None

    kernel = compute_kernel_convex(kernel_size)
    inner_mask = np.zeros_like(im_bw)
    cv2.drawContours(inner_mask, kept, -1, 255, thickness=cv2.FILLED)
    closed = cv2.morphologyEx(inner_mask, cv2.MORPH_CLOSE, kernel)
    conv_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept_conv = [c for c in conv_contours if cv2.contourArea(c) > cnt_threshold]

    area = sum(cv2.contourArea(c) for c in kept) * pixel_size ** 2
    perimeter = sum(cv2.arcLength(c, True) for c in kept) * pixel_size
    perimeter_convex = (
        sum(cv2.arcLength(c, True) for c in kept_conv) * pixel_size if kept_conv else 0.0
    )
    lgi = perimeter / perimeter_convex if perimeter_convex else 0.0
    compactness = compactness_2D(area, perimeter)
    return area, perimeter, perimeter_convex, lgi, compactness


def correct_metrics_in_workbook(
    xlsx_path: Path,
    image_dir: Path,
    pixel_size: float,
    kernel_size: int,
    cnt_threshold: float,
) -> int:
    """Overwrite the core-metric cells of each data row with single-pass values.

    Looks up the area/perimeter/perimeter_convex/LGI/Compactness columns by
    header, then for every row whose ``File`` names an image in ``image_dir``
    recomputes those five metrics via :func:`single_pass_metrics`. Rows whose
    ``File`` is not an image (the appended metadata rows) are left untouched.
    Returns the number of rows corrected.
    """
    metric_cols = ("area", "perimeter", "perimeter_convex", "LGI", "Compactness")
    wb = load_workbook(xlsx_path)
    ws = wb.active

    headers: dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        value = ws.cell(row=1, column=col).value
        if isinstance(value, str):
            headers[value.strip()] = col

    missing = [name for name in ("File", *metric_cols) if name not in headers]
    if missing:
        logging.warning("Skipping recompute for %s: missing columns %s", xlsx_path.name, missing)
        return 0

    file_col = headers["File"]
    corrected = 0
    for row in range(2, ws.max_row + 1):
        name = ws.cell(row=row, column=file_col).value
        if not isinstance(name, str):
            continue
        img_path = image_dir / name.strip()
        if not img_path.is_file():  # metadata rows ("PixelSize:", etc.) are not images
            continue
        result = single_pass_metrics(img_path, pixel_size, kernel_size, cnt_threshold)
        values = result if result is not None else (0.0, 0.0, 0.0, 0.0, 0.0)
        if result is None:
            logging.warning("No contour > %s for %s; wrote zeros.", cnt_threshold, name)
        for metric_name, metric_value in zip(metric_cols, values):
            ws.cell(row=row, column=headers[metric_name], value=metric_value)
        corrected += 1

    wb.save(xlsx_path)
    return corrected


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


def add_scalebar_metadata(
    xlsx_path: Path,
    *,
    measured_pixels: float | None,
    real_world_length: float | None,
    unit: str,
) -> None:
    """Append optional scale-from-scalebar metadata rows to the report workbook."""
    if measured_pixels is None or real_world_length is None:
        return
    wb = load_workbook(xlsx_path)
    ws = wb.active
    ws.append(["ScalebarMeasuredPixels:", measured_pixels])
    ws.append(["ScalebarRealWorldLength:", real_world_length])
    ws.append(["ScalebarRealWorldUnit:", unit])
    wb.save(xlsx_path)


def add_total_depth_column(xlsx_path: Path, unit: str) -> int:
    """Insert ``total_depth_<unit>`` as count * mean_depth for each data row."""
    wb = load_workbook(xlsx_path)
    ws = wb.active

    headers: dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        value = ws.cell(row=1, column=col).value
        if isinstance(value, str):
            headers[value.strip()] = col

    count_col = headers.get("Sulci_count")
    mean_col = headers.get(f"mean_depth_{unit}")
    total_name = f"total_depth_{unit}"
    if count_col is None or mean_col is None:
        logging.warning(
            "Skipping total-depth insertion for %s: missing Sulci_count or %s",
            xlsx_path.name,
            total_name,
        )
        return 0

    total_col = headers.get(total_name)
    if total_col is None:
        insert_at = mean_col
        ws.insert_cols(insert_at)
        ws.cell(row=1, column=insert_at, value=total_name)
        total_col = insert_at
        if mean_col >= insert_at:
            mean_col += 1

    updated = 0
    for row_idx in range(2, ws.max_row + 1):
        first_cell = ws.cell(row=row_idx, column=1).value
        if first_cell in (None, "") or is_metadata_row(first_cell):
            continue
        count_value = ws.cell(row=row_idx, column=count_col).value
        mean_value = ws.cell(row=row_idx, column=mean_col).value
        if count_value in (None, "") or mean_value in (None, ""):
            ws.cell(row=row_idx, column=total_col, value=None)
            continue
        try:
            total_value = float(count_value) * float(mean_value)
        except (TypeError, ValueError):
            ws.cell(row=row_idx, column=total_col, value=None)
            continue
        cell = ws.cell(row=row_idx, column=total_col, value=total_value)
        cell.number_format = "0.000000"
        updated += 1

    wb.save(xlsx_path)
    return updated


def run_folder(
    input_dir: Path,
    output_dir: Path,
    week: int,
    axis: str,
    *,
    pixel_size: float,
    scalebar_measured_pixels: float | None,
    scalebar_real_world_length: float | None,
    kernel_size: int,
    cnt_threshold: float,
    unit: str,
    auto_scalebar: bool = False,
    single_pass_metrics_enabled: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-folder calibration: prefer the images' own embedded scalebar over the
    # single configured pixel size, since each week/axis is rendered at its own
    # zoom. Falls back to the configured pixel_size when no bar is detected.
    eff_pixel_size = pixel_size
    detected_bar_px: int | None = None
    if auto_scalebar and scalebar_real_world_length:
        eff_pixel_size, detected_bar_px = detect_folder_pixel_size(
            find_image_files(input_dir), scalebar_real_world_length, pixel_size
        )
        if detected_bar_px:
            logging.info(
                "Auto-scalebar week=%s axis=%s: %d px = %.4f %s -> %.6f %s/pixel",
                week, axis, detected_bar_px, scalebar_real_world_length, unit,
                eff_pixel_size, unit,
            )
        else:
            logging.warning(
                "No scalebar detected in %s; using configured %.6f %s/pixel",
                input_dir, eff_pixel_size, unit,
            )

    process_on_images_batch(
        str(input_dir),
        str(output_dir),
        pixel_size=eff_pixel_size,
        kernel_size=kernel_size,
        cnt_threshold=cnt_threshold,
        unit=unit,
    )

    default_xlsx = output_dir / "Batch_Allmarks.xlsx"
    final_xlsx = output_dir / f"week{week}_{axis}_Batch_Allmarks.xlsx"
    if final_xlsx.exists():
        final_xlsx.unlink()
    default_xlsx.replace(final_xlsx)
    add_scalebar_metadata(
        final_xlsx,
        measured_pixels=(float(detected_bar_px) if detected_bar_px else scalebar_measured_pixels),
        real_world_length=scalebar_real_world_length,
        unit=unit,
    )
    n_total = add_total_depth_column(final_xlsx, unit)
    logging.info("Added/updated total-depth values in %d data row(s) in %s", n_total, final_xlsx.name)

    # Recompute the core metrics single-pass BEFORE the summary table is built,
    # so the appended means reflect the corrected (retry-free) values.
    if single_pass_metrics_enabled:
        n = correct_metrics_in_workbook(
            final_xlsx, input_dir, eff_pixel_size, kernel_size, cnt_threshold
        )
        logging.info("Single-pass recompute corrected %d data row(s) in %s", n, final_xlsx.name)

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

    if settings["auto_scalebar"]:
        logging.info(
            "Auto-scalebar ON: pixel size measured per folder from the embedded "
            "%.6f %s scalebar (configured %.6f %s/pixel used only as fallback).",
            settings["scalebar_real_world_length"],
            settings["unit"],
            settings["pixel_size"],
            settings["unit"],
        )
    elif settings["scalebar_measured_pixels"] is not None:
        logging.info(
            "Using scalebar calibration: %.6f %s/pixel from %.6f px = %.6f %s",
            settings["pixel_size"],
            settings["unit"],
            settings["scalebar_measured_pixels"],
            settings["scalebar_real_world_length"],
            settings["unit"],
        )
    else:
        logging.info(
            "Using direct pixel size: %.6f %s/pixel",
            settings["pixel_size"],
            settings["unit"],
        )
    if settings["single_pass_metrics"]:
        logging.info(
            "Single-pass metrics ON: area/perimeter/perimeter_convex/LGI/Compactness "
            "recomputed without the LGI retry after each batch run."
        )

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
                    scalebar_measured_pixels=settings["scalebar_measured_pixels"],
                    scalebar_real_world_length=settings["scalebar_real_world_length"],
                    kernel_size=settings["kernel_size"],
                    cnt_threshold=settings["cnt_threshold"],
                    unit=settings["unit"],
                    auto_scalebar=settings["auto_scalebar"],
                    single_pass_metrics_enabled=settings["single_pass_metrics"],
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
