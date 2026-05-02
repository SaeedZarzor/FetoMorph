from __future__ import annotations

import argparse
import io
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage


SULCUS_CLASSES = ("Primary", "Secondary", "Tertiary", "Unclassified")
CORE_METRICS = ("area", "perimeter", "LGI", "Compactness")
COUNT_METRICS = ("Sulci_count", "Primary_count", "Secondary_count", "Tertiary_count", "Unclassified_count")
METADATA_LABELS = {
    "PixelSize:",
    "PixelSizeUnits:",
    "KernelSize:",
    "ScalebarMeasuredPixels:",
    "ScalebarRealWorldLength:",
    "ScalebarRealWorldUnit:",
    "PixelSizeUnits",
    "KernelSize",
}
WORKBOOK_PATTERN = re.compile(r"week(?P<week>\d+)_(?P<axis>[A-Za-z]+)_Batch_Allmarks\.xlsx$", re.IGNORECASE)
ANALYSIS_SHEET_NAME = "Analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze each master measurement report workbook independently and "
            "write summary stats plus boxplots back into the same Excel file."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("measurements"),
        help="Root folder containing week/axis measurement report workbooks.",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["axial", "coronal", "sagittal"],
        help="Axis names to include. Defaults to axial, coronal, and sagittal.",
    )
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        help="Optional weeks to include, for example: --weeks 24 25 26",
    )
    parser.add_argument(
        "--report-glob",
        default="week*_Batch_Allmarks.xlsx",
        help="Glob used below input-root to discover workbooks.",
    )
    parser.add_argument(
        "--sheet-name",
        default=ANALYSIS_SHEET_NAME,
        help="Worksheet name used for the generated analysis output.",
    )
    return parser.parse_args()


def discover_workbooks(input_root: Path, report_glob: str, axes: set[str], weeks: set[int] | None) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(input_root.rglob(report_glob)):
        match = WORKBOOK_PATTERN.match(path.name)
        if not match:
            continue
        axis = match.group("axis").lower()
        week = int(match.group("week"))
        if axis not in axes:
            continue
        if weeks is not None and week not in weeks:
            continue
        paths.append(path)
    return paths


def is_data_row(file_value: object) -> bool:
    if not isinstance(file_value, str):
        return False
    text = file_value.strip()
    if not text:
        return False
    if text in METADATA_LABELS:
        return False
    if text.endswith(":"):
        return False
    return True


def clean_dataframe(df: pd.DataFrame, workbook_path: Path) -> pd.DataFrame:
    if "File" not in df.columns:
        raise ValueError(f"{workbook_path} does not contain a 'File' column.")

    keep_cols = []
    for col in df.columns:
        if col in {"Metric", "Mean"}:
            continue
        if col is None:
            continue
        if isinstance(col, str) and col.startswith("Unnamed:"):
            continue
        keep_cols.append(col)

    df = df[keep_cols].copy()
    df = df[df["File"].map(is_data_row)].copy()
    return df.reset_index(drop=True)


def parse_workbook(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df = clean_dataframe(df, path)

    match = WORKBOOK_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Workbook name does not match expected pattern: {path.name}")

    df["week"] = int(match.group("week"))
    df["axis"] = match.group("axis").lower()
    df["workbook"] = path.name
    return df


def infer_depth_unit(df: pd.DataFrame) -> str:
    for col in df.columns:
        if not isinstance(col, str):
            continue
        match = re.search(r"_([A-Za-z]+)$", col)
        if match and col.startswith(("min_depth_", "mean_depth_", "Primary_v1_", "Secondary_v1_")):
            return match.group(1)
    return "mm"


def round_count_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def existing_columns(df: pd.DataFrame, names: Iterable[str]) -> list[str]:
    return [name for name in names if name in df.columns]


def expand_sulcus_values(row: pd.Series, prefix: str, unit: str) -> list[float]:
    count_col = f"{prefix}_count"
    count_raw = row.get(count_col)
    if pd.isna(count_raw):
        return []

    count = int(round(float(count_raw)))
    if count <= 0:
        return []

    raw_cols = [f"{prefix}_v1_{unit}", f"{prefix}_v2_{unit}", f"{prefix}_v3_{unit}"]
    raw_values = [
        float(row[col])
        for col in raw_cols
        if col in row.index and pd.notna(row[col])
    ]
    if raw_values:
        return raw_values[:count]

    min_col = f"{prefix}_min_{unit}"
    max_col = f"{prefix}_max_{unit}"
    mean_col = f"{prefix}_mean_{unit}"
    min_value = row.get(min_col)
    max_value = row.get(max_col)
    mean_value = row.get(mean_col)

    if pd.notna(min_value) and pd.notna(max_value) and pd.notna(mean_value):
        min_float = float(min_value)
        max_float = float(max_value)
        mean_float = float(mean_value)
        if count == 1:
            return [mean_float]
        if count == 2:
            return [min_float, max_float]
        filler_count = count - 2
        filler = ((mean_float * count) - min_float - max_float) / filler_count
        return [min_float] + [filler] * filler_count + [max_float]

    if pd.notna(mean_value):
        return [float(mean_value)] * count
    if pd.notna(min_value) and pd.notna(max_value):
        if count == 1:
            return [float(min_value)]
        return [float(min_value)] + [float(max_value)] * (count - 1)
    return []


def build_sulcus_value_table(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        for prefix in SULCUS_CLASSES:
            for value in expand_sulcus_values(row, prefix, unit):
                rows.append(
                    {
                        "slice_file": row["File"],
                        "sulcus_class": prefix.lower(),
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def metric_summary(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "n": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else math.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                "min": float(values.min()) if not values.empty else math.nan,
                "max": float(values.max()) if not values.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def rounded_counts_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["File"] + existing_columns(df, COUNT_METRICS)
    rounded = df[cols].copy()
    for col in existing_columns(df, COUNT_METRICS):
        rounded[f"{col}_rounded"] = round_count_series(df[col])
    return rounded


def count_summary(rounded_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in rounded_df.columns:
        if not col.endswith("_rounded"):
            continue
        values = pd.to_numeric(rounded_df[col], errors="coerce").dropna().astype(int)
        rows.append(
            {
                "metric": col,
                "n": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else math.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                "min": int(values.min()) if not values.empty else pd.NA,
                "max": int(values.max()) if not values.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def per_class_sulcus_summary(sulcus_values_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_name in [name.lower() for name in SULCUS_CLASSES]:
        values = pd.to_numeric(
            sulcus_values_df.loc[sulcus_values_df["sulcus_class"] == class_name, "value"],
            errors="coerce",
        ).dropna()
        rows.append(
            {
                "metric": f"{class_name}_sulcus_values",
                "n": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else math.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                "min": float(values.min()) if not values.empty else math.nan,
                "max": float(values.max()) if not values.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_boxplot(values: pd.Series, title: str, ylabel: str) -> io.BytesIO | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.boxplot(clean, patch_artist=True, boxprops={"facecolor": "#9ecae1"})
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    image_bytes = io.BytesIO()
    fig.savefig(image_bytes, format="png", dpi=180)
    plt.close(fig)
    image_bytes.seek(0)
    return image_bytes


def plot_grouped_sulcus_boxplot(sulcus_values_df: pd.DataFrame, ylabel: str) -> io.BytesIO | None:
    grouped = []
    labels = []
    for class_name in [name.lower() for name in SULCUS_CLASSES]:
        values = pd.to_numeric(
            sulcus_values_df.loc[sulcus_values_df["sulcus_class"] == class_name, "value"],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue
        grouped.append(values)
        labels.append(class_name)

    if not grouped:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.boxplot(grouped, labels=labels, patch_artist=True)
    ax.set_title("Sulcus value distributions by class")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    image_bytes = io.BytesIO()
    fig.savefig(image_bytes, format="png", dpi=180)
    plt.close(fig)
    image_bytes.seek(0)
    return image_bytes


def write_dataframe(ws, start_row: int, start_col: int, title: str, df: pd.DataFrame) -> int:
    ws.cell(row=start_row, column=start_col, value=title)
    if df.empty:
        ws.cell(row=start_row + 1, column=start_col, value="No data")
        return start_row + 3

    header_row = start_row + 1
    for offset, col_name in enumerate(df.columns):
        ws.cell(row=header_row, column=start_col + offset, value=str(col_name))

    for row_offset, row in enumerate(df.itertuples(index=False), start=1):
        for col_offset, value in enumerate(row):
            ws.cell(row=header_row + row_offset, column=start_col + col_offset, value=normalize_excel_value(value))

    return header_row + len(df) + 2


def normalize_excel_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value


def replace_analysis_sheet(workbook_path: Path, sheet_name: str):
    wb = load_workbook(workbook_path)
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(title=sheet_name)
    return wb, ws


def add_image(ws, image_bytes: io.BytesIO | None, anchor: str, width: int = 520, height: int = 360) -> None:
    if image_bytes is None:
        return
    img = XLImage(image_bytes)
    img.width = width
    img.height = height
    ws.add_image(img, anchor)


def analyze_workbook(path: Path, sheet_name: str) -> None:
    df = parse_workbook(path)
    depth_unit = infer_depth_unit(df)

    for col in existing_columns(df, CORE_METRICS):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rounded_counts_df = rounded_counts_table(df)
    core_summary_df = metric_summary(df, CORE_METRICS)
    count_summary_df = count_summary(rounded_counts_df)
    sulcus_values_df = build_sulcus_value_table(df, depth_unit)
    sulcus_summary_df = per_class_sulcus_summary(sulcus_values_df)

    wb, ws = replace_analysis_sheet(path, sheet_name)

    match = WORKBOOK_PATTERN.match(path.name)
    assert match is not None
    week = int(match.group("week"))
    axis = match.group("axis").lower()

    ws["A1"] = "Workbook Analysis"
    ws["A2"] = "Workbook"
    ws["B2"] = path.name
    ws["A3"] = "Week"
    ws["B3"] = week
    ws["A4"] = "Axis"
    ws["B4"] = axis
    ws["A5"] = "Slice rows analyzed"
    ws["B5"] = len(df)
    ws["A6"] = "Note"
    ws["B6"] = (
        "When a sulcus class count exceeds 3, approximate raw sulcus values are "
        "reconstructed from count + min/max/mean for summary stats and boxplots."
    )

    next_row = 8
    next_row = write_dataframe(ws, next_row, 1, "Core Metric Summary", core_summary_df)
    next_row = write_dataframe(ws, next_row, 1, "Rounded Sulcus Counts Per Slice", rounded_counts_df)
    next_row = write_dataframe(ws, next_row, 1, "Rounded Sulcus Count Summary", count_summary_df)
    next_row = write_dataframe(ws, next_row, 1, "Sulcus Value Summary", sulcus_summary_df)

    core_metric_plots = [
        ("area", "P2"),
        ("perimeter", "P24"),
        ("LGI", "P46"),
        ("Compactness", "P68"),
    ]
    for metric, anchor in core_metric_plots:
        if metric not in df.columns:
            continue
        add_image(
            ws,
            plot_boxplot(df[metric], f"{metric} distribution", metric),
            anchor,
        )

    sulcus_plot_anchors = {
        "primary": "Y2",
        "secondary": "Y24",
        "tertiary": "Y46",
        "unclassified": "Y68",
    }
    for class_name, anchor in sulcus_plot_anchors.items():
        class_values = sulcus_values_df.loc[sulcus_values_df["sulcus_class"] == class_name, "value"]
        add_image(
            ws,
            plot_boxplot(
                class_values,
                f"{class_name.capitalize()} sulcus values distribution",
                f"depth ({depth_unit})",
            ),
            anchor,
        )

    add_image(
        ws,
        plot_grouped_sulcus_boxplot(sulcus_values_df, f"depth ({depth_unit})"),
        "AJ2",
        width=620,
        height=360,
    )

    wb.save(path)


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    axes = {axis.strip().lower() for axis in args.axes}
    weeks = set(args.weeks) if args.weeks else None

    report_paths = discover_workbooks(input_root, args.report_glob, axes, weeks)
    if not report_paths:
        raise SystemExit(f"No matching workbooks found under {input_root}")

    for path in report_paths:
        analyze_workbook(path, args.sheet_name)
        print(f"Updated workbook: {path}")

    print(f"Analyzed {len(report_paths)} workbooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
