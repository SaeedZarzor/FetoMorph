"""Excel ingestion and column normalisation for FetoMorph optimisation.

Reads one or more Excel files produced by the measurement pipelines,
normalises metric column names (handles typos and variant spellings),
strips metadata rows, and concatenates the results into a single
``pandas.DataFrame`` ready for the NSGA optimiser.
"""

import os
from typing import List, Tuple
import pandas as pd
from PySide6.QtWidgets import QMessageBox


STANDARD_METRIC_COLUMNS = [
    "area",
    "LGI",
    "SulciCount",
    "MaxDepth",
    "MinDepth",
    "MeanDepth",
]

_ALIASES = {
    "area": "area",
    "lgi": "LGI",
    "perimeterrate": "LGI",
    "sulcicount": "SulciCount",
    "sulcicounts": "SulciCount",
    "maxdepth": "MaxDepth",
    "maxdpeth": "MaxDepth",  # frequent typo
    "mindepth": "MinDepth",
    "meandepth": "MeanDepth",
}


def _norm_col_name(name: str) -> str:
    """Lowercase, strip whitespace/punctuation for fuzzy column matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(":", "")
    )


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical metric names using ``_ALIASES``."""
    rename_map = {}
    for col in df.columns:
        key = _norm_col_name(col)
        target = _ALIASES.get(key)
        if target is not None and col != target:
            rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def get_max_sulcicount(df: pd.DataFrame):
    """Return the maximum SulciCount value in *df*, or ``None``."""
    if "SulciCount" not in df.columns:
        return None
    vals = pd.to_numeric(df["SulciCount"], errors="coerce").dropna()
    if vals.empty:
        return None
    return int(vals.max())


def get_max_celldensity(df: pd.DataFrame):
    """Return the maximum CellDensity value in *df*, or ``None``."""
    if "CellDensity" not in df.columns:
        return None
    vals = pd.to_numeric(df["CellDensity"], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.max())



def conver_excel(file_paths: list[str]) -> tuple[pd.DataFrame, int | None, float | None]:
    """Load and concatenate measurement Excel files for optimisation.

    Strips metadata rows (``PixelSize:``, etc.), normalises column names,
    and returns the merged DataFrame along with the maximum SulciCount
    and CellDensity values found (used as constraint upper bounds).

    Args:
        file_paths: List of ``.xlsx`` paths produced by FetoMorph.

    Returns:
        Tuple of ``(df, max_sulci_count, max_cell_density)``.
    """
    df = {}
    ignore_values = ["PixelSize:", "PixelSizeUnits:", "KernelSize:"]

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Path does not exist: {file_path}")
        elif not os.path.isfile(file_path):
            print(f"Path is not a file: {file_path}")
        else:
            temp_df = pd.read_excel(file_path)
            temp_df.columns = temp_df.columns.str.strip()
            temp_df = _normalize_metric_columns(temp_df)
            # remove rows where File contains metadata labels
            if "File" in temp_df.columns:
                temp_df = temp_df[~temp_df["File"].isin(ignore_values)].copy()
            # Keep origin so multi-file runs can be mapped back to the right source file.
            temp_df["__source_excel_path"] = file_path
            temp_df["__source_excel_name"] = os.path.basename(file_path)
            df[file_path] = temp_df
            print(f"The file: {file_path} has been loaded")

        
    df1 = pd.concat(df.values(), axis=0, ignore_index=True)

    n_rows = len(df1)
    print(f"[Read Excel] Total rows loaded: {n_rows}")
    if n_rows == 0:
        QMessageBox.critical(None, "Error", "No valid data found in the provided Excel files.")
        print("No valid data found in the provided Excel files.")
        return pd.DataFrame(), None, None

    max_sulci_count = get_max_sulcicount(df1)
    if max_sulci_count is not None:
        print(f"[Read Excel] Maximum SulciCount found: {max_sulci_count}")
    else:
        print("[Read Excel] SulciCount is missing or non-numeric.")

    max_cell_density = get_max_celldensity(df1)
    if max_cell_density is not None:
        print(f"[Read Excel] Maximum CellDensity found: {max_cell_density}")
    else:
        print("[Read Excel] CellDensity is missing or non-numeric.")

    return df1, max_sulci_count, max_cell_density
