"""Excel ingestion and column normalisation for FetoMorph optimisation.

Reads one or more Excel files produced by the measurement pipelines,
normalises metric column names (handles typos and variant spellings),
strips metadata rows, and concatenates the results into a single
``pandas.DataFrame`` ready for the NSGA optimiser.
"""

import re

from deps import *


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
    # Spec-layout ("Mean results" table) names written by
    # write_results_workbook. Unit suffixes like "(mm)" are already
    # stripped by _norm_col_name.
    "section": "File",
    "filename": "File",
    "maxsulcidepth": "MaxDepth",
    "minsulcidepth": "MinDepth",
    "meansulcidepth": "MeanDepth",
    "celldensity": "CellDensity",
}

# Per-subtype sulci counts in the spec layout; summed into "SulciCount",
# and used as the weights when reconstructing a mean depth.
_SULCI_COUNT_PARTS = (
    "PrimarySulciCount",
    "SecondarySulciCount",
    "TertiarySulciCount",
    "UnclassifiedSulciCount",
)
_SUBTYPE_MEAN_DEPTHS = (
    "PrimaryMeanDepth",
    "SecondaryMeanDepth",
    "TertiaryMeanDepth",
    "UnclassifiedMeanDepth",
)

# Trailing unit suffix, e.g. "MeanSulciDepth (mm)" → "MeanSulciDepth".
_UNIT_SUFFIX_RE = re.compile(r"\([^)]*\)\s*$")

# Individual per-sulcus depth columns, e.g. "Primary_depth_1",
# "Tertiary_depth_10". The trailing \d+ deliberately excludes the
# "..._norm" variants, which are normalised ratios rather than lengths
# and must never be mixed into a depth in mm.
_PER_SULCUS_DEPTH_RE = re.compile(
    r"^(?:Primary|Secondary|Tertiary|Unclassified)_depth_\d+$"
)


def _norm_col_name(name: str) -> str:
    """Lowercase, strip whitespace/punctuation for fuzzy column matching."""
    return (
        _UNIT_SUFFIX_RE.sub("", str(name).strip())
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


def _read_structured_workbook(file_path: str) -> pd.DataFrame | None:
    """Read a workbook written by :func:`write_results_workbook`.

    Those files put a title, metadata, Parameters and Totals blocks
    *above* the "Mean results" table and start the data in column B, so
    a plain ``pd.read_excel`` sees ``Unnamed: 0`` headers and no metric
    columns. Every spec-layout sheet in the workbook is parsed and the
    per-slice rows concatenated.

    Returns ``None`` when the file is not in the spec layout, so the
    caller can fall back to reading it as a flat table.
    """
    try:
        from openpyxl import load_workbook

        from helpers.results_excel_format import parse_results_worksheet
    except Exception as ex:  # pragma: no cover - openpyxl is a hard dep
        print(f"[Read Excel] Spec-layout reader unavailable: {ex}")
        return None

    try:
        wb = load_workbook(file_path, data_only=True, read_only=False)
    except Exception as ex:
        print(f"[Read Excel] Could not open {file_path}: {ex}")
        return None

    frames = []
    try:
        for ws in wb.worksheets:
            try:
                parsed = parse_results_worksheet(ws)
            except Exception as ex:
                print(f"[Read Excel] Skipping sheet '{ws.title}': {ex}")
                continue
            rows = parsed.get("rows") or []
            if not rows:
                continue  # e.g. an embedded per-slice image tab
            sub = pd.DataFrame(rows)
            # Sheet-level metadata: the spec layout carries the source
            # image name / folder in the header, not in every row.
            sub["__source_sheet"] = ws.title
            if parsed.get("file_name"):
                sub["__source_file_name"] = parsed["file_name"]
            if parsed.get("folder"):
                sub["__source_folder"] = parsed["folder"]
            frames.append(sub)
    finally:
        wb.close()

    if not frames:
        return None
    return pd.concat(frames, axis=0, ignore_index=True)


def _derive_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the flat metrics the optimiser needs from spec-layout columns.

    The spec layout splits sulci into Primary/Secondary/Tertiary/
    Unclassified, so the single ``SulciCount`` and ``MeanDepth`` the
    optimiser expects no longer exist and are reconstructed here.
    """
    if "SulciCount" not in df.columns:
        parts = [c for c in _SULCI_COUNT_PARTS if c in df.columns]
        if parts:
            counts = [pd.to_numeric(df[c], errors="coerce").fillna(0) for c in parts]
            df["SulciCount"] = sum(counts).astype(int)

    # The aggregate Max/Min/MeanSulciDepth columns are absent whenever the
    # exporter dropped them, so fall back to scanning every individual
    # sulcus depth across all four classes (Primary/Secondary/Tertiary/
    # Unclassified) and taking the extremes across the whole row.
    depth_cols = [c for c in df.columns if _PER_SULCUS_DEPTH_RE.match(str(c))]
    if depth_cols:
        depths = df[depth_cols].apply(pd.to_numeric, errors="coerce")
        if "MaxDepth" not in df.columns:
            df["MaxDepth"] = depths.max(axis=1, skipna=True)
        if "MinDepth" not in df.columns:
            df["MinDepth"] = depths.min(axis=1, skipna=True)
        if "MeanDepth" not in df.columns:
            # Mean over the individual sulci — exact, and it needs no
            # per-class count weighting.
            df["MeanDepth"] = depths.mean(axis=1, skipna=True)

    if "MeanDepth" not in df.columns:
        # Weight each subtype's mean by its own count — a plain average of
        # the four means would misweight slices with lopsided sulci counts.
        pairs = [
            (m, c)
            for m, c in zip(_SUBTYPE_MEAN_DEPTHS, _SULCI_COUNT_PARTS)
            if m in df.columns and c in df.columns
        ]
        if pairs:
            total_depth = 0.0
            total_count = 0.0
            for mean_col, count_col in pairs:
                means = pd.to_numeric(df[mean_col], errors="coerce")
                counts = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
                counts = counts.where(means.notna(), 0)
                total_depth = total_depth + means.fillna(0) * counts
                total_count = total_count + counts
            df["MeanDepth"] = (total_depth / total_count).where(total_count > 0)

    for col in STANDARD_METRIC_COLUMNS + ["CellDensity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # A slice with zero detected sulci has no folding, so its sulci depth is
    # 0 rather than missing — the early gestational weeks are near-smooth and
    # the exporter drops the depth columns entirely when nothing is detected.
    # Filling only where the count is known to be 0 keeps a genuinely unknown
    # depth as NaN (those rows get dropped by the optimiser instead).
    if "SulciCount" in df.columns:
        no_sulci = pd.to_numeric(df["SulciCount"], errors="coerce") == 0
        if no_sulci.any():
            for col in ("MaxDepth", "MinDepth", "MeanDepth"):
                if col not in df.columns:
                    df[col] = float("nan")
                df.loc[no_sulci & df[col].isna(), col] = 0.0
    return df


# Columns that identify a row rather than measure it, so they can never be
# optimised even though some of them read as numeric.
NON_METRIC_COLUMNS = {"File", "Section", "Slice", "GestationalWeek"}

# Per-sulcus columns ("Primary_depth_3", "Unclassified_depth_2_norm"): one
# value per *sulcus*, not per slice, and how many exist changes from file to
# file. Offering them as objectives would make the dialog's contents depend on
# whichever file happened to be loaded, so they are filtered out. The
# aggregates derived from them (Max/Min/MeanDepth) are offered instead.
_PER_SULCUS_COLUMN_RE = re.compile(r"_depth_\d+(?:_norm)?$", re.IGNORECASE)


def get_optimizable_columns(df: pd.DataFrame) -> list[str]:
    """Return the per-slice numeric columns usable as objectives/constraints.

    Any column of *df* that holds at least one number qualifies, so metrics
    added to the exporter later show up in the optimisation dialog without
    code changes. Bookkeeping columns (``__source_excel_path``), identifiers
    (:data:`NON_METRIC_COLUMNS`) and per-sulcus values are excluded.

    Args:
        df: Merged measurement DataFrame, as returned by :func:`conver_excel`.

    Returns:
        Column names in the order they appear in *df*.
    """
    columns = []
    for col in df.columns:
        name = str(col)
        if name.startswith("__") or name in NON_METRIC_COLUMNS:
            continue
        if _PER_SULCUS_COLUMN_RE.search(name):
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().any():
            columns.append(name)
    return columns


def get_column_ranges(
    df: pd.DataFrame, columns: list[str] | None = None
) -> dict[str, tuple[float, float]]:
    """Return ``{column: (min, max)}`` over the numeric values of each column.

    Used to bound the constraint spin boxes to what the data can actually
    satisfy. Columns with no numeric values are omitted.
    """
    if columns is None:
        columns = get_optimizable_columns(df)
    ranges: dict[str, tuple[float, float]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        ranges[col] = (float(vals.min()), float(vals.max()))
    return ranges


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
            # Files written by write_results_workbook (Batch_Allmarks and the
            # other measurement exports) put the table under title/Parameters/
            # Totals blocks, so they must be parsed structurally. Older flat
            # exports fall back to a plain read.
            temp_df = _read_structured_workbook(file_path)
            structured = temp_df is not None
            if temp_df is None:
                temp_df = pd.read_excel(file_path)
            temp_df.columns = temp_df.columns.astype(str).str.strip()
            temp_df = _normalize_metric_columns(temp_df)
            temp_df = _derive_metric_columns(temp_df)
            # remove rows where File contains metadata labels
            if "File" in temp_df.columns:
                temp_df = temp_df[~temp_df["File"].isin(ignore_values)].copy()
            # Keep origin so multi-file runs can be mapped back to the right source file.
            temp_df["__source_excel_path"] = file_path
            temp_df["__source_excel_name"] = os.path.basename(file_path)
            df[file_path] = temp_df
            print(f"The file: {file_path} has been loaded")
            layout = "spec-layout" if structured else "flat table"
            resolved = [c for c in temp_df.columns if not str(c).startswith("__")]
            missing_std = [c for c in STANDARD_METRIC_COLUMNS
                           if c not in temp_df.columns]
            print(f"[Read Excel]   layout: {layout}, rows: {len(temp_df)}")
            print(f"[Read Excel]   columns after normalisation: {resolved}")
            if missing_std:
                print(f"[Read Excel]   WARNING missing standard metrics: "
                      f"{missing_std}")

        
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
