"""Gestational-week reference statistics loaded from a CSV.

Mirrors the per-week structure documented in ``profile_info.md`` (metadata
plus three metric-summary tables) for gestational weeks 24-38.
"""

from dataclasses import dataclass
from pathlib import Path

import math
import pandas as pd


@dataclass(frozen=True)
class MetricStats:
    """Summary statistics for a single metric within a week."""
    n: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None


@dataclass(frozen=True)
class WeekProfile:
    """All reference values for a single gestational week."""
    # metadata
    week: int
    axis: str
    slice_rows_analyzed: int
    pixel_size_units: str
    kernel_size: int
    # core metric summary
    area: MetricStats
    perimeter: MetricStats
    lgi: MetricStats
    compactness: MetricStats
    # rounded sulcus count summary
    sulci_count: MetricStats
    primary_count: MetricStats
    secondary_count: MetricStats
    tertiary_count: MetricStats
    unclassified_count: MetricStats
    # sulcus value summary
    primary_sulcus_values: MetricStats
    secondary_sulcus_values: MetricStats
    tertiary_sulcus_values: MetricStats
    unclassified_sulcus_values: MetricStats


_METRIC_FIELDS: tuple[str, ...] = (
    "area",
    "perimeter",
    "lgi",
    "compactness",
    "sulci_count",
    "primary_count",
    "secondary_count",
    "tertiary_count",
    "unclassified_count",
    "primary_sulcus_values",
    "secondary_sulcus_values",
    "tertiary_sulcus_values",
    "unclassified_sulcus_values",
)


class GestationalWeekProfile:
    """Reference statistics indexed by gestational week (24-38).

    Construct with the path to a CSV in the wide format described in the
    plan: one row per week, columns ``week, axis, slice_rows_analyzed,
    pixel_size_units, kernel_size`` followed by ``{metric}_n``,
    ``{metric}_mean``, ``{metric}_std``, ``{metric}_min``, ``{metric}_max``
    for every metric listed on :class:`WeekProfile`.
    """

    VALID_RANGE = range(24, 39)

    def __init__(self, csv_path: str | Path) -> None:
        self._profiles: dict[int, WeekProfile] = self._load(Path(csv_path))

    def get(self, week: int) -> WeekProfile | None:
        """Return the profile for *week*, or ``None`` if not loaded."""
        return self._profiles.get(int(week))

    def weeks(self) -> list[int]:
        """Sorted list of gestational weeks available in the registry."""
        return sorted(self._profiles)

    @staticmethod
    def _load(path: Path) -> dict[int, WeekProfile]:
        df = pd.read_csv(path)
        profiles: dict[int, WeekProfile] = {}
        for _, row in df.iterrows():
            metric_stats = {
                name: MetricStats(
                    n=int(row[f"{name}_n"]),
                    mean=_clean(row[f"{name}_mean"]),
                    std=_clean(row[f"{name}_std"]),
                    min=_clean(row[f"{name}_min"]),
                    max=_clean(row[f"{name}_max"]),
                )
                for name in _METRIC_FIELDS
            }
            profile = WeekProfile(
                week=int(row["week"]),
                axis=str(row["axis"]),
                slice_rows_analyzed=int(row["slice_rows_analyzed"]),
                pixel_size_units=str(row["pixel_size_units"]),
                kernel_size=int(row["kernel_size"]),
                **metric_stats,
            )
            profiles[profile.week] = profile
        return profiles


def _clean(value) -> float | None:
    """Map pandas NaN / empty cells to ``None``; otherwise cast to float."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f
