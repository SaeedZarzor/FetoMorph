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
    kernel_size_mm: float
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
    """Reference statistics indexed by (gestational week, axis).

    Construct with the path to a CSV in the wide format: one row per
    (week, axis) pair, columns ``week, axis, slice_rows_analyzed,
    pixel_size_units, kernel_size_mm`` followed by ``{metric}_n``,
    ``{metric}_mean``, ``{metric}_std``, ``{metric}_min``, ``{metric}_max``
    for every metric listed on :class:`WeekProfile`.
    """

    VALID_RANGE = range(24, 39)
    AXES = ("axial", "coronal", "sagittal")

    def __init__(self, csv_path: str | Path) -> None:
        self._profiles: dict[tuple[int, str], WeekProfile] = self._load(Path(csv_path))

    def get(self, week: int, axis: str | None = None) -> WeekProfile | None:
        """Return the profile for *week* and *axis*.

        If *axis* is ``None``, return the first available axis for that week.
        """
        if axis is not None:
            return self._profiles.get((int(week), axis.lower()))
        for a in self.AXES:
            p = self._profiles.get((int(week), a))
            if p is not None:
                return p
        return None

    def weeks(self) -> list[int]:
        """Sorted, deduplicated list of gestational weeks available."""
        return sorted({w for w, _ in self._profiles})

    def axes_for_week(self, week: int) -> list[str]:
        """Return the available axes for a given week."""
        return [a for a in self.AXES if (int(week), a) in self._profiles]

    @staticmethod
    def _load(path: Path) -> dict[tuple[int, str], WeekProfile]:
        df = pd.read_csv(path)
        profiles: dict[tuple[int, str], WeekProfile] = {}
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
            kernel_size_mm = (
                _clean(row["kernel_size_mm"])
                if "kernel_size_mm" in row.index
                else _clean(row.get("kernel_size"))
            )
            profile = WeekProfile(
                week=int(row["week"]),
                axis=str(row["axis"]),
                slice_rows_analyzed=int(row["slice_rows_analyzed"]),
                pixel_size_units=str(row["pixel_size_units"]),
                kernel_size_mm=float(kernel_size_mm) if kernel_size_mm is not None else 0.0,
                **metric_stats,
            )
            profiles[(profile.week, profile.axis.lower())] = profile
        return profiles


METRIC_MAP: dict[str, str] = {
    "Area": "area",
    "Perimeter": "perimeter",
    "LGI": "lgi",
    "Compactness": "compactness",
    "PrimarySulciCount": "primary_count",
    "SecondarySulciCount": "secondary_count",
    "TertiarySulciCount": "tertiary_count",
    "PrimaryMeanDepth": "primary_sulcus_values",
    "SecondaryMeanDepth": "secondary_sulcus_values",
    "TertiaryMeanDepth": "tertiary_sulcus_values",
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "area": 1.0,
    "perimeter": 0.0,  # excluded due to high corrloation with area 
    "lgi": 2.0,
    "compactness": 1.0,
    "primary_count": 1.5,
    "secondary_count": 1.5,
    "tertiary_count": 1,
    "primary_sulcus_values": 1.5,
    "secondary_sulcus_values": 1.5,
    "tertiary_sulcus_values": 1,
}

RANGE_PENALTY = 0.0  # Gaussian: similarity multiplier (λ) when a metric falls outside [min, max] (decreases more penalty as λ approaches 0) typical values might be 0.5 or 0.25 for moderate or strong penalty, respectively
OOR_BETA = 1.0  # Mahalanobis: additive distance penalty (β) per out-of-range metric (increases more penalty) trypocal values might be 0.5, 1.0 or 2.0 for moderate or strong penalty, respectively


@dataclass
class GASPResult:
    """Full GASP output for a single gestational week."""
    week: int
    gasp: float
    per_metric: dict[str, float]
    z_scores: dict[str, float]
    out_of_range: dict[str, bool]


@dataclass
class GASPSummary:
    """Aggregate GASP results across all weeks.

    The scoring method is **Week-specific Gaussian Similarity Scoring**,
    a form of standardised Euclidean similarity (equivalent to a diagonal
    Mahalanobis distance with week-specific variance).  Each metric is
    compared relative to its own gestational-week-specific distribution,
    converted to a Gaussian similarity, optionally penalised if the value
    falls outside the observed [min, max] range, and combined into a
    single GASP score per week via a weighted average.
    """
    results: list[GASPResult]
    best_week: int
    max_gasp: float
    estimated_ga: float
    confidence: str


def compute_similarity_scores(
    measured: dict,
    registry: GestationalWeekProfile,
    axis: str,
    *,
    method: str = "gaussian",
    weights: dict[str, float] | None = None,
    apply_range_penalty: bool = True,
    beta: float | None = None,
) -> GASPSummary:
    """Week-specific Gaussian Similarity Scoring (GASP).

    Two scoring methods are available (selected via *method*):

    **gaussian** (default) — averages per-metric similarity scores:
        S_{w,i} = exp(-z_{w,i}² / 2)
        GASP_w  = weighted_mean(S_{w,i})

    **mahalanobis** — sums weighted squared z-scores plus a β-scaled
    out-of-range count, then converts to a single similarity value
    (diagonal Mahalanobis distance with range penalty):
        d²_w    = Σ_i α_i · z_{w,i}²  +  β · Σ_i I_{w,i}
        GASP_w  = exp(-d²_w / (2 · Σ_i α_i))

    where I_{w,i} ∈ {0, 1} flags whether x_i is outside [min_{w,i},
    max_{w,i}], and *beta* defaults to :data:`OOR_BETA`.

    Both methods compute per-metric z-scores identically:
        z_{w,i} = (x_i - μ_{w,i}) / σ_{w,i}

    If *apply_range_penalty* is True and x_i falls outside [min, max],
    the per-metric similarity is multiplied by :data:`RANGE_PENALTY`
    (gaussian), or β is added per OOR metric to d² (mahalanobis).

    Returns a :class:`GASPSummary` with per-week results, the best-
    matching week, a continuous estimated GA, and a confidence label.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS
    b = beta if beta is not None else OOR_BETA
    use_mahal = method.lower().startswith("mahal")

    week_results: list[GASPResult] = []

    for week in registry.weeks():
        ref = registry.get(week, axis)
        if ref is None:
            continue

        per_metric: dict[str, float] = {}
        z_scores: dict[str, float] = {}
        out_of_range: dict[str, bool] = {}
        for meas_key, ref_field in METRIC_MAP.items():
            val = measured.get(meas_key)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue

            stats: MetricStats = getattr(ref, ref_field)
            if stats.mean is None or stats.std is None or stats.std < 1e-12:
                continue

            z = (val - stats.mean) / stats.std
            sim = math.exp(-0.5 * z * z)

            oor = False
            if apply_range_penalty and stats.min is not None and stats.max is not None:
                if val < stats.min or val > stats.max:
                    sim *= RANGE_PENALTY
                    oor = True

            z_scores[ref_field] = round(z, 4)
            per_metric[ref_field] = round(sim, 4)
            out_of_range[ref_field] = oor

        if not per_metric:
            gasp = 0.0
        elif use_mahal:
            total_w = sum(w.get(k, 1.0) for k in z_scores)
            d_sq = sum(w.get(k, 1.0) * z_scores[k] ** 2 for k in z_scores)
            if apply_range_penalty:
                n_oor = sum(1 for is_oor in out_of_range.values() if is_oor)
                d_sq += b * n_oor
            gasp = math.exp(-d_sq / (2.0 * max(total_w, 1e-12)))
        else:
            total_w = sum(w.get(k, 1.0) for k in per_metric)
            gasp = sum(v * w.get(k, 1.0) for k, v in per_metric.items()) / total_w

        week_results.append(GASPResult(
            week=week,
            gasp=round(gasp, 4),
            per_metric=per_metric,
            z_scores=z_scores,
            out_of_range=out_of_range,
        ))

    week_results.sort(key=lambda r: r.gasp, reverse=True)

    if not week_results:
        return GASPSummary(
            results=[], best_week=0, max_gasp=0.0,
            estimated_ga=0.0, confidence="none")

    best = week_results[0]
    max_gasp = best.gasp

    gasp_sum = sum(r.gasp for r in week_results)
    if gasp_sum > 0:
        estimated_ga = sum(r.week * r.gasp for r in week_results) / gasp_sum
    else:
        estimated_ga = float(best.week)

    scores = [r.gasp for r in week_results]
    if len(scores) >= 2 and max_gasp > 0:
        second = scores[1]
        ratio = max_gasp / max(second, 1e-12)
        mean_score = sum(scores) / len(scores)
        spread = max_gasp - mean_score
        if ratio >= 1.5 and spread >= 0.15:
            confidence = "high"
        elif ratio >= 1.15 or spread >= 0.08:
            confidence = "moderate"
        else:
            confidence = "low"
    else:
        confidence = "low"

    return GASPSummary(
        results=week_results,
        best_week=best.week,
        max_gasp=round(max_gasp, 4),
        estimated_ga=round(estimated_ga, 2),
        confidence=confidence,
    )


def _clean(value) -> float | None:
    """Map pandas NaN / empty cells to ``None``; otherwise cast to float."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f
