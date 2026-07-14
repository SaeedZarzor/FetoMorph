"""Gestational-week reference statistics loaded from a CSV.

Mirrors the per-week structure documented in ``profile_info.md`` (metadata
plus three metric-summary tables) for gestational weeks 24-36.
"""

from dataclasses import dataclass
from pathlib import Path

import math
import re
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
    # aggregate sulcus summaries (all classes pooled)
    sulci_count: MetricStats   # per-slice total sulcus count
    sulci_depth: MetricStats   # pooled per-sulcus depth across all classes
    sulci_count_normalized: MetricStats
    sulci_depth_normalized: MetricStats
    # rounded sulcus count summary (per class)
    primary_count: MetricStats
    secondary_count: MetricStats
    tertiary_count: MetricStats
    unclassified_count: MetricStats
    primary_count_normalized: MetricStats
    secondary_count_normalized: MetricStats
    tertiary_count_normalized: MetricStats
    unclassified_count_normalized: MetricStats
    # sulcus value summary
    primary_sulcus_values: MetricStats
    secondary_sulcus_values: MetricStats
    tertiary_sulcus_values: MetricStats
    unclassified_sulcus_values: MetricStats
    primary_sulcus_values_normalized: MetricStats
    secondary_sulcus_values_normalized: MetricStats
    tertiary_sulcus_values_normalized: MetricStats
    unclassified_sulcus_values_normalized: MetricStats


_METRIC_FIELDS: tuple[str, ...] = (
    "area",
    "perimeter",
    "lgi",
    "compactness",
    "sulci_count",
    "sulci_depth",
    "sulci_count_normalized",
    "sulci_depth_normalized",
    "primary_count",
    "secondary_count",
    "tertiary_count",
    "unclassified_count",
    "primary_count_normalized",
    "secondary_count_normalized",
    "tertiary_count_normalized",
    "unclassified_count_normalized",
    "primary_sulcus_values",
    "secondary_sulcus_values",
    "tertiary_sulcus_values",
    "unclassified_sulcus_values",
    "primary_sulcus_values_normalized",
    "secondary_sulcus_values_normalized",
    "tertiary_sulcus_values_normalized",
    "unclassified_sulcus_values_normalized",
)


def _metric_stats(row, name: str) -> MetricStats:
    """Build :class:`MetricStats` for *name* from a CSV row.

    Tolerant of a missing metric: a CSV without the ``{name}_n`` column (e.g. an
    older reference, or one built before ``sulci_depth`` was added) yields an
    empty ``MetricStats`` rather than raising, so such rows still load.
    """
    n_col = f"{name}_n"
    if n_col not in row.index or pd.isna(row[n_col]):
        return MetricStats(n=0, mean=None, std=None, min=None, max=None)
    return MetricStats(
        n=int(row[n_col]),
        mean=_clean(row.get(f"{name}_mean")),
        std=_clean(row.get(f"{name}_std")),
        min=_clean(row.get(f"{name}_min")),
        max=_clean(row.get(f"{name}_max")),
    )


class GestationalWeekProfile:
    """Reference statistics indexed by (gestational week, axis).

    Construct with the path to a CSV in the wide format: one row per
    (week, axis) pair, columns ``week, axis, slice_rows_analyzed,
    pixel_size_units, kernel_size_mm`` followed by ``{metric}_n``,
    ``{metric}_mean``, ``{metric}_std``, ``{metric}_min``, ``{metric}_max``
    for every metric listed on :class:`WeekProfile`.
    """

    VALID_RANGE = range(24, 37)
    AXES = ("axial", "coronal", "sagittal")

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.is_cropped_reference = "cropped" in self.csv_path.stem.lower()
        self._profiles: dict[tuple[int, str], WeekProfile] = self._load(self.csv_path)

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

    @classmethod
    def _load(cls, path: Path) -> dict[tuple[int, str], WeekProfile]:
        df = pd.read_csv(path)
        profiles: dict[tuple[int, str], WeekProfile] = {}
        for _, row in df.iterrows():
            # Only score weeks inside VALID_RANGE; reference CSVs may carry rows
            # for weeks outside the supported range.
            if int(row["week"]) not in cls.VALID_RANGE:
                continue
            metric_stats = {
                name: _metric_stats(row, name) for name in _METRIC_FIELDS
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
    # Aggregate sulcus metrics — the measured values are derived per slice from
    # the per-class counts / mean depths by :func:`_augment_aggregate_metrics`.
    "TotalSulciCount": "sulci_count",
    "MeanSulciDepth": "sulci_depth",
}

# Metrics used for normalization-based (unit-free) comparison. Area, perimeter
# and the raw (mm) depths are deliberately excluded. LGI / Compactness / total
# sulcal count are already scale-free and compared as-is; the depth and per-class
# count metrics use the ``*_normalized`` reference columns, with the measured
# values derived by :func:`_augment_normalized_metrics` (depth ÷ the slice's own
# max sulcus depth; per-class count ÷ the slice's total count). A ``*_normalized``
# column missing from the CSV drops that metric out during scoring.
NORMALIZED_METRIC_MAP: dict[str, str] = {
    "LGI": "lgi",
    "Compactness": "compactness",
    "TotalSulciCount": "sulci_count",
    "NormSulciDepth": "sulci_depth_normalized",
    "PrimaryDepthNorm": "primary_sulcus_values_normalized",
    "SecondaryDepthNorm": "secondary_sulcus_values_normalized",
    "TertiaryDepthNorm": "tertiary_sulcus_values_normalized",
    "UnclassifiedDepthNorm": "unclassified_sulcus_values_normalized",
    "PrimaryCountNorm": "primary_count_normalized",
    "SecondaryCountNorm": "secondary_count_normalized",
    "TertiaryCountNorm": "tertiary_count_normalized",
    "UnclassifiedCountNorm": "unclassified_count_normalized",
}

# Measured normalized-depth key -> the raw (mean-depth) measured key it scales.
_DEPTH_NORM_SOURCES: dict[str, str] = {
    "NormSulciDepth": "MeanSulciDepth",
    "PrimaryDepthNorm": "PrimaryMeanDepth",
    "SecondaryDepthNorm": "SecondaryMeanDepth",
    "TertiaryDepthNorm": "TertiaryMeanDepth",
    "UnclassifiedDepthNorm": "UnclassifiedMeanDepth",
}

# Measured normalized-count key -> the raw per-class count measured key it scales.
_COUNT_NORM_SOURCES: dict[str, str] = {
    "PrimaryCountNorm": "PrimarySulciCount",
    "SecondaryCountNorm": "SecondarySulciCount",
    "TertiaryCountNorm": "TertiarySulciCount",
    "UnclassifiedCountNorm": "UnclassifiedSulciCount",
}

# Measured-dict keys for the two aggregate sulcus metrics, and the per-class
# keys they are derived from.
TOTAL_COUNT_KEY = "TotalSulciCount"
MEAN_DEPTH_KEY = "MeanSulciDepth"
_CLASS_COUNT_KEYS = (
    "PrimarySulciCount", "SecondarySulciCount",
    "TertiarySulciCount", "UnclassifiedSulciCount",
)
_CLASS_DEPTH_PAIRS = (
    ("PrimarySulciCount", "PrimaryMeanDepth"),
    ("SecondarySulciCount", "SecondaryMeanDepth"),
    ("TertiarySulciCount", "TertiaryMeanDepth"),
    ("UnclassifiedSulciCount", "UnclassifiedMeanDepth"),
)


def _augment_aggregate_metrics(measured: dict) -> dict:
    """Return *measured* with the two aggregate sulcus metrics filled in.

    ``TotalSulciCount`` = Σ per-class counts (the slice's total sulcus count).
    ``MeanSulciDepth``  = Σ(count·mean_depth) / Σ count across classes — the
    slice's overall mean sulcus depth, comparable to the pooled per-sulcus
    ``sulci_depth`` reference the same way ``PrimaryMeanDepth`` compares to
    ``primary_sulcus_values``. Values the caller already supplied explicitly are
    never overwritten; classes missing from *measured* are simply skipped.
    """
    out = dict(measured)

    if out.get(TOTAL_COUNT_KEY) is None:
        counts = [c for c in (_clean(out.get(k)) for k in _CLASS_COUNT_KEYS)
                  if c is not None]
        if counts:
            out[TOTAL_COUNT_KEY] = sum(counts)
        else:
            # Fallback: the metrics-store overall count (cropped slices only
            # record UnclassifiedSulciCount, but "SulciCount" is always set).
            overall_count = _clean(out.get("SulciCount"))
            if overall_count is not None:
                out[TOTAL_COUNT_KEY] = overall_count

    if out.get(MEAN_DEPTH_KEY) is None:
        total_depth = 0.0
        total_count = 0.0
        for count_key, depth_key in _CLASS_DEPTH_PAIRS:
            count = _clean(out.get(count_key))
            depth = _clean(out.get(depth_key))
            if count is not None and depth is not None and count > 0:
                total_depth += count * depth
                total_count += count
        if total_count > 0:
            out[MEAN_DEPTH_KEY] = total_depth / total_count
        else:
            # Fallback: the metrics-store overall mean depth. Cropped slices are
            # all "unclassified", for which no per-class mean depth is recorded,
            # so the per-class derivation above yields nothing.
            overall_mean = _clean(out.get("MeanDepth"))
            if overall_mean is not None:
                out[MEAN_DEPTH_KEY] = overall_mean

    return out


def _stats_has_values(stats: MetricStats | None) -> bool:
    return (
        stats is not None
        and stats.n > 0
        and stats.mean is not None
        and stats.std is not None
        and stats.std >= 1e-12
    )


_PER_SULCUS_DEPTH_RE = re.compile(
    r"^(?:Primary|Secondary|Tertiary|Unclassified)_depth_\d+$")


def _max_sulcus_depth(measured: dict) -> float | None:
    """The slice's own maximum sulcus depth, used to normalize depths.

    Prefers an explicit max — ``MaxSulciDepth`` or the metrics-store ``MaxDepth``
    column (which ``MetricsStore.record_metric_for`` fills with ``max(sulci_depth)``);
    otherwise takes the max over the per-sulcus depth columns the batch exports
    (``Primary_depth_1`` … ``Unclassified_depth_N``). Returns ``None`` when none
    is available (e.g. mesh / manual-entry paths carry no per-sulcus values), so
    the depth metrics are ignored rather than mis-normalized.
    """
    for key in ("MaxSulciDepth", "MaxDepth"):
        explicit = _clean(measured.get(key))
        if explicit is not None and explicit > 0:
            return explicit
    depths = [
        d for d in (
            _clean(v) for k, v in measured.items()
            if isinstance(k, str) and _PER_SULCUS_DEPTH_RE.match(k)
        )
        if d is not None
    ]
    return max(depths) if depths else None


def _augment_normalized_metrics(measured: dict) -> dict:
    """Return *measured* with the normalized (scale-free) metrics filled in.

    Sulcal depth is normalized by the slice's OWN maximum sulcus depth
    (``normalized_sulcal_depth = sulcal_depth / max_sulcal_depth``) so it is
    invariant to the (per-crop) pixel spacing; per-class sulcal count is
    normalized by the slice's total sulcus count. The reference statistics for
    these come from the ``*_normalized`` CSV columns, so a missing column simply
    drops the metric during scoring. Values already present are not overwritten.
    """
    out = _augment_aggregate_metrics(measured)  # TotalSulciCount, MeanSulciDepth

    max_depth = _max_sulcus_depth(out)
    if max_depth and max_depth > 0:
        for norm_key, src_key in _DEPTH_NORM_SOURCES.items():
            src = _clean(out.get(src_key))
            if src is not None and out.get(norm_key) is None:
                out[norm_key] = src / max_depth

    total = _clean(out.get(TOTAL_COUNT_KEY))
    if total and total > 0:
        for norm_key, src_key in _COUNT_NORM_SOURCES.items():
            src = _clean(out.get(src_key))
            if src is not None and out.get(norm_key) is None:
                out[norm_key] = src / total

    return out

DEFAULT_WEIGHTS: dict[str, float] = {
    "area": 1.0,
    "perimeter": 0.0,  # excluded due to high corrloation with area 
    "lgi": 2.0,
    "compactness": 0.5,
    "primary_count": 1.0,
    "secondary_count": 1.0,
    "tertiary_count": 1,
    "primary_sulcus_values": 1.0,
    "secondary_sulcus_values": 1.0,
    "tertiary_sulcus_values": 1.0,
    "unclassified_sulcus_values": 1.0,
    "sulci_count": 1.5,
    "sulci_depth": 1.5,
    "sulci_depth_normalized": 1.5,
    "sulci_count_normalized": 1.5,
    "primary_count_normalized": 1.0,
    "secondary_count_normalized": 1.0,
    "tertiary_count_normalized": 1.0,
    "unclassified_count_normalized": 1.0,
    "primary_sulcus_values_normalized": 1.0,
    "secondary_sulcus_values_normalized": 1.0,
    "tertiary_sulcus_values_normalized": 1.0,
    "unclassified_sulcus_values_normalized": 1.0,
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
    use_normalized: bool = False,
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
    normalized_mode = bool(use_normalized or getattr(registry, "is_cropped_reference", False))

    # Derive the aggregate / normalized measured metrics from the per-class
    # counts and depths so they are scored even when the caller only supplies
    # the per-class values.
    measured = (
        _augment_normalized_metrics(measured) if normalized_mode
        else _augment_aggregate_metrics(measured)
    )

    week_results: list[GASPResult] = []

    for week in registry.weeks():
        ref = registry.get(week, axis)
        if ref is None:
            continue

        per_metric: dict[str, float] = {}
        z_scores: dict[str, float] = {}
        out_of_range: dict[str, bool] = {}
        metric_map = NORMALIZED_METRIC_MAP if normalized_mode else METRIC_MAP
        for meas_key, ref_field in metric_map.items():
            val = measured.get(meas_key)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue

            stats: MetricStats = getattr(ref, ref_field)
            if not _stats_has_values(stats):
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
