"""Compare two groups of pre-computed feature dicts (e.g. two leagues' clips).

Intentionally has no audio-processing dependencies (no librosa/parselmouth):
it only operates on plain dicts/numbers, so it can be reused for any
two-group numeric comparison, not just audio features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats

DEFAULT_METRICS = [
    "attack_time_s",
    "spectral_centroid_hz",
    "spectral_rolloff85_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    "zero_crossing_rate",
    "f0_median_hz",
    "f1_median_hz",
    "f2_median_hz",
]


@dataclass
class MetricComparison:
    metric: str
    mean_a: float
    std_a: float
    n_a: int
    mean_b: float
    std_b: float
    n_b: int
    t_stat: float | None
    p_value: float | None


def _values(items: Iterable[dict], metric: str) -> list[float]:
    return [it[metric] for it in items if it.get(metric) is not None]


def compare_groups(
    group_a: list[dict],
    group_b: list[dict],
    metrics: list[str] | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> list[MetricComparison]:
    """Welch's t-test (unequal variance) per metric between two groups of dicts."""
    metrics = metrics or DEFAULT_METRICS
    results = []
    for metric in metrics:
        va = _values(group_a, metric)
        vb = _values(group_b, metric)
        if not va or not vb:
            continue
        t_stat = p_value = None
        if len(va) >= 2 and len(vb) >= 2:
            t_stat, p_value = stats.ttest_ind(va, vb, equal_var=False)
            t_stat, p_value = float(t_stat), float(p_value)
        results.append(
            MetricComparison(
                metric=metric,
                mean_a=float(np.mean(va)), std_a=float(np.std(va)), n_a=len(va),
                mean_b=float(np.mean(vb)), std_b=float(np.std(vb)), n_b=len(vb),
                t_stat=t_stat, p_value=p_value,
            )
        )
    return results


def format_table(comparisons: list[MetricComparison], label_a: str = "A", label_b: str = "B") -> str:
    lines = [f"{'metric':28s} {label_a+' (mean±sd)':>22s} {label_b+' (mean±sd)':>22s} {'p':>10s}"]
    for c in comparisons:
        pa = f"{c.mean_a:.3f}±{c.std_a:.3f} (n={c.n_a})"
        pb = f"{c.mean_b:.3f}±{c.std_b:.3f} (n={c.n_b})"
        p = f"{c.p_value:.4f}" if c.p_value is not None else "n/a"
        lines.append(f"{c.metric:28s} {pa:>22s} {pb:>22s} {p:>10s}")
    return "\n".join(lines)
