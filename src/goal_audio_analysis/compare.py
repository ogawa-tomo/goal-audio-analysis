"""Compare two groups (label_a/label_b): scalar statistics and curve-shape overlays.

Generic: works on whatever per-clip scalar values or `(x, y)` curves a
`clip/*.py` method produced (or `groupstats.py`'s summaries of them);
doesn't know which method produced them. Two groups only, by design --
see `reports/premier_vs_laliga.md` for why this project doesn't need
more, and revisit this file if a future project does.

Has no audio-processing dependencies itself (no librosa/parselmouth):
the scalar-comparison half only operates on plain dicts/numbers, so it
can be reused for any two-group numeric comparison, not just audio
features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import groupstats

DEFAULT_COLOR_A = "#5e2ca5"
DEFAULT_COLOR_B = "#d6001c"

DEFAULT_METRICS = [
    "spectral_centroid_hz",
    "spectral_rolloff85_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    "zero_crossing_rate",
    "f0_median_hz",
    "f1_median_hz",
    "f2_median_hz",
    "hnr_db",
]

REFERENCE_VOWELS_HZ = {
    "i": (300, 2300),
    "e": (500, 2000),
    "ae": (660, 1700),
    "a": (800, 1300),
    "o": (500, 800),
    "u": (320, 800),
}


@dataclass
class ScalarComparison:
    metric: str
    mean_a: float
    std_a: float
    n_a: int
    mean_b: float
    std_b: float
    n_b: int
    t_stat: Optional[float]
    p_value: Optional[float]


def compare_scalar(values_a: list[float], values_b: list[float], metric: str = "") -> ScalarComparison:
    """Welch's t-test (unequal variance) between two groups' scalar values."""
    summary_a = groupstats.aggregate_scalar(values_a)
    summary_b = groupstats.aggregate_scalar(values_b)
    t_stat = p_value = None
    if len(values_a) >= 2 and len(values_b) >= 2:
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        t_stat, p_value = float(t_stat), float(p_value)
    return ScalarComparison(
        metric=metric,
        mean_a=summary_a.mean, std_a=summary_a.std, n_a=summary_a.n,
        mean_b=summary_b.mean, std_b=summary_b.std, n_b=summary_b.n,
        t_stat=t_stat, p_value=p_value,
    )


def _values(items, metric):
    return [it[metric] for it in items if it.get(metric) is not None]


def compare_groups(
    group_a: list[dict],
    group_b: list[dict],
    metrics: list[str] | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> list[ScalarComparison]:
    """Convenience: run `compare_scalar` for each of `metrics` across two groups of per-clip dicts."""
    metrics = metrics or DEFAULT_METRICS
    results = []
    for metric in metrics:
        va = _values(group_a, metric)
        vb = _values(group_b, metric)
        if not va or not vb:
            continue
        results.append(compare_scalar(va, vb, metric))
    return results


def format_table(comparisons: list[ScalarComparison], label_a: str = "A", label_b: str = "B") -> str:
    lines = [f"{'metric':28s} {label_a+' (mean±sd)':>22s} {label_b+' (mean±sd)':>22s} {'p':>10s}"]
    for c in comparisons:
        pa = f"{c.mean_a:.3f}±{c.std_a:.3f} (n={c.n_a})"
        pb = f"{c.mean_b:.3f}±{c.std_b:.3f} (n={c.n_b})"
        p = f"{c.p_value:.4f}" if c.p_value is not None else "n/a"
        lines.append(f"{c.metric:28s} {pa:>22s} {pb:>22s} {p:>10s}")
    return "\n".join(lines)


def plot_bar_comparison(
    group_a: list[dict],
    group_b: list[dict],
    metrics: list[tuple[str, str]],
    out_path: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    title: str = "",
) -> Path:
    """`metrics` is a list of (key, display_title) pairs."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, disp_title) in zip(axes, metrics):
        va = _values(group_a, key)
        vb = _values(group_b, key)
        means = [np.mean(va), np.mean(vb)]
        stds = [np.std(va), np.std(vb)]
        ax.bar([0, 1], means, yerr=stds, capsize=6, color=[DEFAULT_COLOR_A, DEFAULT_COLOR_B], width=0.6)
        for i, vv in enumerate([va, vb]):
            jitter = np.random.uniform(-0.08, 0.08, size=len(vv))
            ax.scatter(np.full(len(vv), i) + jitter, vv, color="black", zorder=5, s=20, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_a, label_b], rotation=15)
        ax.set_title(disp_title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94] if title else None)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_curve_comparison(
    curves_a: list[tuple[np.ndarray, np.ndarray]],
    curves_b: list[tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    xlabel: str = "",
    ylabel: str = "",
    x_max: float | None = None,
    normalize: bool = True,
    title: str = "",
) -> Path:
    """Overlay each group's per-clip curves plus the group mean.

    Generic over what the curve represents (frequency-axis shape, a
    time-axis series, ...) -- pass `xlabel`/`ylabel`/`x_max` to label it
    and `normalize` to match `groupstats.aggregate_curve`'s meaning
    (True for frequency-axis magnitude curves, False for something
    already gain-independent like HNR's dB values). Individual clips are
    drawn thin/faint; the group mean (via `groupstats.aggregate_curve`)
    is drawn bold.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for curves, label, color in [(curves_a, label_a, DEFAULT_COLOR_A), (curves_b, label_b, DEFAULT_COLOR_B)]:
        for x, y in curves:
            plotted_y = y / y.sum() if normalize else y
            ax.plot(x, plotted_y, color=color, alpha=0.25, linewidth=1)
        if curves:
            x_ref, mean_curve = groupstats.aggregate_curve(curves, normalize=normalize)
            ax.plot(x_ref, mean_curve, color=color, linewidth=2.5, label=f"{label} (mean, n={len(curves)})")

    if x_max is not None:
        ax.set_xlim(0, x_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_formant_chart(
    group_a: list[dict],
    group_b: list[dict],
    out_path: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    f1_key: str = "f1_median_hz",
    f2_key: str = "f2_median_hz",
    title: str = "Formants (F1-F2)",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 7))

    for group, label, color in [(group_a, label_a, DEFAULT_COLOR_A), (group_b, label_b, DEFAULT_COLOR_B)]:
        f1 = _values(group, f1_key)
        f2 = _values(group, f2_key)
        ax.scatter(f2, f1, color=color, s=120, label=label, edgecolor="black", zorder=5)
        if f1 and f2:
            ax.scatter(np.mean(f2), np.mean(f1), color=color, s=400, marker="X",
                       edgecolor="black", linewidth=2, zorder=6)

    for v, (f1r, f2r) in REFERENCE_VOWELS_HZ.items():
        ax.scatter(f2r, f1r, color="gray", s=40, marker="s", zorder=3)
        ax.annotate(f"/{v}/", (f2r, f1r), textcoords="offset points", xytext=(6, 4), fontsize=10, color="gray")

    ax.set_xlabel("F2 (Hz) -- front <-> back")
    ax.set_ylabel("F1 (Hz) -- open <-> close")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
