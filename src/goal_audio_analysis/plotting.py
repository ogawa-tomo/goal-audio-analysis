"""Visualizations for group comparisons produced by `compare.py`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_COLOR_A = "#5e2ca5"
DEFAULT_COLOR_B = "#d6001c"

REFERENCE_VOWELS_HZ = {
    "i": (300, 2300),
    "e": (500, 2000),
    "ae": (660, 1700),
    "a": (800, 1300),
    "o": (500, 800),
    "u": (320, 800),
}


def _values(items, metric):
    return [it[metric] for it in items if it.get(metric) is not None]


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


def _plot_curve_overlay(
    curves_a: list[tuple[np.ndarray, np.ndarray]],
    curves_b: list[tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    label_a: str,
    label_b: str,
    freq_max_hz: float,
    title: str,
    ylabel: str,
) -> Path:
    """Shared implementation for `plot_spectrum_overlay`/`plot_formant_envelope_overlay`.

    Each clip's curve is normalized to sum to 1 before averaging, so the
    plot compares *shape* rather than raw loudness -- keeping per-
    recording gain differences from dominating the comparison. Individual
    clips are drawn thin/faint; the group mean is drawn bold.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for curves, label, color in [(curves_a, label_a, DEFAULT_COLOR_A), (curves_b, label_b, DEFAULT_COLOR_B)]:
        normalized = []
        freqs_ref = None
        for freqs, values in curves:
            if freqs_ref is None:
                freqs_ref = freqs
            normalized.append(values / values.sum())
            ax.plot(freqs, normalized[-1], color=color, alpha=0.25, linewidth=1)
        if normalized:
            mean_curve = np.mean(normalized, axis=0)
            ax.plot(freqs_ref, mean_curve, color=color, linewidth=2.5,
                    label=f"{label} (mean, n={len(normalized)})")

    ax.set_xlim(0, freq_max_hz)
    ax.set_xlabel("Frequency (Hz)")
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


def plot_spectrum_overlay(
    curves_a: list[tuple[np.ndarray, np.ndarray]],
    curves_b: list[tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    freq_max_hz: float = 6000.0,
    title: str = "Spectrum shape (post-onset window)",
) -> Path:
    """Overlay each group's raw post-onset average-spectrum shape plus the group mean.

    `curves_a`/`curves_b` are lists of `(freqs, spectrum)` pairs, e.g. from
    `features.spectrum_curve` -- the raw post-onset window (what
    spectral_centroid_hz etc. are computed from), not a difference against
    a pre-onset baseline, so it still reflects whatever ambient audio was
    already present in that window.
    """
    return _plot_curve_overlay(
        curves_a, curves_b, out_path, label_a, label_b, freq_max_hz, title,
        ylabel="Normalized magnitude (shape, arbitrary units)",
    )


def plot_formant_envelope_overlay(
    curves_a: list[tuple[np.ndarray, np.ndarray]],
    curves_b: list[tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    freq_max_hz: float = 5500.0,
    title: str = "Formant (LPC) envelope shape (post-onset window)",
) -> Path:
    """Overlay each group's LPC spectral-envelope shape plus the group mean.

    `curves_a`/`curves_b` are lists of `(freqs, envelope)` pairs from
    `features.formant_envelope_curve` -- the resonance shape F1/F2 are
    picked from as peaks, rather than the FFT magnitude spectrum
    `plot_spectrum_overlay` shows.
    """
    return _plot_curve_overlay(
        curves_a, curves_b, out_path, label_a, label_b, freq_max_hz, title,
        ylabel="Normalized LPC envelope (shape, arbitrary units)",
    )


def plot_hnr_over_time_overlay(
    curves_a: list[tuple[np.ndarray, np.ndarray]],
    curves_b: list[tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    title: str = "HNR over time (post-onset window)",
) -> Path:
    """Overlay each group's HNR-over-time curve plus the group mean.

    `curves_a`/`curves_b` are lists of `(times, hnr_db)` pairs, e.g. from
    `features.hnr_curve`. Unlike the frequency-axis overlays above,
    curves are NOT normalized before averaging: HNR is already a dB
    ratio, roughly independent of recording gain, so raw values are
    directly comparable across clips. The group mean is computed with
    `nanmean`, truncated to the shortest clip's length in that group, and
    ignores individual undefined (NaN) frames.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for curves, label, color in [(curves_a, label_a, DEFAULT_COLOR_A), (curves_b, label_b, DEFAULT_COLOR_B)]:
        for times, values in curves:
            ax.plot(times, values, color=color, alpha=0.25, linewidth=1)
        if curves:
            min_len = min(len(t) for t, _ in curves)
            stacked = np.array([v[:min_len] for _, v in curves])
            times_ref = curves[0][0][:min_len]
            mean_curve = np.nanmean(stacked, axis=0)
            ax.plot(times_ref, mean_curve, color=color, linewidth=2.5,
                    label=f"{label} (mean, n={len(curves)})")

    ax.set_xlabel("Time since onset (s)")
    ax.set_ylabel("HNR (dB)")
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
