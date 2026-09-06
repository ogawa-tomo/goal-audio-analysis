"""Summarize one group's worth of per-clip (`clip/`) results.

Generic: doesn't know or care which `clip/*.py` method produced its
input, only whether that input is a scalar or an `(x, y)` curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalarSummary:
    mean: float
    std: float
    n: int


def aggregate_scalar(values: list[float]) -> ScalarSummary:
    """Summarize a group's per-clip scalar values into mean/std/n."""
    return ScalarSummary(mean=float(np.mean(values)), std=float(np.std(values)), n=len(values))


def aggregate_curve(
    curves: list[tuple[np.ndarray, np.ndarray]],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a group's per-clip curves into one group-mean curve.

    `normalize`: whether each curve is divided by its own sum before
    averaging -- appropriate for frequency-axis magnitude curves (e.g.
    `clip.spectral.curve`), so per-recording gain differences don't
    dominate the comparison. Set `False` for something already on a
    comparable absolute scale regardless of recording gain, like
    `clip.hnr.curve`'s dB values.

    Curves are truncated to the shortest one's length before averaging
    (they may differ, e.g. a clip whose window is shorter than intended
    -- see `reports/premier_vs_laliga.md` section 3), and `nan` frames
    (e.g. HNR's undefined frames) are ignored via `nanmean` rather than
    propagating into the whole average. Assumes every curve shares the
    same kind of x-axis grid; the first curve's x-values (truncated to
    that same length) are returned as the reference.
    """
    ys = [y / y.sum() if normalize else y for _x, y in curves]
    min_len = min(len(y) for y in ys)
    stacked = np.array([y[:min_len] for y in ys])
    mean_curve = np.nanmean(stacked, axis=0)
    x_ref = curves[0][0][:min_len]
    return x_ref, mean_curve
