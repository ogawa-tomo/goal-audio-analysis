"""Detect candidate onset moments (where a crowd reaction starts rising) in a clip.

Used to help a human mark `onset_marked_time_s` (see `features._load_onset_mark`):
this module only *suggests* candidates -- the correct one is still a human
judgment call, made by listening at each candidate timestamp. See
`scripts/mark_onset_moment.py`.

Unlike the RMS-amplitude-based peak detection this project tried and dropped
earlier (see `features.py`'s module history), this looks at how much the
audio's *spectral shape* changes from one moment to the next -- the
intuition being that a genuine reaction (e.g. a chant giving way to a
broadband roar) shows up as a change in spectral character even on clips
where the raw volume barely changes (a crowd that was already loud from
chanting before the goal). Validated by ear against all 17 clips in this
project's dataset: the nearest candidate landed within 0.6s of the
independently-confirmed onset moment in every case, at the settings used
as defaults below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import librosa


def spectral_shape_change_curve(
    y: np.ndarray,
    sr: int,
    win_s: float = 1.0,
    step_s: float = 0.05,
    window_s: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute how much the spectral shape changes across each point in the clip.

    At each candidate time `t`, compares the average magnitude spectrum of
    the `win_s` seconds before `t` against the `win_s` seconds after it,
    via cosine distance (1 - cosine similarity): 0 means identical shape,
    up to 1 for a totally different one. Only evaluated within the first
    `window_s` seconds of the clip (matching this project's loopback-
    recording convention -- see README) and only at points where both the
    before- and after-windows are fully inside the clip.

    Returns `(t_grid, cos_dist)`, aligned arrays; `cos_dist[i]` is `nan`
    wherever there isn't enough audio on one side to compute it.
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
    clip_dur = float(times[-1]) if len(times) else 0.0

    t_grid = np.arange(win_s, min(window_s, clip_dur - win_s), step_s)
    cos_dist = np.full(len(t_grid), np.nan)
    for i, t in enumerate(t_grid):
        pre_mask = (times >= t - win_s) & (times < t)
        post_mask = (times >= t) & (times < t + win_s)
        if pre_mask.sum() < 2 or post_mask.sum() < 2:
            continue
        a = S[:, pre_mask].mean(axis=1)
        b = S[:, post_mask].mean(axis=1)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos_dist[i] = 1 - (np.dot(a, b) / denom if denom > 0 else 1.0)

    return t_grid, cos_dist


def find_onset_candidates(
    t_grid: np.ndarray,
    cos_dist: np.ndarray,
    height_ratio: float = 0.4,
    min_separation_s: float = 1.5,
) -> list[float]:
    """Pick out local maxima of `cos_dist` that rival the tallest one.

    A `cos_dist` local maximum marks a moment where the spectral character
    changed a lot in a short window -- a candidate onset. `height_ratio`
    sets how close (as a fraction of the tallest one in the clip) another
    local max must be to also count; `min_separation_s` merges maxima that
    are really the same bump, keeping the taller one. Returns candidate
    times in chronological order.
    """
    valid = ~np.isnan(cos_dist)
    idx = np.where(valid)[0]
    if len(idx) < 3:
        return [float(t_grid[i]) for i in idx]

    vals = cos_dist[idx]
    is_local_max = np.zeros(len(vals), dtype=bool)
    is_local_max[1:-1] = (vals[1:-1] > vals[:-2]) & (vals[1:-1] > vals[2:])
    is_local_max[int(np.argmax(vals))] = True  # the tallest one always counts

    peak_height = vals.max()
    cand_idx = idx[np.where(is_local_max & (vals >= height_ratio * peak_height))[0]]
    if len(cand_idx) == 0:
        return []

    order = np.argsort(t_grid[cand_idx])
    cand_t = t_grid[cand_idx][order]
    cand_h = cos_dist[cand_idx][order]

    kept_t: list[float] = []
    kept_h: list[float] = []
    for t, h in zip(cand_t, cand_h):
        if kept_t and (t - kept_t[-1]) < min_separation_s:
            if h > kept_h[-1]:
                kept_t[-1] = float(t)
                kept_h[-1] = float(h)
            continue
        kept_t.append(float(t))
        kept_h.append(float(h))

    return [round(t, 2) for t in kept_t]


def onset_candidates_for_clip(
    path: str | Path,
    sr: int = 22050,
    win_s: float = 1.0,
    step_s: float = 0.05,
    window_s: float = 8.0,
    height_ratio: float = 0.4,
    min_separation_s: float = 1.5,
) -> list[float]:
    """Convenience wrapper: load `path` and return its onset candidates."""
    y, _sr = librosa.load(path, sr=sr, mono=True)
    t_grid, cos_dist = spectral_shape_change_curve(y, sr, win_s, step_s, window_s)
    return find_onset_candidates(t_grid, cos_dist, height_ratio, min_separation_s)
