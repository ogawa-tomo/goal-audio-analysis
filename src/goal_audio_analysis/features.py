"""Acoustic feature extraction for a single audio clip.

Designed for short (few-second) crowd-noise clips such as a football goal
celebration, but nothing here is football-specific.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import librosa


@dataclass
class ClipFeatures:
    file: str
    peak_time_s: float
    default_peak_time_s: float
    candidate_peak_times_s: list[float]
    selected_peak_time_s: Optional[float]
    human_corrected: bool
    attack_time_s: float
    decay_time_s: Optional[float]
    spectral_centroid_hz: float
    spectral_rolloff85_hz: float
    spectral_bandwidth_hz: float
    spectral_flatness: float
    zero_crossing_rate: float
    f0_median_hz: Optional[float]
    f0_mean_hz: Optional[float]
    voiced_fraction: float
    f1_median_hz: Optional[float] = None
    f2_median_hz: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _load_selected_peak(path: Path) -> Optional[float]:
    """Load `selected_peak_time_s` from `<path>`'s sidecar `.mark.json`, if any.

    Produced by `scripts/mark_goal_moment.py`: a human listens at each of
    `candidate_peak_times_s` and picks the one that is actually the target
    event (e.g. the goal). The precise timestamp comes from the algorithm's
    own candidate detection, not from the human typing/reacting in real
    time -- a human's own sense of "the moment" carries reaction-time noise,
    but picking *which* of a handful of well-separated candidates is
    correct does not. When present, this value -- not the algorithm's own
    top (loudest) candidate -- becomes the analysis anchor for everything
    downstream (attack/decay, spectral window).
    """
    mark_path = path.with_suffix(".mark.json")
    if not mark_path.exists():
        return None
    try:
        data = json.loads(mark_path.read_text(encoding="utf-8"))
        return float(data["selected_peak_time_s"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None


def _smooth_rms(rms: np.ndarray, times: np.ndarray, smooth_window_s: float = 0.3) -> np.ndarray:
    """Moving-average-smooth the RMS envelope over a `smooth_window_s`-wide window.

    This matters because a brief single-frame click (e.g. a dropped/
    duplicated buffer during loopback recording) can otherwise register as
    a louder instantaneous peak than a genuine multi-second crowd swell.
    Smoothing dilutes an isolated click's contribution while barely
    affecting a sustained rise.
    """
    if len(times) > 1 and smooth_window_s > 0:
        frame_dt = float(times[1] - times[0])
        smooth_frames = max(1, int(round(smooth_window_s / frame_dt)))
    else:
        smooth_frames = 1
    if smooth_frames > 1:
        kernel = np.ones(smooth_frames) / smooth_frames
        return np.convolve(rms, kernel, mode="same")
    return rms


def _nearest_index(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - t)))


def _global_max_in_window(rms_smooth: np.ndarray, times: np.ndarray, peak_search_window_s: float) -> int:
    """Index of the tallest point in the smoothed envelope within the search window.

    Restricting the search window matters: in real broadcast clips a loud
    stadium PA announcement / goal siren / jingle can occur a few seconds
    *after* the crowd's own vocal reaction and would otherwise be picked up
    as the "peak" instead of the actual cheer.
    """
    mask = times <= peak_search_window_s
    masked = np.where(mask, rms_smooth, -np.inf)
    return int(np.argmax(masked))


def _find_amplitude_candidates(
    rms_smooth: np.ndarray,
    times: np.ndarray,
    peak_search_window_s: float,
    peak_height: float,
    height_ratio: float = 0.8,
) -> list[float]:
    """Find local maxima within the search window that rival the main peak.

    A clean goal reaction should have one dominant swell. An earlier crowd
    reaction to a near-miss/developing chance can produce a second, almost
    as loud, local maximum -- this is a sign a naive "loudest point" search
    might lock onto the wrong one. Returns raw (unmerged) candidate times;
    merging near-duplicates is `_merge_close_candidates`'s job.
    """
    mask = times <= peak_search_window_s
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return [float(times[idx[0]])] if len(idx) else []

    windowed = rms_smooth[idx]
    windowed_times = times[idx]

    is_local_max = np.zeros(len(windowed), dtype=bool)
    is_local_max[1:-1] = (windowed[1:-1] > windowed[:-2]) & (windowed[1:-1] > windowed[2:])
    # the global max within the window always counts, even if it sits at an edge
    is_local_max[int(np.argmax(windowed))] = True

    candidate_idx = np.where(is_local_max & (windowed >= height_ratio * peak_height))[0]
    return [float(windowed_times[i]) for i in candidate_idx]


def _merge_close_candidates(
    cand_times: list[float], rms_smooth: np.ndarray, times: np.ndarray, min_separation_s: float
) -> list[float]:
    """Merge candidates closer together than `min_separation_s`, keeping the taller one."""
    if not cand_times:
        return []
    order = np.argsort(cand_times)
    sorted_times = np.array(cand_times)[order]
    heights = np.array([rms_smooth[_nearest_index(times, t)] for t in sorted_times])

    kept_times: list[float] = []
    kept_heights: list[float] = []
    for t, h in zip(sorted_times, heights):
        if kept_times and (t - kept_times[-1]) < min_separation_s:
            if h > kept_heights[-1]:
                kept_times[-1] = float(t)
                kept_heights[-1] = float(h)
            continue
        kept_times.append(float(t))
        kept_heights.append(float(h))

    return [round(t, 2) for t in kept_times]


def _find_candidates(
    rms_smooth: np.ndarray,
    times: np.ndarray,
    peak_search_window_s: float,
    height_ratio: float = 0.8,
    min_separation_s: float = 2.0,
) -> tuple[list[float], float]:
    """Enumerate all candidate goal-moment times.

    Returns `(candidate_times, default_time)`: `candidate_times` is every
    detected candidate in chronological order; `default_time` is the
    loudest one -- the same choice a pure "find the peak" algorithm would
    have made, kept as the fallback when no human selection exists yet
    (see `_load_selected_peak`) and as a reference point for measuring
    whether a human's selection actually changed anything
    (`human_corrected`).

    A sustained-elevation detector (finding plateaus with no single sharp
    peak) was tried and dropped: on this project's real clips, widening
    `peak_search_window_s` far enough to actually see a sustained
    celebration through to its end was already enough for the amplitude
    detector below to catch a rivaling micro-peak within it (crowd noise
    is never perfectly flat). The sustained detector added nothing beyond
    that on any of the 17 clips tested, while flagging most of them with
    a spurious second candidate from ordinary decay-tail texture.
    """
    top_idx = _global_max_in_window(rms_smooth, times, peak_search_window_s)
    peak_height = rms_smooth[top_idx]

    amp_candidates = _find_amplitude_candidates(rms_smooth, times, peak_search_window_s, peak_height, height_ratio)
    merged = _merge_close_candidates(amp_candidates, rms_smooth, times, min_separation_s)
    if not merged:
        merged = [round(float(times[top_idx]), 2)]

    default_time = max(merged, key=lambda t: rms_smooth[_nearest_index(times, t)])
    return merged, default_time


def _attack_decay(rms: np.ndarray, times: np.ndarray, peak_idx: int, peak_rms: float) -> tuple[float, Optional[float]]:
    thresh = 0.1 * peak_rms
    pre = rms[: peak_idx + 1]
    above = np.where(pre >= thresh)[0]
    attack_start_idx = int(above[0]) if len(above) else 0
    attack_time = float(times[peak_idx] - times[attack_start_idx])

    half = 0.5 * peak_rms
    post = rms[peak_idx:]
    below = np.where(post <= half)[0]
    decay_time = float(times[peak_idx + below[0]] - times[peak_idx]) if len(below) else None

    return attack_time, decay_time


def analyze_clip(
    path: str | Path,
    sr: int = 22050,
    peak_search_window_s: float = 6.0,
    smooth_window_s: float = 0.3,
    spectral_window_pre_s: float = 0.5,
    spectral_window_post_s: float = 2.5,
    with_formants: bool = True,
    candidate_height_ratio: float = 0.8,
    candidate_min_separation_s: float = 2.0,
) -> ClipFeatures:
    """Extract acoustic features from one audio clip.

    First, every candidate goal-moment time within the first
    `peak_search_window_s` seconds of the clip is enumerated (see
    `_find_candidates`). If a human has selected the correct one
    (`scripts/mark_goal_moment.py`, stored in `<clip>.mark.json`), that
    selection is used as the analysis anchor; otherwise the loudest
    candidate is used as a best-effort default. Spectral / pitch / formant
    features are then computed on a window centered on that anchor
    (`spectral_window_pre_s` before it to `spectral_window_post_s` after
    it).

    The default `peak_search_window_s=6.0` is meant to be paired with clips
    cut via `extract.extract_clip`'s defaults (`lead_s=3.0`,
    `duration_s=9.0`): together they give a symmetric +/-3s tolerance
    around an imprecise goal timestamp, while keeping the search window's
    *end* relative to the timestamp (timestamp + 3s) unchanged from the
    original lead_s=1.0/duration_s=7.0/peak_search_window_s=4.0 combination.
    That "timestamp + 3s" boundary is what matters for avoiding a later
    false peak (e.g. a stadium PA/jingle after the cheer), so widening the
    *pre*-timestamp margin alone does not increase that particular risk --
    only extending the search window's far end would.
    """
    path = Path(path)
    y, _sr = librosa.load(path, sr=sr, mono=True)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    rms_smooth = _smooth_rms(rms, times, smooth_window_s)

    candidate_peak_times_s, default_peak_time_s = _find_candidates(
        rms_smooth, times, peak_search_window_s,
        height_ratio=candidate_height_ratio, min_separation_s=candidate_min_separation_s,
    )

    selected_peak_time_s = _load_selected_peak(path)
    peak_time = selected_peak_time_s if selected_peak_time_s is not None else default_peak_time_s
    human_corrected = selected_peak_time_s is not None and abs(selected_peak_time_s - default_peak_time_s) > 1e-6

    peak_idx = _nearest_index(times, peak_time)
    peak_rms = float(rms[peak_idx])
    attack_time, decay_time = _attack_decay(rms, times, peak_idx, peak_rms)

    win_start = max(0.0, peak_time - spectral_window_pre_s)
    win_end = peak_time + spectral_window_post_s
    s_idx = int(win_start * sr)
    e_idx = min(len(y), int(win_end * sr))
    y_win = y[s_idx:e_idx]

    centroid = librosa.feature.spectral_centroid(y=y_win, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y_win, sr=sr, roll_percent=0.85)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y_win, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y_win)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_win)[0]

    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        y_win, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
    )
    f0_voiced = f0[~np.isnan(f0)]

    f1_median = f2_median = None
    if with_formants:
        f1_median, f2_median = _formants(y_win, sr)

    return ClipFeatures(
        file=path.name,
        peak_time_s=round(peak_time, 2),
        default_peak_time_s=round(default_peak_time_s, 2),
        candidate_peak_times_s=candidate_peak_times_s,
        selected_peak_time_s=selected_peak_time_s,
        human_corrected=human_corrected,
        attack_time_s=round(attack_time, 3),
        decay_time_s=round(decay_time, 3) if decay_time is not None else None,
        spectral_centroid_hz=round(float(np.mean(centroid)), 1),
        spectral_rolloff85_hz=round(float(np.mean(rolloff)), 1),
        spectral_bandwidth_hz=round(float(np.mean(bandwidth)), 1),
        spectral_flatness=round(float(np.mean(flatness)), 5),
        zero_crossing_rate=round(float(np.mean(zcr)), 5),
        f0_median_hz=round(float(np.median(f0_voiced)), 1) if len(f0_voiced) else None,
        f0_mean_hz=round(float(np.mean(f0_voiced)), 1) if len(f0_voiced) else None,
        voiced_fraction=round(float(np.mean(~np.isnan(f0))), 3),
        f1_median_hz=f1_median,
        f2_median_hz=f2_median,
    )


def _formants(y_win: np.ndarray, sr: int) -> tuple[Optional[float], Optional[float]]:
    """Estimate median F1/F2 via Praat's Burg method (parselmouth).

    Note: formant tracking assumes a single vocal tract. Applied to a crowd
    of overlapping voices this approximates "where the ensemble's spectral
    envelope peaks", not a literal individual's formants -- treat it as a
    directional signal, not a precise phonetic measurement.
    """
    import parselmouth

    snd = parselmouth.Sound(y_win, sampling_frequency=sr)
    formant = snd.to_formant_burg(
        time_step=0.01, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )

    n_frames = int(formant.get_number_of_frames())
    f1_vals, f2_vals = [], []
    for i in range(1, n_frames + 1):
        t = formant.get_time_from_frame_number(i)
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        if f1 is not None and not np.isnan(f1) and 150 < f1 < 1200:
            f1_vals.append(f1)
        if f2 is not None and not np.isnan(f2) and 500 < f2 < 3500:
            f2_vals.append(f2)

    f1_median = round(float(np.median(f1_vals)), 1) if f1_vals else None
    f2_median = round(float(np.median(f2_vals)), 1) if f2_vals else None
    return f1_median, f2_median


def analyze_directory(dir_path: str | Path, pattern: str = "*.wav", **kwargs) -> list[ClipFeatures]:
    """Run `analyze_clip` on every file matching `pattern` in `dir_path`."""
    dir_path = Path(dir_path)
    return [analyze_clip(p, **kwargs) for p in sorted(dir_path.glob(pattern))]
