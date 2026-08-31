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


class MissingMarkError(ValueError):
    """Raised when a clip has no human-specified goal moment."""


@dataclass
class ClipFeatures:
    file: str
    peak_time_s: float
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


def _load_human_mark(path: Path) -> Optional[float]:
    """Load `human_marked_time_s` from `<path>`'s sidecar `.mark.json`, if any.

    Produced by `scripts/mark_goal_moment.py`: a human listens to the clip
    (rewinding/re-checking as needed via a precise playback tool) and
    reports the exact moment of the target event (e.g. the goal). This
    value is used as-is, verbatim, as the analysis anchor -- there is no
    algorithmic peak detection to fall back on or cross-check against.

    An earlier design tried to have the algorithm find the moment itself
    (a single loudest point, then several candidate peaks for a human to
    choose from). Both were dropped after repeatedly finding real clips
    where the correct moment was not the loudest point in the clip, and
    was not even among the algorithm's own candidate peaks (a goal
    celebration that builds to a sustained plateau rather than a sharp
    peak has no standout instant for an amplitude-based detector to find).
    A careful, verified human read of the clip is the only mechanism this
    project found that reliably gets it right.
    """
    mark_path = path.with_suffix(".mark.json")
    if not mark_path.exists():
        return None
    try:
        data = json.loads(mark_path.read_text(encoding="utf-8"))
        return float(data["human_marked_time_s"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None


def _nearest_index(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - t)))


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
    spectral_window_pre_s: float = 0.5,
    spectral_window_post_s: float = 2.5,
    with_formants: bool = True,
) -> ClipFeatures:
    """Extract acoustic features from one audio clip.

    Requires a human-specified goal moment (`scripts/mark_goal_moment.py`,
    stored in `<clip>.mark.json`) -- raises `MissingMarkError` if the clip
    hasn't been marked yet. There is no algorithmic fallback: see
    `_load_human_mark` for why. Spectral / pitch / formant features are
    computed on a window centered on that marked moment
    (`spectral_window_pre_s` before it to `spectral_window_post_s` after
    it).
    """
    path = Path(path)

    human_marked_time_s = _load_human_mark(path)
    if human_marked_time_s is None:
        raise MissingMarkError(
            f"{path.name} has no goal-moment mark. Run "
            f"`python scripts/mark_goal_moment.py {path}` first."
        )
    peak_time = human_marked_time_s

    y, _sr = librosa.load(path, sr=sr, mono=True)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

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
