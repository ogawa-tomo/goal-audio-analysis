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
    zero_crossing_rate_pre: Optional[float]
    zero_crossing_rate_delta: Optional[float]
    f0_median_hz: Optional[float]
    f0_mean_hz: Optional[float]
    voiced_fraction: float
    f1_median_hz: Optional[float] = None
    f2_median_hz: Optional[float] = None
    increase_centroid_hz: Optional[float] = None
    increase_rolloff85_hz: Optional[float] = None
    increase_bandwidth_hz: Optional[float] = None
    increase_flatness: Optional[float] = None

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


def _increase_spectrum(
    y: np.ndarray,
    sr: int,
    peak_time: float,
    pre_start_s: float,
    pre_end_s: float,
    post_start_s: float,
    post_end_s: float,
) -> Optional[tuple[float, float, float, float]]:
    """Characterize what newly appeared in the audio around `peak_time`.

    Rather than describing the post-goal window's spectral content on its
    own (what the other spectral_* fields do), this compares it against a
    pre-goal baseline window and looks only at the *increase* -- energy
    present after but not before -- per frequency bin. This isolates the
    reaction itself from whatever ambient noise (commentary, an
    already-ongoing chant, stadium acoustics) was already present before
    the goal, which the raw post-goal window can't distinguish. Found to
    show a substantially larger, still statistically significant
    Premier-vs-LaLiga gap than the plain post-goal spectral centroid/
    rolloff on this project's 17-clip dataset.

    Returns `(centroid_hz, rolloff85_hz, bandwidth_hz, flatness)` computed
    on the clipped-positive difference spectrum, or `None` if either
    window is empty (e.g. `peak_time` too close to the start of the clip)
    or the post-goal window is not louder than the baseline anywhere.

    `flatness` here leans low almost by construction: most frequency bins
    have zero increase (post <= pre there), and a geometric mean is highly
    sensitive to near-zero values. Treat it as a rough indicator of how
    concentrated vs. spread the added energy is, not as directly
    comparable to `spectral_flatness`.
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)

    pre_mask = (times >= max(0.0, peak_time + pre_start_s)) & (times < max(0.0, peak_time + pre_end_s))
    post_mask = (times >= max(0.0, peak_time + post_start_s)) & (times < peak_time + post_end_s)
    if pre_mask.sum() < 2 or post_mask.sum() < 2:
        return None

    pre_spec = S[:, pre_mask].mean(axis=1)
    post_spec = S[:, post_mask].mean(axis=1)
    increase = np.clip(post_spec - pre_spec, 0, None)
    total = increase.sum()
    if total <= 0:
        return None

    centroid = float(np.sum(freqs * increase) / total)
    bandwidth = float(np.sqrt(np.sum(increase * (freqs - centroid) ** 2) / total))
    cumsum = np.cumsum(increase)
    idx = int(np.searchsorted(cumsum, 0.85 * total))
    rolloff85 = float(freqs[min(idx, len(freqs) - 1)])
    eps = 1e-10
    geo_mean = np.exp(np.mean(np.log(increase + eps)))
    arith_mean = np.mean(increase) + eps
    flatness = float(geo_mean / arith_mean)

    return centroid, rolloff85, bandwidth, flatness


def _pre_goal_window(y: np.ndarray, sr: int, peak_time: float, pre_start_s: float, pre_end_s: float) -> Optional[np.ndarray]:
    """Slice out the pre-goal baseline window used by `_increase_spectrum` and the zero_crossing_rate delta."""
    pre_s = max(0.0, peak_time + pre_start_s)
    pre_e = max(0.0, peak_time + pre_end_s)
    if pre_e - pre_s < 0.1:
        return None
    return y[int(pre_s * sr):int(pre_e * sr)]


def analyze_clip(
    path: str | Path,
    sr: int = 22050,
    spectral_window_pre_s: float = 0.5,
    spectral_window_post_s: float = 2.5,
    increase_pre_start_s: float = -3.0,
    increase_pre_end_s: float = -0.5,
    with_formants: bool = True,
) -> ClipFeatures:
    """Extract acoustic features from one audio clip.

    Requires a human-specified goal moment (`scripts/mark_goal_moment.py`,
    stored in `<clip>.mark.json`) -- raises `MissingMarkError` if the clip
    hasn't been marked yet. There is no algorithmic fallback: see
    `_load_human_mark` for why. Spectral / pitch / formant features are
    computed on a window centered on that marked moment
    (`spectral_window_pre_s` before it to `spectral_window_post_s` after
    it). The `increase_*` fields instead describe what's *new* relative to
    a pre-goal baseline window (`increase_pre_start_s` to
    `increase_pre_end_s` before the marked moment) -- see
    `_increase_spectrum`. `zero_crossing_rate_pre`/`_delta` apply the same
    before/after comparison to zero-crossing rate, which (unlike the
    spectral_* fields) is a single time-domain scalar rather than a
    per-frequency-bin quantity, so it's compared as a plain before/after
    delta instead of being recomputed on a difference spectrum.
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

    zcr_post_val = float(np.mean(zcr))
    y_pre = _pre_goal_window(y, sr, peak_time, increase_pre_start_s, increase_pre_end_s)
    if y_pre is not None and len(y_pre) > 0:
        zcr_pre_val = float(np.mean(librosa.feature.zero_crossing_rate(y=y_pre)[0]))
        zcr_delta_val = zcr_post_val - zcr_pre_val
    else:
        zcr_pre_val = zcr_delta_val = None

    increase = _increase_spectrum(
        y, sr, peak_time,
        increase_pre_start_s, increase_pre_end_s,
        -spectral_window_pre_s, spectral_window_post_s,
    )
    if increase is not None:
        inc_centroid, inc_rolloff85, inc_bandwidth, inc_flatness = increase
    else:
        inc_centroid = inc_rolloff85 = inc_bandwidth = inc_flatness = None

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
        zero_crossing_rate=round(zcr_post_val, 5),
        zero_crossing_rate_pre=round(zcr_pre_val, 5) if zcr_pre_val is not None else None,
        zero_crossing_rate_delta=round(zcr_delta_val, 5) if zcr_delta_val is not None else None,
        f0_median_hz=round(float(np.median(f0_voiced)), 1) if len(f0_voiced) else None,
        f0_mean_hz=round(float(np.mean(f0_voiced)), 1) if len(f0_voiced) else None,
        voiced_fraction=round(float(np.mean(~np.isnan(f0))), 3),
        f1_median_hz=f1_median,
        f2_median_hz=f2_median,
        increase_centroid_hz=round(inc_centroid, 1) if inc_centroid is not None else None,
        increase_rolloff85_hz=round(inc_rolloff85, 1) if inc_rolloff85 is not None else None,
        increase_bandwidth_hz=round(inc_bandwidth, 1) if inc_bandwidth is not None else None,
        increase_flatness=round(inc_flatness, 6) if inc_flatness is not None else None,
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
