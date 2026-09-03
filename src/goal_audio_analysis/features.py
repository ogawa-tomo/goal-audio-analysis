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
import scipy.signal


class MissingMarkError(ValueError):
    """Raised when a clip has no human-specified goal moment."""


@dataclass
class ClipFeatures:
    file: str
    onset_time_s: float
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
    hnr_db: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _load_onset_mark(path: Path) -> Optional[float]:
    """Load `onset_marked_time_s` from `<path>`'s sidecar `.mark.json`, if any.

    The moment the crowd's reaction *starts rising*, human-confirmed --
    everything in `analyze_clip` is anchored on this. An earlier design
    also used a separate amplitude-peak anchor (`human_marked_time_s`) for
    an attack/decay-time pair (rise/fall speed relative to the loudest
    instant), but both were dropped: for continuously-loud crowd audio, the
    10%/50%-of-peak thresholds they were defined against frequently never
    occurred within the clip at all (decay_time_s came back null for 14 of
    17 clips in this project's dataset) or only at a degenerate point
    (the very start of the recording), making the whole measurement
    meaningless on this kind of data -- not just noisy. Since attack/decay
    speed wasn't the object of interest here anyway (this project compares
    the reaction's spectral *character*, not its temporal dynamics), the
    fix was to drop the metric and the peak anchor it depended on, not to
    patch the threshold logic.

    Onset values here were obtained by taking the spectral-shape-change
    candidate (see the onset-detection experiments in this project's
    history) nearest to each clip's now-removed peak mark, then confirming
    by ear and by inspecting the candidate/mark plot for all 17 clips.
    """
    mark_path = path.with_suffix(".mark.json")
    if not mark_path.exists():
        return None
    try:
        data = json.loads(mark_path.read_text(encoding="utf-8"))
        return float(data["onset_marked_time_s"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None


def _load_post_onset_window(path: Path, sr: int, spectral_window_s: float) -> np.ndarray:
    """Load `path` and return its post-onset analysis window `[onset, onset + spectral_window_s)`.

    Shared by `spectrum_curve`/`formant_envelope_curve`/`hnr_curve`, which
    each describe this same window from a different angle. Raises
    `MissingMarkError` if `path` has no onset mark.
    """
    onset_marked_time_s = _load_onset_mark(path)
    if onset_marked_time_s is None:
        raise MissingMarkError(
            f"{path.name} has no onset mark. Add onset_marked_time_s to "
            f"{path.with_suffix('.mark.json').name}."
        )
    y, _sr = librosa.load(path, sr=sr, mono=True)
    s_idx = int(onset_marked_time_s * sr)
    e_idx = min(len(y), int((onset_marked_time_s + spectral_window_s) * sr))
    return y[s_idx:e_idx]


def spectrum_curve(
    path: str | Path,
    sr: int = 22050,
    spectral_window_s: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(freqs, spectrum)` -- one clip's raw post-onset average spectrum.

    This is the same window `analyze_clip`'s spectral_centroid_hz etc. are
    computed from -- `[onset, onset + spectral_window_s)` -- but returns
    the actual per-bin shape instead of a centroid/rolloff/bandwidth/
    flatness summary.
    """
    y_win = _load_post_onset_window(Path(path), sr, spectral_window_s)

    S = np.abs(librosa.stft(y_win, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    return freqs, S.mean(axis=1)


def hnr_curve(
    path: str | Path,
    sr: int = 22050,
    spectral_window_s: float = 3.0,
    time_step: float = 0.01,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return `(times, hnr_db)` -- one clip's HNR over time, within the post-onset window.

    `hnr_db` (the scalar field in `analyze_clip`) is the median of this
    same curve. Returned frame-by-frame here instead, since HNR -- unlike
    the spectrum/envelope curves -- isn't defined per frequency: it's a
    per-time-frame measure of periodicity from Praat's cross-correlation
    method (see `_harmonicity`), so time is the only axis that makes
    sense for it. `times` is seconds since the start of the post-onset
    window (i.e. since the onset itself). Undefined frames (Praat's -200
    sentinel) come back as `nan` here rather than being dropped, so gaps
    in periodicity stay visible instead of silently vanishing.

    Returns `None` if the window is too short to yield any frame.
    """
    import parselmouth

    y_win = _load_post_onset_window(Path(path), sr, spectral_window_s)

    snd = parselmouth.Sound(y_win, sampling_frequency=sr)
    harmonicity = snd.to_harmonicity_cc(time_step=time_step)
    times = harmonicity.xs()
    if len(times) == 0:
        return None
    values = harmonicity.values.flatten()
    values = np.where(values == -200.0, np.nan, values)
    return times, values


def formant_envelope_curve(
    path: str | Path,
    sr: int = 22050,
    spectral_window_s: float = 3.0,
    order: int = 10,
    max_formant_hz: float = 5500.0,
    frame_length_s: float = 0.025,
    hop_length_s: float = 0.01,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return `(freqs, envelope)` -- the LPC spectral envelope underlying F1/F2.

    `_formants` (F1/F2) picks peaks off this kind of envelope, frame by
    frame via Praat/parselmouth's Burg-method tracker, then reduces the
    result to a per-clip median. This instead returns the *envelope
    itself*, averaged across frames of the post-onset window -- so the
    resonance shape can be inspected or plotted directly, the same way
    `spectrum_curve` exposes the FFT spectrum shape.

    Computed independently of `_formants` via librosa's own Burg-method
    LPC rather than reusing parselmouth, so this approximates -- but
    isn't guaranteed bit-identical to -- what actually produced F1/F2.
    Audio is resampled to `2 * max_formant_hz` before LPC (mirroring
    Praat's own downsampling, since a formant above the Nyquist rate
    can't be estimated), and each frame gets a light pre-emphasis (0.97)
    to roughly match Praat's `pre_emphasis_from` boost -- without it, the
    LPC fit is dominated by low-frequency crowd rumble instead of the
    formant structure.

    Returns `None` if the post-onset window is too short to yield a
    single analysis frame.
    """
    y_win = _load_post_onset_window(Path(path), sr, spectral_window_s)

    sr_lpc = int(2 * max_formant_hz)
    y_lpc = librosa.resample(y_win, orig_sr=sr, target_sr=sr_lpc)

    frame_length = int(frame_length_s * sr_lpc)
    hop_length = int(hop_length_s * sr_lpc)
    if len(y_lpc) < frame_length:
        return None

    n_freqs = 512
    freqs = np.linspace(0, sr_lpc / 2, n_freqs)
    window = scipy.signal.windows.hamming(frame_length)

    responses = []
    for start in range(0, len(y_lpc) - frame_length + 1, hop_length):
        frame = y_lpc[start:start + frame_length] * window
        frame = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
        try:
            a = librosa.lpc(frame, order=order)
        except (librosa.util.exceptions.ParameterError, np.linalg.LinAlgError):
            continue
        _, h = scipy.signal.freqz([1.0], a, worN=freqs, fs=sr_lpc)
        responses.append(np.abs(h))

    if not responses:
        return None
    return freqs, np.mean(responses, axis=0)


def _relative_window(y: np.ndarray, sr: int, anchor_time: float, start_s: float, end_s: float) -> Optional[np.ndarray]:
    """Slice out `[anchor_time + start_s, anchor_time + end_s)`, clamped to not start before 0.

    Used for the pre/post-goal baseline windows behind the
    zero_crossing_rate before/after delta.
    """
    w_s = max(0.0, anchor_time + start_s)
    w_e = max(0.0, anchor_time + end_s)
    if w_e - w_s < 0.1:
        return None
    return y[int(w_s * sr):int(w_e * sr)]


def analyze_clip(
    path: str | Path,
    sr: int = 22050,
    spectral_window_s: float = 3.0,
    zcr_window_s: float = 2.0,
    with_formants: bool = True,
) -> ClipFeatures:
    """Extract acoustic features from one audio clip.

    Requires a human-confirmed onset mark (`onset_marked_time_s`) in
    `<clip>.mark.json` -- where the reaction starts rising -- raises
    `MissingMarkError` if missing. There is no algorithmic fallback: see
    `_load_onset_mark` for why.

    Everything here describes the post-onset reaction's spectral
    *character* on its own terms -- the plain spectral_*/f0/f1/f2/hnr_db
    fields, all computed over `[onset, onset + spectral_window_s)` -- plus
    zero_crossing_rate_pre/_delta, which additionally compare that window
    against `zcr_window_s` seconds immediately before the onset. An
    earlier design also compared the *whole* post-onset spectrum against a
    pre-onset baseline (increase_centroid_hz etc., an "increase spectrum")
    to isolate the reaction from preexisting ambient audio (commentary, an
    already-ongoing chant) -- dropped once the question of interest
    shifted from "what did the goal add" to "what does the post-goal
    crowd sound like", at which point that preexisting audio stopped being
    a confound to remove and became part of what's being described. See
    `reports/premier_vs_laliga.md` 4.5 section for the fuller history.
    """
    path = Path(path)

    onset_marked_time_s = _load_onset_mark(path)
    if onset_marked_time_s is None:
        raise MissingMarkError(
            f"{path.name} has no onset mark. Add onset_marked_time_s to "
            f"{path.with_suffix('.mark.json').name}."
        )
    onset_time = onset_marked_time_s

    y, _sr = librosa.load(path, sr=sr, mono=True)

    win_start = onset_time
    win_end = onset_time + spectral_window_s
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
    y_pre = _relative_window(y, sr, onset_time, -zcr_window_s, 0.0)
    y_post_for_delta = _relative_window(y, sr, onset_time, 0.0, zcr_window_s)
    if y_pre is not None and len(y_pre) > 0 and y_post_for_delta is not None and len(y_post_for_delta) > 0:
        zcr_pre_val = float(np.mean(librosa.feature.zero_crossing_rate(y=y_pre)[0]))
        zcr_delta_val = float(np.mean(librosa.feature.zero_crossing_rate(y=y_post_for_delta)[0])) - zcr_pre_val
    else:
        zcr_pre_val = zcr_delta_val = None

    f1_median = f2_median = hnr_median = None
    if with_formants:
        f1_median, f2_median = _formants(y_win, sr)
        hnr_median = _harmonicity(y_win, sr)

    return ClipFeatures(
        file=path.name,
        onset_time_s=round(onset_time, 2),
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
        hnr_db=hnr_median,
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


def _harmonicity(y_win: np.ndarray, sr: int) -> Optional[float]:
    """Estimate median harmonics-to-noise ratio (HNR, dB) via Praat's cross-correlation method.

    HNR measures how periodic/tonal a sound is versus how noise-like: a
    high value means a strong, stable periodic component dominates (e.g.
    a crowd chanting roughly in unison), a low or negative value means
    broadband noise dominates (e.g. an unstructured roar with no shared
    pitch). Like `_formants`, this treats the crowd as if it were one
    periodic source, which is only an approximation -- but unlike F0, it
    doesn't require pitch-tracking to succeed on a frame to produce a
    value, so it stays defined even where `voiced_fraction` is low.

    The cross-correlation method (`to_harmonicity_cc`) is used rather
    than the autocorrelation one, per Praat's own guidance that cc is
    more robust on noisy/non-speech signals. Undefined frames come back
    from parselmouth as the sentinel value -200.0 and are excluded before
    taking the median. Returns `None` if every frame is undefined.
    """
    import parselmouth

    snd = parselmouth.Sound(y_win, sampling_frequency=sr)
    harmonicity = snd.to_harmonicity_cc()
    values = harmonicity.values.flatten()
    values = values[values != -200.0]
    if len(values) == 0:
        return None
    return round(float(np.median(values)), 2)


def analyze_directory(dir_path: str | Path, pattern: str = "*.wav", **kwargs) -> list[ClipFeatures]:
    """Run `analyze_clip` on every file matching `pattern` in `dir_path`."""
    dir_path = Path(dir_path)
    return [analyze_clip(p, **kwargs) for p in sorted(dir_path.glob(pattern))]
