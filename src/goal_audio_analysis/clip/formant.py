"""Formant (F1/F2, vowel-color) analysis of an already-windowed clip (see `window.py`)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import scipy.signal


@dataclass
class FormantFeatures:
    f1_median_hz: Optional[float]
    f2_median_hz: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(window_wav_path: str | Path, sr: int = 22050) -> FormantFeatures:
    """Estimate median F1/F2 via Praat's Burg method (parselmouth).

    Note: formant tracking assumes a single vocal tract. Applied to a crowd
    of overlapping voices this approximates "where the ensemble's spectral
    envelope peaks", not a literal individual's formants -- treat it as a
    directional signal, not a precise phonetic measurement.
    """
    import parselmouth

    y, _sr = librosa.load(window_wav_path, sr=sr, mono=True)

    snd = parselmouth.Sound(y, sampling_frequency=sr)
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
    return FormantFeatures(f1_median_hz=f1_median, f2_median_hz=f2_median)


def curve(
    window_wav_path: str | Path,
    sr: int = 22050,
    order: int = 10,
    max_formant_hz: float = 5500.0,
    frame_length_s: float = 0.025,
    hop_length_s: float = 0.01,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return `(freqs, envelope)` -- the LPC spectral envelope `analyze()`'s F1/F2 are picked from as peaks.

    Computed independently of `analyze()` via librosa's own Burg-method
    LPC rather than reusing parselmouth, so this approximates -- but
    isn't guaranteed bit-identical to -- what actually produced F1/F2.
    Audio is resampled to `2 * max_formant_hz` before LPC (mirroring
    Praat's own downsampling, since a formant above the Nyquist rate
    can't be estimated), and each frame gets a light pre-emphasis (0.97)
    to roughly match Praat's `pre_emphasis_from` boost -- without it, the
    LPC fit is dominated by low-frequency crowd rumble instead of the
    formant structure.

    Returns `None` if `window_wav_path` is too short to yield a single
    analysis frame.
    """
    y, _sr = librosa.load(window_wav_path, sr=sr, mono=True)

    sr_lpc = int(2 * max_formant_hz)
    y_lpc = librosa.resample(y, orig_sr=sr, target_sr=sr_lpc)

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
