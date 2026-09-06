"""Cut a clip's analysis window out into its own audio file.

Everything downstream (`spectral.py`, `formant.py`, `f0.py`, `zcr.py`,
`hnr.py`) analyzes only the file this produces -- none of them know
about the original clip, the onset mark, or where either lives. This
also makes the window itself something a human can listen to directly,
to confirm it actually captures the reaction.

Regenerate this on every run rather than treating it as a durable cache:
it's cheap to produce, and always overwriting it is what keeps it from
ever going stale relative to whatever onset time was most recently
confirmed.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import soundfile as sf


def extract_window(
    clip_path: str | Path,
    onset_time_s: float,
    window_s: float,
    out_path: str | Path,
    sr: int = 22050,
) -> Path:
    """Cut `[onset_time_s, onset_time_s + window_s)` out of `clip_path` and write it to `out_path`."""
    y, _sr = librosa.load(clip_path, sr=sr, mono=True)
    s_idx = int(onset_time_s * sr)
    e_idx = min(len(y), int((onset_time_s + window_s) * sr))
    y_win = y[s_idx:e_idx]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile defaults to 16-bit PCM for WAV, which would quantize y_win
    # (float32 from librosa.load) on every write -- a small but real and
    # entirely avoidable precision loss for downstream analysis. FLOAT
    # keeps this a lossless round-trip; still plays fine in any current
    # browser or media player.
    sf.write(str(out_path), y_win, sr, subtype="FLOAT")
    return out_path
