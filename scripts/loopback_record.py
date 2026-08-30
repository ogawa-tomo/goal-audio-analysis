"""Loopback-record the system's audio output to a WAV file.

**Windows-only.** This script relies on WASAPI loopback (via the
`soundcard` library), which is a Windows/WASAPI-specific mechanism for
capturing "what you hear" without extra setup. It does **not** work on
macOS (CoreAudio has no built-in loopback route -- capturing system audio
there requires installing a virtual audio driver such as BlackHole and
routing output through it, which this script does not currently support)
or inside WSL (no path to the Windows OS audio mixer from there).

This is a data-preparation helper, deliberately kept OUT of the
goal_audio_analysis library/CLI (see README: audio acquisition is a
separate concern from analysis). Run it with a native Windows Python
(not WSL).

Usage:
    python scripts/loopback_record.py OUT.wav --duration 20 --countdown 3

Typical workflow:
    1. Open the source video in your browser and pause it at the point
       just before the moment you want to capture (e.g. a few seconds
       before a goal).
    2. Run this script with an appropriate --duration and --countdown.
    3. During the countdown, switch to the browser and hit play so that
       playback starts right as recording begins.
    4. The resulting WAV can be fed straight into
       `goal-audio extract-clip` / `goal-audio analyze`.

Before recording:
    - Turn OFF any OS/driver audio enhancements (loudness equalization,
      spatial/surround virtualization, etc.) -- Sound settings > your
      output device > Properties > Enhancements/Spatial sound > disable.
    - Set the system volume to a fixed, known level (100% recommended,
      as long as nothing clips) and leave it there for all recordings
      in a comparison, so results stay comparable across clips.
    - Close other apps that might play sound or trigger notifications
      during the recording window.

Requires: pip install soundcard
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf


def record_loopback(out_path: str | Path, duration_s: float, samplerate: int = 48000) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speaker = sc.default_speaker()
    mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)

    frames = []
    with mic.recorder(samplerate=samplerate) as recorder:
        n_chunks = int(duration_s * samplerate / 1024) + 1
        for _ in range(n_chunks):
            frames.append(recorder.record(numframes=1024))

    audio = np.concatenate(frames, axis=0)
    n_samples = int(duration_s * samplerate)
    audio = audio[:n_samples]

    sf.write(str(out_path), audio, samplerate)
    return out_path


def main(argv=None):
    if platform.system() != "Windows":
        print(
            f"error: this script only supports Windows (WASAPI loopback). "
            f"Detected platform: {platform.system()}. See the module docstring "
            f"for why macOS/WSL aren't supported yet.",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="output WAV path")
    parser.add_argument("--duration", type=float, required=True, help="how many seconds to record")
    parser.add_argument("--countdown", type=float, default=3.0, help="seconds to wait before recording starts")
    parser.add_argument("--samplerate", type=int, default=48000)
    args = parser.parse_args(argv)

    print(f"Recording device: {sc.default_speaker().name}")
    for remaining in range(int(args.countdown), 0, -1):
        print(f"  starting in {remaining}...")
        time.sleep(1)

    print(f"Recording for {args.duration:.1f}s -> {args.out}")
    out = record_loopback(args.out, args.duration, args.samplerate)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
