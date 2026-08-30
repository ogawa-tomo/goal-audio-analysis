r"""録音済みのゴール音声を再生しながら、人間が知覚したタイミングを記録する。

**Windows専用。**(キー入力検知に標準ライブラリ`msvcrt`を使用しているため)

これは、`goal-audio analyze`によるピーク検出(音量ベース・アルゴリズム的)とは独立に、
「人間が実際に聞いて、ゴールだと感じた瞬間」を記録するための補助スクリプト。両者を
突き合わせることで、ピーク検出が別の場面(スルーパスへの歓声、演出音など)を誤って
捉えていないかを検証できる。

## 重要: 再生の遅延について

`soundcard`のスピーカー再生には、`play()`を呼んでから実際に音が出るまでに**数秒単位の
遅延**があることが分かっている(このスクリプトの初期バージョンで実際に発生し、検証で
判明した)。単純に「再生開始からの経過時間」を測るだけでは、この遅延の分だけ実際の
音声内容より遅れた値になってしまう。

この対策として、**再生と同時にループバック録音も行い**、録音された音声そのものの
立ち上がり位置と、元ファイルの立ち上がり位置を比較して遅延を自動計測し、記録した
打刻(生の値)から差し引いて補正している。キー入力の打刻自体は「録音済みサンプル数」を
基準にしており(壁時計時間ではない)、これは実際にスピーカーから出た音の量に対応するため、
再生側の起動遅延の影響を受けにくい。

これも`loopback_record.py`と同様、音声データを扱う準備工程の一部として、意図的に
goal_audio_analysisライブラリ/CLIには含めていない。検出結果との突き合わせ(数値比較)は
ライブラリ側(`features.analyze_clip`が自動で行う。sidecarファイルが存在すれば読み込む)
で行う。

セットアップは`loopback_record.py`と共通(同じ仮想環境・同じ`scripts/requirements.txt`を使う)。

使い方:
    python scripts/mark_goal_moment.py <clip.wav>

流れ:
    1. 指定したWAVファイルをスピーカーから再生する(同時にループバック録音も行われる)
    2. 再生を聞きながら、ゴールだと感じた瞬間にスペースキーかEnterキーを押す
       (複数回押した場合は最後の入力を採用する。聞き直して押し直してよい)
    3. 再生終了後、録音した音声と元ファイルを比較して再生の遅延を自動計測し、
       打刻を補正した上で表示する。同じ場所に`<clip>.mark.json`として保存する
       (例: `goal1.wav` -> `goal1.mark.json`)
    4. 以降`goal-audio analyze`を実行すると、このファイルを自動的に読み込み、
       アルゴリズムのピーク検出結果と比較する
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf

if platform.system() == "Windows":
    import msvcrt


def _mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim == 2 else data


def _rms_curve(data: np.ndarray, samplerate: int, frame_s: float = 0.05, hop_s: float = 0.025):
    mono = _mono(data).astype(np.float64)
    frame = max(1, int(frame_s * samplerate))
    hop = max(1, int(hop_s * samplerate))
    n = len(mono)
    times, rms = [], []
    i = 0
    while i + frame <= n:
        window = mono[i:i + frame]
        rms.append(float(np.sqrt(np.mean(window ** 2))))
        times.append(i / samplerate)
        i += hop
    return np.array(times), np.array(rms)


def _find_onset(times: np.ndarray, rms: np.ndarray, search_window_s: float = 10.0, threshold_frac: float = 0.15) -> float:
    """First time RMS rises past `threshold_frac` of its own max within the
    first `search_window_s` seconds -- i.e. where the clip's content starts,
    as opposed to leading silence.
    """
    mask = times <= search_window_s
    if not mask.any():
        return 0.0
    peak = float(np.max(rms[mask]))
    if peak <= 0:
        return 0.0
    thresh = threshold_frac * peak
    above = np.where(rms[mask] >= thresh)[0]
    return float(times[mask][above[0]]) if len(above) else 0.0


def play_record_and_mark(wav_path: Path, chunk_frames: int = 1024):
    """Play `wav_path` while simultaneously loopback-recording it. Returns
    `(raw_marks_s, recorded_audio, samplerate, original_audio)`.

    `raw_marks_s` are timestamped against recorded-sample count (i.e. real
    elapsed time since the recorder itself started), not wall-clock time
    since the playback call -- this sidesteps the playback pipeline's own
    startup latency, but still needs the onset-based correction in `main()`
    to line up with the original file's own content timeline.
    """
    data, samplerate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    speaker = sc.default_speaker()
    mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)

    marks: list[float] = []
    recorded_chunks: list[np.ndarray] = []

    def do_playback():
        with speaker.player(samplerate=samplerate) as player:
            idx = 0
            n = len(data)
            while idx < n:
                player.play(data[idx: idx + chunk_frames])
                idx += chunk_frames

    playback_thread = threading.Thread(target=do_playback)
    total_duration_s = len(data) / samplerate
    record_seconds = total_duration_s + 2.0  # margin for playback startup latency + tail

    recorded_samples = 0
    with mic.recorder(samplerate=samplerate) as recorder:
        playback_thread.start()
        n_chunks = int(record_seconds * samplerate / chunk_frames) + 1
        for _ in range(n_chunks):
            chunk = recorder.record(numframes=chunk_frames)
            recorded_chunks.append(chunk)
            recorded_samples += chunk_frames
            elapsed = recorded_samples / samplerate
            while msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b" ", b"\r"):
                    marks.append(round(elapsed, 3))
                    print(f"  marked at {elapsed:.3f}s (raw, before latency correction)")
    playback_thread.join()

    recorded_audio = np.concatenate(recorded_chunks, axis=0)
    return marks, recorded_audio, samplerate, data


def main(argv=None):
    if platform.system() != "Windows":
        print(
            "error: this script only supports Windows (msvcrt-based key detection, "
            f"WASAPI loopback). Detected platform: {platform.system()}.",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="再生するWAVファイルのパス")
    args = parser.parse_args(argv)

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"error: file not found: {clip_path}", file=sys.stderr)
        return 1

    print(f"Playing (+ measuring playback latency via loopback): {clip_path}")
    print("ゴールだと感じた瞬間にスペースキー(またはEnter)を押してください...")
    raw_marks, recorded_audio, samplerate, original_audio = play_record_and_mark(clip_path)

    if not raw_marks:
        print("マークなし。終了します(sidecarファイルは作成しません)。")
        return 0

    orig_times, orig_rms = _rms_curve(original_audio, samplerate)
    rec_times, rec_rms = _rms_curve(recorded_audio, samplerate)
    orig_onset = _find_onset(orig_times, orig_rms)
    rec_onset = _find_onset(rec_times, rec_rms)
    latency_s = rec_onset - orig_onset

    print(f"再生パイプラインの遅延を自動計測: {latency_s:.3f}s "
          f"(元ファイルの立ち上がり={orig_onset:.3f}s, 録音の立ち上がり={rec_onset:.3f}s)")

    corrected_marks = [round(m - latency_s, 3) for m in raw_marks]
    human_marked_time_s = corrected_marks[-1]
    print(f"補正後のタイミング: {human_marked_time_s:.3f}s (全マーク補正後: {corrected_marks})")

    mark_path = clip_path.with_suffix(".mark.json")
    mark_path.write_text(
        json.dumps(
            {
                "human_marked_time_s": human_marked_time_s,
                "all_marks_s": corrected_marks,
                "raw_marks_s": raw_marks,
                "measured_latency_s": round(latency_s, 3),
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved: {mark_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
