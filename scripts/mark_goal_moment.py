r"""録音済みのゴール音声を再生しながら、人間が知覚したタイミングを記録する。

**Windows専用。**(キー入力検知に標準ライブラリ`msvcrt`を使用しているため)

これは、`goal-audio analyze`によるピーク検出(音量ベース・アルゴリズム的)とは独立に、
「人間が実際に聞いて、ゴールだと感じた瞬間」を記録するための補助スクリプト。両者を
突き合わせることで、ピーク検出が別の場面(スルーパスへの歓声、演出音など)を誤って
捉えていないかを検証できる。

これも`loopback_record.py`と同様、音声データを扱う準備工程の一部として、意図的に
goal_audio_analysisライブラリ/CLIには含めていない。検出結果との突き合わせ(数値比較)は
ライブラリ側(`features.analyze_clip`が自動で行う。sidecarファイルが存在すれば読み込む)
で行う。

セットアップは`loopback_record.py`と共通(同じ仮想環境・同じ`scripts/requirements.txt`を使う)。

使い方:
    python scripts/mark_goal_moment.py <clip.wav>

流れ:
    1. 指定したWAVファイルをスピーカーから再生する
    2. 再生を聞きながら、ゴールだと感じた瞬間にスペースキーかEnterキーを押す
       (複数回押した場合は最後の入力を採用する。聞き直して押し直してよい)
    3. 再生終了後、採用したタイミングを表示し、同じ場所に
       `<clip>.mark.json`として保存する(例: `goal1.wav` -> `goal1.mark.json`)
    4. 以降`goal-audio analyze`を実行すると、このファイルを自動的に読み込み、
       アルゴリズムのピーク検出結果と比較する
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf

if platform.system() == "Windows":
    import msvcrt


def play_and_mark(wav_path: Path, chunk_frames: int = 1024) -> list[float]:
    """Play `wav_path` through the default speaker; return elapsed-time (s)
    marks for every space/Enter keypress during playback.
    """
    data, samplerate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    speaker = sc.default_speaker()

    marks: list[float] = []
    start = time.time()
    with speaker.player(samplerate=samplerate) as player:
        idx = 0
        n = len(data)
        while idx < n:
            chunk = data[idx: idx + chunk_frames]
            player.play(chunk)
            idx += chunk_frames
            while msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b" ", b"\r"):
                    elapsed = round(time.time() - start, 3)
                    marks.append(elapsed)
                    print(f"  marked at {elapsed:.3f}s")
    return marks


def main(argv=None):
    if platform.system() != "Windows":
        print(
            "error: this script only supports Windows (msvcrt-based key detection). "
            f"Detected platform: {platform.system()}.",
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

    print(f"Playing: {clip_path}")
    print("ゴールだと感じた瞬間にスペースキー(またはEnter)を押してください...")
    marks = play_and_mark(clip_path)

    if not marks:
        print("マークなし。終了します(sidecarファイルは作成しません)。")
        return 0

    human_marked_time_s = marks[-1]
    print(f"採用するタイミング: {human_marked_time_s:.3f}s (全マーク: {marks})")

    mark_path = clip_path.with_suffix(".mark.json")
    mark_path.write_text(
        json.dumps(
            {"human_marked_time_s": human_marked_time_s, "all_marks_s": marks},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved: {mark_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
