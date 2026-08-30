"""システムの音声出力をWAVファイルにループバック録音する。

**Windows専用。** このスクリプトはWASAPIループバック(`soundcard`ライブラリ経由)を
利用しており、これは追加のセットアップなしに「聞こえている音」を録音できる
Windows/WASAPI固有の仕組み。**macOSでは動作しない**(CoreAudioには標準のループバック
機構がなく、システム音声を録音するには`BlackHole`等の仮想オーディオデバイスの導入と
そこへの出力ルーティングが必要で、このスクリプトは現時点でそれに対応していない)。
WSL内でも動作しない(WSLからはWindowsのOSオーディオミキサーにアクセスする経路がない)。

これは音声データを準備するための補助スクリプトであり、意図的に
goal_audio_analysisライブラリ/CLIには含めていない(READMEの通り、音声データの取得は
解析とは別の関心事として切り離している)。WSLではなく、Windowsネイティブの
Pythonで実行すること。

使い方:
    python scripts/loopback_record.py OUT.wav --duration 20 --countdown 3

典型的な使い方の流れ:
    1. ブラウザで元動画を開き、切り出したい瞬間(例: ゴールの数秒前)の直前で一時停止する
    2. 適切な--duration/--countdownを指定してこのスクリプトを実行する
    3. カウントダウン中にブラウザに切り替え、録音開始と同時に再生が始まるよう再生ボタンを押す
    4. できあがったWAVファイルはそのまま`goal-audio extract-clip` / `goal-audio analyze`
       に渡せる

録音前に:
    - OS/ドライバの音声エフェクト(ラウドネス補正、空間/サラウンド仮想化など)をすべて
      OFFにする -- サウンド設定 > 出力デバイス > プロパティ > 拡張機能/空間オーディオ
      から無効化
    - システム音量を固定の既知の値にする(クリッピングしない範囲で100%を推奨)。
      比較対象となる録音全体で同じ音量のまま統一すること(結果を比較可能に保つため)
    - 録音時間中に音を鳴らしたり通知を出したりする可能性のある他のアプリは閉じておく

必要なパッケージ: pip install soundcard
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
