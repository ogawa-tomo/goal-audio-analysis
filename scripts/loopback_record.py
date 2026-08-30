r"""システムの音声出力をWAVファイルにループバック録音する。

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

セットアップ(初回のみ、Windows側のPowerShellで実行。仮想環境(venv)を使う前提の手順。
他のPython環境と依存関係が衝突しないようにするため):
    0. (このリポジトリをWSL側(\\wsl.localhost\...)に置いている場合の注意)
       venv(正確にはpipのブートストラップ)はUNCパス上ではうまく動かないことを確認して
       いる。そのため**venv自体はWindowsのローカルディスク上に作ること**。スクリプトの
       実行や、requirements.txtの参照はWSL側のパスのままで問題ない(パッケージの
       インストール元・実行対象がUNCパスというだけなら問題なく動く)
    1. Windows用のPython(3.9以降を想定。WSL内のPythonではない)がインストール済みか
       確認する。未インストールなら python.org からインストールする
       > python --version
    2. このスクリプト専用の仮想環境を、Windowsのローカルディスク上(例: ユーザー
       フォルダ直下)に作成する。パス・フォルダ名は任意
       > python -m venv $env:USERPROFILE\.venv-goal-audio-loopback
    3. 仮想環境を有効化する(PowerShellのセッションを開くたび毎回必要。変数から組み立てた
       パスをスクリプトとして実行するので、先頭に呼び出し演算子`&`が必要)
       > & "$env:USERPROFILE\.venv-goal-audio-loopback\Scripts\Activate.ps1"
    4. 依存パッケージをインストールする(scripts/requirements.txt に一覧がある:
       soundcard, soundfile, numpy)。この時点でプロンプトの先頭に
       (.venv-goal-audio-loopback) と表示されていることを確認しておくこと
       (仮想環境が有効化されている印)。<リポジトリのパス>は実際の場所に置き換える
       (WSL側に置いている場合は \\wsl.localhost\<ディストリ名>\home\<ユーザー名>\goal-audio-analysis 等)
       > pip install -r <リポジトリのパス>\scripts\requirements.txt
    5. 動作確認(3秒だけ試し録りしてみる。出力先はカレントディレクトリのtest.wav)
       > python <リポジトリのパス>\scripts\loopback_record.py test.wav --duration 3

使い方(2回目以降は、まず仮想環境を有効化してから実行する):
    & "$env:USERPROFILE\.venv-goal-audio-loopback\Scripts\Activate.ps1"
    python <リポジトリのパス>\scripts\loopback_record.py OUT.wav --duration 20 --countdown 3

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

必要なパッケージ: venv上で pip install -r scripts/requirements.txt (詳細は上記セットアップ参照)
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
    parser.add_argument("out", help="出力WAVファイルのパス")
    parser.add_argument(
        "--duration", type=float, required=True,
        help="録音する長さ(単位: 秒)。カウントダウン終了後、この秒数だけ録音する",
    )
    parser.add_argument(
        "--countdown", type=float, default=3.0,
        help="録音開始までの待機時間(単位: 秒、デフォルト3秒)。この間にブラウザに切り替えて再生を始める",
    )
    parser.add_argument(
        "--samplerate", type=int, default=48000,
        help="サンプリングレート(単位: Hz、デフォルト48000)",
    )
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
