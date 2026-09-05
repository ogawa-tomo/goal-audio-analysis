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
    3. 仮想環境内のpython.exeを指す変数を用意する(PowerShellのセッションを開くたび
       毎回必要)。**`Activate.ps1`による有効化は使わない** -- 既定の実行ポリシーだと
       "スクリプトの実行が無効になっている"というエラー(PSSecurityException)で
       弾かれる環境が多いため。venv内のpython.exeをフルパスで直接呼べば、有効化
       (=PowerShellスクリプトの実行)自体が不要になりこの問題を回避できる
       > $venvPython = "$env:USERPROFILE\.venv-goal-audio-loopback\Scripts\python.exe"
    4. 依存パッケージをインストールする(scripts/requirements.txt に一覧がある:
       soundcard, soundfile, numpy)。<リポジトリのパス>は実際の場所に置き換える
       (WSL側に置いている場合は \\wsl.localhost\<ディストリ名>\home\<ユーザー名>\goal-audio-analysis 等)
       > & $venvPython -m pip install -r <リポジトリのパス>\scripts\requirements.txt
    5. 動作確認(3秒だけ試し録りしてみる)。**出力先は必ず絶対パスで指定する** --
       相対パスのままだと、実行時のカレントディレクトリが書き込み権限のない場所
       (例: PowerShellを開いた直後の既定ディレクトリが `C:\WINDOWS\System32` に
       なっているケース)だと `soundfile.LibsndfileError: ... System error.` で
       失敗することがある
       > & $venvPython <リポジトリのパス>\scripts\loopback_record.py "$env:USERPROFILE\test.wav" --duration 3

使い方(2回目以降は、まず`$venvPython`変数を再定義してから実行する。PowerShellの
セッションをまたぐと変数は消えるため):
    $venvPython = "$env:USERPROFILE\.venv-goal-audio-loopback\Scripts\python.exe"
    & $venvPython <リポジトリのパス>\scripts\loopback_record.py "<絶対パスでの出力先>\OUT.wav" --duration 14 --countdown 3

典型的な使い方の流れ:
    1. ブラウザで元動画を開き、捉えたい瞬間(例: ゴール)の**5秒前**で一時停止する
       (ちょうど`--lead`相当の秒数ぴったりにすると、再生ボタンを押すタイミングが
       前後にズレたときに耐性がなくなる。カウントダウン中はブラウザに切り替えて
       手元のカウントダウンが見えなくなるため、早く/遅く押す両方向のズレが起こりうる
       ことに注意)
    2. `--duration 14`を指定してこのスクリプトを実行する(録音14秒。詳しい根拠はREADMEの
       「音声データの用意について」参照)
    3. カウントダウン中にブラウザに切り替え、**カウントダウンが終わる少し前に**再生ボタンを
       押す(ちょうど終わったタイミングではなく、心持ち早めに)。こうすると、録音開始時点で
       既に動画の音声が流れている状態になり、録音が完全な無音から始まらずに済む。多少早く
       押しすぎても、`--duration 14`には十分な余裕があるので問題ない
    4. できあがったWAVファイルは、そのまま`goal-audio mark-onset`でマークしてから
       `goal-audio analyze`に渡せる(すでに短く切り出し済みなので`extract_clip.py`は不要)

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
import warnings
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf


def record_loopback(out_path: str | Path, duration_s: float, samplerate: int = 48000) -> tuple[Path, int]:
    """Record `duration_s` seconds of loopback audio to `out_path`.

    Returns `(out_path, discontinuity_count)`. `discontinuity_count` is how
    many times `soundcard` raised its "data discontinuity in recording"
    warning during capture (a dropped/duplicated audio buffer, typically a
    WASAPI hiccup) -- a nonzero count means the recording may contain a
    brief click or glitch and is worth inspecting (or re-recording) before
    trusting it for analysis. Note that `analyze`'s peak search already
    smooths the RMS envelope specifically to avoid such single-frame clicks
    being mistaken for a genuine crowd swell, but a discontinuity elsewhere
    in the file could still be worth a second look.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speaker = sc.default_speaker()
    mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)

    frames = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with mic.recorder(samplerate=samplerate) as recorder:
            n_chunks = int(duration_s * samplerate / 1024) + 1
            for _ in range(n_chunks):
                frames.append(recorder.record(numframes=1024))
        discontinuity_count = sum(1 for w in caught if "discontinuity" in str(w.message))

    audio = np.concatenate(frames, axis=0)
    n_samples = int(duration_s * samplerate)
    audio = audio[:n_samples]

    sf.write(str(out_path), audio, samplerate)
    return out_path, discontinuity_count


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
    out, discontinuity_count = record_loopback(args.out, args.duration, args.samplerate)
    print(f"saved: {out}")
    if discontinuity_count:
        print(
            f"WARNING: {discontinuity_count} recording discontinuity(ies) detected during capture. "
            f"This clip may contain a brief click/glitch -- inspect it (e.g. a spectrogram) or "
            f"consider re-recording before trusting the analysis.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
