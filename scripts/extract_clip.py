r"""ローカルの音声ファイルから、ゴール前後の短いクリップを切り出す。

音声データそのものの取得(ダウンロード・録音等)はこのスクリプトの対象外 --
README の「音声データの用意について」を参照(YouTube 等が原音源の場合は
ループバック録音、`scripts/loopback_record.py` を使う方法を推奨している)。

このプロジェクトの実際のデータセット(17クリップ)は、ゴールごとに
`loopback_record.py` を個別実行して直接録音したもので、このスクリプトは
経由していない。手元に既に長い音声ファイルがある場合(テレビ録画・CC
ライセンス音源のダウンロード等)に、そこから短いクリップを切り出すための
代替手段として用意している。

## 使い方

    python scripts/extract_clip.py <src.wav> <out.wav> <center_seconds>

`<center_seconds>` はゴール等、捉えたい瞬間のおおよそのタイムスタンプ
(秒数)。解析の基準点(`onset_marked_time_s`)は後で
`scripts/mark_onset_moment.py` で別途正確に指定するため、ここでの
タイムスタンプの精度は重要ではない -- クリップのどこかに目的の瞬間が
収まっていればよい。

Depends on the external `ffmpeg` executable being on PATH.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def extract_clip(
    src_wav: str | Path,
    out_wav: str | Path,
    center_s: float,
    lead_s: float = 3.0,
    duration_s: float = 9.0,
) -> Path:
    """Cut a `duration_s`-long clip from `src_wav`, starting `lead_s` seconds
    before `center_s` (e.g. before a goal timestamp).
    """
    src_wav = Path(src_wav)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    start = max(0.0, center_s - lead_s)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_wav),
            "-ss", str(start),
            "-t", str(duration_s),
            str(out_wav),
        ],
        check=True,
    )
    return out_wav


def check_tools_available() -> None:
    """Raise RuntimeError if ffmpeg is not on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("required tool not found on PATH: ffmpeg")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="切り出し元の音声ファイル")
    parser.add_argument("out", help="出力先のWAVファイル")
    parser.add_argument("center", type=float, help="捉えたい瞬間のタイムスタンプ(秒数、おおよそでよい)")
    parser.add_argument(
        "--lead", type=float, default=3.0,
        help="centerの何秒前からクリップを開始するか(既定3.0)",
    )
    parser.add_argument(
        "--duration", type=float, default=9.0,
        help="クリップの長さ(秒、既定9.0)",
    )
    args = parser.parse_args(argv)

    try:
        check_tools_available()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    out = extract_clip(args.src, args.out, center_s=args.center, lead_s=args.lead, duration_s=args.duration)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
