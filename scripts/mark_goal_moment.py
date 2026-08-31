r"""ゴールの瞬間を人間が指定し、解析の基準点として登録する。

ここで指定した時刻は、そのまま(補正なしで)`goal-audio analyze`の解析基準点として
使われる -- アルゴリズムによる自動検出や候補との照合は一切行わない。**マークして
いないクリップは`goal-audio analyze`がエラーで拒否する**(このプロジェクトでは、
音量ベースの自動検出を何通りも試したが、いずれも「音量が一番大きい瞬間」や「音量の
変化が一番急な瞬間」が、実際のゴールの瞬間と一致しない実例が繰り返し見つかったため、
自動検出そのものを廃止した)。

## 使い方

1. 対象のクリップを、0.01秒精度で再生位置を確認できるツール(ブラウザの`<audio>`要素を
   使った簡易ツールなど、リポジトリ外で管理)で再生する
2. **一度聞いて反応した時刻をそのまま使わないこと。** 巻き戻し・微調整(±0.1秒など)を
   繰り返して、確信を持てる時刻に絞り込んでから確定すること。ここで指定した数値は
   そのまま解析に使われるため、ここでの精度がそのまま解析結果の精度になる
3. 確定した時刻を、以下のように渡す

    python scripts/mark_goal_moment.py <clip.wav> <time>

`<time>` は秒数(例: `4.2`)、または `分:秒` 形式(例: `0:04.2`)のどちらでも指定できる。
これで、`<clip>`と同じ場所に`<clip>.mark.json`が保存される。

## 設計の経緯

これで3回目の設計変更になる。

1. (v1) スクリプト自身がスピーカーで再生し、リアルタイムでキー入力を検知する方式
   -- 「`play()`を呼んでから実際に音が聞こえるまでの遅延」という技術的なバグに
   悩まされ、精度も安定しなかったため廃止
2. (v3) 市販のメディアプレイヤーで再生・一時停止し、表示された時刻をそのまま
   読み取って渡す方式に単純化。ただし実際の運用(一度だけ再生して反応した瞬間を
   申告する)では、反応速度のズレが計測値に混入するという弱点が残っていた
3. その弱点への対策として、いったんは「アルゴリズムが検出した複数の候補から
   人間が選ぶ」方式(候補同士は離れているため反応速度に左右されにくい)を試した。
   しかし実際のデータで、正解がそもそも候補に含まれないケース(音量が緩やかに
   盛り上がる持続的な展開で、目立った山がない)が繰り返し見つかり、変化率ベースの
   検出に切り替える等の改良も試したが、どれも別のクリップで悪化するなど汎化しな
   かった。結局、**「この数値がそのまま使われる」と明確に意識した上で、慎重に
   確認して確定してもらう**という運用(今のこのバージョン)に戻すのが、最も単純で
   確実という結論になった

このスクリプト自体はOS依存がなく、WSL側からでも実行できる。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def parse_time(value: str) -> float:
    """Parse a time string as either plain seconds ("4.2") or "M:SS[.ms]" ("0:04.2")."""
    value = value.strip()
    if re.match(r"^\d+:\d+(\.\d+)?$", value):
        minutes_str, seconds_str = value.split(":")
        return int(minutes_str) * 60 + float(seconds_str)
    return float(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="対象のWAVファイルのパス")
    parser.add_argument("time", help="ゴールの瞬間の再生位置。秒数(例: 4.2)または 分:秒(例: 0:04.2)")
    args = parser.parse_args(argv)

    clip_path = Path(args.clip)
    if not clip_path.exists():
        print(f"error: file not found: {clip_path}", file=sys.stderr)
        return 1

    try:
        human_marked_time_s = round(parse_time(args.time), 3)
    except ValueError:
        print(f"error: could not parse time: {args.time!r} (expected seconds or M:SS)", file=sys.stderr)
        return 1

    mark_path = clip_path.with_suffix(".mark.json")
    mark_path.write_text(
        json.dumps({"human_marked_time_s": human_marked_time_s}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved: {mark_path} (human_marked_time_s={human_marked_time_s})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
