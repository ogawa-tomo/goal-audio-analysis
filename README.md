# goal-audio-analysis

ゴール時の歓声など、群衆の音声データを解析するためのツール。

- **`features.py`**: 1クリップの音声から音響特徴量(音量ピーク・立ち上がり時間・スペクトル重心・
  ロールオフ・帯域幅・F0・フォルマントF1/F2)を抽出する
- **`compare.py`**: 特徴量(dictのリスト)を2群受け取り、Welchのt検定で比較する。音声処理には
  依存しないので、音声以外の2群比較にも流用できる
- **`extract.py`**: `ffmpeg`を使ってローカルの音声ファイルからクリップを切り出す。**音声データ
  そのものの取得(ダウンロード・録音等)はこのツールの対象外**(下記「音声データの用意について」参照)
- **`plotting.py`**: 比較結果の棒グラフ・母音(F1-F2)図を生成する
- **`cli.py`**: 上記をコマンドラインから呼び出せるようにするエントリポイント

## セットアップ

```bash
pip install -e .
```

`ffmpeg`実行ファイルもPATH上に必要(クリップ切り出しに使用)。

## 音声データの用意について

このツールは「手元にある音声ファイルからクリップを切り出して解析する」ところから始まる。
**音声ファイルそのものをどう用意するかはツールの対象外**であり、ソースごとに以下の点に注意する。

- **YouTube等の動画サイトが原音源の場合**: 多くのサービス(YouTube含む)は利用規約で
  公式のダウンロード機能以外でのダウンロードを禁止している。**通常の再生(ストリーミング)を
  行いながら、OSのループバック録音機能や画面録画ツールでシステムの音声出力を録音する方法**
  を使うこと。この方法であれば「許可されていない方法でのダウンロード」という規約条項には
  触れない(通常の視聴と同じリクエストしかサーバーに送っていないため)。ただし著作権法上の
  整理は別途必要になるので、下記「著作権・利用規約について」を参照
  - 録音時はOSの音声エフェクト(ラウドネス補正・空間オーディオ等)やAGC(自動ゲイン調整)・
    ノイズ抑制を無効化し、システム音量を固定しておくこと。これらが有効だとスペクトル特徴量
    (重心・ロールオフ等)やRMSエンベロープ(立ち上がり時間・ピーク検出)が歪む可能性がある
  - 録音・保存部分だけを自動化する補助スクリプトを [`scripts/loopback_record.py`](scripts/loopback_record.py)
    に用意している(**WSLではなくWindows側のPythonで実行すること**。WASAPIループバックは
    OSのオーディオミキサーに直接アクセスするため、WSL内からは扱えない)。ブラウザでの
    再生開始タイミングだけは手動で合わせる想定(完全自動化はしていない)。使い方は
    スクリプト冒頭のdocstringを参照。事前に `pip install -r scripts/requirements.txt`
    (Windows側のPython環境に)が必要
- **WOWOW・U-NEXT等の有料配信サービスが原音源の場合**: 利用規約でストリーミングの録画・
  録音自体を明確に禁止していることが多く、YouTubeより制約が厳しい。BS放送等、テレビ放送を
  対応レコーダーで録画したもの(コピーワンス等の正規のコピー制御下にある録画)であれば、
  そこからのループバック録音は比較的問題が少ないと考えられるが、配信サービス(VOD)からの
  録画・録音は規約違反となる可能性が高いため避けること
- **Creative Commons等でライセンスが明示された音源(Freesoundなど)の場合**: ライセンス条件
  (帰属表示の要否、商用利用可否など)を確認した上でダウンロードして問題ない。実際に使用した
  音源とライセンスの記録例は `data/raw/SOURCES.md` を参照(音源ファイル自体はGit管理外)

## 使い方

```bash
# 1. (音声データを用意する。上記「音声データの用意について」を参照。ここではdata/raw/source.wav
#    に用意できているものとする)

# 2. ゴールタイムスタンプ(秒)を中心にクリップを切り出す
goal-audio extract-clip data/raw/source.wav data/clips/goal1.wav 74

# 3. 特徴量抽出(複数クリップまとめて)
goal-audio analyze data/clips/premier_*.wav -o results/premier_features.json
goal-audio analyze data/clips/laliga_*.wav -o results/laliga_features.json

# 4. 2群比較(表を標準出力 + グラフをresults/以下に保存)
goal-audio compare results/premier_features.json results/laliga_features.json \
  --label-a Premier --label-b LaLiga -o results/
```

Pythonから直接呼ぶ場合:

```python
from goal_audio_analysis import features, compare, plotting

f = features.analyze_clip("data/clips/goal1.wav")

group_a = [features.analyze_clip(p).to_dict() for p in premier_clips]
group_b = [features.analyze_clip(p).to_dict() for p in laliga_clips]
comparisons = compare.compare_groups(group_a, group_b, label_a="Premier", label_b="LaLiga")
print(compare.format_table(comparisons))
```

## ディレクトリ構成

```
.
├── src/goal_audio_analysis/   # ライブラリ本体
├── data/
│   ├── raw/                    # 用意した元音声(Git管理外。SOURCES.mdのみ例外)
│   └── clips/                  # 切り出したクリップ(Git管理外)
├── results/                    # 解析結果(json/png、Git管理対象)
└── reports/                    # 調査記録・レポート
```

`data/`以下の音声ファイルは`.gitignore`で除外している。フォルダ構造だけ`.gitkeep`で保持している。

## 既知の注意点

- ピーク検出はクリップ先頭4秒以内に制限している(既定値)。それより後にスタジアムの演出音
  (ジングル・サイレン等)が入ると誤検出することがあったため
- フォルマント(F1/F2)解析はPraat(Burg法)を使用しているが、本来1人の声道を前提とした手法。
  群衆音への適用は「集団としてのスペクトル包絡のピーク」の近似であり、厳密な音声学的
  フォルマントではないことに留意
- F0(pYIN)は群衆のブロードバンドな歓声には本質的に不安定。有声フレーム率(`voiced_fraction`)
  が低いクリップのF0値は参考程度に扱うこと

詳しい調査記録は [`reports/`](reports/) を参照。

## 著作権・利用規約について

- 以前のバージョンでは`extract.py`に`yt-dlp`を使ったYouTube直接ダウンロード機能を含んでいたが、
  YouTubeの利用規約(公式のダウンロード機能以外でのダウンロード禁止)に抵触する行為を助長する
  コードであったため**削除した**。音声データの用意方法は上記「音声データの用意について」を参照
- **`data/`以下の音声ファイルはGit管理に含めない**(`.gitignore`)。取得した音声・切り出した
  クリップを外部に再配布・公開しないことを設計上の前提にしている
- 特徴量抽出(スペクトル重心・F0・フォルマント等の数値算出)自体は、日本の著作権法30条の4
  (著作物を「享受」する目的でない利用=情報解析)により、権利者の許諾なく行える可能性が高い。
  ただし、これは法的助言ではなく一般的な整理であり、個別のケースについて確信が持てない場合は
  専門家に確認すること
- スペクトログラム画像など、特定の音声から直接生成した視覚的な成果物を公開する場合は、
  「引用」(著作権法32条)の要件(公表済みの著作物であること、自分の著作物が「主」・引用部分が
  「従」の関係であること、必要最小限の範囲であること、出所明示)を満たすよう注意する
- 上記の理由から、本リポジトリは**Privateで運用**している。Publicにする場合は、音声非公開の
  徹底と、視覚的な成果物(スペクトログラム等)の要否を改めて見直すこと
