# DL-Scaffold

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-latest-green.svg)](https://github.com/astral-sh/uv)
[![Lightning](https://img.shields.io/badge/Lightning-2.5+-792ee5.svg)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd.svg)](https://hydra.cc/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Lightning + Hydra + Wandb**を活用した、プロダクション対応の深層学習プロジェクトテンプレートです。実験管理、再現性、スケーラビリティを重視した設計で、すぐに研究開発を開始できます。

## クイックスタート

### このテンプレートを使用する

1. GitHubで「Use this template」ボタンをクリックして新しいリポジトリを作成
2. 新しいリポジトリをクローン
3. セットアップスクリプトを実行

```bash
# 新しいリポジトリをクローン
git clone https://github.com/yourusername/project-name.git
cd project-name

# セットアップ
make setup
```

セットアップスクリプトは以下を実行します：
- すべての `project_name` を実際のプロジェクト名に更新（途中でプロジェクト名を入力するように求められます）
- uvを使用してPython環境を初期化
- Rovo Dev CLIをインストール
- GitHub CLI（`gh`）をインストール（途中でログインを求められます）
- すべての依存関係をインストール
- pre-commitフックを設定
- 初期テストを実行

### 手動セットアップ（代替方法）

手動セットアップを希望する場合：

```bash
# プロジェクト名を更新
python scripts/update_project_name.py your_project_name

# uvをインストール（まだインストールしていない場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Pythonバージョンを設定
uv python pin 3.12

# 依存関係をインストール
uv sync --all-extras

# pre-commitフックをインストール
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# テストを実行
uv run pytest
```

## 主な特徴

### 実験管理
- **Experiment中心の設計** - すべての設定を1ファイルで管理
- **Hydra設定システム** - 階層的で柔軟な設定管理
- **再現性の保証** - シード固定、決定論的実行、設定の自動保存

### Lightning統合
- **PyTorch Lightning 2.5+** - モダンな深層学習フレームワーク
- **自動最適化** - 分散学習、混合精度、勾配累積
- **豊富なコールバック** - EarlyStopping, ModelCheckpoint, RichProgressBar
- **柔軟なロガー** - Wandb, TensorBoard対応

### 設定管理
- **Model Variants** - モデル設定のバリエーション管理
- **Data Variants** - データセット設定のバリエーション管理
- **Experiment Configs** - 完全な実験設定の定義
- **簡単なオーバーライド** - コマンドラインから自由に調整

### 開発ツール
- **[uv](https://github.com/astral-sh/uv)** - 高速なPythonパッケージマネージャー
- **[Ruff](https://github.com/astral-sh/ruff)** - 超高速リンター・フォーマッター
- **[mypy](https://mypy-lang.org/)** - 厳格な型チェック
- **[pytest](https://pytest.org/)** - テストフレームワーク
-  GitHub CLIによるワンコマンドPR・Issue作成
-  キャッシュ最適化された実行環境

### 包括的ドキュメント
-  **動的agents.md** - プロジェクトと共に進化する知識ベース
-  **専門ガイド** - ML/バックエンドプロジェクト対応
-  **協働戦略ガイド** - 人間とRovo Dev CLIの効果的な連携方法
-  **メモリ更新プロトコル** - ドキュメント品質管理フレームワーク

## プロジェクト構造

```
DL-Scaffold/
├── .project-root                # プロジェクトルートマーカー
├── configs/                      # Hydra設定ファイル
│   ├── train.yaml               # メイントレーニング設定
│   ├── eval.yaml                # 評価設定
│   ├── experiment/              # 🔬 実験設定（推奨）
│   │   ├── mnist_baseline.yaml  # MNISTベースライン
│   │   ├── mnist_large.yaml     # 大規模モデル実験
│   │   └── mnist_dev.yaml       # 開発・デバッグ用
│   ├── model_variant/           # モデル設定バリアント
│   │   ├── mnist_simple.yaml    # 128 hidden units
│   │   └── mnist_large.yaml     # 256 hidden units
│   ├── data_variant/            # データ設定バリアント
│   │   ├── mnist_standard.yaml
│   │   └── mnist_large_batch.yaml
│   ├── callbacks/               # コールバック設定
│   │   ├── default.yaml
│   │   ├── early_stopping.yaml
│   │   └── model_checkpoint.yaml
│   ├── trainer/                 # Trainer設定
│   │   ├── default.yaml (CPU)
│   │   ├── gpu.yaml
│   │   ├── ddp.yaml (分散学習)
│   │   └── mps.yaml (Apple Silicon)
│   └── logger/                  # ロガー設定
│       └── wandb.yaml
├── project_name/                # メインパッケージ
│   ├── data/                    # DataModules
│   │   └── mnist_datamodule.py
│   ├── models/                  # LightningModules
│   │   └── mnist_module.py
│   └── utils/                   # ユーティリティ
│       └── logging_utils.py
├── scripts/                     # トレーニング・評価スクリプト
│   ├── train.py
│   └── eval.py
├── tests/                       # テスト
├── docs/                        # ドキュメント
└── data/                        # データディレクトリ（自動作成）
```

## 実験の実行

### 基本的な使い方

```bash
# 事前定義された実験を実行（推奨）
uv run python scripts/train.py experiment=mnist_baseline

# 開発モード（少ないデータで高速テスト）
uv run python scripts/train.py experiment=mnist_dev

# パラメータをオーバーライド
uv run python scripts/train.py experiment=mnist_baseline trainer.max_epochs=20

# 複数のパラメータを変更
uv run python scripts/train.py experiment=mnist_baseline \
  trainer.max_epochs=50 \
  model.lr=0.0001 \
  data.batch_size=256
```

### GPU/MPS使用

```bash
# GPU使用
uv run python scripts/train.py experiment=mnist_baseline trainer=gpu

# Apple Silicon (MPS)使用
uv run python scripts/train.py experiment=mnist_baseline trainer=mps

# 分散学習（複数GPU）
uv run python scripts/train.py experiment=mnist_baseline trainer=ddp trainer.devices=4
```

### Wandbロギング

```bash
# Wandbを有効化
uv run python scripts/train.py experiment=mnist_baseline logger=wandb

# Wandbのプロジェクト名を指定
uv run python scripts/train.py experiment=mnist_baseline logger=wandb \
  logger.wandb.project=my-project \
  logger.wandb.name=experiment-001
```

### モデル評価

```bash
# 保存されたチェックポイントで評価
uv run python scripts/eval.py \
  experiment=mnist_baseline \
  ckpt_path=/path/to/checkpoint.ckpt
```

## 新しい実験の作成

### ステップ1: Model Variantを定義

`configs/model_variant/my_model.yaml`:
```yaml
_target_: project_name.models.mnist_module.MNISTLightningModule

input_size: 28
hidden_dim: 512  # カスタマイズ
num_classes: 10
lr: 0.0005
weight_decay: 1e-4
```

### ステップ2: Experiment設定を作成

`configs/experiment/my_experiment.yaml`:
```yaml
# @package _global_

defaults:
  - /model_variant@model: my_model
  - /data_variant@data: mnist_standard
  - override /data: mnist
  - override /callbacks: default
  - override /trainer: gpu
  - override /logger: wandb

tags: ["custom", "experiment"]

seed: 42
train: true
test: true

trainer:
  max_epochs: 100
  precision: "16-mixed"  # 混合精度
```

### ステップ3: 実行

```bash
uv run python scripts/train.py experiment=my_experiment
```

## 開発

### テストの実行

```bash
# すべてのテストを実行（単体・プロパティ・統合）
make test

# カバレッジ付きで実行
make test-cov

# テスト種別で実行
uv run pytest tests/unit/ -v           # 単体テスト
uv run pytest tests/property/ -v       # プロパティベーステスト
uv run pytest tests/integration/ -v    # 統合テスト

# 特定のテストを実行
uv run pytest tests/unit/test_helpers.py -v
```

### コード品質

```bash
# コードをフォーマット
make format

# コードをリント
make lint

# 型チェック
make typecheck

# すべてのチェックを順番に実行
make check

# pre-commitで完全チェック
make check-all
```

### パフォーマンス測定・プロファイリング

```bash
# ローカルベンチマーク実行
make benchmark

# プロファイリング実行（cProf使用）
make profile
```

### GitHub統合

```bash
# プルリクエスト作成
make pr TITLE="新機能追加" BODY="説明" LABEL="enhancement"
make pr TITLE="バグ修正" BODY="修正内容" LABEL="bug"

# イシュー作成
make issue TITLE="機能要求" BODY="詳細" LABEL="enhancement"
make issue TITLE="バグ報告" BODY="再現手順" LABEL="bug"

# 直接gh CLIを使用
gh pr create --title "タイトル" --body "本文" --label "ラベル"
gh issue create --title "タイトル" --body "本文" --label "ラベル"
```

### その他のコマンド

```bash
# 利用可能なコマンドを表示
make help

# キャッシュファイルの削除
make clean

# セキュリティスキャン
make security

# 依存関係の脆弱性チェック
make audit
```

### 依存関係の管理

```bash
# ランタイム依存関係を追加
uv add requests

# 開発依存関係を追加
uv add --dev pytest-mock

# ドキュメント関連依存関係を追加
uv sync --extra docs

# すべての依存関係を同期
uv sync --all-extras

# 依存関係を更新
uv lock --upgrade
```

## 新規プロジェクト設定チェックリスト

### 基本プロジェクト設定
- [ ] **プロジェクト名更新**: `make setup`実行またはスクリプトで一括変更
- [ ] **作者情報更新**: `pyproject.toml`の`authors`セクション
- [ ] **ライセンス選択**: LICENSEファイルを適切なライセンスに更新
- [ ] **README.md更新**: プロジェクト固有の説明・機能・使用方法
- [ ] **agents.md カスタマイズ**: プロジェクト概要をテンプレートから更新

### 開発環境・品質設定
- [ ] **依存関係調整**: プロジェクトに必要な追加パッケージの導入
- [ ] **型チェック厳格さ**: 必要に応じて段階的に`mypy`設定を調整
- [ ] **リントルール**: プロジェクトに合わせた`ruff`設定のカスタマイズ
- [ ] **テストカバレッジ**: `pytest`カバレッジ要件の調整
- [ ] **プロファイリング**: パフォーマンス要件に応じたベンチマーク設定

### GitHubリポジトリ・セキュリティ設定
- [ ] **ブランチ保護**: `main`ブランチの保護ルール有効化
- [ ] **PR必須レビュー**: Pull Request作成時のレビュー要求設定
- [ ] **ステータスチェック**: CI・型チェック・テストの必須化
- [ ] **Dependabot**: 自動依存関係更新の有効化
- [ ] **Issues/Projects**: 必要に応じてプロジェクト管理機能の有効化
- [ ] **Secrets管理**: 必要なAPIキーや認証情報の安全な設定

### ドキュメント・協働設定
- [ ] **agents.md詳細化**: プロジェクト固有の開発ルール・制約の追加
- [ ] **専門ガイド選択**: ML/バックエンドなど該当するガイドのインポート
- [ ] **チーム規約**: `docs/team-rules.md`などチーム固有ルールの追加
- [ ] **協働メトリクス**: 効率指標の初期値設定・測定開始

## カスタマイズ

### 型チェックの厳格さ調整

mypyのstrictモードが最初から厳しすぎる場合：

```toml
# pyproject.toml - 基本設定から開始
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true

# 段階的により厳格な設定を有効化
[[tool.mypy.overrides]]
module = ["project_name.core.*"]
strict = true  # まずコアモジュールにstrictモードを適用
```

### リントルールの変更

```toml
# pyproject.toml
[tool.ruff.lint]
# 必要に応じてルールコードを追加・削除
select = ["E", "F", "I"]  # 基本から開始
ignore = ["E501"]  # 行の長さはフォーマッターが処理
```

### テストカバレッジ要件の変更

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = [
    "--cov-fail-under=60",  # 初期要件を低めに設定
]
```

## 外部リソース・参考資料

### 開発ツール公式ドキュメント
- **[uv ドキュメント](https://docs.astral.sh/uv/)** - Pythonパッケージ管理
- **[Ruff ドキュメント](https://docs.astral.sh/ruff/)** - リント・フォーマッター
- **[mypy ドキュメント](https://mypy.readthedocs.io/)** - 型チェッカー
- **[pytest ドキュメント](https://docs.pytest.org/en/stable/)** - テストフレームワーク
- **[Hypothesis ドキュメント](https://hypothesis.readthedocs.io/)** - プロパティベーステスト

### Python・型ヒント
- **[PEP 695 - Type Parameter Syntax](https://peps.python.org/pep-0695/)** - 新型構文仕様
- **[TypedDict Guide](https://docs.python.org/3/library/typing.html#typing.TypedDict)** - 型安全な辞書
- **[Python 3.12 リリースノート](https://docs.python.org/3/whatsnew/3.12.html)** - 新機能一覧

---

## ライセンス

このプロジェクトはApache-2.0ライセンスの下でライセンスされています。
