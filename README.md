# Modern ViT

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-latest-green.svg)](https://github.com/astral-sh/uv)
[![Lightning](https://img.shields.io/badge/Lightning-2.5+-792ee5.svg)](https://lightning.ai/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd.svg)](https://hydra.cc/)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Vision Transformer (ViT)** と **Mixture of Experts (MoE)** の実装を含む、モダンな深層学習プロジェクトです。Lightning + Hydra + Wandb を活用し、実験管理、再現性、スケーラビリティを重視した設計で、Imagenette データセットでの ViT 学習をすぐに開始できます。

## クイックスタート

### セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/Kotomiya07/modern-vit.git
cd modern-vit

# セットアップスクリプトを実行
bash scripts/setup.sh
```

セットアップスクリプトは以下を実行します：

- uv を使用して Python 環境を初期化
- すべての依存関係をインストール
- pre-commit フックを設定
- 必要なディレクトリを作成

### 手動セットアップ（代替方法）

手動セットアップを希望する場合：

```bash
# uvをインストール（まだインストールしていない場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# Pythonバージョンを設定
uv python pin 3.12

# 依存関係をインストール
uv sync --all-extras

# pre-commitフックをインストール
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# データディレクトリを作成
mkdir -p data logs

# テストを実行
uv run pytest
```

## 主な特徴

### Vision Transformer 実装

- **ViT (Vision Transformer)** - パッチベースの Transformer アーキテクチャ
- **MoE (Mixture of Experts)** - スパースなエキスパート選択機構
- **Expert Choice MoE** - エキスパート主導の選択メカニズム
- **マルチスケール対応** - Small, Base サイズのモデル設定

### サポートデータセット

- **MNIST** - 手書き数字認識（ベースライン実装）
- **Imagenette** - ImageNet のサブセット（10 クラス）
- **自動ダウンロード** - Hugging Face Datasets による管理
- **柔軟なデータ設定** - バッチサイズ、augmentation 設定の切り替え

### 実験管理

- **Experiment 中心の設計** - すべての設定を 1 ファイルで管理
- **Hydra 設定システム** - 階層的で柔軟な設定管理
- **再現性の保証** - シード固定、決定論的実行、設定の自動保存
- **rootutils 統合** - `.project-root`マーカーによる自動パス解決

### Lightning 統合

- **PyTorch Lightning 2.5+** - モダンな深層学習フレームワーク
- **自動最適化** - 分散学習、混合精度、勾配累積
- **豊富なコールバック** - EarlyStopping, ModelCheckpoint, RichProgressBar
- **柔軟なロガー** - Wandb, TensorBoard 対応

### 開発ツール

- **[uv](https://github.com/astral-sh/uv)** - 高速な Python パッケージマネージャー
- **[Ruff](https://github.com/astral-sh/ruff)** - 超高速リンター・フォーマッター
- **[mypy](https://mypy-lang.org/)** - 厳格な型チェック
- **[pytest](https://pytest.org/)** - テストフレームワーク
- **rootutils** - ディレクトリ非依存の実行環境

## プロジェクト構造

```
modern-vit/
├── .project-root                     # プロジェクトルートマーカー
├── configs/                          # Hydra設定ファイル
│   ├── train.yaml                    # メイントレーニング設定
│   ├── eval.yaml                     # 評価設定
│   ├── experiment/                   # 🔬 実験設定（推奨）
│   │   ├── mnist_*.yaml              # MNISTベースライン実験
│   │   ├── vit_imagenette_dev.yaml   # 開発・デバッグ用（小型）
│   │   ├── vit_imagenette_baseline.yaml  # ベースライン
│   │   └── vit_imagenette_*_moe*.yaml    # MoE系実験
│   ├── model_variant/                # モデル設定バリアント
│   │   ├── mnist_*.yaml              # MNISTモデル設定
│   │   ├── vit_imagenette_small*.yaml    # ViT Small
│   │   ├── vit_imagenette_base*.yaml     # ViT Base
│   │   └── *_moe*.yaml               # MoE系モデル設定
│   ├── data_variant/                 # データ設定バリアント
│   │   ├── mnist_*.yaml
│   │   └── imagenette_standard.yaml
│   ├── callbacks/                    # コールバック設定
│   │   ├── default.yaml
│   │   ├── early_stopping.yaml
│   │   └── model_checkpoint.yaml
│   ├── trainer/                      # Trainer設定
│   │   ├── default.yaml (CPU)
│   │   ├── gpu.yaml
│   │   ├── ddp.yaml (分散学習)
│   │   └── mps.yaml (Apple Silicon)
│   └── logger/                       # ロガー設定
│       └── wandb.yaml
├── modern_vit/                       # メインパッケージ
│   ├── data/                         # DataModules
│   │   └── imagenette_datamodule.py  # Imagenetteデータセット
│   ├── models/                       # LightningModules
│   │   ├── config.py                 # モデル設定クラス
│   │   ├── mnist_module.py           # MNISTモジュール
│   │   └── vit_module.py             # ViT/MoEモジュール
│   └── utils/                        # ユーティリティ
│       ├── logging_utils.py          # ログユーティリティ
│       └── parameter_utils.py        # パラメータ計算
├── scripts/                          # トレーニング・評価スクリプト
│   ├── train.py                      # 学習スクリプト
│   ├── eval.py                       # 評価スクリプト
│   ├── test_imagenette.py            # Imagenetteテスト
│   └── check_moe.py                  # MoE動作確認
├── tests/                            # テスト
│   └── unit/                         # ユニットテスト
├── docs/                             # ドキュメント
│   ├── IMAGENETTE_QUICKSTART.md      # Imagenetteクイックスタート
│   ├── imagenette-vit-guide.md       # ViTガイド
│   ├── MOE_CONFIGS.md                # MoE設定ガイド
│   └── MOE_VERIFICATION.md           # MoE検証ガイド
└── data/                             # データディレクトリ（自動作成）
```

## 実験の実行

### Imagenette (ViT) の学習

```bash
# 開発モード（少ないデータで高速テスト）
uv run python scripts/train.py experiment=vit_imagenette_dev

# ベースラインモデル（ViT Small）
uv run python scripts/train.py experiment=vit_imagenette_baseline

# MoE統合モデル
uv run python scripts/train.py experiment=vit_imagenette_baseline_moe

# MoE全層適用モデル
uv run python scripts/train.py experiment=vit_imagenette_baseline_moe_all
```

### MNIST の学習

```bash
# MNISTベースライン
uv run python scripts/train.py experiment=mnist_baseline

# 開発モード
uv run python scripts/train.py experiment=mnist_dev

# 大規模モデル
uv run python scripts/train.py experiment=mnist_large
```

### パラメータのカスタマイズ

```bash
# パラメータをオーバーライド
uv run python scripts/train.py experiment=vit_imagenette_baseline trainer.max_epochs=20

# 複数のパラメータを変更
uv run python scripts/train.py experiment=vit_imagenette_baseline \
  trainer.max_epochs=50 \
  model.lr=0.0001 \
  data.batch_size=64
```

### GPU/MPS 使用

```bash
# GPU使用
uv run python scripts/train.py experiment=vit_imagenette_baseline trainer=gpu

# Apple Silicon (MPS)使用
uv run python scripts/train.py experiment=vit_imagenette_baseline trainer=mps

# 分散学習（複数GPU）
uv run python scripts/train.py experiment=vit_imagenette_baseline trainer=ddp trainer.devices=4
```

### Wandb ロギング

```bash
# Wandbを有効化
uv run python scripts/train.py experiment=vit_imagenette_baseline logger=wandb

# Wandbのプロジェクト名を指定
uv run python scripts/train.py experiment=vit_imagenette_baseline logger=wandb \
  logger.wandb.project=modern-vit \
  logger.wandb.name=vit-baseline-001
```

### モデル評価

```bash
# 保存されたチェックポイントで評価
uv run python scripts/eval.py \
  experiment=vit_imagenette_baseline \
  ckpt_path=/path/to/checkpoint.ckpt
```

### データセットとモデルのテスト

```bash
# Imagenetteデータセットの動作確認
uv run python scripts/test_imagenette.py

# MoE動作の確認
uv run python scripts/check_moe.py
```

## 新しい実験の作成

### ViT モデルのカスタマイズ例

#### ステップ 1: Model Variant を定義

`configs/model_variant/vit_imagenette_custom.yaml`:

```yaml
_target_: modern_vit.models.vit_module.ViTLightningModule

# モデルアーキテクチャ
image_size: 224
patch_size: 16
num_classes: 10
dim: 384 # 埋め込み次元
depth: 8 # Transformerブロック数
heads: 6 # アテンションヘッド数
mlp_dim: 1536 # FFN中間次元
dropout: 0.1
emb_dropout: 0.1

# 学習設定
lr: 0.0003
weight_decay: 0.05
warmup_epochs: 5

# オプション: MoE設定
use_moe: false
num_experts: 0
expert_capacity_factor: 1.0
```

#### ステップ 2: Experiment 設定を作成

`configs/experiment/vit_imagenette_custom.yaml`:

```yaml
# @package _global_

defaults:
  - /model_variant@model: vit_imagenette_custom
  - /data_variant@data: imagenette_standard
  - override /data: imagenette
  - override /callbacks: default
  - override /trainer: gpu
  - override /logger: wandb

tags: ["vit", "custom", "imagenette"]

seed: 42
train: true
test: true

trainer:
  max_epochs: 100
  precision: "16-mixed"
  gradient_clip_val: 1.0

logger:
  wandb:
    project: "modern-vit"
    name: "vit-custom-experiment"
```

#### ステップ 3: 実行

```bash
uv run python scripts/train.py experiment=vit_imagenette_custom
```

## 開発

### テストの実行

```bash
# すべてのテストを実行
uv run pytest

# カバレッジ付きで実行
uv run pytest --cov=modern_vit --cov-report=html

# 単体テストを実行
uv run pytest tests/unit/ -v

# 特定のテストを実行
uv run pytest tests/unit/test_vit_module.py -v

# Makefileを使用
make test
```

### コード品質

```bash
# コードをフォーマット
uv run ruff format .

# コードをリント
uv run ruff check .

# 型チェック
uv run mypy modern_vit

# pre-commitで完全チェック
uv run pre-commit run --all-files

# Makefileを使用
make format    # フォーマット
make lint      # リント
make typecheck # 型チェック
```

### 依存関係の管理

```bash
# ランタイム依存関係を追加
uv add torch torchvision

# 開発依存関係を追加
uv add --dev pytest-mock

# 特定のextraをインストール
uv sync --extra vision     # Vision関連（timm等）
uv sync --extra nlp        # NLP関連（transformers, datasets等）
uv sync --extra wandb-logging  # Wandbロギング
uv sync --extra dev        # 開発ツール

# すべての依存関係を同期
uv sync --all-extras

# 依存関係を更新
uv lock --upgrade
```

## モデルとデータセット

### サポートモデル

- **ViT (Vision Transformer)**: パッチベースの画像分類モデル
  - Small: dim=384, depth=8, heads=6
  - Base: dim=768, depth=12, heads=12
- **MoE (Mixture of Experts)**: スパース専門家モデル

  - 標準 MoE: トークンが専門家を選択
  - Expert Choice MoE: 専門家がトークンを選択
  - 最終層のみ / 全層適用の設定が可能

- **MNIST 分類器**: シンプルな MLP ベースライン

### データセット

- **Imagenette**: ImageNet の 10 クラスサブセット

  - 224x224 にリサイズ
  - 自動ダウンロード（Hugging Face Datasets）
  - Train: 9,469 枚 / Val: 3,925 枚

- **MNIST**: 手書き数字認識
  - 28x28 グレースケール
  - Train: 60,000 枚 / Test: 10,000 枚

## ドキュメント

詳細なガイドは`docs/`ディレクトリを参照してください：

- [Imagenette Quickstart](docs/IMAGENETTE_QUICKSTART.md) - Imagenette 学習のクイックスタート
- [Imagenette ViT Guide](docs/imagenette-vit-guide.md) - ViT 実装の詳細ガイド
- [MoE Configs](docs/MOE_CONFIGS.md) - MoE 設定の詳細
- [MoE Verification](docs/MOE_VERIFICATION.md) - MoE 動作検証方法

## プロジェクト設計

### rootutils 統合

このプロジェクトは`rootutils`を使用して、どのディレクトリからでもスクリプトを実行できます。

```python
# scripts/train.py, scripts/eval.py等
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
```

これにより：

- プロジェクトルートが自動検出される
- Python パスが自動設定される
- `import modern_vit`がどこからでも動作する

### Hydra 設定の階層構造

設定は階層的に構成されており、柔軟な組み合わせが可能です：

```
experiment/
  └─ 完全な実験設定（推奨エントリーポイント）
      ├─ model_variant/ （モデルアーキテクチャ）
      ├─ data_variant/  （データセット設定）
      ├─ trainer/       （学習設定）
      ├─ callbacks/     （コールバック）
      └─ logger/        （ロギング）
```

### 実験中心設計

`configs/experiment/`の各ファイルが完全な実験を定義します。これにより：

- 実験の再現性が保証される
- 設定の共有が容易になる
- バージョン管理がしやすくなる

## 参考資料

### フレームワーク・ライブラリ

- **[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)** - 深層学習フレームワーク
- **[Hydra](https://hydra.cc/)** - 設定管理フレームワーク
- **[Weights & Biases](https://docs.wandb.ai/)** - 実験管理・可視化
- **[rootutils](https://github.com/ashleve/rootutils)** - プロジェクトルート管理

### Vision Transformer 関連

- **[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)** - ViT 原論文
- **[Mixture of Experts](https://arxiv.org/abs/1701.06538)** - MoE 原論文
- **[timm](https://github.com/huggingface/pytorch-image-models)** - PyTorch Image Models

### 開発ツール

- **[uv](https://docs.astral.sh/uv/)** - Python パッケージ管理
- **[Ruff](https://docs.astral.sh/ruff/)** - リント・フォーマッター
- **[pytest](https://docs.pytest.org/)** - テストフレームワーク

## トラブルシューティング

### データセットのダウンロードエラー

```bash
# Hugging Face Datasetsのキャッシュをクリア
rm -rf ~/.cache/huggingface/datasets/frgfm___imagenette
```

### CUDA Out of Memory

バッチサイズを減らすか、勾配累積を使用：

```bash
uv run python scripts/train.py experiment=vit_imagenette_baseline \
  data.batch_size=32 \
  trainer.accumulate_grad_batches=2
```

### MoE の動作確認

```bash
# MoE層の動作を確認
uv run python scripts/check_moe.py
```

## コントリビューション

プルリクエストを歓迎します。大きな変更の場合は、まず issue を開いて変更内容を議論してください。

## ライセンス

このプロジェクトは Apache-2.0 ライセンスの下でライセンスされています。詳細は [LICENSE](LICENSE) ファイルを参照してください。
