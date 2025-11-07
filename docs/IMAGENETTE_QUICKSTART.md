# Imagenette ViT Training - Quick Start

このプロジェクトでは、Hugging Face DatasetsのImagenetteデータセットを使用してViTモデルを学習できます。

## セットアップ

### 1. 必要なパッケージをインストール

```bash
# NLPグループにdatasetsが含まれています
uv sync --extra nlp
```

または、個別にインストール:

```bash
uv add datasets pillow
```

### 2. データセットとモデルのテスト

```bash
# データモジュールのテスト (データのダウンロード含む)
uv run python scripts/test_imagenette.py
```

### 3. 開発用実験を実行

```bash
# 小型モデルで素早く動作確認
uv run python scripts/train.py experiment=vit_imagenette_dev
```

### 4. 完全な学習を実行

```bash
# ベースラインモデルで完全な学習
uv run python scripts/train.py experiment=vit_imagenette_baseline
```

## 設定の詳細

詳しい設定やカスタマイズ方法については、以下を参照してください:

- [Imagenette ViT Training Guide](docs/imagenette-vit-guide.md) - 完全なガイド

## ファイル構成

作成されたファイル:

```tree
modern_vit/data/
├── __init__.py
└── imagenette_datamodule.py     # Imagenetteデータモジュール

configs/
├── data/
│   └── imagenette.yaml
├── data_variant/
│   └── imagenette_standard.yaml # データバリアント
├── model/
│   └── vit_imagenette.yaml      # モデル設定
├── model_variant/
│   ├── vit_imagenette_base.yaml # ベースモデル
│   └── vit_imagenette_small.yaml# 小型モデル
└── experiment/
    ├── vit_imagenette_baseline.yaml # ベースライン実験
    └── vit_imagenette_dev.yaml      # 開発用実験

scripts/
└── test_imagenette.py           # テストスクリプト

docs/
└── imagenette-vit-guide.md      # 詳細ガイド
```

## トラブルシューティング

### パッケージがインストールされていない場合

```bash
uv sync --extra nlp
```

### OOMエラーが発生した場合

```bash
# バッチサイズを減らす
uv run python scripts/train.py experiment=vit_imagenette_dev data.batch_size=8
```

### データセットのダウンロードが遅い場合

初回実行時のみデータセットのダウンロードが必要です。`data/`ディレクトリにキャッシュされます。
