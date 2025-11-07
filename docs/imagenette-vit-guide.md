# Imagenette ViT Training Guide

このガイドでは、Hugging Face DatasetsのImagenetteデータセットを使用してViTモデルを学習・評価する方法を説明します。

## データセットについて

- **データセット**: `frgfm/imagenette` (Hugging Face Datasets)
- **クラス数**: 10クラス (ImageNetのサブセット)
- **画像サイズ**: 224x224にクロップ/リサイズ
- **正規化**: ImageNet標準 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## 必要なパッケージ

以下のパッケージが必要です:

```bash
uv add datasets pillow torchvision
```

## 設定ファイル

以下の設定ファイルが作成されています:

### データ設定
- `configs/data/imagenette.yaml` - ベースデータ設定
- `configs/data_variant/imagenette_standard.yaml` - 標準データバリアント

### モデル設定
- `configs/model/vit_imagenette.yaml` - ベースモデル設定
- `configs/model_variant/vit_imagenette_base.yaml` - ベースモデル (dim=384, depth=12)
- `configs/model_variant/vit_imagenette_small.yaml` - 小型モデル (dim=256, depth=6)

### 実験設定
- `configs/experiment/vit_imagenette_baseline.yaml` - ベースライン実験
- `configs/experiment/vit_imagenette_dev.yaml` - 開発用実験 (高速イテレーション)

## 学習の実行

### 1. 開発用実験 (推奨: まず試す)

小型モデルで少ないバッチ数で実行し、動作確認します:

```bash
uv run python scripts/train.py experiment=vit_imagenette_dev
```

特徴:
- 小型モデル (dim=256, depth=6)
- 訓練バッチ制限: 100バッチ
- 検証バッチ制限: 20バッチ
- エポック数: 10
- バッチサイズ: 16

### 2. ベースライン実験

完全な学習を実行:

```bash
uv run python scripts/train.py experiment=vit_imagenette_baseline
```

特徴:
- ベースモデル (dim=384, depth=12)
- 完全なデータセット
- エポック数: 30
- バッチサイズ: 32

### 3. カスタム設定

設定をコマンドラインでオーバーライド:

```bash
# バッチサイズを変更
uv run python scripts/train.py experiment=vit_imagenette_baseline data.batch_size=64

# エポック数を変更
uv run python scripts/train.py experiment=vit_imagenette_baseline trainer.max_epochs=50

# GPUを使用
uv run python scripts/train.py experiment=vit_imagenette_baseline trainer.accelerator=gpu

# 学習率を変更
uv run python scripts/train.py experiment=vit_imagenette_baseline model.config.lr=1e-4
```

## 評価の実行

学習済みモデルでテストセットを評価:

```bash
uv run python scripts/eval.py \
  data=imagenette \
  model=vit_imagenette \
  ckpt_path=logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/last.ckpt
```

## モデルアーキテクチャ

### ViT Base (vit_imagenette_base)
- Embedding次元: 384
- レイヤー数: 12
- ヘッド数: 6
- パッチサイズ: 16x16
- パラメータ数: 約22M

### ViT Small (vit_imagenette_small)
- Embedding次元: 256
- レイヤー数: 6
- ヘッド数: 4
- パッチサイズ: 16x16
- パラメータ数: 約7M

## 期待される結果

Imagenetteは比較的簡単なデータセットなので、以下の精度が期待されます:

- ViT Small: 約80-85% (10エポック程度)
- ViT Base: 約85-90% (30エポック程度)

## トラブルシューティング

### OOM (Out of Memory) エラー

バッチサイズを減らす:

```bash
uv run python scripts/train.py experiment=vit_imagenette_baseline data.batch_size=16
```

または小型モデルを使用:

```bash
uv run python scripts/train.py experiment=vit_imagenette_dev
```

### データセットのダウンロードが遅い

初回実行時、データセットは自動的にダウンロードされます。`data/`ディレクトリにキャッシュされます。

### num_workersエラー

Mac/Windowsの場合、num_workersを0に設定:

```bash
uv run python scripts/train.py experiment=vit_imagenette_baseline data.num_workers=0
```

## 次のステップ

### MoE (Mixture of Experts) を有効化

より高度なモデルを試す場合:

```yaml
# configs/model_variant/vit_imagenette_moe.yaml を作成
config:
  image_size: 224
  patch_size: 16
  channels: 3
  num_classes: 10
  dim: 384
  depth: 12
  heads: 6
  n_kv_heads: 6
  multiple_of: 256
  
  # MoE設定
  n_experts: 8
  n_experts_per_tok: 2
  router_config:
    type: "token_choice"
    k: 2
  moe_layers: [3, 6, 9]  # レイヤー3, 6, 9でMoEを使用
  
  lr: 1e-3
  weight_decay: 0.05
  aux_loss_weight: 0.01
```

### WandBロギングを有効化

```bash
uv run python scripts/train.py experiment=vit_imagenette_baseline logger=wandb
```

## ファイル構造

```
configs/
├── data/
│   └── imagenette.yaml              # データ基本設定
├── data_variant/
│   └── imagenette_standard.yaml     # データバリアント
├── model/
│   └── vit_imagenette.yaml          # モデル基本設定
├── model_variant/
│   ├── vit_imagenette_base.yaml     # ベースモデル
│   └── vit_imagenette_small.yaml    # 小型モデル
└── experiment/
    ├── vit_imagenette_baseline.yaml # ベースライン実験
    └── vit_imagenette_dev.yaml      # 開発用実験

modern_vit/
└── data/
    ├── __init__.py
    └── imagenette_datamodule.py     # データモジュール実装
```
