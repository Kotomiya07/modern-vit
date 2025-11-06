# DL-Scaffold チュートリアル

このチュートリアルでは、DL-Scaffoldを使って深層学習プロジェクトを始める方法を学びます。

## 🎯 目次

1. [初めての実験を実行](#初めての実験を実行)
2. [新しいモデルを作成](#新しいモデルを作成)
3. [カスタムデータセットを追加](#カスタムデータセットを追加)
4. [実験設定をカスタマイズ](#実験設定をカスタマイズ)
5. [分散学習](#分散学習)
6. [Wandbで実験を追跡](#wandbで実験を追跡)

## 初めての実験を実行

### 1. 環境のセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/Kotomiya07/DL-Scaffold.git
cd DL-Scaffold

# 依存関係をインストール
uv sync
```

### 2. サンプル実験を実行

```bash
# MNISTベースライン実験（開発モード）
uv run python scripts/train.py experiment=mnist_dev

# 期待される出力:
# - 10 training batches
# - 5 validation batches
# - 約30秒で完了
# - 精度: ~60-70%
```

### 3. フルトレーニング

```bash
# 完全なトレーニング（10エポック）
uv run python scripts/train.py experiment=mnist_baseline

# 期待される結果:
# - 約5-10分で完了（CPUの場合）
# - 検証精度: ~98%以上
# - チェックポイント保存: logs/train/runs/...
```

## 新しいモデルを作成

### ステップ1: LightningModuleを実装

`project_name/models/custom_model.py`:

```python
"""Custom model for your task."""

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import Accuracy


class CustomModel(nn.Module):
    """Your custom PyTorch model."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomLightningModule(LightningModule):
    """Lightning wrapper for custom model."""
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = CustomModel(input_dim, hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=output_dim)
        self.val_acc = Accuracy(task="multiclass", num_classes=output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val/loss", loss)
        self.log("val/acc", self.val_acc)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
```

### ステップ2: Model Variantを作成

`configs/model_variant/custom.yaml`:

```yaml
_target_: project_name.models.custom_model.CustomLightningModule

input_dim: 784
hidden_dim: 256
output_dim: 10
lr: 0.001
```

### ステップ3: Experimentを作成

`configs/experiment/custom_baseline.yaml`:

```yaml
# @package _global_

defaults:
  - /model_variant@model: custom
  - /data_variant@data: mnist_standard
  - override /data: mnist
  - override /callbacks: default
  - override /trainer: default

tags: ["custom", "baseline"]

seed: 42
train: true
test: true

trainer:
  max_epochs: 10
```

### ステップ4: 実行

```bash
uv run python scripts/train.py experiment=custom_baseline
```

## カスタムデータセットを追加

### ステップ1: DataModuleを実装

`project_name/data/custom_datamodule.py`:

```python
"""Custom DataModule."""

from pathlib import Path
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    """Your custom dataset."""
    
    def __init__(self, data_path: Path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # Load your data here
        
    def __len__(self):
        # Return dataset size
        pass
        
    def __getitem__(self, idx):
        # Return sample
        pass


class CustomDataModule(LightningDataModule):
    """DataModule for custom dataset."""
    
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = CustomDataset(Path(self.hparams.data_dir) / "train")
        self.val_dataset = CustomDataset(Path(self.hparams.data_dir) / "val")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
```

### ステップ2: Data設定を作成

`configs/data/custom.yaml`:

```yaml
_target_: project_name.data.custom_datamodule.CustomDataModule

data_dir: ${paths.data_dir}/custom
batch_size: 32
num_workers: 0
```

`configs/data_variant/custom_standard.yaml`:

```yaml
batch_size: 32
num_workers: 0
```

### ステップ3: 使用

```bash
uv run python scripts/train.py \
  experiment=custom_baseline \
  data=custom
```

## 実験設定をカスタマイズ

### ハイパーパラメータの調整

```bash
# 学習率を変更
uv run python scripts/train.py experiment=mnist_baseline model.lr=0.0001

# バッチサイズを変更
uv run python scripts/train.py experiment=mnist_baseline data.batch_size=256

# エポック数を変更
uv run python scripts/train.py experiment=mnist_baseline trainer.max_epochs=50
```

### Early Stoppingのカスタマイズ

`configs/callbacks/custom_early_stopping.yaml`:

```yaml
early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: "val/loss"  # または "val/acc"
  patience: 10
  mode: "min"  # lossの場合は "min", accの場合は "max"
  min_delta: 0.001
```

実験設定で使用:

```yaml
defaults:
  - override /callbacks: custom_early_stopping
```

## 分散学習

### 単一ノード、複数GPU

```bash
# DDPで4GPUを使用
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer=ddp \
  trainer.devices=4
```

### 混合精度学習

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer=gpu \
  trainer.precision="16-mixed"
```

### 設定ファイルで指定

`configs/experiment/mnist_distributed.yaml`:

```yaml
# @package _global_

defaults:
  - mnist_baseline
  - override /trainer: ddp

trainer:
  devices: 4
  precision: "16-mixed"
  max_epochs: 20
```

## Wandbで実験を追跡

### ステップ1: Wandbにログイン

```bash
uv run wandb login
```

### ステップ2: 実験を実行

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  logger=wandb
```

### ステップ3: カスタム設定

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  logger=wandb \
  logger.wandb.project=my-dl-project \
  logger.wandb.name=baseline-exp-001 \
  logger.wandb.tags="[baseline,mnist,v1]"
```

### ステップ4: オフラインモード

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  logger=wandb \
  logger.wandb.offline=true
```

## よくある質問

### Q: 学習が遅い場合は?

A: 以下を試してください:

1. GPUを使用: `trainer=gpu`
2. ワーカー数を増やす: `data.num_workers=4`
3. バッチサイズを大きく: `data.batch_size=256`
4. 混合精度: `trainer.precision="16-mixed"`

### Q: メモリ不足エラーが出る場合は?

A: 以下を試してください:

1. バッチサイズを小さく: `data.batch_size=32`
2. 勾配累積を使用: `trainer.accumulate_grad_batches=4`
3. モデルサイズを小さく

### Q: 実験結果はどこに保存される?

A: `logs/train/runs/` ディレクトリに保存されます。各実行は日時でフォルダ分けされます。

### Q: チェックポイントから再開するには?

A: 

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  ckpt_path=/path/to/checkpoint.ckpt
```

## 次のステップ

- [Examples](examples.md) - より詳細な使用例
- [ML Project Guide](ml-project-guide.md) - ベストプラクティス
- [Lightning Documentation](https://lightning.ai/docs/) - Lightning公式ドキュメント
- [Hydra Documentation](https://hydra.cc/) - Hydra公式ドキュメント
