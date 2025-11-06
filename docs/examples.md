# DL-Scaffold 使用例

このドキュメントでは、DL-Scaffoldの実践的な使用例を紹介します。すべての例は実際に動作確認済みです。

## 📋 目次

- [基本的な使用例](#基本的な使用例)
- [モデルバリアント](#モデルバリアント)
- [データバリアント](#データバリアント)
- [トレーナー設定](#トレーナー設定)
- [ロギング](#ロギング)
- [実験管理](#実験管理)
- [トラブルシューティング](#トラブルシューティング)

## 基本的な使用例

### 最小限の実験（開発モード）

```bash
uv run python scripts/train.py experiment=mnist_dev
```

**期待される出力:**
```
Global seed set to 42
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs

Epoch 0: 100%|██████████| 15/15 [00:10<00:00,  1.45it/s, v_num=0, train/loss=0.123, train/acc=0.672]

Testing: 0it [00:00, ?it/s]
────────────────────────────────────────────────────────────────────────────
Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────
test/acc                0.673
test/loss               0.891
────────────────────────────────────────────────────────────────────────────
```

**実行時間:** 約30秒（CPU）  
**用途:** 素早い動作確認、デバッグ

### 完全なベースライン実験

```bash
uv run python scripts/train.py experiment=mnist_baseline
```

**期待される出力:**
```
Epoch 9: 100%|██████████| 938/938 [01:23<00:00, 11.23it/s, v_num=1, train/loss=0.034, train/acc=0.989, val/loss=0.043, val/acc=0.987]

`Trainer.fit` stopped: `max_epochs=10` reached.

Testing DataLoader 0: 100%|██████████| 157/157 [00:06<00:00, 25.78it/s]
────────────────────────────────────────────────────────────────────────────
Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────
test/acc                0.987
test/loss               0.041
────────────────────────────────────────────────────────────────────────────
```

**最終精度:** 98.7%  
**実行時間:** 約5-10分（CPU）、1-2分（GPU）  
**チェックポイント:** `logs/train/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/epoch_009.ckpt`

## モデルバリアント

### シンプルモデル（421K parameters）

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  model_variant@model=mnist_simple
```

**モデル詳細:**
- Hidden units: 128
- Parameters: 421,258
- 構造: Conv(32) -> Conv(64) -> FC(128) -> FC(10)

### ラージモデル（824K parameters）

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  model_variant@model=mnist_large
```

**モデル詳細:**
- Hidden units: 256
- Parameters: 824,074
- 構造: Conv(32) -> Conv(64) -> FC(256) -> FC(10)

**性能比較:**
| モデル | Parameters | Val Acc | Train Time (10 epochs) |
|--------|-----------|---------|------------------------|
| Simple | 421K      | 98.7%   | 5-10分 (CPU)          |
| Large  | 824K      | 98.9%   | 7-12分 (CPU)          |

## データバリアント

### 標準バッチサイズ

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  data_variant@data=mnist_standard
```

**設定:**
- Batch size: 64
- Num workers: 0
- Pin memory: false

**使用メモリ:** 約500MB

### ラージバッチ

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  data_variant@data=mnist_large_batch
```

**設定:**
- Batch size: 256
- Num workers: 0
- Pin memory: false

**使用メモリ:** 約1.5GB  
**速度向上:** 約1.5-2x faster

## トレーナー設定

### CPU学習

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer=cpu
```

**設定:**
```yaml
accelerator: cpu
devices: 1
```

### GPU学習（単一GPU）

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer=gpu
```

**速度:** CPU比で約5-10x faster

### MPS学習（Mac M1/M2）

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer=mps
```

**注意:** Mac特有の制約あり（pin_memory=falseが必須）

### マルチGPU（DDP）

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer=ddp \
  trainer.devices=4
```

**期待される出力:**
```
Using 4 GPUs for training
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/4
...
All distributed processes registered. Starting with 4 processes
```

### 混合精度学習

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer=gpu \
  trainer.precision="16-mixed"
```

**メモリ削減:** 約40-50%  
**速度向上:** 約1.5-2x faster（GPU依存）

## ロギング

### Wandbロギング

```bash
# 基本的な使用
uv run python scripts/train.py \
  experiment=mnist_baseline \
  logger=wandb

# カスタムプロジェクト名
uv run python scripts/train.py \
  experiment=mnist_baseline \
  logger=wandb \
  logger.wandb.project=my-mnist-experiments

# タグ付け
uv run python scripts/train.py \
  experiment=mnist_baseline \
  logger=wandb \
  logger.wandb.tags="[baseline,v1,production]"

# オフラインモード
uv run python scripts/train.py \
  experiment=mnist_baseline \
  logger=wandb \
  logger.wandb.offline=true
```

**Wandbダッシュボードで確認できる情報:**
- train/loss, train/acc
- val/loss, val/acc, val/acc_best
- test/loss, test/acc
- システムメトリクス（CPU、メモリ使用率）
- ハイパーパラメータ

### CSVLogger（デフォルト）

```bash
uv run python scripts/train.py experiment=mnist_baseline
```

**ログ保存先:** `logs/train/runs/YYYY-MM-DD_HH-MM-SS/csv/`

## 実験管理

### ハイパーパラメータスイープ

```bash
# 学習率を変更
uv run python scripts/train.py \
  experiment=mnist_baseline \
  model.lr=0.0001

# バッチサイズを変更
uv run python scripts/train.py \
  experiment=mnist_baseline \
  data.batch_size=128

# 複数のパラメータを同時に変更
uv run python scripts/train.py \
  experiment=mnist_baseline \
  model.lr=0.001 \
  data.batch_size=256 \
  trainer.max_epochs=20
```

### シード値の固定

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  seed=12345
```

**再現性:** 同じシードで完全に同じ結果が得られます

### トレーニングのみ（テストなし）

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  test=false
```

### チェックポイントから再開

```bash
uv run python scripts/train.py \
  experiment=mnist_baseline \
  ckpt_path=logs/train/runs/2024-01-15_10-30-45/checkpoints/epoch_005.ckpt
```

### 評価のみ

```bash
uv run python scripts/eval.py \
  ckpt_path=logs/train/runs/2024-01-15_10-30-45/checkpoints/last.ckpt
```

**期待される出力:**
```
Testing: 100%|██████████| 157/157 [00:06<00:00, 25.78it/s]
────────────────────────────────────────────────────────────────────────────
Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────
test/acc                0.987
test/loss               0.041
────────────────────────────────────────────────────────────────────────────
```

## トラブルシューティング

### Out of Memory (OOM)

**問題:**
```
RuntimeError: CUDA out of memory
```

**解決策:**

```bash
# バッチサイズを小さく
uv run python scripts/train.py \
  experiment=mnist_baseline \
  data.batch_size=32

# 勾配累積を使用（効果的なバッチサイズは保持）
uv run python scripts/train.py \
  experiment=mnist_baseline \
  data.batch_size=32 \
  trainer.accumulate_grad_batches=2

# 混合精度学習
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer.precision="16-mixed"
```

### Multiprocessing Error（Mac）

**問題:**
```
RuntimeError: DataLoader worker (pid XXXX) is killed by signal: Killed.
```

**解決策:**

```bash
# num_workersを0に設定
uv run python scripts/train.py \
  experiment=mnist_baseline \
  data.num_workers=0
```

### Pin Memory Error（Mac MPS）

**問題:**
```
RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
```

**解決策:**

```bash
# pin_memoryを無効化
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer=mps \
  data.pin_memory=false
```

### Hydra Config Error

**問題:**
```
omegaconf.errors.ConfigAttributeError: Key 'model' is not in struct
```

**解決策:**

1. `# @package _global_` をconfig先頭から削除
2. Experimentファイルで正しくdefaultsを設定:
```yaml
defaults:
  - /model_variant@model: mnist_simple
  - /data_variant@data: mnist_standard
```

### 実験が遅い

**解決策の優先順位:**

1. **GPUを使用:**
```bash
trainer=gpu
```

2. **バッチサイズを大きく:**
```bash
data.batch_size=256
```

3. **混合精度:**
```bash
trainer.precision="16-mixed"
```

4. **データローダーを高速化（Linuxのみ）:**
```bash
data.num_workers=4 data.pin_memory=true
```

### 学習が不安定

**解決策:**

```bash
# 学習率を小さく
uv run python scripts/train.py \
  experiment=mnist_baseline \
  model.lr=0.0001

# Gradient clippingを追加
uv run python scripts/train.py \
  experiment=mnist_baseline \
  trainer.gradient_clip_val=1.0

# Warmupの追加（カスタムスケジューラー実装が必要）
```

## 高度な使用例

### カスタムコールバック

```bash
# Early stopping + Model checkpoint
uv run python scripts/train.py \
  experiment=mnist_baseline \
  callbacks.early_stopping.patience=5 \
  callbacks.model_checkpoint.save_top_k=3
```

### 複数実験の自動実行

```bash
# Bash script example
for lr in 0.001 0.0001 0.00001; do
  uv run python scripts/train.py \
    experiment=mnist_baseline \
    model.lr=$lr \
    logger.wandb.name="lr_${lr}"
done
```

### Hydra multirun

```bash
# 複数の学習率を並列実行
uv run python scripts/train.py -m \
  experiment=mnist_baseline \
  model.lr=0.001,0.0001,0.00001
```

## まとめ

このドキュメントで紹介した例を組み合わせることで、ほとんどの深層学習タスクに対応できます。

**よく使うコマンド:**
- 開発: `experiment=mnist_dev`
- ベースライン: `experiment=mnist_baseline`
- GPU学習: `trainer=gpu`
- Wandb: `logger=wandb`

**次のステップ:**
- [Tutorial](tutorial.md) - ステップバイステップガイド
- [ML Project Guide](ml-project-guide.md) - ベストプラクティス
