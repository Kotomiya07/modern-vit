# MoE設定ファイル一覧

このドキュメントでは、MoE（Mixture of Experts）を使用したモデルの設定ファイルについて説明します。

## モデルバリアント設定

### 1. `vit_imagenette_small_moe.yaml`
**用途**: 開発・テスト用の小さなモデルにMoEを適用

**特徴**:
- モデルサイズ: Small (dim=256, depth=6)
- エキスパート数: 4
- ルーター: Token-Choice (k=2)
- MoE適用レイヤー: 全レイヤー (0-5)

**使用例**:
```bash
uv run python scripts/train.py experiment=vit_imagenette_dev_moe
```

### 2. `vit_imagenette_base_moe.yaml`
**用途**: Baseモデルの深いレイヤーにMoEを適用

**特徴**:
- モデルサイズ: Base (dim=384, depth=12)
- エキスパート数: 8
- ルーター: Token-Choice (k=2)
- MoE適用レイヤー: 深いレイヤー (4-11)

**使用例**:
```bash
uv run python scripts/train.py experiment=vit_imagenette_baseline_moe
```

### 3. `vit_imagenette_base_moe_all.yaml`
**用途**: Baseモデルの全レイヤーにMoEを適用（最大容量）

**特徴**:
- モデルサイズ: Base (dim=384, depth=12)
- エキスパート数: 8
- ルーター: Token-Choice (k=2)
- MoE適用レイヤー: 全レイヤー (0-11)

**使用例**:
```bash
uv run python scripts/train.py experiment=vit_imagenette_baseline_moe_all
```

### 4. `vit_imagenette_small_moe_expert_choice.yaml`
**用途**: Expert-Choiceルーターを使用した実験

**特徴**:
- モデルサイズ: Small (dim=256, depth=6)
- エキスパート数: 4
- ルーター: Expert-Choice (k=2)
- MoE適用レイヤー: 全レイヤー (0-5)

**使用例**:
```bash
uv run python scripts/train.py model=vit_imagenette model.variant=vit_imagenette_small_moe_expert_choice
```

## 実験設定

### 1. `vit_imagenette_dev_moe.yaml`
開発用のMoE実験設定。小さなモデルで高速に反復実験が可能。

### 2. `vit_imagenette_baseline_moe.yaml`
ベースラインモデルの深いレイヤーにMoEを適用した実験設定。

### 3. `vit_imagenette_baseline_moe_all.yaml`
ベースラインモデルの全レイヤーにMoEを適用した実験設定（最大容量）。

## MoE設定パラメータ

### 主要パラメータ

- **`n_experts`**: エキスパートの数
  - Smallモデル: 4
  - Baseモデル: 8

- **`n_experts_per_tok`**: 各トークンが使用するエキスパート数
  - デフォルト: 2

- **`router_config`**: ルーター設定
  - `type`: `"token_choice"` または `"expert_choice"`
  - `k`: 選択するエキスパート/トークン数

- **`moe_layers`**: MoEを適用するレイヤーのインデックスリスト
  - 例: `[0, 1, 2, 3, 4, 5]` (全レイヤー)
  - 例: `[4, 5, 6, 7, 8, 9, 10, 11]` (深いレイヤーのみ)

- **`aux_loss_weight`**: 補助損失の重み
  - デフォルト: 0.01
  - 負荷分散を促進するために使用

## 使用方法

### 実験設定を使用する場合

```bash
# 開発用MoE実験
uv run python scripts/train.py experiment=vit_imagenette_dev_moe

# ベースラインMoE実験（深いレイヤー）
uv run python scripts/train.py experiment=vit_imagenette_baseline_moe

# ベースラインMoE実験（全レイヤー）
uv run python scripts/train.py experiment=vit_imagenette_baseline_moe_all
```

### モデルバリアントを直接指定する場合

```bash
uv run python scripts/train.py \
  model=vit_imagenette \
  model.variant=vit_imagenette_small_moe \
  data=imagenette
```

### MoEの動作確認

```bash
# MoEの動作を確認
uv run python scripts/check_moe.py configs/experiment/vit_imagenette_dev_moe.yaml
```

## 推奨設定

### 開発・テスト段階
- `vit_imagenette_small_moe.yaml` を使用
- 小さなモデルで高速に実験可能

### 本番実験
- `vit_imagenette_base_moe.yaml` を使用
- 深いレイヤーにMoEを適用し、パフォーマンスと効率のバランスを取る

### 最大容量が必要な場合
- `vit_imagenette_base_moe_all.yaml` を使用
- 全レイヤーにMoEを適用し、最大のモデル容量を確保

## 注意事項

1. **メモリ使用量**: MoEを使用すると、エキスパート数に応じてメモリ使用量が増加します
2. **訓練時間**: MoEモデルは通常のモデルより訓練時間が長くなる可能性があります
3. **補助損失**: `aux_loss_weight` を調整して、負荷分散を制御できます
4. **エキスパート使用状況**: `scripts/check_moe.py` を使用して、エキスパートの使用状況を確認してください

