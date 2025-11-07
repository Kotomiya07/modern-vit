# MoE動作確認ガイド

このドキュメントでは、MoE（Mixture of Experts）が正しく機能していることを確認する方法を説明します。

## 確認方法

### 1. ユニットテストの実行

MoEの動作を確認するための包括的なテストが用意されています：

```bash
uv run pytest tests/unit/test_vit_module.py::test_moe_expert_usage -v
uv run pytest tests/unit/test_vit_module.py::test_moe_forward_output_shape -v
uv run pytest tests/unit/test_vit_module.py::test_moe_auxiliary_loss -v
uv run pytest tests/unit/test_vit_module.py::test_moe_expert_diversity -v
uv run pytest tests/unit/test_vit_module.py::test_moe_gradient_flow -v
```

すべてのテストを実行：

```bash
uv run pytest tests/unit/test_vit_module.py -v
```

### 2. デバッグスクリプトの使用

`scripts/check_moe.py` を使用して、実際のモデル設定でMoEの動作を確認できます：

```bash
uv run python scripts/check_moe.py configs/experiment/vit_imagenette_dev.yaml
```

このスクリプトは以下を表示します：
- 各MoEレイヤーでのエキスパート使用状況
- エキスパートごとのトークン割り当て数と割合
- ルーティングスコアの平均値
- 負荷分散の評価

### 3. プログラムから直接確認

モデルを使用する際に、MoEレイヤーから直接統計を取得できます：

```python
from modern_vit.models.vit_module import MoE, TokenChoiceRouter
import torch

# MoEレイヤーを作成
router = TokenChoiceRouter(dim=256, n_experts=4, n_experts_per_tok=2)
moe = MoE(dim=256, hidden_dim=1024, multiple_of=256, n_experts=4, router=router)

# 入力を作成
x = torch.randn(2, 64, 256)  # (batch_size, seq_len, dim)

# 統計を取得
stats = moe.get_expert_usage_stats(x)

print(f"Expert counts: {stats['expert_counts']}")
print(f"Expert fractions: {stats['expert_fractions']}")
print(f"All experts used: {stats['all_experts_used']}")
```

## 確認すべきポイント

### ✅ 正常な動作の指標

1. **複数のエキスパートが使用されている**
   - すべてのエキスパートが使用されることが理想的ですが、初期化時は一部のみ使用されることもあります
   - 訓練が進むと、より多くのエキスパートが使用されるようになります

2. **補助損失（auxiliary loss）が計算されている**
   - MoEレイヤーがある場合、`aux_loss > 0` であるべきです
   - 補助損失は負荷分散を促進するために使用されます

3. **エキスパート間で負荷が分散されている**
   - 各エキスパートに割り当てられるトークンの割合が均等に近いことが理想的です
   - 理想的な割合は `1 / n_experts` です

4. **勾配が正しく流れている**
   - ルーターとエキスパートのパラメータに勾配が存在することを確認します

### ⚠️ 問題の可能性がある指標

1. **一部のエキスパートのみが使用されている**
   - 初期化時は正常ですが、訓練後も続く場合は問題の可能性があります
   - 補助損失の重みを調整することを検討してください

2. **補助損失が0または非常に大きい**
   - 補助損失が0の場合、MoEが正しく動作していない可能性があります
   - 補助損失が非常に大きい場合、負荷分散がうまく機能していない可能性があります

3. **出力形状が正しくない**
   - MoEの出力は入力と同じ形状であるべきです
   - `(batch_size, seq_len, dim)` の形状を確認してください

## トラブルシューティング

### 問題: 一部のエキスパートが使用されない

**原因**: ルーターの初期化が偏っている、または補助損失の重みが不適切

**解決策**:
- 補助損失の重みを調整: `config.aux_loss_weight` を増やす
- ルーターの初期化を確認

### 問題: 補助損失が0

**原因**: MoEレイヤーが設定されていない、またはルーターが正しく動作していない

**解決策**:
- `moe_layers` 設定を確認
- ルーターのタイプとパラメータを確認

### 問題: 出力形状が正しくない

**原因**: MoEの実装に問題がある可能性

**解決策**:
- テストを実行して実装を確認
- 入力と出力の形状をデバッグ出力で確認

## 参考

- `modern_vit/models/vit_module.py`: MoEの実装
- `tests/unit/test_vit_module.py`: MoEのテスト
- `scripts/check_moe.py`: デバッグスクリプト

