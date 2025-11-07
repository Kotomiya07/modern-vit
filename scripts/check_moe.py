"""スクリプト: MoEの動作を確認するためのデバッグツール."""

import sys
from pathlib import Path

import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from modern_vit.models.config import ViTLightningModuleConfig
from modern_vit.models.vit_module import ViTLightningModule


def print_expert_stats(stats: dict, layer_idx: int) -> None:
    """エキスパート使用統計を表示."""
    print(f"\n=== Layer {layer_idx} MoE Statistics ===")
    print(f"Total tokens: {stats['n_tokens']}")
    print(f"Experts per token (k): {stats['k']}")
    print(f"All experts used: {stats['all_experts_used']}")
    print("\nExpert usage:")
    for i, (count, fraction, score) in enumerate(
        zip(
            stats["expert_counts"],
            stats["expert_fractions"],
            stats["routing_scores"],
        )
    ):
        print(f"  Expert {i}: {count:4d} tokens ({fraction * 100:5.2f}%), mean score: {score:.4f}")


def check_moe_functionality(config: DictConfig) -> None:
    """MoEの動作を確認."""
    print("=" * 60)
    print("MoE Functionality Check")
    print("=" * 60)

    # モデルを初期化
    lightning_config = ViTLightningModuleConfig(
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        num_classes=config.model.num_classes,
        dim=config.model.dim,
        depth=config.model.depth,
        heads=config.model.heads,
        n_kv_heads=config.model.n_kv_heads,
        multiple_of=config.model.multiple_of,
        n_experts=config.model.n_experts,
        router_config=config.model.router_config,
        moe_layers=config.model.moe_layers,
        channels=config.model.channels,
    )

    model = ViTLightningModule(lightning_config)
    model.eval()

    # テスト用の入力を作成
    batch_size = 2
    input_tensor = torch.randn(
        batch_size,
        config.model.channels,
        config.model.image_size,
        config.model.image_size,
    )

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"MoE layers: {config.model.moe_layers}")
    print(f"Number of experts: {config.model.n_experts}")

    # Forward pass
    with torch.no_grad():
        logits, aux_loss = model(input_tensor)

    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.6f}")

    # 各MoEレイヤーの統計を取得
    moe_layers = config.model.moe_layers
    if moe_layers:
        print("\n" + "=" * 60)
        print("Expert Usage Statistics per Layer")
        print("=" * 60)

        # 中間層の出力を取得するためにhookを使用
        stats_by_layer = []

        def make_hook(layer_idx: int):
            def hook(module, input, output):
                if hasattr(module, "get_expert_usage_stats"):
                    stats = module.get_expert_usage_stats(input[0])
                    stats_by_layer.append((layer_idx, stats))

            return hook

        hooks = []
        layer_idx = 0
        for block in model.model.layers:
            if layer_idx in moe_layers:
                hook = block.feed_forward.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)
            layer_idx += 1

        # Forward pass with hooks
        with torch.no_grad():
            _ = model(input_tensor)

        # 統計を表示
        for layer_idx, stats in stats_by_layer:
            print_expert_stats(stats, layer_idx)

        # Hookを削除
        for hook in hooks:
            hook.remove()

        # 全体的な評価
        print("\n" + "=" * 60)
        print("Overall Assessment")
        print("=" * 60)

        all_experts_used = all(stats["all_experts_used"] for _, stats in stats_by_layer)
        if all_experts_used:
            print("✓ All experts are being used across all MoE layers")
        else:
            print("⚠ Some experts are not being used in some layers")
            print("  This may be normal for random initialization, but should improve with training")

        # 負荷分散の評価
        print("\nLoad balancing assessment:")
        for layer_idx, stats in stats_by_layer:
            fractions = stats["expert_fractions"]
            ideal_fraction = 1.0 / len(fractions)
            max_deviation = abs(fractions - ideal_fraction).max()
            print(f"  Layer {layer_idx}: Max deviation from ideal ({ideal_fraction:.3f}): {max_deviation:.3f}")
    else:
        print("\n⚠ No MoE layers configured!")

    print("\n" + "=" * 60)
    print("Check completed!")
    print("=" * 60)


def main() -> None:
    """メイン関数."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_moe.py <config_path>")
        print("Example: python scripts/check_moe.py configs/experiment/vit_imagenette_dev.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    from hydra import compose, initialize_config_dir

    config_dir = Path("configs")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=str(config_path.relative_to(config_dir)))

    check_moe_functionality(cfg)


if __name__ == "__main__":
    main()
