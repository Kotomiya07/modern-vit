"""Unit tests for the ViT module."""

from typing import Any

import pytest
import torch

from modern_vit.models.config import ViTLightningModuleConfig
from modern_vit.models.vit_module import ViTLightningModule


@pytest.mark.parametrize(
    "router_config",
    [
        {"type": "token_choice", "k": 2},
        {"type": "expert_choice", "k": 2},
    ],
)
def test_vit_module_instantiation(router_config: dict[str, Any]) -> None:
    """Test that the ViTLightningModule can be instantiated with different routers."""
    config = ViTLightningModuleConfig(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        n_kv_heads=4,
        multiple_of=256,
        n_experts=4,
        router_config=router_config,
        moe_layers=(0, 2, 4),  # Use MoE in some layers
    )
    model = ViTLightningModule(config)
    assert isinstance(model, ViTLightningModule)
    
    # Also test forward pass to ensure router works
    input_tensor = torch.randn(2, 3, 32, 32)
    logits, aux_loss = model(input_tensor)
    assert logits.shape == (2, 10)
    assert isinstance(aux_loss, torch.Tensor)


def test_vit_module_forward_pass() -> None:
    """Test the forward pass and auxiliary loss calculation."""
    config = ViTLightningModuleConfig(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        n_kv_heads=4,
        multiple_of=256,
        n_experts=4,
        router_config={"type": "token_choice", "k": 2},
        moe_layers=(0, 1, 2, 3, 4, 5),  # Use MoE in all layers
    )
    model = ViTLightningModule(config)
    input_tensor = torch.randn(2, 3, 32, 32)
    logits, aux_loss = model(input_tensor)

    assert logits.shape == (2, 10)
    assert isinstance(aux_loss, torch.Tensor)
    assert aux_loss.item() > 0


def test_vit_module_no_moe() -> None:
    """Test that the model works without any MoE layers."""
    config = ViTLightningModuleConfig(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        n_kv_heads=4,
        multiple_of=256,
        n_experts=4,
        moe_layers=(),  # No MoE layers
    )
    model = ViTLightningModule(config)
    input_tensor = torch.randn(2, 3, 32, 32)
    logits, aux_loss = model(input_tensor)

    assert logits.shape == (2, 10)
    assert aux_loss.item() == 0
