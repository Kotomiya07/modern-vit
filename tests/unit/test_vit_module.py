"""Unit tests for the ViT module."""

from typing import Any

import pytest
import torch

from modern_vit.models.config import ViTLightningModuleConfig
from modern_vit.models.vit_module import ExpertChoiceRouter, MoE, TokenChoiceRouter, ViTLightningModule


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


def test_moe_expert_usage() -> None:
    """Test that MoE uses multiple experts and distributes tokens."""
    dim = 256
    hidden_dim = 1024
    multiple_of = 256
    n_experts = 4
    n_experts_per_tok = 2
    
    router = TokenChoiceRouter(dim, n_experts, n_experts_per_tok)
    moe = MoE(dim, hidden_dim, multiple_of, n_experts, router)
    
    # Create input with multiple tokens
    bsz, seqlen = 2, 64
    x = torch.randn(bsz, seqlen, dim)
    
    # Get expert usage statistics
    stats = moe.get_expert_usage_stats(x)
    
    # Check that statistics are computed correctly
    assert "expert_counts" in stats
    assert "expert_fractions" in stats
    assert "routing_scores" in stats
    assert "all_experts_used" in stats
    
    # Check that all experts are used (or at least most of them)
    # With random routing, it's unlikely that all experts are used, but we should use multiple
    expert_counts = stats["expert_counts"]
    assert expert_counts.sum() == stats["n_tokens"] * stats["k"]
    
    # Check that fractions sum to 1.0
    assert abs(stats["expert_fractions"].sum() - 1.0) < 1e-6
    
    # Check that at least 2 experts are used (with k=2 and random routing)
    assert (expert_counts > 0).sum() >= 2


def test_moe_forward_output_shape() -> None:
    """Test that MoE forward pass returns correct output shape."""
    dim = 256
    hidden_dim = 1024
    multiple_of = 256
    n_experts = 4
    n_experts_per_tok = 2
    
    router = TokenChoiceRouter(dim, n_experts, n_experts_per_tok)
    moe = MoE(dim, hidden_dim, multiple_of, n_experts, router)
    
    bsz, seqlen = 2, 64
    x = torch.randn(bsz, seqlen, dim)
    
    output, aux_loss = moe(x)
    
    # Check output shape
    assert output.shape == (bsz, seqlen, dim)
    assert isinstance(aux_loss, torch.Tensor)
    assert aux_loss.item() >= 0


def test_moe_auxiliary_loss() -> None:
    """Test that MoE auxiliary loss is computed correctly."""
    dim = 256
    hidden_dim = 1024
    multiple_of = 256
    n_experts = 4
    n_experts_per_tok = 2
    
    router = TokenChoiceRouter(dim, n_experts, n_experts_per_tok)
    moe = MoE(dim, hidden_dim, multiple_of, n_experts, router)
    
    bsz, seqlen = 2, 64
    x = torch.randn(bsz, seqlen, dim)
    
    _, aux_loss = moe(x)
    
    # Auxiliary loss should be non-negative
    assert aux_loss.item() >= 0
    
    # With balanced routing, auxiliary loss should be reasonable
    # (not too high, not zero)
    assert aux_loss.item() < 100.0  # Reasonable upper bound


def test_moe_expert_diversity() -> None:
    """Test that MoE uses diverse experts across different inputs."""
    dim = 256
    hidden_dim = 1024
    multiple_of = 256
    n_experts = 4
    n_experts_per_tok = 2
    
    router = TokenChoiceRouter(dim, n_experts, n_experts_per_tok)
    moe = MoE(dim, hidden_dim, multiple_of, n_experts, router)
    
    # Test with multiple different inputs
    bsz, seqlen = 2, 64
    all_expert_usage = set()
    
    for _ in range(10):
        x = torch.randn(bsz, seqlen, dim)
        stats = moe.get_expert_usage_stats(x)
        used_experts = set(torch.nonzero(torch.tensor(stats["expert_counts"])).flatten().tolist())
        all_expert_usage.update(used_experts)
    
    # With multiple random inputs, we should see diversity in expert usage
    # At least 2 different experts should be used across all inputs
    assert len(all_expert_usage) >= 2


def test_moe_gradient_flow() -> None:
    """Test that gradients flow through MoE layer."""
    dim = 256
    hidden_dim = 1024
    multiple_of = 256
    n_experts = 4
    n_experts_per_tok = 2
    
    router = TokenChoiceRouter(dim, n_experts, n_experts_per_tok)
    moe = MoE(dim, hidden_dim, multiple_of, n_experts, router)
    
    bsz, seqlen = 2, 64
    x = torch.randn(bsz, seqlen, dim, requires_grad=True)
    
    output, aux_loss = moe(x)
    loss = output.sum() + aux_loss
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    
    # Check that router parameters have gradients
    assert router.gate.weight.grad is not None
    assert router.gate.weight.grad.abs().sum() > 0


def test_expert_choice_router_zero_sum_normalization() -> None:
    """Test that ExpertChoiceRouter handles zero-sum normalization without NaN.
    
    This test verifies the fix for the bug where division by zero occurs
    when a token isn't selected by any expert, causing NaN values in routing weights.
    """
    dim = 256
    n_experts = 2
    top_k_tokens = 2
    
    router = ExpertChoiceRouter(dim, n_experts, top_k_tokens)
    
    # Create a scenario where n_experts * top_k_tokens < n_tokens
    # This means some tokens won't be selected by any expert
    n_tokens = 10  # More tokens than can be selected (2 experts * 2 tokens = 4 tokens selected)
    x = torch.randn(n_tokens, dim)
    
    # Run the router
    top_k_scores, top_k_indices, aux_loss = router(x)
    
    # Check that there are no NaN values in the output
    assert not torch.isnan(top_k_scores).any(), "Routing scores contain NaN values"
    assert not torch.isinf(top_k_scores).any(), "Routing scores contain Inf values"
    
    # Check that scores are properly normalized for tokens that were selected
    # For tokens that were selected, scores should sum to 1.0
    # For tokens that were not selected, scores should be 0
    score_sums = top_k_scores.sum(dim=-1)
    
    # Tokens with non-zero scores should have scores summing to 1.0
    selected_tokens_mask = score_sums > 0
    if selected_tokens_mask.any():
        selected_scores_sum = score_sums[selected_tokens_mask]
        torch.testing.assert_close(
            selected_scores_sum, 
            torch.ones_like(selected_scores_sum),
            atol=1e-6,
            rtol=1e-5,
            msg="Selected tokens should have scores summing to 1.0"
        )
    
    # Unselected tokens should have zero scores
    unselected_tokens_mask = ~selected_tokens_mask
    if unselected_tokens_mask.any():
        unselected_scores = top_k_scores[unselected_tokens_mask]
        assert (unselected_scores == 0).all(), "Unselected tokens should have zero scores"
    
    # Verify that the expected number of tokens were selected
    n_selected_tokens = selected_tokens_mask.sum().item()
    expected_selected = min(n_experts * top_k_tokens, n_tokens)
    assert n_selected_tokens == expected_selected, \
        f"Expected {expected_selected} tokens to be selected, but got {n_selected_tokens}"
