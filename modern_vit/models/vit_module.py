"""Vision Transformer (ViT) with modern components."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from einops import rearrange
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import Accuracy, MaxMetric, MeanMetric

from modern_vit.models.config import ViTConfig, ViTLightningModuleConfig


class RMSNorm(nn.Module):
    """Root Mean Square Normalization.

    Args:
        dim: The dimension of the input tensor.
        eps: A small value to avoid division by zero.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Apply the normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    """Precompute the frequency tensor for rotary positional embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Grouped Query Attention."""

    def __init__(self, n_heads: int, n_kv_heads: int, dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None) -> Tensor:
        """Forward pass."""
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Grouped Query Attention
        xk = xk.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)
        xv = xv.repeat_interleave(self.n_heads // self.n_kv_heads, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim**0.5)
        if mask is not None:
            scores += mask
        scores = nn.functional.softmax(scores, dim=-1).type_as(xq)

        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class AbstractRouter(nn.Module, ABC):
    """Abstract base class for MoE routers."""

    def __init__(self, dim: int, n_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, n_experts, bias=False)

    @abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Should return scores, indices, and auxiliary loss."""
        raise NotImplementedError


class TokenChoiceRouter(AbstractRouter):
    """Router that selects the top-k experts for each token."""

    def __init__(self, dim: int, n_experts: int, n_experts_per_tok: int) -> None:
        super().__init__(dim, n_experts)
        self.n_experts_per_tok = n_experts_per_tok

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for token-choice routing."""
        logits = self.gate(x)
        scores = nn.functional.softmax(logits, dim=-1)

        # Get top-k experts for each token
        top_k_scores, top_k_indices = torch.topk(scores, self.n_experts_per_tok, dim=-1)
        top_k_scores /= top_k_scores.sum(dim=-1, keepdim=True)

        # Calculate auxiliary loss for load balancing
        _, n_experts = scores.shape
        # Fraction of router probability mass allocated to each expert
        tokens_per_expert = scores.mean(dim=0)
        # Fraction of tokens for which the expert is in top-k
        selected_mask = torch.zeros_like(scores)
        selected_mask.scatter_(1, top_k_indices, 1.0)
        router_prob_per_expert = selected_mask.mean(dim=0)
        aux_loss = (tokens_per_expert * router_prob_per_expert).sum() * n_experts

        return top_k_scores, top_k_indices, aux_loss


class ExpertChoiceRouter(AbstractRouter):
    """Router that selects the top-k tokens for each expert."""

    def __init__(self, dim: int, n_experts: int, top_k_tokens: int) -> None:
        super().__init__(dim, n_experts)
        self.top_k_tokens = top_k_tokens

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for expert-choice routing.

        In expert-choice routing, each expert selects top-k tokens.
        Returns scores and indices in a format compatible with token-choice routing
        for consistency with MoE.forward() implementation.
        """
        logits = self.gate(x)  # Shape: (n_tokens, n_experts)
        # Transpose to (n_experts, n_tokens) for expert-choice routing
        logits_T = logits.transpose(0, 1)
        scores = nn.functional.softmax(logits_T, dim=-1)  # Softmax over tokens for each expert

        # Get top-k tokens for each expert
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k_tokens, dim=-1)
        # Shape: (n_experts, k)

        # Normalize scores
        top_k_scores /= top_k_scores.sum(dim=-1, keepdim=True)

        # Convert to token-choice format: (n_tokens, k) where each token knows which experts selected it
        # This is a simplified conversion - full expert-choice routing would require
        # different MoE.forward() implementation
        n_tokens = x.shape[0]
        n_experts = logits.shape[1]

        # Create token-choice compatible format
        # For each token, find which experts selected it
        token_scores = torch.zeros(n_tokens, n_experts, device=x.device)
        token_indices = torch.zeros(n_tokens, n_experts, dtype=torch.long, device=x.device)

        for expert_idx in range(n_experts):
            selected_token_indices = top_k_indices[expert_idx]  # Shape: (k,)
            selected_scores = top_k_scores[expert_idx]  # Shape: (k,)
            token_scores[selected_token_indices, expert_idx] = selected_scores
            token_indices[selected_token_indices, expert_idx] = expert_idx

        # Take top-k experts per token (in case a token is selected by multiple experts)
        top_k_scores_final, top_k_indices_final = torch.topk(token_scores, min(n_experts, self.top_k_tokens), dim=-1)
        top_k_scores_final /= top_k_scores_final.sum(dim=-1, keepdim=True)

        return top_k_scores_final, top_k_indices_final, torch.tensor(0.0, device=x.device)


class MoE(nn.Module):
    """Mixture of Experts layer."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        n_experts: int,
        router: AbstractRouter,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList([FeedForward(dim, hidden_dim, multiple_of) for _ in range(n_experts)])
        self.router = router

    def get_expert_usage_stats(self, x: Tensor) -> dict[str, Any]:
        """Get statistics about expert usage for debugging.

        Args:
            x: Input tensor of shape (bsz, seqlen, dim)

        Returns:
            Dictionary containing:
                - expert_counts: Number of tokens assigned to each expert
                - expert_fractions: Fraction of tokens assigned to each expert
                - routing_scores: Mean routing scores for each expert
                - all_experts_used: Whether all experts are used
        """
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        top_k_scores, top_k_indices, _ = self.router(x_flat)
        n_tokens = x_flat.shape[0]
        k = top_k_indices.shape[1]
        n_experts = len(self.experts)

        # Count tokens assigned to each expert
        flat_indices = top_k_indices.view(-1)
        expert_counts = torch.zeros(n_experts, device=x.device, dtype=torch.long)
        expert_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))

        # Calculate fractions
        expert_fractions = expert_counts.float() / (n_tokens * k)

        # Calculate mean routing scores per expert
        flat_scores = top_k_scores.view(-1)
        expert_scores_sum = torch.zeros(n_experts, device=x.device)
        expert_scores_count = torch.zeros(n_experts, device=x.device, dtype=torch.long)
        expert_scores_sum.scatter_add_(0, flat_indices, flat_scores)
        expert_scores_count.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
        expert_scores_mean = expert_scores_sum / (expert_scores_count + 1e-8)  # Avoid division by zero

        return {
            "expert_counts": expert_counts.cpu().numpy(),
            "expert_fractions": expert_fractions.cpu().numpy(),
            "routing_scores": expert_scores_mean.cpu().numpy(),
            "all_experts_used": (expert_counts > 0).all().item(),
            "n_tokens": n_tokens,
            "k": k,
        }

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        n_tokens = x_flat.shape[0]
        top_k_scores, top_k_indices, aux_loss = self.router(x_flat)
        k = top_k_indices.shape[1]

        # Group tokens by expert for efficient batch processing
        # Create token-to-expert mapping: (n_tokens * k,) -> expert_id
        flat_indices = top_k_indices.view(-1)  # Shape: (n_tokens * k,)
        flat_token_indices = torch.arange(n_tokens, device=x.device).repeat_interleave(k)
        flat_k_indices = torch.arange(k, device=x.device).repeat(n_tokens)

        # Process each expert with batched inputs
        output = torch.zeros_like(x_flat)
        for expert_idx in range(len(self.experts)):
            # Find all (token_idx, k_idx) pairs assigned to this expert
            expert_mask = flat_indices == expert_idx
            if not expert_mask.any():
                continue

            # Get token indices and k indices assigned to this expert
            token_indices = flat_token_indices[expert_mask]
            k_indices = flat_k_indices[expert_mask]

            # Batch process all tokens assigned to this expert
            expert_inputs = x_flat[token_indices]
            expert_outputs = self.experts[expert_idx](expert_inputs)

            # Get corresponding scores
            expert_scores = top_k_scores[token_indices, k_indices].unsqueeze(-1)

            # Weighted sum: add expert outputs to corresponding token positions
            output.index_add_(
                0,
                token_indices,
                (expert_outputs * expert_scores).type_as(x),
            )

        return output.view(bsz, seqlen, dim), aux_loss


class ViTBlock(nn.Module):
    """Vision Transformer Block."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        n_experts: int,
        router_config: dict[str, Any],
        use_moe: bool,
    ) -> None:
        super().__init__()
        self.attention = Attention(n_heads, n_kv_heads, dim)
        if use_moe:
            router_type = router_config.get("type", "token_choice")
            if router_type == "token_choice":
                router = TokenChoiceRouter(dim, n_experts, router_config.get("k", 2))
            elif router_type == "expert_choice":
                router = ExpertChoiceRouter(dim, n_experts, router_config.get("k", 2))
            else:
                msg = f"Unknown router type: {router_type}"
                raise ValueError(msg)
            self.feed_forward = MoE(
                dim,
                dim * 4,
                multiple_of,
                n_experts,
                router,
            )
        else:
            self.feed_forward = FeedForward(dim, dim * 4, multiple_of)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        ff_out = self.feed_forward(self.ffn_norm(h))
        aux_loss = torch.tensor(0.0, device=h.device)
        if isinstance(ff_out, tuple):
            ff_out, aux_loss = ff_out

        out = h + ff_out
        return out, aux_loss


class ViT(nn.Module):
    """Vision Transformer with GQA, SwiGLU, RoPE, RMSNorm, and MoE."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Linear(config.channels * config.patch_size * config.patch_size, config.dim)
        self.layers = nn.ModuleList(
            [
                ViTBlock(
                    config.dim,
                    config.heads,
                    config.n_kv_heads,
                    config.multiple_of,
                    config.n_experts,
                    config.router_config,
                    use_moe=i in config.moe_layers,
                )
                for i in range(config.depth)
            ]
        )
        self.norm = RMSNorm(config.dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(config.dim, config.num_classes)

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.dim // config.heads,
                (config.image_size // config.patch_size) ** 2,
            ),
        )

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        p = self.patch_size
        x = rearrange(
            img,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=p,
            p2=p,
        )
        x = self.patch_embedding(x)
        _, n, _ = x.shape

        # Ensure freqs_cis is on the same device as input
        freqs_cis = self.freqs_cis.to(x.device)[:n]

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux_loss = layer(x, freqs_cis, mask=None)
            total_aux_loss += aux_loss

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling

        x = self.to_latent(x)
        return self.linear_head(x), total_aux_loss


class ViTLightningModule(LightningModule):
    """LightningModule for the ViT model."""

    def __init__(self, config: ViTLightningModuleConfig) -> None:
        super().__init__()
        self.save_hyperparameters()

        router_config = config.router_config
        if router_config is None:
            router_config = {"type": "token_choice", "k": config.n_experts_per_tok}

        vit_config = ViTConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            n_kv_heads=config.n_kv_heads,
            multiple_of=config.multiple_of,
            n_experts=config.n_experts,
            router_config=router_config,
            moe_layers=config.moe_layers,
            channels=config.channels,
        )

        self.model = ViT(vit_config)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        return self.model(x)

    def on_train_start(self) -> None:
        """Called at the beginning of training."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform a single model step on a batch of data."""
        x, y = batch
        logits, aux_loss = self.forward(x)
        main_loss = self.criterion(logits, y)
        total_loss = main_loss + self.hparams.config.aux_loss_weight * aux_loss
        preds = torch.argmax(logits, dim=1)
        return total_loss, main_loss, aux_loss, preds, y

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Perform a single training step."""
        total_loss, main_loss, aux_loss, preds, targets = self.model_step(batch)

        self.train_loss(total_loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/main_loss", main_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/aux_loss", aux_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single validation step."""
        total_loss, main_loss, aux_loss, preds, targets = self.model_step(batch)

        self.val_loss(total_loss)
        self.val_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/main_loss", main_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/aux_loss", aux_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Perform a single test step."""
        total_loss, main_loss, aux_loss, preds, targets = self.model_step(batch)

        self.test_loss(total_loss)
        self.test_acc(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/main_loss", main_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/aux_loss", aux_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning-rate schedulers."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.config.lr,
            weight_decay=self.hparams.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/acc",
            },
        }
