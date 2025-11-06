"""Vision Transformer (ViT) with modern components."""

from abc import ABC, abstractmethod
from typing import Any

from einops import rearrange
import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torchmetrics import Accuracy, MaxMetric, MeanMetric


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

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None
    ) -> Tensor:
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

        # Calculate auxiliary loss
        n_tokens, n_experts = scores.shape
        # For each expert, what's the fraction of tokens assigned to it?
        tokens_per_expert = scores.mean(dim=0)
        # What's the probability that a token is assigned to an expert?
        router_prob_per_expert = (scores > 0).float().mean(dim=0)
        aux_loss = (tokens_per_expert * router_prob_per_expert).sum() * n_experts

        top_k_scores, top_k_indices = torch.topk(
            scores, self.n_experts_per_tok, dim=-1
        )
        top_k_scores /= top_k_scores.sum(dim=-1, keepdim=True)
        return top_k_scores, top_k_indices, aux_loss


class ExpertChoiceRouter(AbstractRouter):
    """Router that selects the top-k tokens for each expert."""

    def __init__(self, dim: int, n_experts: int, top_k_tokens: int) -> None:
        super().__init__(dim, n_experts)
        self.top_k_tokens = top_k_tokens

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass for expert-choice routing."""
        logits = self.gate(x)
        scores = nn.functional.softmax(logits, dim=0)  # Apply softmax over tokens
        top_k_scores, top_k_indices = torch.topk(
            scores, self.top_k_tokens, dim=0
        )
        return top_k_scores, top_k_indices, torch.tensor(0.0)


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
        self.experts = nn.ModuleList(
            [
                FeedForward(dim, hidden_dim, multiple_of)
                for _ in range(n_experts)
            ]
        )
        self.router = router

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        top_k_scores, top_k_indices, aux_loss = self.router(x_flat)

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = top_k_indices == i
            if mask.any():
                expert_indices = mask.nonzero(as_tuple=True)
                expert_inputs = x_flat[expert_indices[0]]
                expert_scores = top_k_scores[expert_indices]
                expert_outputs = expert(expert_inputs)
                output.index_add_(
                    0,
                    expert_indices[0],
                    (expert_outputs * expert_scores.unsqueeze(-1)).type_as(x),
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
                raise ValueError(f"Unknown router type: {router_type}")
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

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        ff_out = self.feed_forward(self.ffn_norm(h))
        aux_loss = torch.tensor(0.0)
        if isinstance(ff_out, tuple):
            ff_out, aux_loss = ff_out

        out = h + ff_out
        return out, aux_loss


class ViT(nn.Module):
    """Vision Transformer with GQA, SwiGLU, RoPE, RMSNorm, and MoE."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        n_kv_heads: int,
        multiple_of: int,
        n_experts: int,
        router_config: dict[str, Any],
        moe_layers: tuple[int, ...] = (),
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(
            channels * patch_size * patch_size, dim
        )
        self.layers = nn.ModuleList(
            [
                ViTBlock(
                    dim,
                    heads,
                    n_kv_heads,
                    multiple_of,
                    n_experts,
                    router_config,
                    use_moe=i in moe_layers,
                )
                for i in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            dim // heads, (image_size // patch_size) ** 2
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
        b, n, _ = x.shape

        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[:n]

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

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        n_kv_heads: int,
        multiple_of: int,
        n_experts: int,
        n_experts_per_tok: int = 2,
        router_config: dict[str, Any] | None = None,
        moe_layers: tuple[int, ...] = (),
        channels: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        aux_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if router_config is None:
            router_config = {"type": "token_choice", "k": n_experts_per_tok}

        self.model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            n_kv_heads=n_kv_heads,
            multiple_of=multiple_of,
            n_experts=n_experts,
            router_config=router_config,
            moe_layers=moe_layers,
            channels=channels,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.model(x)

    def on_train_start(self) -> None:
        """Called at the beginning of training."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform a single model step on a batch of data."""
        x, y = batch
        logits, aux_loss = self.forward(x)
        main_loss = self.criterion(logits, y)
        total_loss = main_loss + self.hparams.aux_loss_weight * aux_loss
        preds = torch.argmax(logits, dim=1)
        return total_loss, main_loss, aux_loss, preds, y

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Perform a single training step."""
        total_loss, main_loss, aux_loss, preds, targets = self.model_step(batch)

        self.train_loss(total_loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/main_loss", main_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/aux_loss", aux_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
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

    def test_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
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
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
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
