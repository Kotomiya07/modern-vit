"""Model parameter counting utilities."""

from lightning import LightningModule
from torch import nn

from modern_vit.models.vit_module import MoE, ViT


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_active_parameters(model: nn.Module) -> int:
    """Count active parameters for MoE models.
    
    For MoE models, active parameters are:
    - All non-MoE parameters (always active)
    - MoE attention parameters (always active)
    - MoE feed-forward parameters scaled by k/n_experts (only k experts active per token)
    
    Args:
        model: PyTorch model (should be ViT or ViTLightningModule)
        
    Returns:
        Estimated number of active parameters
    """
    if isinstance(model, LightningModule):
        # Extract the underlying ViT model
        if hasattr(model, "model") and isinstance(model.model, ViT):
            vit_model = model.model
        else:
            # Fallback: return total parameters if not ViT
            return count_parameters(model)
    elif isinstance(model, ViT):
        vit_model = model
    else:
        # Fallback: return total parameters if not ViT
        return count_parameters(model)
    
    total_params = 0
    active_params = 0
    
    # Count patch embedding and head
    total_params += count_parameters(vit_model.patch_embedding)
    active_params += count_parameters(vit_model.patch_embedding)
    
    total_params += count_parameters(vit_model.norm)
    active_params += count_parameters(vit_model.norm)
    
    total_params += count_parameters(vit_model.to_latent)
    active_params += count_parameters(vit_model.to_latent)
    
    total_params += count_parameters(vit_model.linear_head)
    active_params += count_parameters(vit_model.linear_head)
    
    # Count layers
    for i, layer in enumerate(vit_model.layers):
        # Attention is always active
        attn_params = count_parameters(layer.attention)
        total_params += attn_params
        active_params += attn_params
        
        # Norm layers are always active
        attn_norm_params = count_parameters(layer.attention_norm)
        ffn_norm_params = count_parameters(layer.ffn_norm)
        total_params += attn_norm_params + ffn_norm_params
        active_params += attn_norm_params + ffn_norm_params
        
        # Feed-forward layer
        if isinstance(layer.feed_forward, MoE):
            # MoE layer: count router + experts separately
            # Router is always active
            router_params = count_parameters(layer.feed_forward.router)
            total_params += router_params
            active_params += router_params
            
            # Experts: only k out of n_experts are active per token
            n_experts = len(layer.feed_forward.experts)
            # Get k from router config
            if hasattr(layer.feed_forward.router, "n_experts_per_tok"):
                k = layer.feed_forward.router.n_experts_per_tok
            elif hasattr(layer.feed_forward.router, "top_k_tokens"):
                k = layer.feed_forward.router.top_k_tokens
            else:
                k = 2  # Default
            
            expert_params = count_parameters(layer.feed_forward.experts[0])
            total_expert_params = expert_params * n_experts
            active_expert_params = expert_params * k
            
            total_params += total_expert_params
            active_params += active_expert_params
        else:
            # Regular feed-forward: always active
            ffn_params = count_parameters(layer.feed_forward)
            total_params += ffn_params
            active_params += ffn_params
    
    return int(active_params)


def format_parameter_count(num_params: int) -> str:
    """Format parameter count in human-readable format.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string (e.g., "1.2M", "500K")
    """
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return str(num_params)


def print_model_parameters(model: LightningModule | nn.Module) -> None:
    """Print model parameter statistics.
    
    Args:
        model: LightningModule or PyTorch model
    """
    total_params = count_parameters(model)
    active_params = count_active_parameters(model)
    
    print("=" * 60)
    print("Model Parameters")
    print("=" * 60)
    print(f"Total parameters:     {total_params:,} ({format_parameter_count(total_params)})")
    print(f"Active parameters:    {active_params:,} ({format_parameter_count(active_params)})")
    
    if total_params > 0:
        active_ratio = (active_params / total_params) * 100
        print(f"Active ratio:         {active_ratio:.2f}%")
    
    print("=" * 60)

