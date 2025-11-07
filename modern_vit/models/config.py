from typing import Any
from dataclasses import dataclass

@dataclass  
class ViTConfig:  
    image_size: int  
    patch_size: int  
    num_classes: int  
    dim: int  
    depth: int  
    heads: int  
    n_kv_heads: int  
    multiple_of: int  
    n_experts: int  
    router_config: dict[str, Any]  
    moe_layers: tuple[int, ...] = ()  
    channels: int = 3  


@dataclass
class ViTLightningModuleConfig:
    """Configuration for ViTLightningModule."""
    
    image_size: int  
    patch_size: int  
    num_classes: int  
    dim: int  
    depth: int  
    heads: int  
    n_kv_heads: int  
    multiple_of: int  
    n_experts: int  
    router_config: dict[str, Any] | None = None
    n_experts_per_tok: int = 2
    moe_layers: tuple[int, ...] = ()  
    channels: int = 3  
    lr: float = 1e-3
    weight_decay: float = 0.0
    aux_loss_weight: float = 0.01
