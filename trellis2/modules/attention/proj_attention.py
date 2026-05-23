"""
View-Aligned Projection Attention Module for TRELLIS2

This module implements the projection-based attention mechanism that combines
global cross-attention with view-aligned projected features.

Supports two modes:
- "proj": Standard projection (DINOv3 only), per-block proj_linear
- "gated_proj": Gated fusion of DINOv3 (semantic) + VAE (color) features
"""

from typing import *
import torch
import torch.nn as nn


class ProjectAttention(nn.Module):
    """
    Projection-based Attention Module with per-block proj_linear.
    
    Combines global cross-attention with view-aligned projected features.
    Each block owns a proj_linear that projects DINOv3 features from
    proj_in_channels (e.g. 1024) to model_channels (e.g. 1536).
    
    The module receives:
    - x: Input features from the transformer
    - context: A dict with keys:
      - 'global': Global image features, shape [B, M, ctx_channels]
      - 'proj': View-aligned projected features, shape [B, N, proj_in_channels]
    
    The output combines the cross-attention result with the projected context.
    """
    def __init__(self, cross_attn_block: nn.Module, channels: int, proj_in_channels: int):
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(proj_in_channels, channels, bias=True)
        
    def forward(self, x: torch.Tensor, context: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if isinstance(context, dict):
            global_context = context['global']
            proj_context = context['proj']
        else:
            global_context, proj_context = context
        
        global_out = self.cross_attn_block(x, global_context)
        proj_out = self.proj_linear(proj_context)
        context_combined = proj_out + global_out
        return context_combined


class GatedProjectAttention(nn.Module):
    """
    Concat-Projection Attention Module for DINOv3 (semantic) + VAE (color) features.
    
    Concatenates DINOv3 and VAE projected features and applies a single linear
    projection to model_channels. This is mathematically equivalent to two
    separate proj_linears + addition, but allows cross-dimensional interactions
    between semantic and color features through the shared weight matrix.
    
    Zero-initialized for stable training: at init, fused=0 so only global
    cross-attention contributes; color+semantic signals are gradually learned.
    
    The module receives:
    - x: Input features from the transformer
    - context: A dict with keys:
      - 'global': Global image features, shape [B, M, ctx_channels]
      - 'proj_semantic': DINOv3 projected features, shape [B, N, dino_channels]
      - 'proj_color': VAE projected features, shape [B, N, vae_channels]
    """
    def __init__(
        self,
        cross_attn_block: nn.Module,
        channels: int,
        dino_in_channels: int,
        vae_in_channels: int,
    ):
        """
        Args:
            cross_attn_block: The underlying cross-attention module
            channels: Model channels (output dimension)
            dino_in_channels: DINOv3 proj feature dimension (e.g. 1024)
            vae_in_channels: VAE latent feature dimension (e.g. 16)
        """
        super().__init__()
        self.cross_attn_block = cross_attn_block
        self.proj_linear = nn.Linear(dino_in_channels + vae_in_channels, channels, bias=True)
        # Zero-init: at start, fused=0, only global cross-attn contributes
        nn.init.zeros_(self.proj_linear.weight)
        nn.init.zeros_(self.proj_linear.bias)
        
    def forward(self, x: torch.Tensor, context: Union[Dict[str, torch.Tensor], Tuple]) -> torch.Tensor:
        if isinstance(context, dict):
            global_context = context['global']
            proj_semantic = context['proj_semantic']
            proj_color = context['proj_color']
        else:
            global_context, proj_semantic, proj_color = context
        
        global_out = self.cross_attn_block(x, global_context)
        fused = self.proj_linear(torch.cat([proj_semantic, proj_color], dim=-1))
        return fused + global_out
