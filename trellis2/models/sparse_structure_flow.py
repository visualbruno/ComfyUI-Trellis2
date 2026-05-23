from typing import *
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype, convert_module_to_f16
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock, ModulatedTransformerCrossBlock_woT
from ..modules.attention import RotaryPositionEmbedder


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        image_attn_mode: Literal["cross", "proj", "gated_proj"] = "cross",
        proj_in_channels: Optional[int] = None,
        vae_in_channels: Optional[int] = None,        
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.image_attn_mode = image_attn_mode
        self.proj_in_channels = proj_in_channels
        self.vae_in_channels = vae_in_channels        
        self.dtype = str_to_dtype(dtype)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)
        elif pe_mode == "rope":
            pos_embedder = RotaryPositionEmbedder(self.model_channels // self.num_heads, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            rope_phases = pos_embedder(coords)
            self.register_buffer("rope_phases", rope_phases)
            
        if pe_mode != "rope":
            self.rope_phases = None

        self.input_layer = nn.Linear(in_channels, model_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                image_attn_mode=image_attn_mode,
                proj_in_channels=proj_in_channels,
                vae_in_channels=vae_in_channels,                
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)
        if self.dtype == torch.float8_e4m3fn:
            self.dtype = torch.bfloat16

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)
            
        elif self.initialization == 'scaled':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            
            # Scaled init for to_out and ffn2
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                # Handle cross, proj, and gated_proj modes
                if self.image_attn_mode in ("proj", "gated_proj"):
                    block.cross_attn.cross_attn_block.to_out.apply(_scaled_init)
                else:
                    block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)
            
            # Initialize input layer to make the initial representation have variance 1
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)
            
            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            
            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, D, H, W]
            t: Timestep tensor [B]
            cond: Conditioning tensor. For "cross" mode: [B, N, D]. 
                  For "proj" mode: dict {'global': global_cond, 'proj': proj_cond} 
                  or tuple of (global_cond, proj_cond)
        
        Returns:
            Output tensor [B, C, D, H, W]
        """        
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        if self.pe_mode == "ape":
            h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        h = manual_cast(h, self.dtype)
        
        # Handle different conditioning modes
        if hasattr(self,'image_attn_mode'):            
            if self.image_attn_mode == 'proj':
                if isinstance(cond, dict):
                    global_cond = cond['global']
                    proj_cond = cond['proj']
                else:
                    global_cond, proj_cond = cond
                global_cond = manual_cast(global_cond, self.dtype)
                proj_cond = manual_cast(proj_cond, self.dtype)
                cond = (global_cond, proj_cond)
            elif self.image_attn_mode == 'gated_proj':
                global_cond = manual_cast(cond['global'], self.dtype)
                proj_semantic = manual_cast(cond['proj_semantic'], self.dtype)
                proj_color = manual_cast(cond['proj_color'], self.dtype)
                cond = {'global': global_cond, 'proj_semantic': proj_semantic, 'proj_color': proj_color}
            else:
                cond = manual_cast(cond, self.dtype)        
        else:
            cond = manual_cast(cond, self.dtype)
            
        for block in self.blocks:
            h = block(h, t_emb, cond, self.rope_phases)
        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution] * 3).contiguous()

        return h

class ModulatedMultiViewCond(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        num_init_tokens: int = 4096,
        dtype: Optional[torch.dtype] = torch.float32,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.cond_blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock_woT(
                channels,
                ctx_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode=attn_mode,
                use_checkpoint=use_checkpoint,
                use_rope=use_rope,
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(4)
        ])
        self.use_fp16 = use_fp16
        if use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.multiview_cond_tokens = nn.Parameter(torch.randn(1, num_init_tokens, channels).to(dtype))
        nn.init.normal_(self.multiview_cond_tokens, std=1e-6)
        self.intermediate_layer_idx = [4, 11, 17, 23]
        if use_fp16:
            self.convert_to_fp16()


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.cond_blocks.apply(convert_module_to_f16)
        self.multiview_cond_tokens = nn.Parameter(self.multiview_cond_tokens.data.to(self.dtype))
    def forward(self, aggregated_tokens_list: List, image_cond: torch.Tensor):

        b = aggregated_tokens_list[0].shape[0]
        patch_start_idx = 5
        idx = 0
        cond = self.multiview_cond_tokens.repeat(b, 1, 1)
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            # x = x.reshape(b, -1, 2048) + torch.cat([image_cond.reshape(b, -1, 1024), image_cond.reshape(b, -1, 1024)],dim=-1)
            x = torch.cat([x.reshape(b, -1, 2048), image_cond.reshape(b, -1, 1024)],dim=-1).to(self.dtype)
            cond = self.cond_blocks[idx](cond, x)
            idx = idx + 1
        return cond