import torch
import torch.nn as nn
from .utils import manual_cast


def chunked_apply(module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0 or x.shape[0] <= chunk_size:
        return module(x)

    # Process first chunk to determine output shape and dtype
    out_0 = module(x[0:chunk_size])
    out_shape = (x.shape[0],) + out_0.shape[1:]
    out = torch.empty(out_shape, device=x.device, dtype=out_0.dtype)
    out[0:chunk_size] = out_0

    # Process remaining chunks
    for i in range(chunk_size, x.shape[0], chunk_size):
        out[i:i+chunk_size] = module(x[i:i+chunk_size])
    return out


class LayerNorm32(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low_vram = False
        self.chunk_size = 65536

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_vram:
            return chunked_apply(self._forward, x, self.chunk_size)
        return self._forward(x)


class GroupNorm32(nn.GroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low_vram = False
        self.chunk_size = 65536

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_vram:
            return chunked_apply(self._forward, x, self.chunk_size)
        return self._forward(x)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
