import torch
import torch.nn as nn
from . import VarLenTensor

__all__ = [
    'SparseLinear'
]


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


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)
        self.low_vram = False
        self.chunk_size = 65536

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        if self.low_vram:
            return input.replace(chunked_apply(super().forward, input.feats, self.chunk_size))
        return input.replace(super().forward(input.feats))
