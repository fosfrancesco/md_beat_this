"""
Transformer with rotary position embedding and SwiGLU feed-forward layers, built on
the attention of beat_this.model.roformer.
"""

import torch.nn.functional as F
from beat_this.model.roformer import Attention, RMSNorm
from torch import nn
from torch.nn import Module, ModuleList


class SwiGLU(Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


class FeedForward(Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        # 2/3 of the hidden size of a non-gated feed-forward with the same `mult`,
        # rounded up to a multiple of 32, to keep a similar parameter count
        hidden_dim = int(2 * dim * mult / 3)
        hidden_dim = 32 * ((hidden_dim + 31) // 32)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, 2 * hidden_dim),
            SwiGLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(Module):
    def __init__(self, *, dim, depth, dim_head, heads, ff_mult, rotary_embed):
        super().__init__()
        self.layers = ModuleList(
            [
                ModuleList(
                    [
                        Attention(
                            dim, heads=heads, dim_head=dim_head, rotary_embed=rotary_embed
                        ),
                        FeedForward(dim, ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
