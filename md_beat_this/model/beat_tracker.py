"""
Model definition for the masked diffusion Beat This! beat tracker.
"""

import math
from collections import OrderedDict

from beat_this.model.beat_tracker import BeatThis
from beat_this.model.roformer import Attention
from einops import rearrange
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import nn

from md_beat_this.model import roformer

# token ids of the beat and downbeat sequences
MASK, PAD, NO_EVENT, EVENT = 0, 1, 2, 3


class MDBeatThis(nn.Module):
    """
    The Beat This! network, additionally conditioned on partially known beat and
    downbeat sequences. It is composed of:
    - a frontend that processes the input spectrogram,
    - embeddings of the beat and downbeat tokens, added to the frontend output,
    - a series of transformer blocks,
    - two linear heads that produce the beat and downbeat logits.

    Args:
        spect_dim (int): The dimension of the input spectrogram (default: 128).
        transformer_dim (int): The dimension of the main transformer blocks (default: 512).
        ff_mult (int): The multiplier for the feed-forward dimension in the transformer blocks (default: 4).
        n_layers (int): The number of transformer blocks (default: 6).
        head_dim (int): The dimension of each attention head for the partial transformers in the frontend and the transformer blocks (default: 32).
        stem_dim (int): The out dimension of the stem convolutional layer (default: 32).
    """

    def __init__(
        self,
        spect_dim: int = 128,
        transformer_dim: int = 512,
        ff_mult: int = 4,
        n_layers: int = 6,
        head_dim: int = 32,
        stem_dim: int = 32,
    ):
        super().__init__()
        # shared rotary embedding for frontend blocks and transformer blocks
        rotary_embed = RotaryEmbedding(head_dim)

        # create the frontend
        # - stem
        stem = BeatThis.make_stem(spect_dim, stem_dim)
        spect_dim //= 4  # frequencies were convolved with stride 4
        # - three frontend blocks
        frontend_blocks = []
        dim = stem_dim
        for _ in range(3):
            frontend_blocks.append(
                self.make_frontend_block(dim, dim * 2, head_dim, rotary_embed)
            )
            dim *= 2
            spect_dim //= 2  # frequencies were convolved with stride 2
        frontend_blocks = nn.Sequential(*frontend_blocks)
        # - linear projection to transformer dimensionality
        concat = Rearrange("b c f t -> b t (c f)")
        linear = nn.Linear(dim * spect_dim, transformer_dim)
        self.frontend = nn.Sequential(
            OrderedDict(stem=stem, blocks=frontend_blocks, concat=concat, linear=linear)
        )

        # create the token embeddings, scaled by sqrt(transformer_dim)
        self.beat_embedding = nn.Embedding(4, transformer_dim, padding_idx=PAD)
        self.downbeat_embedding = nn.Embedding(4, transformer_dim, padding_idx=PAD)
        self.embedding_scale = math.sqrt(transformer_dim)

        # create the transformer blocks
        assert (
            transformer_dim % head_dim == 0
        ), "transformer_dim must be divisible by head_dim"
        self.transformer_blocks = roformer.Transformer(
            dim=transformer_dim,
            depth=n_layers,
            heads=transformer_dim // head_dim,
            dim_head=head_dim,
            ff_mult=ff_mult,
            rotary_embed=rotary_embed,
        )

        # create the output heads
        self.beat_head = nn.Linear(transformer_dim, 1)
        self.downbeat_head = nn.Linear(transformer_dim, 1)

    @staticmethod
    def make_frontend_block(
        in_dim: int,
        out_dim: int,
        head_dim: int,
        rotary_embed: RotaryEmbedding,
    ) -> nn.Module:
        return nn.Sequential(
            OrderedDict(
                partial=PartialFTTransformer(
                    dim=in_dim,
                    dim_head=head_dim,
                    n_head=in_dim // head_dim,
                    rotary_embed=rotary_embed,
                ),
                conv2d=nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=(2, 3),
                    stride=(2, 1),
                    padding=(0, 1),
                    bias=False,
                ),
                norm=nn.BatchNorm2d(out_dim),
                activation=nn.GELU(),
            )
        )

    def forward(self, spect, beat_tokens, downbeat_tokens):
        """
        Args:
            spect (torch.Tensor): Spectrogram of shape (batch, time, bins).
            beat_tokens (torch.Tensor): Beat token ids of shape (batch, time).
            downbeat_tokens (torch.Tensor): Downbeat token ids of shape (batch, time).

        Returns:
            dict: Beat and downbeat logits, each of shape (batch, time).
        """
        x = self.frontend(spect)
        x = (
            x
            + self.beat_embedding(beat_tokens) * self.embedding_scale
            + self.downbeat_embedding(downbeat_tokens) * self.embedding_scale
        )
        x = self.transformer_blocks(x)
        return {
            "beat": self.beat_head(x).squeeze(-1),
            "downbeat": self.downbeat_head(x).squeeze(-1),
        }

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # the released checkpoints store the attention output projections as
        # `to_out.weight` instead of `to_out.0.weight`, and no rotary frequencies
        for key in [k for k in state_dict if k.endswith(".to_out.weight")]:
            state_dict[key.replace(".to_out.", ".to_out.0.")] = state_dict.pop(key)
        for name, freqs in self.named_parameters(remove_duplicate=False):
            if name.endswith("rotary_embed.freqs"):
                state_dict.setdefault(prefix + name, freqs.detach())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class PartialFTTransformer(nn.Module):
    """
    Takes a (batch, channels, freqs, time) input, applies self-attention and
    a feed-forward block once across frequencies and once across time.
    Returns a tensor of the same shape as the input.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        n_head: int,
        rotary_embed: RotaryEmbedding,
    ):
        super().__init__()

        assert dim % dim_head == 0, "dim must be divisible by dim_head"
        assert dim // dim_head == n_head, "n_head must be equal to dim // dim_head"
        # frequency directed partial transformer
        self.attnF = Attention(
            dim, heads=n_head, dim_head=dim_head, rotary_embed=rotary_embed
        )
        self.ffF = roformer.FeedForward(dim)
        # time directed partial transformer
        self.attnT = Attention(
            dim, heads=n_head, dim_head=dim_head, rotary_embed=rotary_embed
        )
        self.ffT = roformer.FeedForward(dim)

    def forward(self, x):
        b = len(x)
        # frequency directed partial transformer
        x = rearrange(x, "b c f t -> (b t) f c")
        x = x + self.attnF(x)
        x = x + self.ffF(x)
        # time directed partial transformer
        x = rearrange(x, "(b t) f c ->(b f) t c", b=b)
        x = x + self.attnT(x)
        x = x + self.ffT(x)
        x = rearrange(x, "(b f) t c -> b c f t", b=b)
        return x
