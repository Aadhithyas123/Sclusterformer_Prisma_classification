import torch
from einops import rearrange
from torch import nn


class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CTAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        x12 = torch.chunk(x, chunks=2, dim=0)
        x1 = x12[0]
        x2 = x12[1]

        q = self.to_q(x2)
        k = self.to_k(x1)
        v = self.to_v(x1)
        qkv = []
        qkv.append(q)
        qkv.append(k)
        qkv.append(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.do1(out)
        return out

class CT_Transformer(nn.Module):
    def __init__(self, h_dim, depth, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                LayerNormalize(h_dim, CTAttention(h_dim, heads=heads, dropout=dropout))
            )

    def forward(self, h_tokens):

        for h_attend_lg in self.layers:
            h_tokens = h_attend_lg(h_tokens)

        return h_tokens


class FusionEncoder(nn.Module):
    """
    Fuse two token streams (high/low) of equal shape (B, L, D) without changing L.
    Keeps your constructor signature for compatibility.
    """
    def __init__(self, depth, h_dim, ct_attn_heads, ct_attn_depth, dropout=0.1, patchsize=13):
        super().__init__()
        self.h_dim = h_dim

        # simple feature mixer after concat([H,L] along D)
        self.mix = nn.Sequential(
            nn.LayerNorm(2 * h_dim),
            nn.Linear(2 * h_dim, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # optional cross-attention block(s)
        self.mha_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=h_dim, num_heads=ct_attn_heads, dropout=dropout, batch_first=True)
            for _ in range(max(1, ct_attn_depth))
        ])
        self.ffn = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(h_dim)

    def forward(self, h_tokens: torch.Tensor, l_tokens: torch.Tensor) -> torch.Tensor:
        """
        h_tokens, l_tokens: (B, L, D)
        returns: fused tokens (B, L, D)
        """
        assert h_tokens.ndim == 3 and l_tokens.ndim == 3, \
            f"Expected (B,L,D), got {h_tokens.shape} and {l_tokens.shape}"
        B1, L1, D1 = h_tokens.shape
        B2, L2, D2 = l_tokens.shape
        assert B1 == B2, f"Batch mismatch {B1} vs {B2}"
        assert L1 == L2, f"Token length mismatch {L1} vs {L2} (inputs must be pre-aligned)"
        assert D1 == D2 == self.h_dim, f"Embed dim mismatch {D1},{D2} vs {self.h_dim}"

        # 1) feature concat along channel, project back to D
        z = torch.cat([h_tokens, l_tokens], dim=2)   # (B, L, 2D)
        z = self.mix(z)                               # (B, L, D)

        # 2) cross-attention refinement (self-attn over fused tokens)
        x = z
        for mha in self.mha_blocks:
            # pre-norm + residual
            x_norm = nn.functional.layer_norm(x, (self.h_dim,))
            attn_out, _ = mha(x_norm, x_norm, x_norm)   # (B, L, D)
            x = x + attn_out
            x = x + self.ffn(x)                         # FFN residual

        return self.norm_out(x)                         # (B, L, D)
