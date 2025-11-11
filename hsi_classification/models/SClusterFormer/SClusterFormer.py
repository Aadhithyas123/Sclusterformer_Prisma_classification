import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from .FS_Attention import FreqSpectralAttentionLayer
from .deform_conv_v3 import DeformConv2d
from .Pseudo3DDeformConv import DeformConv3d
from .crossAttention import FusionEncoder
import math
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


def pairwise_euclidean_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    em = torch.norm(x1.unsqueeze(-2) - x2.unsqueeze(-3), dim=-1)

    sim = torch.exp(-em)

    return sim


class Cluster3D(nn.Module):
    def __init__(self, patch_size=13, dim=256, out_dim=256, proposal_w=2, proposal_h=2, fold_w=1, fold_h=1, heads=4,
                 head_dim=24, return_center=False):
        super().__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool3d((1, proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.rule1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x):  # [b, n, c]
        x = rearrange(x, "b (w h) c -> b c w h", w=self.patch_size, h=self.patch_size)
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        x = x.unsqueeze(2)
        value = value.unsqueeze(2)
        centers = rearrange(self.centers_proposal(x),
                            'b c d w h -> b (c d) w h')
        value_centers = rearrange(self.centers_proposal(value), 'b c d w h -> b (w h) (c d)')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        sim = self.rule1(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c d w h -> b (w h) (c d)')
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (mask.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        out = rearrange(out, "b c w h -> b (w h) c")
        return out


class Cluster2D(nn.Module):
    def __init__(self, patch_size, dim=768, out_dim=768, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4,
                 head_dim=24,
                 return_center=False):
        super().__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
        self.rule2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        x = rearrange(x, "b (w h) c -> b c w h", w=self.patch_size, h=self.patch_size)
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')
        b, c, ww, hh = centers.shape
        sim = self.rule2(
            self.sim_beta +
            self.sim_alpha * pairwise_euclidean_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                mask.sum(dim=-1, keepdim=True) + 1.0)

        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        if self.fold_w > 1 and self.fold_h > 1:
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        out = rearrange(out, "b c w h -> b (w h) c")
        return out
# inside models/SClusterFormer/SClusterFormer.py (class SClusterFormer)



class GroupedPixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))

        x = x.flatten(2).transpose(1, 2)

        after_feature_map_size = self.ifm_size

        return x, after_feature_map_size


class PixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1, i=0):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1 if i == 0 else 2,
                              padding=1 if i == 0 else (3 // 2, 3 // 2))
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))

        after_feature_map_size = x.shape[2]

        x = x.flatten(2).transpose(1, 2)

        return x, after_feature_map_size


class Block(nn.Module):
    def __init__(self, patch_size, dim, num_heads, mlp_ratio=4, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Cluster3D(patch_size=patch_size, dim=dim, out_dim=dim, proposal_w=4, proposal_h=4, fold_w=1,
                              fold_h=1, heads=num_heads, head_dim=24)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Block2D(nn.Module):
    def __init__(self, patch_size, dim, num_heads, mlp_ratio=4, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Cluster2D(patch_size=patch_size, dim=dim, out_dim=dim, proposal_w=4, proposal_h=4, fold_w=1,
                              fold_h=1, heads=num_heads, head_dim=24)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

"""
    Here we have updated the implementation of multi-scale operations, achieving refined feature extraction through a lighter-weight shared weight approach.
"""
class MultiScaleDeformConv3D_FSA(nn.Module):
    def __init__(self, deform_conv: nn.Module, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.deform_conv = deform_conv
        self.out_channels = deform_conv.outc * 30

        self.scale_convs = nn.ModuleList([
            nn.Conv3d(deform_conv.outc, deform_conv.outc, kernel_size=1, groups=deform_conv.outc)
            for _ in kernel_sizes
        ])

        self.fuse = nn.Conv3d(
            deform_conv.outc * len(kernel_sizes),
            deform_conv.outc,
            kernel_size=1
        )

        self.attention = FreqSpectralAttentionLayer(
            channel=self.out_channels,
            dct_h=kernel_sizes[-1],
            dct_w=kernel_sizes[-1],
            reduction=16,
            freq_sel_method='top2'
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        feat_base = self.deform_conv(x)
        feats = []
        for idx, k in enumerate(self.kernel_sizes):
            if k == self.kernel_sizes[0]:
                feat_scaled = feat_base
            else:
                scale = k / self.kernel_sizes[0]
                feat_scaled = F.avg_pool3d(
                    feat_base,
                    kernel_size=(1, int(scale), int(scale)),
                    stride=(1, int(scale), int(scale)),
                    ceil_mode=True
                )
                feat_scaled = F.interpolate(
                    feat_scaled, size=(D, H, W),
                    mode='trilinear', align_corners=False
                )

            feat_scaled = self.scale_convs[idx](feat_scaled)
            feats.append(feat_scaled)

        multi_scale = torch.cat(feats, dim=1)
        multi_scale = self.fuse(multi_scale)

        out = feat_base + multi_scale

        B, Ck, D, H, W = out.shape
        out_4d = out.view(B, Ck * D, H, W)
        out_attn = self.attention(out_4d)
        out_final = out_attn.view(B, Ck, D, H, W)

        return out_final



class MultiScaleDeformConv2D(nn.Module):
    def __init__(self, deform_conv: nn.Module, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.deform_conv = deform_conv
        self.fuse = nn.Conv2d(deform_conv.outc * 3, deform_conv.outc, kernel_size=1)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.shape
        feats = []

        for k in self.kernel_sizes:
            scale = k / self.kernel_sizes[0]
            if scale != 1.0:
                x_scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            else:
                x_scaled = x

            feat = self.deform_conv(x_scaled)
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(feat)

        out = torch.cat(feats, dim=1)
        out = self.fuse(out) + feats[1]

        return out
    
# Inside class SClusterFormer  (add above forward)
def _run_upper(self, x4):
    """
    x4: (B,C,H,W). Try 5D first (B,1,C,H,W); if it fails, fallback to 4D.
    Cache the decision in self._upper_mode.
    """
    if not hasattr(self, "_upper_mode"):
        try:
            out = self.forward_features_Upper(x4.unsqueeze(1))   # try 5D
            self._upper_mode = "5d"
            print("[SClusterFormer] forward_features_Upper expects 5D (B,1,C,H,W)")
            return out
        except Exception as e:
            out = self.forward_features_Upper(x4)                # fallback 4D
            self._upper_mode = "4d"
            print("[SClusterFormer] forward_features_Upper expects 4D (B,C,H,W)")
            return out
    # cached path
    return self.forward_features_Upper(x4.unsqueeze(1) if self._upper_mode == "5d" else x4)

def _run_lower(self, x4):
    """
    x4: (B,C,H,W). Try 5D first (B,1,C,H,W); if it fails, fallback to 4D.
    Cache the decision in self._lower_mode.
    """
    if not hasattr(self, "_lower_mode"):
        try:
            out = self.forward_features_Lower(x4.unsqueeze(1))   # try 5D
            self._lower_mode = "5d"
            print("[SClusterFormer] forward_features_Lower expects 5D (B,1,C,H,W)")
            return out
        except Exception as e:
            out = self.forward_features_Lower(x4)                # fallback 4D
            self._lower_mode = "4d"
            print("[SClusterFormer] forward_features_Lower expects 4D (B,C,H,W)")
            return out
    # cached path
    return self.forward_features_Lower(x4.unsqueeze(1) if self._lower_mode == "5d" else x4)
def compute_class_weights(y_masked: np.ndarray, n_classes: int) -> torch.Tensor:
    # y_masked is an (H,W) map with -1 for bg, 0..K-1 for classes, or a flat 1D array of train labels 0..K-1
    if y_masked.ndim > 1:
        yy = y_masked[y_masked >= 0].ravel()
    else:
        yy = y_masked[y_masked >= 0]
    counts = np.bincount(yy, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (counts * n_classes)
    return torch.tensor(weights, dtype=torch.float32)






import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# assumes these imports exist in your module:
# from .Pseudo3DDeformConv import DeformConv3d
# from .deform_conv_v3 import DeformConv2d
# from .CrossAttention import FusionEncoder
# from .blocks import GroupedPixelEmbedding, PixelEmbedding, Block, Block2D
# (rename paths to match your repo)

class SClusterFormer(nn.Module):
    def __init__(self,
                 img_size=224,
                 pca_components=30,       # P
                 emap_components=8,       # E
                 num_classes=12,
                 num_stages=3,
                 n_groups=(32, 32, 32),
                 embed_dims=(256, 128, 64),
                 num_heads=(8, 8, 8),
                 mlp_ratios=(1, 1, 1),
                 depths=(2, 2, 2),
                 patchsize=17):
        super().__init__()

        # ---- critical channel counts for forward() ----
        self.reducedbands = int(pca_components)         # P (PCA channels)
        self.emapbands    = int(emap_components)        # E (EMAP channels)
        self.num_stages   = int(num_stages)

        print(f"[SClusterFormer.__init__] P={self.reducedbands}, E={self.emapbands}, img={img_size}")

        # ---- padding (ReplicationPad3d wants 5D: N,C,D,H,W) ----
        new_bands = max(1, math.ceil(max(1, self.reducedbands) / n_groups[0]) * n_groups[0])
        pad_d = max(0, new_bands - max(1, self.reducedbands))
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_d))

        # ===================== MDC-FSA (deformable) =====================
        # PCA path uses 3D deform conv, input 5D (B,1,P,H,W)
        self.deform_conv_layer_pca = MultiScaleDeformConv3D_FSA(
            DeformConv3d(inc=1, outc=1, kernel_size=3, padding=1, bias=False, modulation=True)
        )

        # EMAP path uses 2D deform conv, input 4D (B,E,H,W)
        if self.emapbands > 0:
            self.deform_conv_layer_emap = MultiScaleDeformConv2D(
                DeformConv2d(inc=self.emapbands, outc=30, kernel_size=9, padding=1, bias=False, modulation=True)
            )
        else:
            self.deform_conv_layer_emap = None

        # ===================== Upper Branch (3D→Transformer) =====================
        for i in range(self.num_stages):
            patch_embed = GroupedPixelEmbedding(
                in_feature_map_size=img_size,
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i]
            )
            block = nn.ModuleList([
                Block(dim=embed_dims[i],
                      num_heads=num_heads[i],
                      mlp_ratio=mlp_ratios[i],
                      drop=0.0,
                      patch_size=img_size)
                for _ in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i+1}", patch_embed)
            setattr(self, f"block{i+1}",       block)
            setattr(self, f"norm{i+1}",        norm)

        # convenience sizes for the lower (2D) branch
        self.embed_img = [
            img_size,
            math.ceil(img_size / 2),
            math.ceil(math.ceil(img_size / 2) / 2)
        ]

        # ===================== Lower Branch (2D→Transformer) =====================
        for i in range(self.num_stages):
            patch_embed2d = PixelEmbedding(
                in_feature_map_size=img_size if i == 0 else self.embed_img[i - 1],
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i],
                i=i
            )
            block2d = nn.ModuleList([
                Block2D(dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=0.0,
                        patch_size=self.embed_img[i])
                for _ in range(depths[i])
            ])
            norm2d = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed2d{i+1}", patch_embed2d)
            setattr(self, f"block2d{i+1}",       block2d)
            setattr(self, f"norm2d{i+1}",        norm2d)

        # ===================== Fusion =====================
        self.coefficients = nn.Parameter(torch.tensor([0.7], dtype=torch.float32))
        self.fusion_encoder = FusionEncoder(
            depth=1,
            h_dim=64,
            ct_attn_heads=4,
            ct_attn_depth=1,
            dropout=0.1,
            patchsize=patchsize
        )

        # project both branches to common dim before fusion/head
        self.fuse_dim = embed_dims[-1]
        self.proj_upper = nn.Identity()
        self.proj_lower = nn.Identity()
        # (we’ll detect and set real Linear the first time if dims differ)

        # final head (logits)
        self.head = nn.Linear(self.fuse_dim, num_classes)

        # runtime routing flags
        self._upper_mode = None   # "5d" or "4d"
        self._lower_mode = None   # "5d" or "4d"
        self._dbg_once   = False
        self._dbg_pca    = False
        self._dbg_emap   = False
        self._fuse_dbg   = False

    # ---------- utils: tokens ↔ grid ----------
    def _tokens_to_grid(self, x):         # (B, L, D) -> (B, D, S, S)
        B, L, D = x.shape
        S = int(math.sqrt(L))
        assert S * S == L, f"Token length {L} is not a perfect square."
        return x.transpose(1, 2).reshape(B, D, S, S)

    def _grid_to_tokens(self, x):         # (B, D, S, S) -> (B, L, D)
        B, D, S, _ = x.shape
        return x.reshape(B, D, S*S).transpose(1, 2)

    def _ensure_tokens(self, z):
        """
        Accepts:
          - (B, D, H, W)   -> tokens (B, HW, D)
          - (B, 1, D, H, W)-> squeeze to (B, D, H, W) then tokens
          - already (B, L, D) -> return as is
        """
        if z.dim() == 5:
            # (B,1,D,H,W) or (B,C,D,H,W) — squeeze channel to 1 if present
            if z.size(1) != 1:
                z = z[:, :1, ...]
            z = z.squeeze(1)
        if z.dim() == 4:
            return self._grid_to_tokens(z)
        if z.dim() == 3:
            return z
        raise RuntimeError(f"_ensure_tokens: unexpected shape {tuple(z.shape)}")

    def _align_token_grids(self, h_tokens, l_tokens, mode="min"):
        """
        Inputs: (B, Lh, D), (B, Ll, D), where Lh=Sh^2, Ll=Sl^2.
        Returns both resized to same L = Sc^2.
        """
        Bh, Lh, Dh = h_tokens.shape
        Bl, Ll, Dl = l_tokens.shape
        assert Bh == Bl, "batch mismatch"
        assert Dh == Dl, "embed dim mismatch"

        Sh = int(math.sqrt(Lh)); Sl = int(math.sqrt(Ll))
        assert Sh * Sh == Lh and Sl * Sl == Ll, "non-square token grids"

        if mode == "min":
            Sc = min(Sh, Sl)
        elif mode == "max":
            Sc = max(Sh, Sl)
        elif mode == "to_h":
            Sc = Sh
        elif mode == "to_l":
            Sc = Sl
        else:
            raise ValueError(f"unknown mode {mode}")

        Hg = self._tokens_to_grid(h_tokens)  # (B, D, Sh, Sh)
        Lg = self._tokens_to_grid(l_tokens)  # (B, D, Sl, Sl)

        if Sh != Sc:
            Hg = F.interpolate(Hg, size=(Sc, Sc), mode="bilinear", align_corners=False)
        if Sl != Sc:
            Lg = F.interpolate(Lg, size=(Sc, Sc), mode="bilinear", align_corners=False)

        Hc = self._grid_to_tokens(Hg)        # (B, Sc*Sc, D)
        Lc = self._grid_to_tokens(Lg)        # (B, Sc*Sc, D)
        return Hc, Lc

    # ---------- branches (kept close to your originals) ----------
    def forward_features_Upper(self, x):
        # expects 5D (B,1,C,H,W) then squeezes to 4D inside your original code
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block       = getattr(self, f"block{i + 1}")
            norm        = getattr(self, f"norm{i + 1}")

            x, s = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = norm(x)

            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()
        return x  # could be (B,D,H,W) or (B,L,D) depending on your blocks

    def forward_features_Lower(self, x):
        # expects 5D (B,1,C,H,W) then squeezes to 4D inside your original code
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed2d{i + 1}")
            block2d     = getattr(self, f"block2d{i + 1}")
            norm        = getattr(self, f"norm2d{i + 1}")

            x, s = patch_embed(x)
            for blk in block2d:
                x = blk(x)
            x = norm(x)

            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()
        return x  # (B,D,H,W) or (B,L,D)

    def forward(self, x):
        """
        x: (B, 1, C_total, H, W), C_total=P+E
        """
        P = self.reducedbands
        E = self.emapbands
        B, one, Ctot, H, W = x.shape
        assert one == 1, f"expected (B,1,C,H,W), got {x.shape}"
        assert Ctot == P + E, f"channel mismatch: x has {Ctot}, but P+E={P+E}"

        # -------- PCA path (3D deform) --------
        x_pca5d = x[:, :, :P, :, :]                       # (B,1,P,H,W)
        if not self._dbg_pca:
            print(f"[SClusterFormer] PCA to 3D deform: {tuple(x_pca5d.shape)} (expect B,1,{P},H,W)")
            self._dbg_pca = True
        x_pca5d = self.deform_conv_layer_pca(x_pca5d)     # (B,1,*,H,W)
        x_up    = self.forward_features_Upper(x_pca5d)    # -> (B,D,H,W) or (B,L,D)

        # -------- EMAP path (2D deform) --------
        if E > 0 and self.deform_conv_layer_emap is not None:
            x_emap5d = x[:, :, P:P+E, :, :]               # (B,1,E,H,W)
            x_emap2d = x_emap5d.squeeze(1)                # (B,E,H,W)
            if not self._dbg_emap:
                print(f"[SClusterFormer] EMAP to 2D deform: {tuple(x_emap2d.shape)} (expect B,{E},H,W)")
                self._dbg_emap = True
            x_emap2d = self.deform_conv_layer_emap(x_emap2d)  # (B,*,H,W)
            x_emap5d = x_emap2d.unsqueeze(1)              # (B,1,*,H,W)
            x_low    = self.forward_features_Lower(x_emap5d)
        else:
            # No EMAP features: create a zero branch matching x_up's spatial size after token alignment
            x_low = None

        # -------- Normalize to (B,L,D) tokens for both branches --------
        x_up_tok  = self._ensure_tokens(x_up)             # (B, Lh, Du)
        if x_low is None:
            # create a dummy zero token grid with same L and D as upper (after projection)
            Du = x_up_tok.size(-1)
            x_low_tok = torch.zeros_like(x_up_tok)
        else:
            x_low_tok = self._ensure_tokens(x_low)        # (B, Ll, Dl)

        # -------- Match embed dims (Du vs Dl) to fuse_dim --------
        Du = x_up_tok.size(-1)
        Dl = x_low_tok.size(-1)

        if isinstance(self.proj_upper, nn.Identity) and Du != self.fuse_dim:
            self.proj_upper = nn.Linear(Du, self.fuse_dim).to(x_up_tok.device)
        if isinstance(self.proj_lower, nn.Identity) and Dl != self.fuse_dim:
            self.proj_lower = nn.Linear(Dl, self.fuse_dim).to(x_low_tok.device)

        x_up_tok  = self.proj_upper(x_up_tok)             # (B, Lh, fuse_dim)
        x_low_tok = self.proj_lower(x_low_tok)            # (B, Ll, fuse_dim)

        # -------- Align token lengths (fixes 16 vs 289, etc.) --------
        x_up_tok, x_low_tok = self._align_token_grids(x_up_tok, x_low_tok, mode="min")

        if not self._fuse_dbg:
            print("[SClusterFormer] after align:",
                  "upper", tuple(x_up_tok.shape),
                  "lower", tuple(x_low_tok.shape))
            self._fuse_dbg = True

        # -------- Fuse --------
        x_cfpf = self.fusion_encoder(x_up_tok, x_low_tok)  # expects (B,L,D) with same L and D

        # -------- Pool over tokens and predict --------
        x_cfpf    = x_cfpf.mean(dim=1)       # (B, D)
        x_up_p    = x_up_tok.mean(dim=1)
        x_low_p   = x_low_tok.mean(dim=1)

        logits_up   = self.head(x_up_p)
        logits_low  = self.head(x_low_p)
        logits_cfpf = self.head(x_cfpf)

        # weighted blend (as in your code)
        x = logits_low * ((1 - self.coefficients) / 2) \
          + logits_cfpf * ((1 - self.coefficients) / 2) \
          + logits_up   * self.coefficients

        return x




