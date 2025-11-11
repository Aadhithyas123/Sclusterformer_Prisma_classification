import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm1d, Conv1d, ReLU, Sequential, AvgPool1d
)

# ---------- Residual 1D Block (no attention path) ----------
class ResidualBlock1d(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=3, red_type="stride", k_size_pool=None):
        super().__init__()
        # padding for k in {3,5,7}
        p = {3:1, 5:2, 7:3}.get(k, k//2)

        self.red_type = red_type
        self.conv1 = Sequential(
            Conv1d(c_in, c_in, kernel_size=k, stride=1, padding=p),
            BatchNorm1d(c_in),
            ReLU(inplace=True),
        )
        self.conv2 = Sequential(
            Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p),
            BatchNorm1d(c_out),
        )

        self.downsample = None
        if c_in != c_out or s != 1:
            self.downsample = Sequential(
                Conv1d(c_in, c_out, kernel_size=1, stride=s),
                BatchNorm1d(c_out),
            )

        self.relu = ReLU(inplace=True)
        if self.red_type != "stride":
            # optional extra pooling if you don't reduce with stride
            self.pool = AvgPool1d(kernel_size=k_size_pool)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        if self.red_type != "stride":
            out = self.pool(out)
        return out


# ---------- Spectral 1D CNN trunk ----------
class CNN1DOpt(nn.Module):
    """
    Expects input (B, 1, C) where C = n_bands (spectrum length).
    Builds n_layers residual blocks; channels double each layer.
    """
    def __init__(self,
                 in_hidden: int,            # first hidden width (e.g., 32)
                 n_layers: int,
                 n_bands: int,
                 last_pooling: bool = True,
                 red_type: str = "stride",  # "stride" uses strides [2,3,3,...]
                 kernel_size=3,
                 momentum: float = 0.1):
        super().__init__()

        self.in_res = in_hidden
        self.n_layers = n_layers
        self.last_pooling = last_pooling
        # allow int or list for kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * n_layers
        else:
            assert len(kernel_size) == n_layers, "kernel_size list must match n_layers"
            self.kernel_size = kernel_size

        # Input: (B,1,C) -> (B,in_hidden,C)
        self.conv_in = Sequential(
            Conv1d(in_channels=1, out_channels=in_hidden, kernel_size=1),
            BatchNorm1d(in_hidden, momentum=momentum),
            ReLU(inplace=True),
        )

        # strides/pools
        if red_type == "stride":
            strides_list = [2] + [3] * max(0, n_layers - 1)
            k_pool_list = [1] * n_layers
        else:
            strides_list = [1] * n_layers
            k_pool_list = [2] + [3] * max(0, n_layers - 1)

        # residual layers
        layers = []
        c_in = in_hidden
        for i in range(n_layers):
            c_out = in_hidden * (2 ** (i + 1))
            layers.append(
                ResidualBlock1d(
                    c_in=c_in,
                    c_out=c_out,
                    k=self.kernel_size[i],
                    s=strides_list[i],
                    red_type=red_type,
                    k_size_pool=k_pool_list[i],
                )
            )
            c_in = c_out
        self.res_layers = nn.ModuleList(layers)

        # Adaptive pooling to 1 if requested
        if self.last_pooling:
            # heuristic: downsampling factor ~ 2 * 3^(n_layers-1)
            denom = (2 * (3 ** max(0, n_layers - 1)))
            k_size_pool = max(1, n_bands // denom)
            self.avg_pool = AvgPool1d(kernel_size=k_size_pool)

        self.out_channels = c_in  # channels after last block

    def forward(self, x):  # x: (B,1,C)
        x = self.conv_in(x)
        for i in range(self.n_layers):
            x = self.res_layers[i](x)
        if self.last_pooling:
            x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)  # (B, out_channels)
        return x


# ---------- Final classifier wrapper ----------
class BaselineCNN1D(nn.Module):
    """
    Baseline 1D CNN for HSI classification.

    Accepts:
      - (B, C, H, W)   OR
      - (B, 1, C, H, W) OR
      - (B, 1, C)      (already spectral)

    Behavior:
      - spatially averages (H,W) -> spectrum (B, C)
      - runs 1D CNN along spectral axis
      - returns logits (B, n_classes)
    """
    def __init__(self,
                 n_bands: int,
                 n_classes: int,
                 n_layers: int = 2,
                 hidden_width: int = 32,
                 kernel_size=3,
                 red_type: str = "stride",
                 last_pooling: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.trunk = CNN1DOpt(
            in_hidden=hidden_width,
            n_layers=n_layers,
            n_bands=n_bands,
            last_pooling=last_pooling,
            red_type=red_type,
            kernel_size=kernel_size,
        )
        feat_dim = self.trunk.out_channels
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes)  # logits (use CrossEntropyLoss)
        )

    def forward(self, x):
        # Normalize shapes to (B,1,C)
        if x.dim() == 5:
            # (B,1,C,H,W) -> (B,C,H,W)
            x = x.squeeze(1)
        if x.dim() == 4:
            # (B,C,H,W) -> (B,C)
            x = x.mean(dim=(-2, -1))
        if x.dim() == 2:
            # (B,C) -> (B,1,C)
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.shape[1] == 1, f"Expected (B,1,C), got {tuple(x.shape)}"

        feats = self.trunk(x)       # (B, feat_dim)
        logits = self.head(feats)   # (B, n_classes)
        return logits
