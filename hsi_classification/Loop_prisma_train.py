
#read_hsi_tif, rasterize_shapefile_to_labels, zscore_per_band, stratified_pixel_splits, make_dataloaders, 
# pad_reflect, evaluate, train_one_epoch, save_classification_map, plus a model already built outside

# prisma_tif_shp_hsi_train.py
# End-to-end HSI classification with .tif data cube + shapefile labels.
# Models: Baseline 1D-CNN (center spectrum) OR SClusterFormer (if available).
# Saves metrics and a classification map image.

import os, time, math, json, random, argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine

import geopandas as gpd
from shapely.geometry import mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

try:
    from thop import profile, clever_format
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False

# Optional: import SClusterFormer if you have it in your repo:
_HAS_SCLUSTER = False
try:
    from models.SClusterFormer.SClusterFormer import SClusterFormer
    _HAS_SCLUSTER = True
except Exception:
    pass
from .util import print_results

# -----------------------
# Utils / Config
# -----------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@dataclass
class TrainCfg:
    model: str = "cnn1d"    # "cnn1d" or "scluster"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    window: int = 15
    val_frac: float = 0.2
    test_frac: float = 0.2
    log_every: int = 50
    label_smoothing: float = 0.05
    # 1D-CNN hyperparams
    cnn_channels: Tuple[int, ...] = (32, 64, 128)
    cnn_kernel: int = 5
    # SCluster (shallow-ish) hyperparams
    d_model: int = 128
    n_heads: int = 4
    pe_dropout: float = 0.0


# -----------------------
# I/O: raster + shapefile
# -----------------------

def read_hsi_tif(path_tif: str) -> Tuple[np.ndarray, dict]:
    """
    Returns cube (H, W, C) float32; profile for transform/CRS.
    """
    with rasterio.open(path_tif) as ds:
        arr = ds.read()  # (C, H, W)
        profile = ds.profile
        nod = ds.nodata
    cube = np.moveaxis(arr, 0, -1).astype(np.float32)  # (H, W, C)
    if nod is not None:
        mask = np.any((arr == nod) | ~np.isfinite(arr), axis=0)
        cube[mask] = 0.0
    return cube, profile  # profile['transform'], profile['crs']

def rasterize_shapefile_to_labels(
    shp_path: str,
    label_field: str,
    profile: dict,
    background: int = 0
) -> np.ndarray:
    """
    Rasterizes polygons to label raster aligned with the .tif (H, W).
    """
    gdf = gpd.read_file(shp_path)
    # Reproject to raster CRS if needed:
    if gdf.crs and profile.get("crs") and gdf.crs != profile["crs"]:
        gdf = gdf.to_crs(profile["crs"])

    # Build shapes: (geom, value)
    shapes = []
    if label_field not in gdf.columns:
        raise ValueError(f"label_field '{label_field}' not found in shapefile columns {list(gdf.columns)}")

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        val = int(row[label_field])
        shapes.append((mapping(geom), val))

    H, W = profile["height"], profile["width"]
    transform: Affine = profile["transform"]
    labels = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=background,
        dtype=np.int32
    )
    return labels

def zscore_per_band(cube: np.ndarray) -> np.ndarray:
    """ cube: (H, W, C) -> z-score per band (ignore zeros-only bands gracefully). """
    H, W, C = cube.shape
    X = cube.reshape(-1, C)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    return Z.reshape(H, W, C).astype(np.float32)

def pad_reflect(x: np.ndarray, pad: int) -> np.ndarray:
    # x: (H, W, C)
    return np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


# -----------------------
# Dataset / Dataloaders
# -----------------------

class HSIPatchDataset(Dataset):
    def __init__(self, cube, gt_id, indices, window=15, return_coords=False):
        # gt_id must be -1 for bg, 0..K-1 for classes
        assert gt_id.min() >= -1, "gt must be -1 (bg) or >=0"
        self.window = window
        self.pad = window // 2
        self.indices = indices
        self.cube_p = pad_reflect(cube, self.pad)
        self.gt = gt_id
        self.H, self.W, self.C = cube.shape
        self.return_coords = return_coords

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        r, c = self.indices[i]
        y = int(self.gt[r, c])
        # safety: ensure no bg sneaks in
        if y < 0:
            raise RuntimeError("Background (-1) reached HSIPatchDataset — indices filter is wrong.")
        rp, cp = r + self.pad, c + self.pad
        patch = self.cube_p[rp-self.pad:rp+self.pad+1, cp-self.pad:cp+self.pad+1, :]
        patch = np.transpose(patch, (2, 0, 1)).copy()   # (C,H,W)
        if self.return_coords:
            return torch.from_numpy(patch), torch.tensor(y), (int(r), int(c))
        return torch.from_numpy(patch), torch.tensor(y)

def build_indices(gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      coords: (N, 2) int indices [r,c] for nonzero labels
      labels: (N,) int labels in 1..K
    """
    mask = gt >= 0
    rows, cols = np.where(mask)
    labels = gt[mask]
    coords = np.stack([rows, cols], axis=1)
    return coords, labels

'''def stratified_pixel_splits(
    gt: np.ndarray,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    coords, labels = build_indices(gt)  # labels in 1..K
    # train/val/test split with stratify on labels
    idx = np.arange(len(labels))
    i_tr, i_te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=labels)
    lab_tr = labels[i_tr]
    i_tr, i_va = train_test_split(i_tr, test_size=val_frac/(1.0 - test_frac), random_state=seed, stratify=lab_tr)

    def idx_to_rc(idxs):
        return [tuple(coords[i]) for i in idxs]

    return {
        "train": idx_to_rc(i_tr),
        "val":   idx_to_rc(i_va),
        "test":  idx_to_rc(i_te)
    }'''
    
def stratified_pixel_splits(
    gt: np.ndarray,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
    min_train: int = 0,
    min_val: int = 0,
    min_test: int = 0,
) -> dict:
    """
    Stratified per-class pixel split for labels in [-1,0..K-1].
    - Background must be < 0 (e.g., -1); only labels >=0 are split.
    - Enforces per-class minimums; if requested totals exceed available,
      it scales down the largest splits first but never below the minimums.
    Returns lists of (r,c) for 'train','val','test'.
    """
    rng = np.random.RandomState(seed)

    # collect coords/labels for foreground only
    rows, cols = np.where(gt >= 0)
    labels = gt[rows, cols].astype(np.int64)           # 0..K-1
    coords = np.stack([rows, cols], axis=1)
    K = int(labels.max()) + 1

    # buckets for indices
    idx_tr, idx_va, idx_te = [], [], []

    for k in range(K):
        cls_idx = np.where(labels == k)[0]
        n = cls_idx.size
        if n == 0:
            continue
        rng.shuffle(cls_idx)

        # desired counts
        want_tr = int(round((1.0 - val_frac - test_frac) * n))
        want_va = int(round(val_frac * n))
        want_te = n - want_tr - want_va

        # apply minimums
        want_tr = max(want_tr, min_train)
        want_va = max(want_va, min_val)
        want_te = max(want_te, min_test)

        # if too many requested, reduce the largest splits first (but keep mins)
        total = want_tr + want_va + want_te
        while total > n:
            # choose the split with largest excess above its minimum
            choices = [
                ("tr", want_tr - min_train),
                ("va", want_va - min_val),
                ("te", want_te - min_test),
            ]
            # pick the one with positive excess and largest value
            choices = [c for c in choices if c[1] > 0]
            if not choices:  # cannot reduce further; just clip
                overflow = total - n
                want_tr -= overflow
                total = n
                break
            choices.sort(key=lambda x: x[1], reverse=True)
            top = choices[0][0]
            if   top == "tr": want_tr -= 1
            elif top == "va": want_va -= 1
            else:             want_te -= 1
            total -= 1

        # if too few (rare after mins), add remainder to train
        if want_tr + want_va + want_te < n:
            want_tr = n - want_va - want_te

        # slice
        a = want_tr
        b = want_tr + want_va
        tr_k = cls_idx[:a]
        va_k = cls_idx[a:b]
        te_k = cls_idx[b:a+want_va+want_te]

        idx_tr.append(tr_k)
        idx_va.append(va_k)
        idx_te.append(te_k)

    # concat and map back to (r,c)
    idx_tr = np.concatenate(idx_tr) if idx_tr else np.empty((0,), dtype=int)
    idx_va = np.concatenate(idx_va) if idx_va else np.empty((0,), dtype=int)
    idx_te = np.concatenate(idx_te) if idx_te else np.empty((0,), dtype=int)

    def to_rc(idxs: np.ndarray):
        return [tuple(coords[i]) for i in idxs.tolist()]

    return {
        "train": to_rc(idx_tr),
        "val":   to_rc(idx_va),
        "test":  to_rc(idx_te),
    }



def make_dataloaders(cube, gt, splits: Dict[str, List[Tuple[int,int]]], batch: int, window: int, num_workers=4):
    ds_tr = HSIPatchDataset(cube, gt, splits["train"], window, return_coords=False)
    ds_va = HSIPatchDataset(cube, gt, splits["val"], window)
    ds_te = HSIPatchDataset(cube, gt, splits["test"], window)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return {"train": dl_tr, "val": dl_va, "test": dl_te}


# -----------------------
# Models
# -----------------------

class BaselineCNN1D(nn.Module):
    """
    Pixel-spectral 1D CNN over the center pixel (C bands).
    Input to forward(): (B, C, H, W) patches
    """
    def __init__(self, n_bands: int, n_classes: int,
                 channels: Tuple[int,...]=(32,64,128), kernel_size: int = 5, dropout=0.1):
        super().__init__()
        k = kernel_size; pad = k // 2
        layers = []
        c_in = 1
        for c_out in channels:
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            x = torch.zeros(1, 1, n_bands)
            y = self.conv(x)
            flat = y.shape[1] * y.shape[2]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, max(64, flat//4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(64, flat//4), n_classes)
        )

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape
        cy, cx = H//2, W//2
        spec = x[:, :, cy, cx]          # (B, C)
        spec = spec.unsqueeze(1)         # (B, 1, C)
        z = self.conv(spec)
        logits = self.head(z)
        return logits

class SClusterWrapper(nn.Module):
    def __init__(self, n_bands: int, n_classes: int, img_size: int,
                 pca_components: int, emap_components: int):
        super().__init__()
        if not _HAS_SCLUSTER:
            raise RuntimeError("models.SClusterFormer not found.")
        self.net = SClusterFormer(
            img_size=img_size,
            pca_components=pca_components,
            emap_components=emap_components,
            num_classes=n_classes,
            n_groups=[16,16,16],
            depths=[2,1,1],
            patchsize=img_size
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        return self.net(x)



# -----------------------
# Training / Eval
# -----------------------

def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    y: 1D array of train labels in [0..n_classes-1]
    returns torch.FloatTensor[n_classes]
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = counts.sum() / (counts * n_classes)   # inverse-freq normalized
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, device, criterion, scheduler=None, log_every=50):
    model.train()
    total, n = 0.0, 0
    t0 = time.time()
    for it, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        # after you compute class weights from TRAIN labels (tensor on device)
        # logits: (B, K), y: (B,)
        if torch.any((y < 0) | (y >= logits.size(1))):
            bad = y[(y < 0) | (y >= logits.size(1))].unique()
            raise RuntimeError(f"Target out of range, unique bad targets={bad.tolist()}, K={logits.size(1)}")


        loss = criterion(logits, y)
        
        

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * x.size(0); n += x.size(0)
        if (it+1) % log_every == 0:
            print(f"  iter {it+1}/{len(loader)}  loss={total/max(n,1):.4f}")
    return total/max(n,1), time.time()-t0
    

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score, f1_score
import numpy as np
import torch
@torch.no_grad()
def evaluate(model, dl, device, n_classes: int, class_ids=None):
    """
    class_ids: iterable of numeric class ids to report (defaults to 0..n_classes-1).
    """
    labels = list(range(n_classes)) if class_ids is None else list(class_ids)
    target_names = [str(c) for c in labels]   # numeric labels as strings

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)  # use only for reporting
            

            

            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # fixed-size confusion matrix over the numeric labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    # per-class recall (diag / row sum), NaN for classes absent in test
    rows = cm.sum(axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_recall = np.where(rows > 0, np.diag(cm) / rows, np.nan)

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,   # shows numeric ids as strings
        zero_division=0,
        digits=4
    )

    absent_test = [lab for lab, r in zip(labels, rows) if r == 0]

    return {
        "cm": cm,
        "oa": oa,
        "kappa": kappa,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "report": report,
        "absent_test": absent_test,
        "labels": labels,
    }


def probe_params_flops(model, C: int, H: int, W: int, device: str):
    model = model.to(device)
    sample = torch.zeros(1, C, H, W, dtype=torch.float32, device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if _HAS_THOP:
        flops, params = profile(model, inputs=(sample,), verbose=False)
        flops, params = clever_format([flops, params], "%.2f")
        return n_params, (flops, params)
    return n_params, None

def colorize_labels(lbl: np.ndarray, K: int) -> np.ndarray:
    """
    Simple colorizer: HSV ramp -> RGB for classes 1..K; 0 -> black.
    """
    H, W = lbl.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    for k in range(1, K+1):
        hue = (k-1)/max(1,K)  # [0,1)
        # quick HSV->RGB (fixed S=0.85, V=0.95)
        s, v = 0.85, 0.95
        hi = int(hue*6.0) % 6
        f = hue*6.0 - math.floor(hue*6.0)
        p = v*(1-s); q=v*(1-s*f); t=v*(1-s*(1-f))
        if   hi == 0: rgb = (v,t,p)
        elif hi == 1: rgb = (q,v,p)
        elif hi == 2: rgb = (p,v,t)
        elif hi == 3: rgb = (p,q,v)
        elif hi == 4: rgb = (t,p,v)
        else:         rgb = (v,p,q)
        out[lbl==k] = rgb
    return (np.clip(out,0,1)*255).astype(np.uint8)

def save_classification_map(pred_1K: np.ndarray, out_png: str):
    """
    pred_1K: (H,W) labels in 0..K where 0 background, colors for 1..K
    """
    K = int(pred_1K.max())
    rgb = colorize_labels(pred_1K, K)
    import imageio
    imageio.imwrite(out_png, rgb)

def debug_labels(gt: np.ndarray, title: str = "[debug labels]"):
    """Print conventions & counts to verify bg and class ids."""
    vals, cnts = np.unique(gt, return_counts=True)
    print(f"{title} unique={vals.tolist()}  n={int(cnts.sum())}")
    if (vals == -1).any():
        print("  -> detected background = -1  (classes should be 0..K-1)")
    elif (vals == 0).any():
        print("  -> detected background = 0   (classes should be 1..K)")
    else:
        print("  -> no explicit bg value detected (check your rasterization)")
    # per-class table (skip bg)
    bg_vals = {-1, 0}
    print("  per-class counts (excluding bg):")
    for v, c in zip(vals, cnts):
        if v in bg_vals: continue
        print(f"    class {int(v):>3}: {int(c)}")

import numpy as np
from collections import Counter

def _count_labels(gt_id, rc_list, K):
    labs = [gt_id[r, c] for (r, c) in rc_list]
    cnt  = Counter(labs)
    # ensure length K (0..K-1), fill missing with 0
    return np.array([cnt.get(i, 0) for i in range(K)], dtype=int)

def debug_split_stats(labels_id, splits, K, id2orig=None, title="Split stats"):
    names = ["train", "val", "test"]
    print(f"\n[{title}]")
    for name in names:
        counts = _count_labels(labels_id, splits[name], K)
        present = np.nonzero(counts)[0].tolist()
        missing = [i for i in range(K) if counts[i] == 0]
        total   = counts.sum()
        pct     = np.round(100.0 * counts / max(1, total), 2)

        if id2orig is not None:
            present_cf = [id2orig[i] for i in present]
            missing_cf = [id2orig[i] for i in missing]
            lab_header = "id(CATFOR)"
        else:
            present_cf, missing_cf = present, missing
            lab_header = "id"

        print(f"\n[{name}] total={total}")
        print(f"  present {lab_header}: {present} ({present_cf})")
        print(f"  missing {lab_header}: {missing} ({missing_cf})")
        print("  counts:", counts.tolist())
        print("  perc  :", pct.tolist())

    # Hard check: no class should be missing in any split
    for name in names:
        counts = _count_labels(labels_id, splits[name], K)
        assert (counts > 0).all(), f"[{name}] some classes are missing!"


def loop_train_test(
    hp,
    cube_tif: str,
    shapefile: str,
    label_field: str,
    model_builder,                 # (n_bands, n_classes, window, pca_components, emap_components) -> nn.Module
    outdir: str = "outputs_prisma"
):
    """
    Adds PCA (fit on TRAIN pixels only) + EMAP on one principal component (PC1 by default), then trains/evals.

    hp keys:
      run_times, window, epochs, batch_size, val_frac, test_frac, seed, lr, weight_decay, device
      use_pca (bool, default True)
      pca_components (int, default 30)
      use_emap (bool, default True)
      emap_thresholds (list[int], default [32, 64, 128, 256])
      emap_on_pc (int, default 0)   # 0-based PC index (PC1=0)
    """
    # --- std imports (kept local so this function is drop-in) ---
    import os, time, math
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.decomposition import PCA
    from scipy.ndimage import maximum_filter, minimum_filter, label as cc_label

    # --- project helpers expected to exist in your repo ---
    # read_hsi_tif: returns (cube(H,W,C), raster_profile)
    # rasterize_shapefile_to_labels: returns (H,W) with label codes
    # zscore_per_band: per-channel standardization
    # stratified_pixel_splits: build {'train','val','test'} lists of (r,c)
    # make_dataloaders: builds PyTorch loaders from cube + labels + splits
    # train_one_epoch, evaluate
    # pad_reflect, build_indices, save_classification_map, print_results
    # set_seed
    

    os.makedirs(outdir, exist_ok=True)

    # --- hyperparameters ---
    run_times   = int(hp.get('run_times', 1))
    window      = int(hp.get('window', 15))
    epochs      = int(hp.get('epochs', 30))
    batch       = int(hp.get('batch', 32))
    val_frac    = float(hp.get('val_frac', 0.2))
    test_frac   = float(hp.get('test_frac', 0.2))
    seed0       = int(hp.get('seed', 42))
    lr          = float(hp.get('lr', 1e-3))
    wd          = float(hp.get('weight_decay', 1e-4))
    device      = hp.get('device', "cuda" if torch.cuda.is_available() else "cpu")

    use_pca         = bool(hp.get('use_pca', True))
    pca_components  = int(hp.get('pca_components', 30))
    use_emap        = bool(hp.get('use_emap', True))
    emap_thresholds = list(hp.get('emap_thresholds', [32, 64, 128, 256]))
    emap_on_pc      = int(hp.get('emap_on_pc', 0))  # 0=PC1

    # ===================== Load & prepare labels =====================
    cube_raw, profile = read_hsi_tif(cube_tif)                     # (H,W,C_full)
    
    labels_raw = rasterize_shapefile_to_labels(shapefile, label_field, profile, background=0)
    debug_labels(labels_raw, title="[debug labels_raw]")

    # Remap CATFOR codes (excluding 0) -> 0..K-1 contiguous ids
    codes = np.unique(labels_raw)
    codes = codes[codes > 0]
    code2id = {int(c): i for i, c in enumerate(sorted(codes.tolist()))}
    labels_id = np.zeros_like(labels_raw, dtype=np.int32) - 1     # -1 background
    mask = labels_raw > 0
    labels_id[mask] = np.vectorize(code2id.get)(labels_raw[mask])
    n_classes = len(code2id)
    id2orig = {v: k for k, v in code2id.items()}
    with open(os.path.join(outdir, "catfor_mapping.json"), "w") as f:
        json.dump({"code2id": code2id, "id2orig": id2orig}, f, indent=2)

    debug_labels(labels_id, title="[debug labels_id]")
    print(f"[debug] n_classes={n_classes}")
    # Normalize full cube once (before PCA)
    cube_raw = zscore_per_band(cube_raw).astype(np.float32)
    H, W, C = cube_raw.shape
    print(f"[data] cube={cube_raw.shape}  classes(K)={n_classes}  labeled_px={np.count_nonzero(labels_id>=0)}")

    # For reporting
    OA, AA, KAPPA, TRAINING_TIME, TESTING_TIME = [], [], [], [], []
    ELEMENT_ACC = np.zeros((run_times, n_classes), dtype=np.float32)

    # coords of all labeled pixels (for dense map later)
    coords_all, _ = build_indices(labels_id)

    # ===================== small helpers =====================
    def fit_pca_on_train(cube_full: np.ndarray, train_rc: list, n_comp: int):
        """Fit PCA ONLY on train pixels, then transform the whole cube."""
        Xtr = np.asarray([cube_full[r, c, :] for (r, c) in train_rc], dtype=np.float32)
        pca = PCA(n_components=n_comp, whiten=True, random_state=0)
        pca.fit(Xtr)
        Z = pca.transform(cube_full.reshape(-1, cube_full.shape[2])).astype(np.float32)
        return pca, Z.reshape(H, W, n_comp)

    def _recon_dilate(seed, mask_img, iters=1):
        out = seed.copy()
        for _ in range(iters):
            out = np.minimum(mask_img, maximum_filter(out, size=3))
        return out

    def _recon_erode(seed, mask_img, iters=1):
        out = seed.copy()
        for _ in range(iters):
            out = np.maximum(mask_img, minimum_filter(out, size=3))
        return out

    def emap_area_on_gray(gray: np.ndarray, thresholds: list) -> np.ndarray:
        """
        Lightweight EMAP-style features on a single gray image:
        For each threshold T:
          - Opening-like (keep bright CC area>=T), Closing-like (keep dark CC area>=T),
          - approximate reconstruction to avoid over-smoothing.
        Returns (H,W, 2*len(T)) z-scored per channel.
        """
        Hh, Ww = gray.shape
        m = float(gray.mean())
        bin_b = (gray > m).astype(np.uint8)  # bright
        bin_d = (gray < m).astype(np.uint8)  # dark
        feats = []
        for thr in thresholds:
            # Opening-like
            lbl_b, n_b = cc_label(bin_b)
            keep_b = np.zeros_like(lbl_b, dtype=bool)
            if n_b > 0:
                areas = np.bincount(lbl_b.ravel())
                ids = np.where(areas >= thr)[0]
                ids = ids[ids != 0]
                if ids.size:
                    keep_b[np.isin(lbl_b, ids)] = True
            opened = np.where(keep_b, gray, gray.min())
            opened = np.minimum(opened, gray)
            opened = _recon_dilate(opened, gray, iters=1)

            # Closing-like
            lbl_d, n_d = cc_label(bin_d)
            keep_d = np.zeros_like(lbl_d, dtype=bool)
            if n_d > 0:
                areas = np.bincount(lbl_d.ravel())
                ids = np.where(areas >= thr)[0]
                ids = ids[ids != 0]
                if ids.size:
                    keep_d[np.isin(lbl_d, ids)] = True
            closed = np.where(keep_d, gray, gray.max())
            closed = np.maximum(closed, gray)
            closed = _recon_erode(closed, gray, iters=1)

            feats.append(opened.astype(np.float32))
            feats.append(closed.astype(np.float32))

        em = np.stack(feats, axis=-1)
        D = em.shape[-1]
        em2 = em.reshape(-1, D)
        mu, sd = em2.mean(0), em2.std(0) + 1e-6
        return ((em2 - mu) / sd).reshape(Hh, Ww, D).astype(np.float32)

    def compute_class_weights(y_np: np.ndarray, K: int) -> torch.Tensor:
        """y_np are 0..K-1 ids from TRAIN split."""
        counts = np.bincount(y_np, minlength=K).astype(np.float32)
        counts[counts == 0] = 1.0
        w = counts.sum() / (counts * K)     # inverse freq normalized
        return torch.tensor(w, dtype=torch.float32)

    # ===================== RUNS =====================
    for run_i in range(run_times):
        seed = seed0 + run_i
        set_seed(seed)
        print(f"\n===== round {run_i+1}/{run_times} (seed {seed}) =====")

        # Splits on remapped labels (background=-1 ignored by splitter)
        #splits = stratified_pixel_splits(labels_id, val_frac=val_frac, test_frac=test_frac, seed=seed)
        
        def _steal_one_of_class(labels_id, splits, src, dst, k):
            # move one (r,c) with label k from src -> dst
            for i, (r,c) in enumerate(splits[src]):
                if labels_id[r, c] == k:
                    splits[dst].append(splits[src].pop(i))
                    return True
            return False

        def ensure_val_has_all_classes(labels_id, splits, n_classes):
            # ensure every class present in val at least once, if possible
            for k in range(n_classes):
                has_in_val = any(labels_id[r, c] == k for (r, c) in splits["val"])
                if has_in_val:
                    continue
                # try to take from train first (prefer keeping test intact)
                moved = _steal_one_of_class(labels_id, splits, "train", "val", k)
                if not moved:
                    _ = _steal_one_of_class(labels_id, splits, "test", "val", k)
            return splits


        
        splits = stratified_pixel_splits(labels_id, val_frac=val_frac, test_frac=test_frac, seed=seed, min_train= 80, min_val=20, min_test=20)
        # --- Debug: check class presence and distribution in splits ---

        # after building splits:
        splits = ensure_val_has_all_classes(labels_id, splits, n_classes)

        
        id2orig = {
            0:1,  1:2,  2:3,  3:4,  4:5,  5:8,
            6:9,  7:11, 8:12, 9:13, 10:14, 11:18
            }

        # --- call right after you build 'splits' ---
        # splits = stratified_pixel_splits(labels_id, val_frac=..., test_frac=..., seed=...)
        debug_split_stats(labels_id, splits, n_classes, id2orig=id2orig, title="Post-stratified splits")
        
        # --- sanity guard: no bg (-1) and no label >= K in any split ---
        for _name in ["train", "val", "test"]:
            ys = np.array([labels_id[r, c] for (r, c) in splits[_name]], dtype=np.int64)
            bad_bg = int((ys < 0).sum())
            bad_hi = int((ys >= n_classes).sum())
            if bad_bg or bad_hi:
                print(f"[error] {_name}: bg={bad_bg}, >=K={bad_hi}, K={n_classes}")
                raise RuntimeError(f"Invalid labels in split '{_name}'")


        # ---- PCA (fit on TRAIN only) ----
        if use_pca:
            _, cube_pca = fit_pca_on_train(cube_raw, splits["train"], pca_components)
            n_spec = pca_components
        else:
            cube_pca = cube_raw
            n_spec = C

        # ---- EMAP on PC[emap_on_pc] ----
        if use_emap:
            pc_idx = int(np.clip(emap_on_pc, 0, n_spec - 1))
            gray = cube_pca[..., pc_idx]
            emap = emap_area_on_gray(gray, emap_thresholds)     # (H,W,E)
            cube_for_model = np.concatenate([cube_pca, emap], axis=2).astype(np.float32)
            n_emap = emap.shape[-1]
        else:
            cube_for_model = cube_pca.astype(np.float32)
            n_emap = 0
            
        def _assert_split_ok(name, rc_list, gt_id, K):
            ys = np.array([gt_id[r, c] for (r, c) in rc_list], dtype=np.int64)
            assert (ys >= 0).all(), f"{name}: has background (-1) samples!"
            assert ys.max() < K,    f"{name}: has label >= K (max={ys.max()}, K={K})"
            assert ys.min() >= 0,   f"{name}: has negative label (min={ys.min()})"

        _assert_split_ok("train", splits["train"], labels_id, n_classes)
        _assert_split_ok("val",   splits["val"],   labels_id, n_classes)
        _assert_split_ok("test",  splits["test"],  labels_id, n_classes)

        # ---- Dataloaders on PCA+EMAP stack ----
        dls = make_dataloaders(cube_for_model, labels_id, splits,
                               batch=batch, window=window, num_workers=4)

        # ---- Model ----
        n_bands_for_model = cube_for_model.shape[2]
        print(f"[loop] n_spec={n_spec if use_pca else 0}, n_emap={n_emap}, n_bands_for_model={n_bands_for_model}")
        
        model = model_builder(
            n_bands_for_model, n_classes, window,
            pca_components=(n_spec if use_pca else 0),
            emap_components=n_emap
        ).to(device)


        optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        total_steps = epochs * max(1, len(dls["train"]))
        warmup_steps = max(10, len(dls["train"]))
        def lr_lambda(step):
            if step < warmup_steps: return float(step) / float(max(1, warmup_steps))
            prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # ---- Class weights from TRAIN labels ----
        train_coords = splits["train"]
        labs_train = np.array([labels_id[r, c] for (r, c) in train_coords], dtype=np.int64)

        # debug + explicit construction (replaces compute_class_weights for transparency)
        counts = np.bincount(labs_train, minlength=n_classes).astype(np.float32)
        print("[debug] train per-class counts:", counts.tolist())
        counts[counts == 0] = 1.0
        w = counts.sum() / (counts * n_classes)     # inverse-freq normalized
        w = np.clip(w, 1e-3, 1e3)                   # keep sane bounds
        class_w = torch.tensor(w, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05).to(device)
        print("[debug] class weights:", class_w.detach().cpu().numpy().round(4).tolist())


        # ---- Train (keep best by val macro-F1) ----
        t0 = time.perf_counter()
        best_val_f1, best_state = -1.0, None
        for ep in range(1, epochs + 1):
            print(f"--- epoch {ep}/{epochs} ---")
            tr_loss, tr_time = train_one_epoch(model, dls["train"], optimizer, device, criterion, scheduler, log_every=50)

            # quick debug on first val batch
            with torch.no_grad():
                for i, (xv, yv) in enumerate(dls["val"]):
                    xv = xv.to(device, non_blocking=True)
                    logits = model(xv)
                    pred = logits.argmax(dim=1)
                    if i == 0:
                        u, cts = torch.unique(pred, return_counts=True)
                        print(f"[debug] first val batch preds: {u.tolist()} counts {cts.tolist()}")
                    break

            va_metrics = evaluate(model, dls["val"], device, n_classes)
            print(f"  train_loss={tr_loss:.4f}  time/epoch={tr_time:.1f}s  val_macroF1={va_metrics['macro_f1']:.4f}  OA={va_metrics['oa']:.4f}")

            if va_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = va_metrics["macro_f1"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        t1 = time.perf_counter()
        TRAINING_TIME.append(t1 - t0)

        # ---- Test ----
        model.load_state_dict(best_state)
        t2 = time.perf_counter()
        te_metrics = evaluate(model, dls["test"], device, n_classes)   # numeric labels 0..K-1
        t3 = time.perf_counter()
        TESTING_TIME.append(t3 - t2)

        print(f"OA: {te_metrics['oa']}")
        print(f"Kappa: {te_metrics['kappa']}")
        print(f"Macro-F1: {te_metrics['macro_f1']}")
        print(te_metrics["report"])

        # Aggregate scores
        OA.append(te_metrics["oa"])
        KAPPA.append(te_metrics["kappa"])
        per_cls_recall = te_metrics["per_class_recall"]          # length K; NaN for empty classes
        AA.append(float(np.nanmean(per_cls_recall)))
        ELEMENT_ACC[run_i, :] = np.nan_to_num(per_cls_recall, nan=0.0)

        if te_metrics.get("absent_test"):
            print(f"[warn] No test samples for classes: {te_metrics['absent_test']}")

        # ---- Dense classification map over all labeled pixels (same PCA+EMAP cube) ----
        model.eval()
        pad = window // 2
        cube_padded = pad_reflect(cube_for_model, pad)
        preds_full = np.zeros_like(labels_id, dtype=np.int32)    # 0..K (0 reserved for background in map view)
        with torch.no_grad():
            for (r, c) in coords_all:
                rp, cp = r + pad, c + pad
                patch = cube_padded[rp - pad:rp + pad + 1, cp - pad:cp + pad + 1, :]
                patch = np.transpose(patch, (2, 0, 1))[None, ...]      # (1,C,H,W)
                logits = model(torch.from_numpy(patch).float().to(device))
                cls = int(logits.argmax(1).item()) + 1   # +1 only for the PNG map, not for loss
     # +1 for visualization map (bg=0)
                preds_full[r, c] = cls

        map_path = os.path.join(outdir, f"clsmap_run{run_i+1}.png")
        save_classification_map(preds_full, map_path)
        print(f"Saved classification map -> {map_path}")

    # ===================== Final aggregation =====================
    print_results(
        n_classes,
        np.array(OA),
        np.array(AA),
        np.array(KAPPA),
        ELEMENT_ACC,
        np.array(TRAINING_TIME),
        np.array(TESTING_TIME)
    )

def _labels_from_indices(labels_1K: np.ndarray, rc_list: list[tuple]) -> np.ndarray:
    out = np.zeros_like(labels_1K)
    for (r, c) in rc_list:
        out[r, c] = labels_1K[r, c]
    return out

def remap_labels_contiguous(gt: np.ndarray, background: int = 0):
    """
    Map unique labels (excluding background) to 0..K-1 contiguous ids.
    Returns: mapped_gt, id2orig, orig2id
    """
    orig = np.unique(gt)
    orig = orig[orig != background]
    orig_sorted = np.sort(orig)
    orig2id = {int(o): i for i, o in enumerate(orig_sorted)}
    id2orig = {i: int(o) for i, o in enumerate(orig_sorted)}

    mapped = np.zeros_like(gt, dtype=np.int32) - 1   # -1 for background
    mask = (gt != background)
    mapped[mask] = np.vectorize(orig2id.get)(gt[mask])

    n_classes = len(orig_sorted)
    return mapped, id2orig, orig2id, n_classes

