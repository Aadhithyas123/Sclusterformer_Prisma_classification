from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
# ============================================================
# Active Learning Strategies (Random, Uncertainty, KMeans, Hybrid)
# ============================================================

class BaseActiveLearningStrategy:
    name: str = "base"

    def select(
        self,
        model: nn.Module,
        cube: np.ndarray,
        labels_id: np.ndarray,
        pool_coords: List[Tuple[int, int]],
        window: int,
        device: str,
        query_size: int,
        batch_size: int = 256,
        num_workers: int = 8,
    ) -> Tuple[List[Tuple[int, int]], float, float]:
        """
        Return: (selected_coords, T_query_inference, T_query_selection)
        """
        raise NotImplementedError


def _collect_pool_logits_and_entropy(
    model: nn.Module,
    cube: np.ndarray,
    labels_id: np.ndarray,
    pool_coords: List[Tuple[int, int]],
    window: int,
    device: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the model on all pool coords (in the same order as pool_coords) and returns:
      logits_arr: [N, C]
      entropy_arr: [N]
      T_inf: inference time (s)
    The mapping index i -> coordinate is simply pool_coords[i].
    """
    if len(pool_coords) == 0:
        return np.zeros((0,)), np.zeros((0,)), 0.0

    # no need for return_coords=True, we rely on ordering == pool_coords
    ds_pool = HSIPatchDataset(cube, labels_id, pool_coords, window, return_coords=False)
    dl_pool = DataLoader(
        ds_pool,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    all_logits: List[np.ndarray] = []
    all_entropy: List[np.ndarray] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, y in dl_pool:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)                      # [B, C]
            probs = torch.softmax(logits, dim=1)    # [B, C]
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)  # [B]

            all_logits.append(logits.cpu().numpy())
            all_entropy.append(entropy.cpu().numpy())

    T_inf = time.perf_counter() - t0

    logits_arr = np.concatenate(all_logits, axis=0)
    entropy_arr = np.concatenate(all_entropy, axis=0)

    # sanity: should match pool size
    assert logits_arr.shape[0] == len(pool_coords)
    assert entropy_arr.shape[0] == len(pool_coords)

    return logits_arr, entropy_arr, T_inf




class RandomSamplingStrategy(BaseActiveLearningStrategy):
    name = "random"

    def select(
        self,
        model,
        cube,
        labels_id,
        pool_coords,
        window,
        device,
        query_size,
        batch_size=256,
        num_workers=8,
    ):
        if len(pool_coords) == 0 or query_size <= 0:
            return [], 0.0, 0.0
        query_size = min(query_size, len(pool_coords))
        selected = random.sample(pool_coords, query_size)
        return selected, 0.0, 0.0  # no inference/selection overhead


class UncertaintySamplingStrategy(BaseActiveLearningStrategy):
    name = "uncertainty"

    def select(
        self,
        model,
        cube,
        labels_id,
        pool_coords,
        window,
        device,
        query_size,
        batch_size=256,
        num_workers=8,
    ):
        if len(pool_coords) == 0 or query_size <= 0:
            return [], 0.0, 0.0

        logits_arr, entropy_arr, T_inf = _collect_pool_logits_and_entropy(
            model, cube, labels_id, pool_coords, window, device,
            batch_size=batch_size, num_workers=num_workers
        )

        query_size = min(query_size, entropy_arr.shape[0])
        t_sel0 = time.perf_counter()
        idx_sorted = np.argsort(-entropy_arr)  # descending entropy
        chosen_idx = idx_sorted[:query_size]

        # map indices back to coords via pool_coords
        selected_coords = [pool_coords[i] for i in chosen_idx]
        T_sel = time.perf_counter() - t_sel0

        return selected_coords, T_inf, T_sel



class KMeansDiversityStrategy(BaseActiveLearningStrategy):
    """
    Diversity sampling via KMeans over logit features (no labeled-set dependence).
    """
    name = "kmeans"

    def select(
        self,
        model,
        cube,
        labels_id,
        pool_coords,
        window,
        device,
        query_size,
        batch_size=256,
        num_workers=8,
    ):
        if len(pool_coords) == 0 or query_size <= 0:
            return [], 0.0, 0.0

        logits_arr, entropy_arr, T_inf = _collect_pool_logits_and_entropy(
            model, cube, labels_id, pool_coords, window, device,
            batch_size=batch_size, num_workers=num_workers
        )

        N = logits_arr.shape[0]
        query_size = min(query_size, N)
        if query_size == 0:
            return [], T_inf, 0.0

        t_sel0 = time.perf_counter()
        kmeans = KMeans(n_clusters=query_size, random_state=0, n_init="auto").fit(logits_arr)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        selected_idx = []
        for k in range(query_size):
            cluster_idx = np.where(labels == k)[0]
            if cluster_idx.size == 0:
                continue
            pts = logits_arr[cluster_idx]
            center = centers[k]
            dists = np.linalg.norm(pts - center, axis=1)
            j = cluster_idx[np.argmin(dists)]
            selected_idx.append(j)

        selected_idx = list(dict.fromkeys(selected_idx))
        if len(selected_idx) > query_size:
            selected_idx = selected_idx[:query_size]

        # map through pool_coords
        selected_coords = [pool_coords[j] for j in selected_idx]
        T_sel = time.perf_counter() - t_sel0
        return selected_coords, T_inf, T_sel



class HybridUncertaintyKMeansStrategy(BaseActiveLearningStrategy):
    """
    2-step hybrid:
      1) keep top-M most uncertain points (M = topk_factor * query_size)
      2) run KMeans diversity on that subset
    Hybrid-UK: Uncertainty-filtered k-means diversity.
    Step 1: rank pool by predictive entropy, keep top M = topk_factor * Q.
    Step 2: run k-means (k=Q) in logit space on that subset and pick
            medoids (closest to cluster centers).
    """
    name = "hybrid"

    def __init__(self, topk_factor: int = 10):
        self.topk_factor = topk_factor

    def select(
        self,
        model,
        cube,
        labels_id,
        pool_coords,
        window,
        device,
        query_size,
        batch_size=256,
        num_workers=8,
    ):
        if len(pool_coords) == 0 or query_size <= 0:
            return [], 0.0, 0.0

        logits_arr, entropy_arr, T_inf = _collect_pool_logits_and_entropy(
            model, cube, labels_id, pool_coords, window, device,
            batch_size=batch_size, num_workers=num_workers
        )

        N = logits_arr.shape[0]
        query_size = min(query_size, N)
        if query_size == 0:
            return [], T_inf, 0.0

        # Step 1: keep top-M uncertain
        M = min(N, self.topk_factor * query_size)
        idx_sorted = np.argsort(-entropy_arr)   # high -> low
        top_idx = idx_sorted[:M]
        feats_sub = logits_arr[top_idx]

        # Step 2: KMeans on subset
        t_sel0 = time.perf_counter()
        kmeans = KMeans(n_clusters=query_size, random_state=0, n_init="auto").fit(feats_sub)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        selected_idx_local = []
        for k in range(query_size):
            cluster_idx = np.where(labels == k)[0]
            if cluster_idx.size == 0:
                continue
            pts = feats_sub[cluster_idx]
            center = centers[k]
            dists = np.linalg.norm(pts - center, axis=1)
            j_local = cluster_idx[np.argmin(dists)]
            selected_idx_local.append(j_local)

        selected_idx_local = list(dict.fromkeys(selected_idx_local))
        if len(selected_idx_local) > query_size:
            selected_idx_local = selected_idx_local[:query_size]

        # convert local indices in feats_sub back to global indices in pool_coords
        selected_coords = [pool_coords[top_idx[j]] for j in selected_idx_local]
        T_sel = time.perf_counter() - t_sel0
        return selected_coords, T_inf, T_sel
import numpy as np

def _softmax_np(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Stable softmax in NumPy.
    logits: (N, K) array
    returns: (N, K) array of probabilities
    """
    # subtract max for numerical stability
    z = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=axis, keepdims=True)

class ClassBalancedEntropyStrategy(BaseActiveLearningStrategy):
    """
    CB-ENT: Class-balanced entropy.
    Scores = predictive entropy * inverse predicted-class frequency
    (computed from current labeled set).
    """
    name = "cb_entropy"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # power for rarity weighting

    def select(
        self,
        model,
        cube,
        labels_id,
        pool_coords,
        window,
        device,
        query_size,
        batch_size=256,
        num_workers=8,
        labeled_coords=None,  # <-- need this
    ):
        if len(pool_coords) == 0 or query_size <= 0:
            return [], 0.0, 0.0

        # 1) get logits & probs & entropy on pool
        logits_arr, entropy_arr, T_inf = _collect_pool_logits_and_entropy(
            model, cube, labels_id, pool_coords, window, device,
            batch_size=batch_size, num_workers=num_workers
        )
        probs = _softmax_np(logits_arr)          # (N,K)
        yhat  = probs.argmax(1)                  # predicted class

        # 2) compute class frequencies from LABELED set
        if labeled_coords is None or len(labeled_coords) == 0:
            # fallback: just entropy
            scores = entropy_arr
        else:
            # labels_id: -1 for unlabeled, 0..K-1 for classes
            labs_L = np.array([labels_id[r, c] for (r, c) in labeled_coords], dtype=np.int64)
            K = probs.shape[1]
            counts = np.bincount(labs_L, minlength=K).astype(np.float32)
            freq = counts / max(1.0, counts.sum())
            freq = np.clip(freq, 1e-9, 1.0)
            rarity = (1.0 / freq) ** self.alpha
            scores = entropy_arr * rarity[yhat]

        # 3) select top-K
        N = len(pool_coords)
        query_size = min(query_size, N)
        idx_sorted = np.argsort(-scores)
        chosen_idx = idx_sorted[:query_size]
        selected_coords = [pool_coords[i] for i in chosen_idx]
        return selected_coords, T_inf, 0.0


def make_al_strategy(name: str) -> BaseActiveLearningStrategy:
    name = name.lower()
    if name in ("random", "rand"):
        return RandomSamplingStrategy()
    if name in ("uncertainty", "entropy"):
        return UncertaintySamplingStrategy()
    if name in ("kmeans", "diversity", "coreset"):
        return KMeansDiversityStrategy()
    if name in ("hybrid", "uncertainty_kmeans"):
        return HybridUncertaintyKMeansStrategy()
    if name in ("cbe", "ClassBalancedEntropy"):
        return ClassBalancedEntropyStrategy()
    raise ValueError(f"Unknown active learning strategy: {name}")

# ============================================================
# Training helper for Active Learning
# ============================================================

def train_model_for_al(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    device: str,
    n_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    log_every: int = 50,
) -> Tuple[Dict[str, torch.Tensor], dict, float]:
    """
    Train model with early selection by validation macro-F1.
    Returns: (best_state_dict, best_val_metrics, T_train_total)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * max(1, len(dl_train))
    warmup_steps = max(10, len(dl_train))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0).to(device)

    best_val_f1 = -1.0
    best_state = None
    best_val_metrics = None

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        print(f"[AL] epoch {ep}/{epochs}")
        tr_loss, _ = train_one_epoch(
            model,
            dl_train,
            optimizer,
            device,
            criterion,
            scheduler,
            log_every=log_every,
            proto_mem=None,
            class_w=class_weights,
        )
        val_metrics = evaluate(model, dl_val, device, n_classes)
        print(
            f"[AL]   train_loss={tr_loss:.4f} "
            f"val_macroF1={val_metrics['macro_f1']:.4f}  OA={val_metrics['oa']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics

    T_train = time.perf_counter() - t0
    return best_state, best_val_metrics, T_train


# ============================================================
# Metrics plots for Active Learning
# ============================================================

def plot_active_learning_curves(
    histories: Dict[str, List[dict]],
    outdir: str,
    suffix: str = "",
):
    """
    histories: dict[str -> list of round dicts]
      each round dict must contain:
        'n_labels', 'macro_f1', 'oa', 'kappa', 'aa', 'T_total'
    Saves:
      al_macroF1_vs_labels*.png
      al_macroF1_vs_time*.png
    """
    os.makedirs(outdir, exist_ok=True)

    # --- Macro-F1 vs #labels ---
    plt.figure()
    for name, hist in histories.items():
        n_labels = [h["n_labels"] for h in hist]
        macro_f1 = [h["macro_f1"] for h in hist]
        plt.plot(n_labels, macro_f1, marker="o", label=name)
    plt.xlabel("# labeled train samples")
    plt.ylabel("Macro-F1 (test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"al_macroF1_vs_labels{suffix}.png"), dpi=150)
    plt.close()

    # --- Macro-F1 vs cumulative time ---
    plt.figure()
    for name, hist in histories.items():
        cum_times = []
        acc = 0.0
        for h in hist:
            acc += h["T_total"]
            cum_times.append(acc)
        macro_f1 = [h["macro_f1"] for h in hist]
        plt.plot(cum_times, macro_f1, marker="o", label=name)
    plt.xlabel("Cumulative time (s)")
    plt.ylabel("Macro-F1 (test)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"al_macroF1_vs_time{suffix}.png"), dpi=150)
    plt.close()
# ============================================================
# Core Active Learning loop for a single run
# ============================================================

def run_active_learning_single_run(
    cube_for_model: np.ndarray,
    labels_id: np.ndarray,
    splits: Dict[str, List[Tuple[int, int]]],
    n_classes: int,
    model_builder,
    device: str,
    window: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    pca_components: int,
    emap_components: int,
    al_strategy_name: str,
    al_init_frac: float,
    al_target_frac: float,
    al_rounds: int,
    al_query_batch: int,
    run_index: int = 0,
) -> List[dict]:
    """
    Runs pool-based AL on the TRAIN split and evaluates on TEST at each round.
    Returns history: list of round dicts with metrics and timings.
    """

    # --- fixed val/test loaders ---
    ds_val = HSIPatchDataset(cube_for_model, labels_id, splits["val"], window)
    ds_test = HSIPatchDataset(cube_for_model, labels_id, splits["test"], window)

    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    # --- full train coords and labels ---
    train_coords_full = list(splits["train"])
    N_train = len(train_coords_full)
    y_train_full = np.array(
        [labels_id[r, c] for (r, c) in train_coords_full],
        dtype=np.int64,
    )

    # --- initial labeled vs pool (stratified wrt classes) ---
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=al_init_frac,
        random_state=42 + run_index,
    )
    idx = np.arange(N_train)
    labeled_coords: List[Tuple[int, int]] = []
    pool_coords: List[Tuple[int, int]] = []

    for lab_idx, pool_idx in sss.split(idx, y_train_full):
        labeled_coords = [train_coords_full[i] for i in lab_idx]
        pool_coords = [train_coords_full[i] for i in pool_idx]

    target_labels = max(len(labeled_coords) + 1, int(al_target_frac * N_train))
    per_round = max(1, (target_labels - len(labeled_coords)) // max(1, al_rounds))

    print(
        f"[AL] N_train={N_train}, init_labeled={len(labeled_coords)}, "
        f"target_labels={target_labels}, ~per_round={per_round}"
    )

    strategy = make_al_strategy(al_strategy_name)
    history: List[dict] = []
    last_T_query_inf = 0.0
    last_T_query_sel = 0.0

    round_idx = 0
    while True:
        n_labels = len(labeled_coords)
        print(
            f"[AL] Round {round_idx} — labeled={n_labels}, "
            f"pool={len(pool_coords)}"
        )

        # --------------------- build train loader ---------------------
        ds_train = HSIPatchDataset(
            cube_for_model, labels_id, labeled_coords, window
        )
        dl_train = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        # --------------------- build model ----------------------------
        n_bands_for_model = cube_for_model.shape[2]
        model = model_builder(
            n_bands_for_model,
            n_classes,
            window,
            pca_components=pca_components,
            emap_components=emap_components,
        ).to(device)

        # class weights from CURRENT labeled set
        y_labeled = np.array(
            [labels_id[r, c] for (r, c) in labeled_coords], dtype=np.int64
        )
        counts = np.bincount(y_labeled, minlength=n_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        w = counts.sum() / (counts * n_classes)
        w = np.clip(w, 1e-3, 1e3)
        class_w = torch.tensor(w, dtype=torch.float32, device=device)

        # --------------------- train & select best by val -------------
        best_state, best_val_metrics, T_train = train_model_for_al(
            model,
            dl_train,
            dl_val,
            device,
            n_classes,
            epochs,
            lr,
            weight_decay,
            class_w,
            log_every=50,
        )
        model.load_state_dict(best_state)

        # --------------------- test ---------------------------
        t2 = time.perf_counter()
        te_metrics = evaluate(model, dl_test, device, n_classes)
        T_test = time.perf_counter() - t2
        aa = float(np.nanmean(te_metrics["per_class_recall"]))

        print(
            f"[AL/Test] round={round_idx} "
            f"OA={te_metrics['oa']:.4f}  AA={aa:.4f}  "
            f"Kappa={te_metrics['kappa']:.4f}  "
            f"Macro-F1={te_metrics['macro_f1']:.4f}"
        )

        history.append(
            {
                "round": round_idx,
                "n_labels": n_labels,
                "oa": te_metrics["oa"],
                "aa": aa,
                "kappa": te_metrics["kappa"],
                "macro_f1": te_metrics["macro_f1"],
                "per_class_recall": te_metrics["per_class_recall"],
                "T_train": T_train,
                "T_test": T_test,
                "T_query_inf": last_T_query_inf if round_idx > 0 else 0.0,
                "T_query_sel": last_T_query_sel if round_idx > 0 else 0.0,
                "T_total": T_train + T_test + (last_T_query_inf + last_T_query_sel if round_idx > 0 else 0.0),
            }
        )

        # stopping conditions
        if (
            len(labeled_coords) >= target_labels
            or len(pool_coords) == 0
            or round_idx >= al_rounds
        ):
            break

        # --------------------- query for next round -------------------
        remaining_to_target = target_labels - len(labeled_coords)
        query_size = min(per_round, remaining_to_target, len(pool_coords))
        print(f"[AL]   querying {query_size} new labels with '{strategy.name}'")

        new_coords, last_T_query_inf, last_T_query_sel = strategy.select(
            model=model,
            cube=cube_for_model,
            labels_id=labels_id,
            pool_coords=pool_coords,
            window=window,
            device=device,
            query_size=query_size,
            batch_size=al_query_batch,
            num_workers=8,
        )

        # update labeled / pool sets
        labeled_coords.extend(new_coords)
        new_set = set(new_coords)
        pool_coords = [rc for rc in pool_coords if rc not in new_set]

        round_idx += 1

    return history


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

def train_one_epoch(model, loader, optimizer, device, criterion,
                    scheduler=None, log_every=50, proto_mem=None, class_w=None):
    model.train()
    total, n = 0.0, 0
    t0 = time.time()

    for it, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)

        # base loss
        loss = criterion(logits, y)

        # === PCL (optional) ===
        if proto_mem is not None:
            backbone = getattr(model, "net", model)
            if hasattr(backbone, "_last_fused"):
                feats = backbone._last_fused                 # (B,D) with grad
                cw_for_pcl = class_w if class_w is not None else None
                pcl = proto_mem.loss(feats, y, class_weights=cw_for_pcl)
                loss = loss + 0.2 * pcl

        # step
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()  # step once per iter

        # update prototypes with no grad
        if proto_mem is not None:
            backbone = getattr(model, "net", model)
            if hasattr(backbone, "_last_fused"):
                with torch.no_grad():
                    proto_mem.update(backbone._last_fused.detach(), y)

        # logging
        total += loss.item() * x.size(0)
        n += x.size(0)
        if (it + 1) % log_every == 0:
            print(f"  iter {it+1}/{len(loader)}  loss={total / max(n,1):.4f}")

    return total / max(n, 1), time.time() - t0

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
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def print_results(
    n_class,
    oa,
    aa,
    kappa,
    macro_f1,
    class_acc,
    traintime,
    testtime,
    querytime=None,
):
    """
    oa, aa, kappa, macro_f1: 1D arrays over runs (values in [0,1])
    class_acc: [n_runs, n_class] per-class recall (or accuracy)
    traintime, testtime, querytime: per-run times (seconds)
    """
    # mean/std in %
    mean_oa = format(np.mean(oa * 100), '.2f')
    std_oa  = format(np.std(oa * 100),  '.2f')

    mean_aa = format(np.mean(aa * 100), '.2f')
    std_aa  = format(np.std(aa * 100),  '.2f')

    mean_kappa = format(np.mean(kappa * 100), '.2f')
    std_kappa  = format(np.std(kappa * 100),  '.2f')

    mean_macro = format(np.mean(macro_f1 * 100), '.2f')
    std_macro  = format(np.std(macro_f1 * 100),  '.2f')

    print('\n===== Aggregated results over runs =====')
    print('train_time: mean =', np.mean(traintime), 's, std =', np.std(traintime))
    print('test_time : mean =', np.mean(testtime), 's, std =', np.std(testtime))

    if querytime is not None:
        print('query_time: mean =', np.mean(querytime), 's, std =', np.std(querytime))

    print('\nPer-class recall (mean ± std over runs):')
    for i in range(n_class):
        m = np.mean(class_acc[:, i]) * 100
        s = np.std(class_acc[:, i]) * 100
        print(f'  Class {i+1}: {m:.2f}±{s:.2f}')

    print('\nOA mean:', mean_oa, 'std:', std_oa)
    print('AA mean:', mean_aa, 'std:', std_aa)
    print('Kappa mean:', mean_kappa, 'std:', std_kappa)
    print('Macro-F1 mean:', mean_macro, 'std:', std_macro)




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



def make_dataloaders(cube, gt, splits: Dict[str, List[Tuple[int,int]]], batch: int, window: int, num_workers=8):
    ds_tr = HSIPatchDataset(cube, gt, splits["train"], window, return_coords=False)
    ds_va = HSIPatchDataset(cube, gt, splits["val"], window)
    ds_te = HSIPatchDataset(cube, gt, splits["test"], window)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return {"train": dl_tr, "val": dl_va, "test": dl_te}

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

from collections import defaultdict

def _count_per_split_per_class(labels_id, splits, K):
    counts = {name: np.zeros(K, dtype=int) for name in ("train","val","test")}
    for name in ("train","val","test"):
        for (r,c) in splits[name]:
            k = int(labels_id[r,c])
            if 0 <= k < K:
                counts[name][k] += 1
    return counts

def _move_one(labels_id, splits, src, dst, k):
    """Move one sample of class k from src to dst. Return True if moved."""
    for i, (r,c) in enumerate(splits[src]):
        if labels_id[r,c] == k:
            splits[dst].append(splits[src].pop(i))
            return True
    return False

def _feasible_min_presence(n_total, split_names=("train","val","test")):
    """
    Given total samples for a class, return dict of desired minimum presence per split.
    If n_total>=3 -> 1 in each split
    If n_total==2 -> 1 in train + 1 in test (val can be 0)
    If n_total==1 -> 1 in train only
    If n_total==0 -> all 0
    """
    want = {n:0 for n in split_names}
    if n_total >= 3:
        for n in split_names: want[n] = 1
    elif n_total == 2:
        want["train"] = 1; want["test"] = 1
    elif n_total == 1:
        want["train"] = 1
    return want

def enforce_min_presence_all_splits(labels_id, splits, K, prefer_order=("train","val","test")):
    """
    Ensure each class meets feasible min presence across splits by moving samples.
    prefer_order: priority order when we must *keep* a class (we avoid stealing from the earlier ones).
    """
    # current counts
    counts = _count_per_split_per_class(labels_id, splits, K)
    totals = counts["train"] + counts["val"] + counts["test"]

    # per class desires
    desires = [ _feasible_min_presence(int(t)) for t in totals ]

    # Try to satisfy desires by moving samples
    for k in range(K):
        want = desires[k]  # dict split->min
        # First, ensure required presence where missing
        for dst in ("train","val","test"):
            need = want[dst]
            have = counts[dst][k]
            while have < need:
                # find a donor split that currently has > want[donor]
                donor_candidates = [s for s in ("train","val","test") if s != dst]
                # prefer not to steal from earlier priority splits
                donor_candidates.sort(key=lambda s: prefer_order.index(s), reverse=True)
                moved = False
                for src in donor_candidates:
                    if counts[src][k] > want[src]:
                        if _move_one(labels_id, splits, src, dst, k):
                            counts[src][k] -= 1
                            counts[dst][k] += 1
                            have += 1
                            moved = True
                            break
                if not moved:
                    # no available donor that stays above its own minimum; break
                    break

    return splits, desires

def check_min_presence(labels_id, splits, K, desires):
    """Assert that achieved presence >= desired presence for each class/split."""
    counts = _count_per_split_per_class(labels_id, splits, K)
    for k in range(K):
        want = desires[k]
        for name in ("train","val","test"):
            if counts[name][k] < want[name]:
                raise AssertionError(
                    f"[{name}] class {k}: have={counts[name][k]} < want={want[name]} "
                    f"(total for class={sum(counts[n][k] for n in ('train','val','test'))})"
                )
                
def loop_train_test(
    hp,
    cube_tif: str,
    shapefile: str,
    label_field: str,
    model_builder,                 # (n_bands, n_classes, window, pca_components, emap_components) -> nn.Module
    outdir: str = "outputs_prisma"
):
    """
    Train/evaluate HSI classifier with or without Active Learning.

    hp keys:
      run_times, window, epochs, batch_size, val_frac, test_frac, seed, lr, weight_decay, device
      use_pca (bool, default True), pca_components (int)
      use_emap (bool, default False), emap_thresholds (list[int]), emap_on_pc (int)

      # Active learning (optional):
      use_active_learning (bool)
      al_strategy ('random','uncertainty','kmeans','hybrid')
      al_init_frac (float in (0,1))
      al_target_frac (float in (0,1])
      al_rounds (int)
      al_query_batch (int)
    """
    import os, time, math
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.decomposition import PCA
    from scipy.ndimage import maximum_filter, minimum_filter, label as cc_label

    os.makedirs(outdir, exist_ok=True)

    # --- hyperparameters ---
    run_times   = int(hp.get('run_times', 1))
    window      = int(hp.get('window', 15))
    epochs      = int(hp.get('epochs', 30))
    batch       = int(hp.get('batch_size', 256))
    val_frac    = float(hp.get('val_frac', 0.2))
    test_frac   = float(hp.get('test_frac', 0.2))
    seed0       = int(hp.get('seed', 42))
    lr          = float(hp.get('lr', 1e-3))
    wd          = float(hp.get('weight_decay', 1e-4))
    device      = hp.get('device', "cuda" if torch.cuda.is_available() else "cpu")

    use_pca         = bool(hp.get('use_pca', True))
    pca_components  = int(hp.get('pca_components', 50))
    use_emap        = bool(hp.get('use_emap', False))
    emap_thresholds = list(hp.get('emap_thresholds', [32, 64, 128, 256]))
    emap_on_pc      = int(hp.get('emap_on_pc', 0))  # 0=PC1

    # per-class min presence in splits (optional)
    min_train = int(hp.get("min_train", 0))
    min_val   = int(hp.get("min_val", 0))
    min_test  = int(hp.get("min_test", 0))

    # --- active learning hyperparams (optional) ---
    use_active_learning = bool(hp.get("use_active_learning", False))
    al_strategy_name    = hp.get("al_strategy", "uncertainty")
    al_init_frac        = float(hp.get("al_init_frac", 0.01))
    al_target_frac      = float(hp.get("al_target_frac", 0.10))
    al_rounds           = int(hp.get("al_rounds", 5))
    al_query_batch      = int(hp.get("al_query_batch", batch))

    # ===================== Load & prepare labels =====================
    cube_raw, profile = read_hsi_tif(cube_tif)                     # (H,W,C_full)

    labels_raw = rasterize_shapefile_to_labels(shapefile, label_field, profile, background=0)
    debug_labels(labels_raw, title="[debug labels_raw]")

    # Remap raw codes (excluding 0) -> 0..K-1 contiguous ids
    codes = np.unique(labels_raw)
    codes = codes[codes > 0]
    code2id = {int(c): i for i, c in enumerate(sorted(codes.tolist()))}
    labels_id = np.zeros_like(labels_raw, dtype=np.int32) - 1     # -1 background
    mask = labels_raw > 0
    labels_id[mask] = np.vectorize(code2id.get)(labels_raw[mask])
    n_classes = len(code2id)
    id2orig = {v: k for k, v in code2id.items()}
    with open(os.path.join(outdir, "catfor_mapping.json"), "w") as f:
        import json
        json.dump({"code2id": code2id, "id2orig": id2orig}, f, indent=2)

    debug_labels(labels_id, title="[debug labels_id]")
    print(f"[debug] n_classes={n_classes}")

    # Normalize full cube once (before PCA)
    cube_raw = zscore_per_band(cube_raw).astype(np.float32)
    H, W, C = cube_raw.shape
    print(f"[data] cube={cube_raw.shape}  classes(K)={n_classes}  labeled_px={np.count_nonzero(labels_id>=0)}")

    # coords of all labeled pixels (for dense map later)
    coords_all, _ = build_indices(labels_id)
    OA, AA, KAPPA, MACRO_F1 = [], [], [], []
    TRAINING_TIME, TESTING_TIME, QUERY_TIME = [], [], []
    ELEMENT_ACC = np.zeros((run_times, n_classes), dtype=np.float32)

    

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

    def _assert_split_ok(name, rc_list, gt_id, K):
        ys = np.array([gt_id[r, c] for (r, c) in rc_list], dtype=np.int64)
        assert (ys >= 0).all(), f"{name}: has background (-1) samples!"
        assert ys.max() < K,    f"{name}: has label >= K (max={ys.max()}, K={K})"
        assert ys.min() >= 0,   f"{name}: has negative label (min={ys.min()})"

    # helpers to ensure val has all classes (if possible)
    def _steal_one_of_class(labels_id, splits, src, dst, k):
        # move one (r,c) with label k from src -> dst
        for i, (r, c) in enumerate(splits[src]):
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

    # ===================== RUNS =====================
    for run_i in range(run_times):
        seed = seed0 + run_i
        set_seed(seed)
        print(f"\n===== round {run_i+1}/{run_times} (seed {seed}) =====")

        # Splits on remapped labels (background=-1 ignored by splitter)
        splits = stratified_pixel_splits(
            labels_id,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            min_train=min_train,
            min_val=min_val,
            min_test=min_test,
        )

        # make sure validation has all classes at least once if possible
        splits = ensure_val_has_all_classes(labels_id, splits, n_classes)

        # enforce feasible per-class presence across all splits
        splits, desires = enforce_min_presence_all_splits(
            labels_id, splits, n_classes, prefer_order=("train", "val", "test")
        )
        check_min_presence(labels_id, splits, n_classes, desires)

        # Optional: print post-fix stats
        debug_split_stats(
            labels_id,
            splits,
            n_classes,
            id2orig=id2orig,
            title=f"Post-stratified splits (run {run_i+1})",
        )

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

        # sanity guard: splits contain only [0..K-1]
        _assert_split_ok("train", splits["train"], labels_id, n_classes)
        _assert_split_ok("val",   splits["val"],   labels_id, n_classes)
        _assert_split_ok("test",  splits["test"],  labels_id, n_classes)

        # ===================== ACTIVE LEARNING BRANCH =====================
        if use_active_learning:
            print(f"[AL] Running Active Learning (strategy='{al_strategy_name}') for run {run_i+1}/{run_times}")

            history = run_active_learning_single_run(
                cube_for_model=cube_for_model,
                labels_id=labels_id,
                splits=splits,
                n_classes=n_classes,
                model_builder=model_builder,
                device=device,
                window=window,
                batch_size=batch,
                epochs=epochs,
                lr=lr,
                weight_decay=wd,
                pca_components=(n_spec if use_pca else 0),
                emap_components=n_emap,
                al_strategy_name=al_strategy_name,
                al_init_frac=al_init_frac,
                al_target_frac=al_target_frac,
                al_rounds=al_rounds,
                al_query_batch=al_query_batch,
                run_index=run_i,
            )

            # use last round as "final" summary for this run
            last = history[-1]
            OA.append(last["oa"])
            KAPPA.append(last["kappa"])
            MACRO_F1.append(last["macro_f1"])
            per_cls_recall = last["per_class_recall"]
            AA.append(float(np.nanmean(per_cls_recall)))
            ELEMENT_ACC[run_i, :] = np.nan_to_num(per_cls_recall, nan=0.0)

            total_train_time = sum(h["T_train"] for h in history)
            total_test_time = sum(h["T_test"] for h in history)
            total_query_time = sum(h["T_query_inf"] + h["T_query_sel"] for h in history)

            TRAINING_TIME.append(total_train_time)
            TESTING_TIME.append(total_test_time)
            QUERY_TIME.append(total_query_time)

            plot_active_learning_curves(
                histories={al_strategy_name: history},
                outdir=outdir,
                suffix=f"_run{run_i+1}",
            )

            # skip standard supervised training in AL mode
            continue




        # ===================== STANDARD SUPERVISED BRANCH =====================
        # Dataloaders on PCA+EMAP stack
        dls = make_dataloaders(
            cube_for_model,
            labels_id,
            splits,
            batch=batch,
            window=window,
            num_workers=8,
        )

        # Model
        n_bands_for_model = cube_for_model.shape[2]
        print(
            f"[loop] n_spec={n_spec if use_pca else 0}, "
            f"n_emap={n_emap}, "
            f"n_bands_for_model={n_bands_for_model}"
        )

        model = model_builder(
            n_bands_for_model,
            n_classes,
            window,
            pca_components=(n_spec if use_pca else 0),
            emap_components=n_emap,
        ).to(device)

        backbone = getattr(model, "net", model)
        warmup_ep = max(1, epochs // 5)
        if hasattr(backbone, "enable_adapters"):
            backbone.enable_adapters(False)

        # Class weights from TRAIN labels
        train_coords = splits["train"]
        labs_train = np.array([labels_id[r, c] for (r, c) in train_coords], dtype=np.int64)
        counts = np.bincount(labs_train, minlength=n_classes).astype(np.float32)
        print("[debug] train per-class counts:", counts.tolist())
        counts[counts == 0] = 1.0

        # keep your class-weighted CE
        w = counts.sum() / (counts * n_classes)
        w = np.clip(w, 1e-3, 1e3)
        class_w = torch.tensor(w, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.0).to(device)

        optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        total_steps = epochs * max(1, len(dls["train"]))
        warmup_steps = max(10, len(dls["train"]))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ---- Train (keep best by val macro-F1) ----
        t0 = time.perf_counter()
        best_val_f1, best_state = -1.0, None
        for ep in range(1, epochs + 1):
            print(f"--- epoch {ep}/{epochs} ---")

            # flip on adapters after warm-up if available
            if ep == warmup_ep:
                backbone = getattr(model, "net", model)
                if hasattr(backbone, "enable_adapters"):
                    backbone.enable_adapters(True)
                    print(f"[info] enabled adapters at epoch {ep}")

            tr_loss, tr_time = train_one_epoch(
                model,
                dls["train"],
                optimizer,
                device,
                criterion,
                scheduler,
                log_every=50,
                proto_mem=None,
                class_w=class_w,
            )

            va_metrics = evaluate(model, dls["val"], device, n_classes)
            print(
                f"  train_loss={tr_loss:.4f}  time/epoch={tr_time:.1f}s  "
                f"val_macroF1={va_metrics['macro_f1']:.4f}  OA={va_metrics['oa']:.4f}"
            )

            if va_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = va_metrics["macro_f1"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        t1 = time.perf_counter()
        TRAINING_TIME.append(t1 - t0)

        # ---- Test ----
        model.load_state_dict(best_state)
        t2 = time.perf_counter()
        te_metrics = evaluate(model, dls["test"], device, n_classes)
        t3 = time.perf_counter()
        TESTING_TIME.append(t3 - t2)

        print(f"OA: {te_metrics['oa']}")
        print(f"Kappa: {te_metrics['kappa']}")
        print(f"Macro-F1: {te_metrics['macro_f1']}")
        print(te_metrics["report"])

        # Aggregate scores
        OA.append(te_metrics["oa"])
        KAPPA.append(te_metrics["kappa"])
        MACRO_F1.append(te_metrics["macro_f1"])

        per_cls_recall = te_metrics["per_class_recall"]          # length K; NaN for empty classes
        AA.append(float(np.nanmean(per_cls_recall)))
        ELEMENT_ACC[run_i, :] = np.nan_to_num(per_cls_recall, nan=0.0)

        # no query in standard supervised mode
        QUERY_TIME.append(0.0)

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
                preds_full[r, c] = cls  # +1 for visualization map (bg=0)

        map_path = os.path.join(outdir, f"clsmap_run{run_i+1}.png")
        save_classification_map(preds_full, map_path)
        print(f"Saved classification map -> {map_path}")

    # ===================== Final aggregation =====================
    query_arg = np.array(QUERY_TIME) if use_active_learning else None
    print_results(
        n_classes,
        np.array(OA),
        np.array(AA),
        np.array(KAPPA),
        np.array(MACRO_F1),
        ELEMENT_ACC,
        np.array(TRAINING_TIME),
        np.array(TESTING_TIME),
        query_arg,
    )
