# run_prisma_experiment.py

import sys
sys.path.append("/home/aadhithya")  # folder that contains hsi_classification

from hsi_classification.AL_Training import loop_train_test

import warnings
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    cohen_kappa_score,
)

import torch.nn as nn
import torch.optim as optim
from operator import truediv

warnings.filterwarnings("ignore")

# from .util import print_results   # <- not needed anymore; handled in Loop_prisma_train_mod1

# ========= MODELS =========
from .models.SClusterFormer.CNN1Dplus import BaselineCNN1D


# If you still need SClusterFormer directly elsewhere, keep this import.
# The wrapper below will import SClusterFormer from the package.



# ========= DATA PATHS =========
CUBE_TIF    = r"/data/hsi/mosaic_south.tif"   # hyperspectral .tif
SHAPEFILE   = r"/data/hsi/gt_south.shp"       # polygons
LABEL_FIELD = "CATFOR"                        # integer labels in 1..K


# ========= EXPERIMENT SETUP =========
RUN_TIMES    = 5
WINDOW       = 15          # patch size (odd)
EPOCHS       = 80
BATCH_SIZE   = 256
VAL_FRAC     = 0.2
TEST_FRAC    = 0.2
SEED         = 42
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR_BASE  = "outputs_prisma"

# PCA + EMAP (as in paper)
USE_PCA         = True
PCA_COMPONENTS  = 50           # try 30/48/64 in ablation
USE_EMAP        = False
EMAP_THRESHOLDS = [32, 64, 128, 256]  # area thresholds (pixels)
EMAP_ON_PC      = 0            # 0 => PC1

# ========= MODEL CHOICE =========
# choose: "cnn1d" (baseline) or "scluster" (SClusterFormer wrapper)
MODEL = "cnn1d" #different classifiers try??


# ========= ACTIVE LEARNING CONFIG =========
# If False -> standard supervised training (no AL, QUERY_TIME=0)
USE_ACTIVE_LEARNING = True

# One of: "random", "uncertainty", "kmeans", "hybrid","cbe"
AL_STRATEGY    = "random"


# Fractions of TRAIN split
AL_INIT_FRAC   = 0.01 #try 0.03   # fraction of train pixels initially labeled
AL_TARGET_FRAC = 0.10   # stop when this fraction of train pixels is labeled

AL_ROUNDS      = 5      # max AL rounds
AL_QUERY_BATCH = 256    # batch size for querying (inference over pool) differnt size effect check?


# ========= OPTIONAL SPLIT MINIMUMS PER CLASS =========
# 0 means "no enforced minimum". You can set 1 to force at least one per class.
MIN_TRAIN_PER_CLASS = 0
MIN_VAL_PER_CLASS   = 0
MIN_TEST_PER_CLASS  = 0


# ========= SClusterFormer WRAPPER =========
class SClusterWrapper(nn.Module):
    def __init__(
        self,
        n_bands,
        n_classes,
        img_size,
        pca_components,
        emap_components,
        d_model=128,
        n_heads=4,
    ):
        super().__init__()
        # n_bands must equal pca + emap channels
        assert pca_components + emap_components == n_bands, \
            f"n_bands={n_bands}, but pca+emap={pca_components + emap_components}"
        print(
            f"[SClusterWrapper] n_bands={n_bands}, "
            f"pca_components={pca_components}, emap_components={emap_components}"
        )

        # Import from top-level package so this script can be run as a module
        from hsi_classification.models.SClusterFormer.SClusterFormer import SClusterFormer

        self.net = SClusterFormer(
            img_size=img_size,
            pca_components=pca_components,
            emap_components=emap_components,
            num_classes=n_classes,
            num_stages=3,
            n_groups=(32, 32, 32),
            embed_dims=(256, 128, 64),
            num_heads=(8, 8, 8),
            mlp_ratios=(1, 1, 1),
            depths=(2, 2, 2),
        )

    def forward(self, x):  # x: (B, C, H, W)
        # SClusterFormer expects (B, 1, C, H, W)
        return self.net(x.unsqueeze(1))


# ========= MAIN RUN FUNCTION =========
def Run_experiment():
    # Build outdir name depending on model + AL strategy
    if USE_ACTIVE_LEARNING:
        exp_tag = f"{MODEL}_AL_{AL_STRATEGY}"
    else:
        exp_tag = f"{MODEL}_SUPERVISED"
    outdir = f"{OUTDIR_BASE}_{exp_tag}"

    # hp dict expected by loop_train_test
    hp = {
        # basic training
        "run_times": RUN_TIMES,
        "window": WINDOW,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "seed": SEED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "device": DEVICE,

        # PCA / EMAP knobs
        "use_pca": USE_PCA,
        "pca_components": PCA_COMPONENTS,
        "use_emap": USE_EMAP,
        "emap_thresholds": EMAP_THRESHOLDS,
        "emap_on_pc": EMAP_ON_PC,

        # per-class minimums in splits (optional)
        "min_train": MIN_TRAIN_PER_CLASS,
        "min_val": MIN_VAL_PER_CLASS,
        "min_test": MIN_TEST_PER_CLASS,

        # active learning controls
        "use_active_learning": USE_ACTIVE_LEARNING,
        "al_strategy": AL_STRATEGY,
        "al_init_frac": AL_INIT_FRAC,
        "al_target_frac": AL_TARGET_FRAC,
        "al_rounds": AL_ROUNDS,
        "al_query_batch": AL_QUERY_BATCH,
    }

    # model_builder: must accept (n_bands, n_classes, window, pca_components, emap_components)
    if MODEL == "cnn1d":

        def model_builder(n_bands, n_classes, window, pca_components, emap_components):
            return BaselineCNN1D(
                n_bands=n_bands,
                n_classes=n_classes,
                n_layers=3,          # try 2–4
                hidden_width=32,     # first conv width (32/48/64)
                kernel_size=5,       # or [5,5,3] per-layer
                red_type="stride",   # or "pool"
                last_pooling=True,
                dropout=0.1,
            )

    elif MODEL == "scluster":

        def model_builder(n_bands, n_classes, window, pca_components, emap_components):
            return SClusterWrapper(
                n_bands=n_bands,
                n_classes=n_classes,
                img_size=window,
                pca_components=pca_components,
                emap_components=emap_components,
                d_model=128,
                n_heads=4,
            )
    else:
        raise ValueError("MODEL must be 'cnn1d' or 'scluster'.")

    # === Launch experiment ===
    loop_train_test(
        hp=hp,
        cube_tif=CUBE_TIF,
        shapefile=SHAPEFILE,
        label_field=LABEL_FIELD,
        model_builder=model_builder,
        outdir=outdir,
    )


if __name__ == "__main__":
    Run_experiment()
    print(time.asctime(time.localtime()))
