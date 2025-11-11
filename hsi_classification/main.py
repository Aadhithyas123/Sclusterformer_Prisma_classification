# run_prisma_experiment.py
#from Active_learning.hsi_classification.Loop_prisma_train import loop_train_test
import sys
sys.path.append("/home/aadhithya")  # folder that contains Active_learning
from hsi_classification.Loop_prisma_train import loop_train_test

#from .Loop_prisma_train import loop_train_test


import warnings, time, torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
from .util import print_results

warnings.filterwarnings('ignore')
from .models.SClusterFormer.BaselineCNN1D import BaselineCNN1D
from .models.SClusterFormer.SClusterFormer import SClusterFormer
# ========= EDIT THESE PATHS =========
CUBE_TIF    = r"/data/hsi/mosaic_south.tif"    # your hyperspectral .tif #1.north,2.center,3.shouth
SHAPEFILE   = r"/data/hsi/gt_south.shp"   # polygons
LABEL_FIELD = "CATFOR"                      # integer labels in 1..K

# ========= EXPERIMENT SETUP =========
RUN_TIMES   = 5
WINDOW      = 15              # patch size (odd)
EPOCHS      = 80
BATCH_SIZE  = 32
VAL_FRAC    = 0.2
TEST_FRAC   = 0.2
SEED        = 42
LR          = 1e-3
WEIGHT_DECAY= 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR      = "outputs_prisma"

# PCA + EMAP (as in paper)
USE_PCA         = True
PCA_COMPONENTS  = 30           # try 30/48/64 in ablation
USE_EMAP        = True
EMAP_THRESHOLDS = [32, 64, 128, 256]  # area thresholds (pixels)
EMAP_ON_PC      = 0            # 0 => PC1

# choose: "cnn1d" (baseline) or "scluster" (SClusterFormer wrapper)
MODEL = "scluster"

# --- add in main.py (or export from Loop_prisma_train.py) ---
import torch
import torch.nn as nn


# main.py (or wherever you defined it)
class SClusterWrapper(nn.Module):
    def __init__(self, n_bands, n_classes, img_size,
                 pca_components, emap_components,
                 d_model=128, n_heads=4):
        super().__init__()
        assert pca_components + emap_components == n_bands
        print(f"[SClusterWrapper] pca_components={pca_components}, emap_components={emap_components}")
        from hsi_classification.models.SClusterFormer.SClusterFormer import SClusterFormer
        self.net = SClusterFormer(
            img_size=img_size,
            pca_components=30,
            emap_components=8,
            num_classes=n_classes,
            num_stages=3,
            n_groups=(32,32,32),
            embed_dims=(256,128,64),
            num_heads=(8,8,8),
            mlp_ratios=(1,1,1),
            depths=(2,2,2),
            )


    def forward(self, x):  # x: (B, C, H, W)
        # SClusterFormer expects (B, 1, C, H, W)
        
        return self.net(x.unsqueeze(1))





def Run_experiment():
    # hp dict expected by the updated loop_train_test
    hp = {
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
        # PCA / EMAP knobs (paper-style)
        "use_pca": USE_PCA,
        "pca_components": PCA_COMPONENTS,
        "use_emap": USE_EMAP,
        "emap_thresholds": EMAP_THRESHOLDS,
        "emap_on_pc": EMAP_ON_PC,
    }

    # model_builder: must accept (n_bands, n_classes, window, pca_components, emap_components)
    if MODEL == "cnn1d":
        def model_builder(n_bands, n_classes, window, pca_components, emap_components):
            # PCA/EMAP counts are provided for completeness; CNN1D just uses total n_bands
            return BaselineCNN1D(n_bands=n_bands, n_classes=n_classes, channels=(32, 64, 128), kernel_size=5)
    elif MODEL == "scluster":
        def model_builder(n_bands, n_classes, window, pca_components, emap_components):
            # SClusterFormer expects to know PCA vs EMAP split internally
            # The wrapper should route channels accordingly.
            return SClusterWrapper(n_bands=n_bands, n_classes=n_classes, img_size=window,pca_components=pca_components, emap_components=emap_components, d_model=128, n_heads=4)
    else:
        raise ValueError("MODEL must be 'cnn1d' or 'scluster'.")

    loop_train_test(
        hp=hp,
        cube_tif=CUBE_TIF,
        shapefile=SHAPEFILE,
        label_field=LABEL_FIELD,
        model_builder=model_builder,
        outdir=OUTDIR,
    )

if __name__ == '__main__':
    Run_experiment()
    print(time.asctime(time.localtime()))
