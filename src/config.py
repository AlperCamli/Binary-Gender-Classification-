# src/config.py

import os
import torch

# === update BASE_PATH to wherever you place your CelebA30k data ===
BASE_PATH    = os.getenv("CELEBA30K_PATH", "/path/to/CelebA30k")
CSV_PATH     = os.path.join(BASE_PATH, "CelebA30k.csv")
IMAGES_DIR   = os.path.join(BASE_PATH, "images/CelebA30k")

BATCH_SIZE   = 128
NUM_EPOCHS   = 10
LEARNING_RATES = [1e-3, 1e-4]

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
