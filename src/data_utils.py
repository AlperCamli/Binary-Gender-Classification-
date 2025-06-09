# src/data_utils.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as transforms

from config import CSV_PATH, IMAGES_DIR, BATCH_SIZE

# transforms to match VGG16 / ImageNet stats
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class CelebADataset(Dataset):
    def __init__(self, csv_file=CSV_PATH, img_dir=IMAGES_DIR, transform=data_transforms):
        self.df        = pd.read_csv(csv_file)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row[0])
        image = Image.open(img_path).convert("RGB")

        # Male column: 1→male, -1→female
        label = 1 if row["Male"] == 1 else 0

        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders():
    full = CelebADataset()
    n     = len(full)
    splits = [int(0.8*n), int(0.1*n)]
    train_ds, val_ds, test_ds = random_split(full, [splits[0], splits[1], n - sum(splits)],
                                             generator=torch.Generator().manual_seed(42))
    return (
      DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
      DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False),
      DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False),
    )
