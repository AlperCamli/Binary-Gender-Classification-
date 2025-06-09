# src/evaluate.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE
from data_utils import get_dataloaders
from model_utils import create_vgg16

def evaluate(model, path_to_weights):
    model.load_state_dict(torch.load(path_to_weights, map_location=DEVICE))
    model.to(DEVICE).eval()

    _, _, test_loader = get_dataloaders()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(DEVICE)
            logits = model(x).squeeze()
            preds  = (torch.sigmoid(logits)>=0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    cm   = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}\n")

    # plot cm
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__=="__main__":
    # replace with your saved checkpoint
    checkpoint = "models/best_finetuned_lr0.001.pth"
    for fine in (False, True):
        for lr in [1e-3, 1e-4]:
            print(f"\n=== Eval: LR={lr}, fine={fine} ===")
            mdl = create_vgg16(fine)
            evaluate(mdl, checkpoint)
