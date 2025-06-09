# src/train.py

import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import DEVICE, NUM_EPOCHS, LEARNING_RATES
from data_utils import get_dataloaders
from model_utils import create_vgg16

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.to(DEVICE)
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_acc = 0.0

    for epoch in range(num_epochs):
        start = time.time()
        # — train —
        model.train()
        running_loss = running_correct = 0
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE).float()
            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits)>=0.5).float()
            running_loss   += loss.item() * x.size(0)
            running_correct+= (preds==y).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc  = running_correct / len(train_loader.dataset)

        # — validate —
        model.eval()
        val_loss = val_correct = 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(DEVICE), y.to(DEVICE).float()
                logits = model(x).squeeze()
                val_loss    += criterion(logits,y).item() * x.size(0)
                val_correct += ((torch.sigmoid(logits)>=0.5).float()==y).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc  = val_correct/ len(val_loader.dataset)

        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc

        print(f"Epoch {epoch+1}/{num_epochs}  "
              f"Train: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}  "
              f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}  "
              f"time={(time.time()-start):.1f}s")

    return history

if __name__ == "__main__":
    train_loader, val_loader, _ = get_dataloaders()
    criterion = nn.BCEWithLogitsLoss()

    for lr in LEARNING_RATES:
        for fine in (False, True):
            print(f"\n→ LR={lr}  fine_tune_last_block={fine}")
            model     = create_vgg16(fine)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            hist      = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
            # you might want to save history or model here...
