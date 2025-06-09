# src/plot_utils.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_history(history, title=""):
    # loss
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"],   label="Val")
    plt.title(f"Loss {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # acc
    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"],   label="Val")
    plt.title(f"Accuracy {title}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
