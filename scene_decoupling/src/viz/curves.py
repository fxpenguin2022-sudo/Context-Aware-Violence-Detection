from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(history_csv: str, out_png: str) -> None:
    rows = list(csv.DictReader(Path(history_csv).open('r', encoding='utf-8')))
    epochs = [int(r['epoch']) for r in rows]

    train_loss = [float(r['train_loss']) for r in rows]
    f1 = [float(r['val_f1_best']) for r in rows]
    acc = [float(r['val_acc_best']) for r in rows]
    p = [float(r['val_precision_best']) for r in rows]
    r = [float(r['val_recall_best']) for r in rows]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, f1, label='val_f1_best')
    plt.plot(epochs, acc, label='val_acc_best')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, p, label='val_precision_best')
    plt.plot(epochs, r, label='val_recall_best')
    plt.grid(True, alpha=0.3)
    plt.legend()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
