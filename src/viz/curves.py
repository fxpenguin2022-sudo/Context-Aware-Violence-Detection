from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(history_csv: str, out_png: str) -> None:
    rows = []
    with Path(history_csv).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    f1_best = [float(r["val_f1_best"]) for r in rows]
    acc_best = [float(r["val_acc_best"]) for r in rows]
    precision_best = [float(r.get("val_precision_best", 0.0)) for r in rows]
    recall_best = [float(r.get("val_recall_best", 0.0)) for r in rows]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, f1_best, label="val_f1_best")
    plt.plot(epochs, acc_best, label="val_acc_best")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, precision_best, label="val_precision_best")
    plt.plot(epochs, recall_best, label="val_recall_best")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.grid(True, alpha=0.3)
    plt.legend()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
