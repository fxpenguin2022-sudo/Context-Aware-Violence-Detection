from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_threshold_scan(scan_json: str, out_png: str) -> None:
    records = json.loads(Path(scan_json).read_text(encoding="utf-8"))
    th = [float(x["threshold"]) for x in records]
    f1 = [float(x["f1"]) for x in records]
    acc = [float(x["acc"]) for x in records]

    plt.figure(figsize=(6, 4))
    plt.plot(th, f1, label="f1")
    plt.plot(th, acc, label="acc")
    best_idx = max(range(len(records)), key=lambda i: (f1[i], acc[i]))
    plt.scatter([th[best_idx]], [f1[best_idx]], marker="o", label="best_f1")
    plt.xlabel("threshold")
    plt.ylabel("metric")
    plt.grid(True, alpha=0.3)
    plt.legend()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
