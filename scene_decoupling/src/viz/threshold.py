from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_threshold_scan(scan_json: str, out_png: str) -> None:
    rows = json.loads(Path(scan_json).read_text(encoding='utf-8'))
    t = [float(x['threshold']) for x in rows]
    f1 = [float(x['f1']) for x in rows]
    acc = [float(x['acc']) for x in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(t, f1, label='f1')
    plt.plot(t, acc, label='acc')
    i = max(range(len(rows)), key=lambda j: (f1[j], acc[j]))
    plt.scatter([t[i]], [f1[i]], label='best_f1')
    plt.grid(True, alpha=0.3)
    plt.legend()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
