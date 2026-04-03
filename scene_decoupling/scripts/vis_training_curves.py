#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse

from scene_decoupling.src.viz.curves import plot_training_curves


def main() -> None:
    p = argparse.ArgumentParser(description='Plot scene-decoupling training curves')
    p.add_argument('--history', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    plot_training_curves(args.history, args.output)


if __name__ == '__main__':
    main()
