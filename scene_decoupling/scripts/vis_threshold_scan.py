#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse

from scene_decoupling.src.viz.threshold import plot_threshold_scan


def main() -> None:
    p = argparse.ArgumentParser(description='Plot scene-decoupling threshold scan')
    p.add_argument('--scan-json', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    plot_threshold_scan(args.scan_json, args.output)


if __name__ == '__main__':
    main()
