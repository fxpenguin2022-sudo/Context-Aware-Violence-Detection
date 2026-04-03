from __future__ import annotations

import logging
from pathlib import Path


def build_logger(name: str, level: str = 'INFO', log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', '%Y-%m-%d %H:%M:%S')

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
