"""Logging setup shared by the pipeline modules."""
from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root ``segmentation`` logger (console + optional file).

    Calling it more than once does not duplicate handlers.
    """
    logger = logging.getLogger("segmentation")
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(console)
    if log_file is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``segmentation`` logger."""
    return logging.getLogger(f"segmentation.{name}")
