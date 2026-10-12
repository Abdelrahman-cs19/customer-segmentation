"""Configuration loading and path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration.

    Args:
        path: Optional path to a YAML file. Defaults to ``configs/config.yaml``.

    Returns:
        The parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is empty or required sections are missing.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file is empty or invalid: {cfg_path}")
    required = {"project", "paths", "data", "cleaning", "features", "preprocessing", "clustering"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Config is missing sections: {sorted(missing)}")
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a ``paths`` entry relative to the project root and create parent dirs for dirs."""
    p = PROJECT_ROOT / cfg["paths"][key]
    return p
