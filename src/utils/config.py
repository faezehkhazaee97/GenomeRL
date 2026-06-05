"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
