"""Carga de configuración y rutas del proyecto."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Ruta absoluta de una carpeta definida en config.paths (se crea si no existe)."""
    p = ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
