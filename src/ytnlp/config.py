"""Carga de configuración y rutas del proyecto.

Las carpetas de datos y reportes se crean bajo `YTNLP_STORAGE` si esa variable de entorno
existe (en Colab apunta a Google Drive, para que nada se pierda al cerrar la sesión).
Si no existe, se usan dentro del repositorio. La muestra sintética siempre vive en el repo.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
IN_REPO_ONLY = {"sample"}


@lru_cache(maxsize=1)
def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def storage_root() -> Path:
    return Path(os.environ["YTNLP_STORAGE"]) if os.getenv("YTNLP_STORAGE") else ROOT


def path(key: str) -> Path:
    """Ruta absoluta de una carpeta definida en config.paths (se crea si no existe)."""
    base = ROOT if key in IN_REPO_ONLY else storage_root()
    p = base / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
