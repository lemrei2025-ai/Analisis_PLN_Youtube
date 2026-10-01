"""Carga de configuración y rutas del proyecto.

Las carpetas de datos y reportes se crean bajo `YTNLP_STORAGE` si esa variable de entorno
existe (en Colab apunta a Google Drive, para que nada se pierda al cerrar la sesión).
Si no existe, se usan dentro del repositorio. La muestra sintética siempre vive en el repo.

La configuración base es `configs/config.yaml` (parte del taller, solo lectura). Cada equipo
guarda sus cambios en `<YTNLP_STORAGE>/config_equipo.yaml` con `set_overrides()`; ese archivo
vive en Drive y se combina con la base en cada sesión.
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OVERRIDES_FILE = "config_equipo.yaml"
IN_REPO_ONLY = {"sample"}


def storage_root() -> Path:
    return Path(os.environ["YTNLP_STORAGE"]) if os.getenv("YTNLP_STORAGE") else ROOT


def overrides_path() -> Path:
    return storage_root() / OVERRIDES_FILE


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Configuración base combinada con los cambios del equipo (si existen)."""
    return _merge(_read_yaml(CONFIG_PATH), _read_yaml(overrides_path()))


def set_overrides(**changes: Any) -> Path:
    """Guarda cambios de configuración del equipo. Las claves anidadas usan doble guion bajo.

    Ejemplo: set_overrides(target="perf_class", api__max_comments_per_video=300)
    """
    base = _read_yaml(CONFIG_PATH)
    current = _read_yaml(overrides_path())
    for key, value in changes.items():
        parts = key.split("__")
        node = base
        for p in parts:
            if not isinstance(node, dict) or p not in node:
                raise KeyError(f"'{'.'.join(parts)}' no existe en configs/config.yaml")
            node = node[p]
        target = current
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = value
    p = overrides_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(current, f, allow_unicode=True, sort_keys=False)
    reload()
    return p


def reset_overrides() -> None:
    """Elimina los cambios del equipo y vuelve a la configuración base."""
    overrides_path().unlink(missing_ok=True)
    reload()


def reload() -> None:
    load_config.cache_clear()


def path(key: str) -> Path:
    """Ruta absoluta de una carpeta definida en config.paths (se crea si no existe)."""
    base = ROOT if key in IN_REPO_ONLY else storage_root()
    p = base / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p
