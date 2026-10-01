"""Utilidades para trabajar el proyecto desde Google Colab.

El código del taller se descarga en modo solo lectura; lo que produce cada equipo (datos,
reportes y configuración) se guarda en Google Drive.

Uso en la segunda celda de cualquier notebook (después de descargar el código):

    from ytnlp.colab import setup
    setup()                       # monta Drive, carga secretos, muestra el entorno

Fuera de Colab (en un computador local) las mismas funciones leen el archivo `.env`
y no montan Drive, así que los notebooks corren igual en ambos entornos.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from ytnlp.config import ROOT, load_config, overrides_path, reload

SECRETS = ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "YOUTUBE_API_KEY")
DEFAULT_DRIVE_FOLDER = "MyDrive/ytnlp-proyecto"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def load_secrets(names: tuple[str, ...] = SECRETS) -> dict[str, bool]:
    """Carga credenciales en variables de entorno. Devuelve cuáles se encontraron (sin valores).

    En Colab: panel izquierdo → Secretos → agregar el nombre y el valor, y activar
    "Acceso del notebook". Localmente: archivo `.env` (ver `.env.example`).
    """
    found: dict[str, bool] = {}
    if in_colab():
        from google.colab import userdata

        for n in names:
            try:
                value = userdata.get(n)
            except Exception:  # noqa: BLE001  secreto inexistente o sin acceso
                value = None
            if value:
                os.environ[n] = value
            found[n] = bool(os.getenv(n))
    else:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        found = {n: bool(os.getenv(n)) for n in names}
    return found


def mount_drive(folder: str = DEFAULT_DRIVE_FOLDER) -> Path | None:
    """Monta Google Drive y guarda allí datos y reportes (variable YTNLP_STORAGE)."""
    if not in_colab():
        return None
    from google.colab import drive

    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
    storage = Path("/content/drive") / folder
    storage.mkdir(parents=True, exist_ok=True)
    os.environ["YTNLP_STORAGE"] = str(storage)
    return storage


def environment_summary() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    gpu = "no"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            gpu = out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "entorno": "Google Colab" if in_colab() else "local",
        "python": platform.python_version(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "gpu": gpu,
        "repo": str(ROOT),
        "datos y reportes": os.getenv("YTNLP_STORAGE", str(ROOT)),
    }


def setup(use_drive: bool = True, drive_folder: str = DEFAULT_DRIVE_FOLDER) -> dict:
    storage = mount_drive(drive_folder) if use_drive else None
    reload()  # la configuración del equipo depende de la carpeta de Drive
    secrets = load_secrets()
    info = environment_summary()
    print("Entorno")
    for k, v in info.items():
        print(f"  {k:<17} {v}")
    cfg = load_config()
    origen = "config_equipo.yaml" if overrides_path().exists() else "base del taller"
    print("Configuración")
    print(f"  {'origen':<17} {origen}")
    print(f"  {'target':<17} {cfg['target']}")
    print(f"  {'comentarios/video':<17} {cfg['api']['max_comments_per_video']}")
    print("Secretos")
    for k, ok in secrets.items():
        print(f"  {k:<17} {'OK' if ok else 'falta'}")
    if use_drive and storage is None and in_colab():
        print("Aviso: Drive no se montó; los datos se perderán al cerrar la sesión.")
    return {"storage": storage, "secrets": secrets, **info}
