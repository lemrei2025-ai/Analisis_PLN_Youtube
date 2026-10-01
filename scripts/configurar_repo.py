"""Configura la dirección del repositorio público del taller (uso del docente, una sola vez).

Reemplaza el marcador USUARIO/REPOSITORIO en los notebooks, el README y las guías por el
usuario y el nombre reales del repositorio en GitHub. Después se suben los cambios.

Uso:
    python scripts/configurar_repo.py usuario/nombre-repo
    python scripts/configurar_repo.py https://github.com/usuario/nombre-repo
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "USUARIO/REPOSITORIO"
FILES = [*ROOT.glob("notebooks/*.ipynb"), ROOT / "README.md", *ROOT.glob("docs/*.md")]


def parse(arg: str) -> str:
    m = re.fullmatch(r"(?:https?://github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?", arg.strip())
    if not m:
        raise SystemExit(f"Formato no válido: {arg!r}. Usar usuario/repositorio.")
    return f"{m.group(1)}/{m.group(2)}"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    repo = parse(sys.argv[1])
    changed = 0
    for f in FILES:
        text = f.read_text(encoding="utf-8")
        if PLACEHOLDER in text:
            f.write_text(text.replace(PLACEHOLDER, repo), encoding="utf-8")
            changed += 1
            print("actualizado:", f.relative_to(ROOT))
    if not changed:
        print("No se encontró el marcador; el repositorio ya estaba configurado.")
    else:
        print(f"Listo: {changed} archivos apuntan a https://github.com/{repo}")


if __name__ == "__main__":
    main()
