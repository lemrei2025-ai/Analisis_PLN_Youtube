"""Pipeline de la etapa de datos: ingesta → validación → limpieza → features → análisis → baseline.

Uso:
    python -m ytnlp.pipeline --source sample|kaggle|api [--skip-model]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ytnlp.analysis import stats
from ytnlp.config import path
from ytnlp.data import clean, validate
from ytnlp.features import build_features
from ytnlp.models import baseline

log = logging.getLogger("ytnlp")


def ingest(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "sample":
        from ytnlp.data.kaggle_source import load

        return load(path("sample"))
    if source == "kaggle":
        from ytnlp.data.kaggle_source import load

        return load()
    if source == "api":
        from ytnlp.data.youtube_api import load_latest

        return load_latest()
    raise ValueError(f"Fuente desconocida: {source}")


def run(source: str, skip_model: bool = False) -> dict:
    reports = path("reports")

    log.info("1/6 Ingesta (%s)", source)
    videos, comments = ingest(source)

    log.info("2/6 Validación")
    res = validate.validate(videos, comments)
    validate.write_report(res.report, reports)
    log.info("   filas descartadas: videos=%s comentarios=%s",
             res.report["reglas_duras"]["videos"]["dropped_rows"],
             res.report["reglas_duras"]["comentarios"]["dropped_rows"])

    log.info("3/6 Limpieza")
    v = clean.clean_videos(res.videos)
    c = clean.clean_comments(res.comments)
    v.to_parquet(path("interim") / "videos.parquet", index=False)
    c.to_parquet(path("interim") / "comments.parquet", index=False)

    log.info("4/6 Features")
    feats = build_features.build(v, c)
    feats.to_parquet(path("processed") / "features.parquet", index=False)
    log.info("   %d videos x %d columnas", *feats.shape)

    log.info("5/6 Análisis estadísticos")
    results = stats.run_all(feats, v, c)
    stats.write_report(results, v, reports)

    out = {"videos": len(v), "comentarios": len(c), "features": feats.shape}
    if not skip_model:
        log.info("6/6 Baselines y curva de aprendizaje")
        b = baseline.run(feats)
        baseline.write_report(b, reports)
        out["baseline"] = {k: b["resultados"][k] for k in b["resultados"]}
    log.info("Listo. Reportes en %s", reports)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sample", "kaggle", "api"], default="sample")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    run(args.source, args.skip_model)


if __name__ == "__main__":
    main()
