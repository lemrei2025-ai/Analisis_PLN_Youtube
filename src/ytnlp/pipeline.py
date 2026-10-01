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

        videos, comments = load_latest()
        return _add_keyword(videos), comments
    raise ValueError(f"Fuente desconocida: {source}")


def _add_keyword(videos: pd.DataFrame) -> pd.DataFrame:
    """Completa la keyword de los videos de la API.

    Primero la toma del dataset de Kaggle (si ya se descargó); si falta, usa la categoría
    de YouTube. Así las extracciones hechas antes de guardar la keyword también funcionan.
    """
    from ytnlp.config import load_config
    from ytnlp.data.kaggle_source import keyword_map, normalize_videos

    v = videos.copy()
    if "keyword" not in v:
        v["keyword"] = pd.NA
    kaggle_csv = path("raw") / "kaggle" / load_config()["kaggle"]["videos_file"]
    if v["keyword"].isna().any() and kaggle_csv.exists():
        mapping = keyword_map(normalize_videos(pd.read_csv(kaggle_csv)))
        v["keyword"] = v["keyword"].fillna(v["video_id"].map(mapping))
    if v["keyword"].isna().any() and "category_id" in v:
        v["keyword"] = v["keyword"].fillna("categoria_" + v["category_id"].astype(str))
    return v


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
    skipped = [k for k, r in results.items() if r.get("no_aplica")]
    if skipped:
        log.warning("   análisis sin calcular con estos datos: %s (ver stats_report.md)", ", ".join(skipped))

    out = {"videos": len(v), "comentarios": len(c), "features": feats.shape}
    if not skip_model:
        log.info("6/6 Baselines y curva de aprendizaje")
        try:
            b = baseline.run(feats)
        except ValueError as err:  # datos insuficientes para entrenar y evaluar
            log.warning("   baselines no calculados: %s", err)
            baseline.write_skipped(str(err), len(feats), reports)
            out["baseline"] = {"no_aplica": str(err)}
        else:
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
