"""Descarga y normalización del dataset de Kaggle "YouTube Statistics".

Archivos originales:
- videos-stats.csv: Unnamed: 0, Title, Video ID, Published At, Keyword, Likes, Comments, Views
- comments.csv:     Unnamed: 0, Video ID, Comment, Likes, Sentiment (0 neg, 1 neutral, 2 pos)

Todo el proyecto trabaja con el esquema canónico en snake_case definido aquí.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pandas as pd

from ytnlp.config import load_config, path

log = logging.getLogger(__name__)

VIDEO_COLUMNS = {
    "Video ID": "video_id",
    "Title": "title",
    "Published At": "published_at",
    "Keyword": "keyword",
    "Likes": "likes",
    "Comments": "comments",
    "Views": "views",
}
COMMENT_COLUMNS = {
    "Video ID": "video_id",
    "Comment": "comment",
    "Likes": "comment_likes",
    "Sentiment": "sentiment",
}


def has_credentials() -> bool:
    """True si hay credenciales de Kaggle: token nuevo o par usuario/clave heredado."""
    return bool(
        os.getenv("KAGGLE_API_TOKEN")
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        or (Path.home() / ".kaggle" / "kaggle.json").exists()
        or (Path.home() / ".kaggle" / "access_token").exists()
    )


def download(dest: Path | None = None) -> Path:
    """Descarga el dataset con kagglehub y copia los CSV a data/raw/kaggle.

    Credenciales aceptadas (en este orden): KAGGLE_API_TOKEN (token nuevo de
    kaggle.com/settings/api), KAGGLE_USERNAME + KAGGLE_KEY o ~/.kaggle/kaggle.json (heredadas).
    En Colab, kagglehub también las lee directamente de los Secretos.
    """
    cfg = load_config()["kaggle"]
    dest = dest or path("raw") / "kaggle"
    dest.mkdir(parents=True, exist_ok=True)
    files = (cfg["videos_file"], cfg["comments_file"])
    if all((dest / f).exists() for f in files):
        log.info("Dataset de Kaggle ya presente en %s", dest)
        return dest
    import kagglehub  # import tardío: solo se necesita al descargar

    cache = Path(kagglehub.dataset_download(cfg["dataset"]))
    for f in files:
        found = next(cache.rglob(f), None)
        if found is None:
            raise FileNotFoundError(f"{f} no está en la descarga de {cfg['dataset']}")
        shutil.copy2(found, dest / f)
    log.info("Dataset descargado en %s", dest)
    return dest


def normalize_videos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=VIDEO_COLUMNS)
    df = df[[c for c in VIDEO_COLUMNS.values() if c in df.columns]].copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    for c in ("likes", "comments", "views"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_comments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COMMENT_COLUMNS)
    df = df[[c for c in COMMENT_COLUMNS.values() if c in df.columns]].copy()
    df["comment"] = df["comment"].astype("string")
    df["comment_likes"] = pd.to_numeric(df["comment_likes"], errors="coerce")
    if "sentiment" in df:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    return df


def load(folder: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()["kaggle"]
    folder = folder or download()
    videos = normalize_videos(pd.read_csv(folder / cfg["videos_file"]))
    comments = normalize_comments(pd.read_csv(folder / cfg["comments_file"]))
    return videos, comments


def sample_ids(videos: pd.DataFrame, n: int, seed: int = 42) -> list[str]:
    """Elige n Video IDs repartidos por igual entre las keywords (muestreo estratificado).

    Si el CSV está agrupado por keyword, tomar los primeros n IDs dejaría casi todos los
    videos en un solo tema y haría imposible comparar grupos.
    """
    v = videos.dropna(subset=["video_id"]).drop_duplicates("video_id")
    v = v.sample(frac=1, random_state=seed)
    if "keyword" in v and v["keyword"].nunique() > 1:
        v = v.assign(_turno=v.groupby("keyword").cumcount()).sort_values("_turno", kind="stable")
    return v["video_id"].head(n).tolist()


def keyword_map(videos: pd.DataFrame) -> dict[str, str]:
    """Diccionario video_id -> keyword, para conservar el tema en los datos de la API."""
    v = videos.dropna(subset=["video_id", "keyword"]).drop_duplicates("video_id")
    return dict(zip(v["video_id"], v["keyword"]))

