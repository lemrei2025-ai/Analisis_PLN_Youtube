"""Descarga y normalización del dataset de Kaggle "YouTube Statistics".

Archivos originales:
- videos-stats.csv: Unnamed: 0, Title, Video ID, Published At, Keyword, Likes, Comments, Views
- comments.csv:     Unnamed: 0, Video ID, Comment, Likes, Sentiment (0 neg, 1 neutral, 2 pos)

Todo el proyecto trabaja con el esquema canónico en snake_case definido aquí.
"""

from __future__ import annotations

import logging
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


def download(dest: Path | None = None) -> Path:
    """Descarga el dataset con la API de Kaggle (requiere ~/.kaggle/kaggle.json)."""
    cfg = load_config()["kaggle"]
    dest = dest or path("raw") / "kaggle"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / cfg["videos_file"]).exists() and (dest / cfg["comments_file"]).exists():
        log.info("Dataset de Kaggle ya presente en %s", dest)
        return dest
    from kaggle.api.kaggle_api_extended import KaggleApi  # import tardío: requiere credenciales

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(cfg["dataset"], path=str(dest), unzip=True)
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
