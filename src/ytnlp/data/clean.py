"""Limpieza de texto y de métricas.

Decisiones documentadas:
- Los emojis se convierten a texto (:fire:) en lugar de eliminarse: llevan señal de sentimiento.
- Las URLs y menciones se reemplazan por tokens <url> y <user>.
- Se eliminan comentarios vacíos y duplicados exactos por video (spam repetido).
- Las métricas nulas de videos NO se imputan con la media: se dejan nulas y el video se
  excluye solo de los análisis que necesitan esa métrica.
"""

from __future__ import annotations

import re
import unicodedata

import emoji
import pandas as pd

from ytnlp.config import load_config

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(.)\1{3,}")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str, demojize: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = URL_RE.sub(" <url> ", t)
    t = MENTION_RE.sub(" <user> ", t)
    if demojize:
        t = emoji.demojize(t, delimiters=(" :", ": "))
    t = REPEAT_RE.sub(r"\1\1\1", t)  # "siiiiiii" -> "siii"
    t = SPACE_RE.sub(" ", t).strip().lower()
    return t


def clean_comments(comments: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()["cleaning"]
    df = comments.copy()
    df["comment_raw"] = df["comment"]
    df["comment"] = df["comment"].map(lambda s: clean_text(s, cfg["demojize"]))
    df["n_tokens"] = df["comment"].str.split().str.len().fillna(0).astype(int)
    df = df[df["n_tokens"] >= cfg["min_comment_tokens"]]
    if cfg["drop_duplicate_comments"]:
        df = df.drop_duplicates(["video_id", "comment"])
    df["comment_likes"] = df["comment_likes"].fillna(0)
    return df.reset_index(drop=True)


def clean_videos(videos: pd.DataFrame) -> pd.DataFrame:
    df = videos.drop_duplicates("video_id").copy()
    if "title" in df:
        df["title_clean"] = df["title"].map(clean_text)
    return df.reset_index(drop=True)
