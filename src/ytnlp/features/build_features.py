"""Construcción de la tabla de features por video.

Una fila = un video. Columnas:
- Objetivos (target_*): nunca se usan como features.
- Features de comentarios (agregadas): sentimiento, polarización, longitud, diversidad léxica...
- Features de contexto: keyword, día/hora de publicación, edad, duración, suscriptores.
- `doc`: comentarios concatenados para modelos de texto (TF-IDF, embeddings).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ytnlp.config import load_config

# Léxico mínimo de respaldo cuando la fuente no trae sentimiento (ej. API).
# Para el proyecto se recomienda reemplazarlo por un modelo multilingüe de Hugging Face.
_POS = set(
    "good great love amazing awesome best nice thanks thank cool helpful excellent perfect "
    "beautiful wow funny fun like :red_heart: :fire: :smiling_face_with_heart-eyes: "
    "bueno buena genial excelente gracias increíble encanta mejor hermoso".split()
)
_NEG = set(
    "bad worst hate boring terrible awful wrong fake stupid dislike sad annoying useless "
    "clickbait :thumbs_down: :angry_face: malo mala peor odio aburrido horrible falso".split()
)

TARGETS = ["target_engagement", "target_log_views", "target_log_likes", "target_perf_class"]
LEAKY = {"likes", "comments", "views", "engagement"}


def lexicon_sentiment(text: str) -> int:
    toks = text.split()
    score = sum(t in _POS for t in toks) - sum(t in _NEG for t in toks)
    return 2 if score > 0 else 0 if score < 0 else 1


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(3)) if len(p) else 0.0


def _ttr(texts: pd.Series) -> float:
    toks = " ".join(texts).split()
    return len(set(toks)) / len(toks) if toks else 0.0


def comment_features(comments: pd.DataFrame) -> pd.DataFrame:
    df = comments.copy()
    if "sentiment" not in df or df["sentiment"].isna().all():
        df["sentiment"] = df["comment"].map(lexicon_sentiment)
        df["sentiment_source"] = "lexicon"
    df["sent_signed"] = df["sentiment"] - 1  # -1 neg, 0 neutral, 1 pos
    df["w"] = np.log1p(df["comment_likes"].fillna(0)) + 1
    df["has_question"] = df["comment"].str.contains(r"\?", regex=True)
    df["has_exclaim"] = df["comment"].str.contains("!", regex=False)
    df["n_emoji"] = df["comment"].str.count(r":[a-z0-9_\-&]+:")

    g = df.groupby("video_id")
    feats = pd.DataFrame(
        {
            "n_comments_sample": g.size(),
            "sent_mean": g["sent_signed"].mean(),
            "sent_std": g["sent_signed"].std().fillna(0),
            "sent_weighted": g.apply(lambda x: np.average(x["sent_signed"], weights=x["w"]), include_groups=False),
            "pct_pos": g["sentiment"].apply(lambda s: (s == 2).mean()),
            "pct_neg": g["sentiment"].apply(lambda s: (s == 0).mean()),
            "polarization": g["sentiment"].apply(
                lambda s: _entropy(np.bincount(s.astype(int), minlength=3).astype(float))
            ),
            "top_comment_sent": g.apply(
                lambda x: x.loc[x["comment_likes"].idxmax(), "sent_signed"], include_groups=False
            ),
            "tokens_mean": g["n_tokens"].mean(),
            "ttr": g["comment"].apply(_ttr),
            "pct_question": g["has_question"].mean(),
            "pct_exclaim": g["has_exclaim"].mean(),
            "emoji_per_comment": g["n_emoji"].mean(),
            "log_comment_likes_sum": np.log1p(g["comment_likes"].sum()),
            "doc": g["comment"].apply(lambda s: " ".join(s)),
        }
    )
    return feats.reset_index()


def context_features(videos: pd.DataFrame) -> pd.DataFrame:
    v = videos.copy()
    ref = (
        pd.to_datetime(v["extracted_at"], utc=True)
        if "extracted_at" in v
        else v["published_at"].max()
    )
    out = pd.DataFrame({"video_id": v["video_id"]})
    out["keyword"] = v["keyword"].fillna("unknown") if "keyword" in v else "unknown"
    out["pub_dow"] = v["published_at"].dt.dayofweek
    out["pub_hour"] = v["published_at"].dt.hour
    out["age_days"] = (ref - v["published_at"]).dt.days.clip(lower=1)
    out["log_age_days"] = np.log1p(out["age_days"])
    if "title" in v:
        out["title_len"] = v["title"].fillna("").str.len()
        out["title_has_question"] = v["title"].fillna("").str.contains(r"\?").astype(int)
        out["title_upper_ratio"] = v["title"].fillna("").map(
            lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s))
        )
    if "duration_s" in v:
        out["log_duration"] = np.log1p(v["duration_s"])
    if "subscribers" in v:
        out["log_subscribers"] = np.log1p(v["subscribers"])
    return out


def targets(videos: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    v = videos
    out = pd.DataFrame({"video_id": v["video_id"]})
    eng = (v["likes"] + v["comments"]) / v["views"]
    out["target_engagement"] = eng.where(v["views"] >= cfg["min_views_for_engagement"])
    out["target_log_views"] = np.log1p(v["views"])
    out["target_log_likes"] = np.log1p(v["likes"])
    q_lo, q_hi = cfg["perf_class_quantiles"]
    lo, hi = out["target_engagement"].quantile([q_lo, q_hi])
    out["target_perf_class"] = pd.cut(
        out["target_engagement"], [-np.inf, lo, hi, np.inf], labels=["bajo", "medio", "alto"]
    ).astype("string")
    return out


def build(videos: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    table = (
        targets(videos)
        .merge(context_features(videos), on="video_id", how="left")
        .merge(comment_features(comments), on="video_id", how="inner")
    )
    assert not (LEAKY & set(table.columns)), "Columnas con fuga de información en features"
    return table
