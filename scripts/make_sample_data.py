"""Genera una muestra SINTÉTICA con el mismo formato del dataset de Kaggle.

Sirve para probar el pipeline y la CI sin credenciales. No usar para conclusiones.
Incluye problemas de calidad sembrados a propósito (nulos, duplicados, huérfanos, valores
inconsistentes) para verificar que la validación los detecta.

Uso: python scripts/make_sample_data.py [--videos 300] [--comments-per-video 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample"

KEYWORDS = ["tech", "news", "gaming", "sports", "how-to", "music", "food", "education"]
POS = ["great video", "love this", "amazing content", "thanks for sharing", "so helpful",
       "best tutorial ever", "this is awesome 🔥", "genial video", "excelente explicación ❤️",
       "learned a lot today", "subscribed!", "perfect timing"]
NEU = ["first time watching", "what camera do you use?", "who is here in 2024", "part 2?",
       "watching from Colombia", "interesting", "ok", "can you review the new model?",
       "minute 3:20", "anyone else?"]
NEG = ["clickbait", "this is boring", "worst video", "wrong information", "too long 👎",
       "audio is terrible", "fake", "dislike", "no me gustó", "waste of time"]


def main(n_videos: int, cpv: int, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)

    ids = [f"vid{i:05d}" for i in range(n_videos)]
    kw = rng.choice(KEYWORDS, n_videos)
    kw_effect = {k: e for k, e in zip(KEYWORDS, rng.normal(0, 0.3, len(KEYWORDS)))}
    quality = rng.normal(0, 1, n_videos)  # calidad latente: mueve sentimiento y engagement
    log_views = rng.normal(11, 2.2, n_videos)
    views = np.round(np.exp(log_views))
    eng = np.exp(-3.6 + 0.35 * quality + np.vectorize(kw_effect.get)(kw) + rng.normal(0, 0.4, n_videos))
    likes = np.round(views * eng * 0.9)
    comments = np.round(views * eng * 0.1)
    published = pd.Timestamp("2022-08-01") + pd.to_timedelta(rng.integers(0, 365, n_videos), "D")

    videos = pd.DataFrame({
        "Title": [f"{k} video #{i}" + ("?" if rng.random() < 0.2 else "") for i, k in enumerate(kw)],
        "Video ID": ids,
        "Published At": published.strftime("%Y-%m-%d"),
        "Keyword": kw,
        "Likes": likes,
        "Comments": comments,
        "Views": views,
    })
    # Problemas sembrados
    videos.loc[rng.choice(n_videos, 3, replace=False), "Likes"] = np.nan
    videos.loc[rng.choice(n_videos, 2, replace=False), "Views"] = np.nan
    videos.loc[5, "Likes"] = videos.loc[5, "Views"] * 2  # likes > views
    videos = pd.concat([videos, videos.iloc[[10, 11]]], ignore_index=True)  # duplicados

    rows = []
    for vid, q in zip(ids, quality):
        p_pos = 1 / (1 + np.exp(-(0.3 + 0.9 * q)))
        p_neg = 0.25 * (1 - p_pos)
        for _ in range(cpv):
            u = rng.random()
            if u < p_pos:
                text, s = rng.choice(POS), 2
            elif u < p_pos + p_neg:
                text, s = rng.choice(NEG), 0
            else:
                text, s = rng.choice(NEU), 1
            rows.append({"Video ID": vid, "Comment": text,
                         "Likes": float(rng.poisson(np.exp(rng.normal(1.5, 1.2)))), "Sentiment": float(s)})
    com = pd.DataFrame(rows)
    com.loc[rng.choice(len(com), 5, replace=False), "Comment"] = np.nan
    com = pd.concat([com, pd.DataFrame([
        {"Video ID": "huerfano01", "Comment": "comment without video", "Likes": 1.0, "Sentiment": 1.0},
        {"Video ID": ids[0], "Comment": "invalid sentiment", "Likes": 0.0, "Sentiment": 5.0},
    ])], ignore_index=True)

    videos.insert(0, "Unnamed: 0", range(len(videos)))
    com.insert(0, "Unnamed: 0", range(len(com)))
    videos.to_csv(OUT / "videos-stats.csv", index=False)
    com.to_csv(OUT / "comments.csv", index=False)
    print(f"Muestra sintética: {len(videos)} videos, {len(com)} comentarios en {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=300)
    ap.add_argument("--comments-per-video", type=int, default=10)
    a = ap.parse_args()
    main(a.videos, a.comments_per_video)
