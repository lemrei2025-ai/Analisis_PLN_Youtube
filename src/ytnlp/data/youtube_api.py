"""Extracción de datos con YouTube Data API v3.

Mejora el dataset de Kaggle en tres frentes:
1. Más comentarios por video (hasta `api.max_comments_per_video`) con likes y respuestas.
2. Variables de contexto: duración, categoría, tags, idioma y suscriptores del canal.
3. Fecha de extracción en cada fila, para construir series de tiempo y medir drift.

Costo de cuota (unidades): videos.list = 1, channels.list = 1, commentThreads.list = 1 por página.
Se evita search.list (limitado a 100 llamadas diarias) partiendo de una lista conocida de Video IDs.

Uso:
    python -m ytnlp.data.youtube_api --from-kaggle
    python -m ytnlp.data.youtube_api --ids dQw4w9WgXcQ,9bZkp7q19f0
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

from ytnlp.config import load_config, path

log = logging.getLogger(__name__)
BASE_URL = "https://www.googleapis.com/youtube/v3"
_DURATION_RE = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


class QuotaExceeded(RuntimeError):
    pass


def parse_duration(iso: str | None) -> float | None:
    """Convierte una duración ISO 8601 (PT1H2M3S) a segundos."""
    if not iso:
        return None
    m = _DURATION_RE.fullmatch(iso)
    if not m:
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return float(d * 86400 + h * 3600 + mi * 60 + s)


def _chunks(items: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


class YouTubeClient:
    def __init__(self, api_key: str | None = None, daily_quota: int | None = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Falta YOUTUBE_API_KEY (ver .env.example)")
        self.quota_left = daily_quota or load_config()["api"]["daily_quota"]
        self.session = requests.Session()

    def _get(self, resource: str, params: dict, cost: int = 1) -> dict:
        if self.quota_left < cost:
            raise QuotaExceeded("Cuota diaria agotada; reanudar mañana")
        params = {**params, "key": self.api_key}
        for attempt in range(4):
            r = self.session.get(f"{BASE_URL}/{resource}", params=params, timeout=30)
            if r.status_code == 200:
                self.quota_left -= cost
                return r.json()
            reason = ""
            try:
                reason = r.json()["error"]["errors"][0]["reason"]
            except Exception:  # noqa: BLE001
                pass
            if reason in {"commentsDisabled", "videoNotFound"}:
                return {"items": [], "_skipped": reason}
            if reason == "quotaExceeded":
                raise QuotaExceeded(reason)
            if r.status_code in {429, 500, 503}:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
        r.raise_for_status()
        return {}

    def videos(self, ids: list[str]) -> pd.DataFrame:
        rows = []
        for batch in _chunks(ids, load_config()["api"]["batch_size"]):
            data = self._get(
                "videos",
                {"part": "snippet,statistics,contentDetails", "id": ",".join(batch)},
            )
            for it in data.get("items", []):
                sn, st, cd = it["snippet"], it.get("statistics", {}), it["contentDetails"]
                rows.append(
                    {
                        "video_id": it["id"],
                        "title": sn.get("title"),
                        "published_at": sn.get("publishedAt"),
                        "channel_id": sn.get("channelId"),
                        "category_id": sn.get("categoryId"),
                        "tags": "|".join(sn.get("tags", [])),
                        "default_language": sn.get("defaultAudioLanguage")
                        or sn.get("defaultLanguage"),
                        "duration_s": parse_duration(cd.get("duration")),
                        "views": st.get("viewCount"),
                        "likes": st.get("likeCount"),
                        "comments": st.get("commentCount"),
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        for c in ("views", "likes", "comments"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def channels(self, channel_ids: list[str]) -> pd.DataFrame:
        rows = []
        for batch in _chunks(sorted(set(channel_ids)), 50):
            data = self._get("channels", {"part": "statistics", "id": ",".join(batch)})
            for it in data.get("items", []):
                st = it.get("statistics", {})
                rows.append(
                    {
                        "channel_id": it["id"],
                        "subscribers": pd.to_numeric(st.get("subscriberCount"), errors="coerce"),
                        "channel_videos": pd.to_numeric(st.get("videoCount"), errors="coerce"),
                    }
                )
        return pd.DataFrame(rows)

    def comments(self, video_id: str, max_comments: int, order: str = "relevance") -> list[dict]:
        out: list[dict] = []
        token = None
        while len(out) < max_comments:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_comments - len(out)),
                "order": order,
                "textFormat": "plainText",
            }
            if token:
                params["pageToken"] = token
            data = self._get("commentThreads", params)
            for it in data.get("items", []):
                top = it["snippet"]["topLevelComment"]["snippet"]
                out.append(
                    {
                        "video_id": video_id,
                        "comment": top.get("textDisplay"),
                        "comment_likes": top.get("likeCount"),
                        "reply_count": it["snippet"].get("totalReplyCount"),
                        "comment_published_at": top.get("publishedAt"),
                    }
                )
            token = data.get("nextPageToken")
            if not token:
                break
        return out


def extract(
    video_ids: list[str],
    out_dir: Path | None = None,
    keywords: dict[str, str] | None = None,
) -> Path:
    """Extrae videos y comentarios. `keywords` (video_id -> tema) conserva la keyword de Kaggle."""
    cfg = load_config()["api"]
    stamp = datetime.now(timezone.utc)
    out_dir = out_dir or path("raw") / "api" / stamp.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = YouTubeClient()

    videos = client.videos(video_ids)
    if videos.empty:
        raise RuntimeError("La API no devolvió videos")
    channels = client.channels(videos["channel_id"].dropna().tolist())
    videos = videos.merge(channels, on="channel_id", how="left")
    videos["extracted_at"] = stamp
    if keywords:
        videos["keyword"] = videos["video_id"].map(keywords)

    rows: list[dict] = []
    try:
        for i, vid in enumerate(videos["video_id"], 1):
            rows.extend(client.comments(vid, cfg["max_comments_per_video"], cfg["order"]))
            if i % 50 == 0:
                log.info("%d/%d videos, cuota restante %d", i, len(videos), client.quota_left)
    except QuotaExceeded:
        log.warning("Cuota agotada; se guardan los comentarios obtenidos hasta ahora")
    comments = pd.DataFrame(rows)
    if not comments.empty:
        comments["extracted_at"] = stamp

    videos.to_parquet(out_dir / "videos.parquet", index=False)
    comments.to_parquet(out_dir / "comments.parquet", index=False)
    log.info("Guardados %d videos y %d comentarios en %s", len(videos), len(comments), out_dir)
    return out_dir


def load_latest() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = path("raw") / "api"
    runs = sorted(p for p in base.glob("*") if (p / "videos.parquet").exists())
    if not runs:
        raise FileNotFoundError("No hay extracciones de la API; ejecutar `make extract`")
    last = runs[-1]
    return pd.read_parquet(last / "videos.parquet"), pd.read_parquet(last / "comments.parquet")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", help="Video IDs separados por coma")
    g.add_argument("--ids-file", help="Archivo con un Video ID por línea")
    g.add_argument("--from-kaggle", action="store_true", help="Usa los IDs del dataset de Kaggle")
    ap.add_argument("--limit", type=int, default=None, help="Máximo de videos a extraer")
    args = ap.parse_args()

    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    elif args.ids_file:
        ids = [x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()]
    keywords = None
    if args.from_kaggle:
        from ytnlp.data import kaggle_source

        kv = kaggle_source.load()[0]
        keywords = kaggle_source.keyword_map(kv)
        ids = kaggle_source.sample_ids(kv, args.limit or len(kv))
    if args.limit:
        ids = ids[: args.limit]
    extract(ids, keywords=keywords)


if __name__ == "__main__":
    main()
