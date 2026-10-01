"""Validación (Pandera) y reporte de calidad de datos.

Dos niveles de reglas:
- Duras (esquema): si una fila las viola se descarta y se reporta.
- Blandas (chequeos de calidad): se miden y se reportan, no descartan filas.

El reporte también emite métricas de monitoreo que se comparan entre extracciones.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMOJI_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

VIDEO_SCHEMA = pa.DataFrameSchema(
    {
        "video_id": pa.Column(str, nullable=False),
        "title": pa.Column(str, nullable=True, required=False),
        "published_at": pa.Column("datetime64[ns, UTC]", nullable=True, coerce=True),
        "keyword": pa.Column(str, nullable=True, required=False),
        "likes": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "comments": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "views": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
    },
    strict=False,
)

COMMENT_SCHEMA = pa.DataFrameSchema(
    {
        "video_id": pa.Column(str, nullable=False),
        "comment": pa.Column(str, nullable=True),
        "comment_likes": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "sentiment": pa.Column(
            float, pa.Check.isin([0, 1, 2]), nullable=True, coerce=True, required=False
        ),
    },
    strict=False,
)


@dataclass
class ValidationResult:
    videos: pd.DataFrame
    comments: pd.DataFrame
    report: dict = field(default_factory=dict)


def _apply_schema(df: pd.DataFrame, schema: pa.DataFrameSchema) -> tuple[pd.DataFrame, dict]:
    try:
        return schema.validate(df, lazy=True), {}
    except SchemaErrors as err:
        fc = err.failure_cases
        bad_idx = fc["index"].dropna().unique()
        summary = (
            fc.groupby(["column", "check"], dropna=False).size().reset_index(name="n")
            .astype({"column": str, "check": str})
            .to_dict("records")
        )
        clean = df.drop(index=[i for i in bad_idx if i in df.index])
        return schema.validate(clean, lazy=False), {"dropped_rows": int(len(bad_idx)), "failures": summary}


def _pct(x: float) -> float:
    return round(100 * float(x), 2)


def quality_checks(videos: pd.DataFrame, comments: pd.DataFrame) -> dict:
    now = pd.Timestamp(datetime.now(timezone.utc))
    text = comments["comment"].fillna("").astype(str)
    tokens = text.str.split().str.len()
    cpv = comments.groupby("video_id").size()

    q: dict = {}
    q["volumen"] = {
        "videos": int(len(videos)),
        "comentarios": int(len(comments)),
        "comentarios_por_video_mediana": float(cpv.median()) if len(cpv) else 0.0,
        "comentarios_por_video_min": int(cpv.min()) if len(cpv) else 0,
        "comentarios_por_video_max": int(cpv.max()) if len(cpv) else 0,
        "videos_sin_comentarios": int((~videos["video_id"].isin(comments["video_id"])).sum()),
    }
    q["completitud_pct_nulos"] = {
        "videos": {c: _pct(videos[c].isna().mean()) for c in videos.columns},
        "comentarios": {c: _pct(comments[c].isna().mean()) for c in comments.columns},
    }
    q["unicidad"] = {
        "videos_duplicados": int(videos.duplicated("video_id").sum()),
        "comentarios_duplicados": int(comments.duplicated(["video_id", "comment"]).sum()),
    }
    q["consistencia"] = {
        "likes_mayor_que_views": int((videos["likes"] > videos["views"]).sum()),
        "comments_mayor_que_views": int((videos["comments"] > videos["views"]).sum()),
        "fechas_futuras": int((videos["published_at"] > now).sum()),
        "views_cero": int((videos["views"] == 0).sum()),
    }
    q["integridad_referencial"] = {
        "comentarios_sin_video": int((~comments["video_id"].isin(videos["video_id"])).sum()),
    }
    q["ruido_texto"] = {
        "pct_vacios": _pct((text.str.strip() == "").mean()),
        "pct_menos_de_3_tokens": _pct((tokens < 3).mean()),
        "pct_con_url": _pct(text.str.contains(URL_RE).mean()),
        "pct_solo_simbolos_o_emojis": _pct(text.str.match(EMOJI_ONLY_RE).mean()),
        "pct_no_ascii_mayoritario": _pct(
            text.map(lambda s: sum(ord(ch) > 127 for ch in s) / max(len(s), 1) > 0.5).mean()
        ),
        "tokens_mediana": float(tokens.median()) if len(tokens) else 0.0,
    }
    bal: dict = {}
    if "keyword" in videos:
        vc = videos["keyword"].value_counts()
        p = vc / vc.sum()
        bal["keywords"] = int(len(vc))
        bal["videos_por_keyword_min"] = int(vc.min())
        bal["videos_por_keyword_max"] = int(vc.max())
        bal["entropia_normalizada_keyword"] = round(
            float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0, 3
        )
    if "sentiment" in comments:
        bal["sentimiento_pct"] = {
            str(int(k)): _pct(v)
            for k, v in comments["sentiment"].value_counts(normalize=True).sort_index().items()
        }
    q["balance"] = bal
    views = videos["views"].dropna()
    if len(views) > 2:
        q["representatividad"] = {
            "views_mediana": float(views.median()),
            "views_p90": float(views.quantile(0.9)),
            "asimetria_log_views": round(float(np.log1p(views).skew()), 3),
            "pct_views_top_10pct_videos": _pct(
                views.nlargest(max(1, len(views) // 10)).sum() / views.sum()
            ),
        }
    if videos["published_at"].notna().any():
        age = (now - videos["published_at"]).dt.days
        q["vigencia"] = {
            "edad_dias_mediana": float(age.median()),
            "edad_dias_min": float(age.min()),
            "edad_dias_max": float(age.max()),
        }
    vocab = set(" ".join(text.str.lower()).split())
    q["monitoreo"] = {
        "timestamp": now.isoformat(),
        "n_videos": int(len(videos)),
        "n_comentarios": int(len(comments)),
        "pct_nulos_comment": _pct(comments["comment"].isna().mean()),
        "sentimiento_medio": float(comments["sentiment"].mean()) if "sentiment" in comments else None,
        "vocabulario": int(len(vocab)),
    }
    return q


def validate(videos: pd.DataFrame, comments: pd.DataFrame) -> ValidationResult:
    videos = videos.copy()
    comments = comments.copy()
    videos["video_id"] = videos["video_id"].astype("string").astype(object)
    comments["video_id"] = comments["video_id"].astype("string").astype(object)
    for df, cols in ((videos, ["title", "keyword"]), (comments, ["comment"])):
        for c in cols:
            if c in df:
                df[c] = df[c].astype(object).where(df[c].notna(), None)

    v_ok, v_err = _apply_schema(videos, VIDEO_SCHEMA)
    c_ok, c_err = _apply_schema(comments, COMMENT_SCHEMA)
    report = {
        "reglas_duras": {"videos": v_err or {"dropped_rows": 0}, "comentarios": c_err or {"dropped_rows": 0}},
        **quality_checks(v_ok, c_ok),
    }
    return ValidationResult(v_ok.reset_index(drop=True), c_ok.reset_index(drop=True), report)


def _md_table(d: dict) -> str:
    lines = ["| Métrica | Valor |", "| --- | --- |"]
    for k, v in d.items():
        lines.append(f"| {k} | {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} |")
    return "\n".join(lines)


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    parts = ["# Reporte de calidad de datos\n"]
    for section, content in report.items():
        parts.append(f"## {section.replace('_', ' ').capitalize()}\n")
        parts.append(_md_table(content) if isinstance(content, dict) else str(content))
        parts.append("")
    md = out_dir / "quality_report.md"
    md.write_text("\n".join(parts), encoding="utf-8")
    _append_monitoring(report["monitoreo"], out_dir)
    return md


def _append_monitoring(metrics: dict, out_dir: Path) -> None:
    """Historial de métricas de monitoreo para detectar drift entre extracciones."""
    hist = out_dir / "monitoring_history.csv"
    row = pd.DataFrame([metrics])
    row.to_csv(hist, mode="a", header=not hist.exists(), index=False)
