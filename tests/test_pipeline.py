"""Pruebas unitarias y de integración del pipeline de datos."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ytnlp.config import ROOT
from ytnlp.data.clean import clean_text
from ytnlp.data.kaggle_source import normalize_comments, normalize_videos
from ytnlp.data.validate import validate
from ytnlp.data.youtube_api import parse_duration
from ytnlp.features.build_features import LEAKY, build


@pytest.fixture(scope="session")
def sample():
    if not (ROOT / "data/sample/videos-stats.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/make_sample_data.py")], check=True)
    v = normalize_videos(pd.read_csv(ROOT / "data/sample/videos-stats.csv"))
    c = normalize_comments(pd.read_csv(ROOT / "data/sample/comments.csv"))
    return v, c


def test_parse_duration():
    assert parse_duration("PT1H2M3S") == 3723
    assert parse_duration("PT45S") == 45
    assert parse_duration("P1DT1M") == 86460
    assert parse_duration(None) is None


def test_clean_text_keeps_emoji_signal():
    t = clean_text("Me encantaaaaa 🔥 https://x.co @juan")
    assert ":fire:" in t and "<url>" in t and "<user>" in t
    assert "encantaaa" in t and "encantaaaa" not in t


def test_validation_detects_seeded_issues(sample):
    v, c = sample
    res = validate(v, c)
    q = res.report
    assert q["reglas_duras"]["comentarios"]["dropped_rows"] >= 1  # sentimiento = 5
    assert q["unicidad"]["videos_duplicados"] == 2
    assert q["consistencia"]["likes_mayor_que_views"] >= 1
    assert q["integridad_referencial"]["comentarios_sin_video"] >= 1
    assert res.comments["sentiment"].isin([0, 1, 2]).all()


def test_features_have_no_leakage(sample):
    from ytnlp.data.clean import clean_comments, clean_videos

    v, c = sample
    res = validate(v, c)
    feats = build(clean_videos(res.videos), clean_comments(res.comments))
    assert not (LEAKY & set(feats.columns))
    assert feats["video_id"].is_unique
    assert feats["polarization"].between(0, 1).all()
    eng = feats["target_engagement"].dropna()
    assert (eng > 0).all() and np.isfinite(eng).all()


def test_pipeline_end_to_end():
    from ytnlp.config import path
    from ytnlp.pipeline import run

    out = run("sample")
    assert out["videos"] > 0
    for f in ("quality_report.md", "stats_report.md", "baseline_report.md"):
        assert (path("reports") / f).exists()


def test_storage_env_redirects_data(tmp_path, monkeypatch):
    from ytnlp.config import ROOT, path

    monkeypatch.setenv("YTNLP_STORAGE", str(tmp_path))
    assert path("reports").is_relative_to(tmp_path)
    assert path("raw").is_relative_to(tmp_path)
    assert path("sample").is_relative_to(ROOT)


def test_kaggle_download_copies_files(tmp_path, monkeypatch):
    """La descarga usa kagglehub y copia los CSV aunque vengan en subcarpetas."""
    import types

    from ytnlp.data import kaggle_source

    cache = tmp_path / "cache" / "versions" / "1"
    cache.mkdir(parents=True)
    (cache / "videos-stats.csv").write_text("a\n1\n")
    (cache / "comments.csv").write_text("b\n2\n")
    fake = types.SimpleNamespace(dataset_download=lambda handle: str(tmp_path / "cache"))
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    dest = kaggle_source.download(tmp_path / "raw")
    assert (dest / "videos-stats.csv").exists() and (dest / "comments.csv").exists()


def test_kaggle_credentials_detection(monkeypatch, tmp_path):
    from ytnlp.data.kaggle_source import has_credentials

    monkeypatch.setenv("HOME", str(tmp_path))
    for v in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert not has_credentials()
    monkeypatch.setenv("KAGGLE_API_TOKEN", "x")
    assert has_credentials()


def test_team_overrides_live_in_storage(tmp_path, monkeypatch):
    from ytnlp import config

    monkeypatch.setenv("YTNLP_STORAGE", str(tmp_path))
    config.reload()
    assert config.load_config()["target"] == "engagement"
    saved = config.set_overrides(target="perf_class", api__max_comments_per_video=300)
    assert saved == tmp_path / "config_equipo.yaml"
    cfg = config.load_config()
    assert cfg["target"] == "perf_class"
    assert cfg["api"]["max_comments_per_video"] == 300
    assert cfg["api"]["batch_size"] == 50  # el resto de la sección se conserva
    with pytest.raises(KeyError):
        config.set_overrides(no_existe=1)
    config.reset_overrides()
    assert config.load_config()["target"] == "engagement"
    monkeypatch.delenv("YTNLP_STORAGE")
    config.reload()


def _fake_api_extraction(storage, ids, with_keyword=False):
    """Escribe una extracción con el formato de la API (sin keyword ni sentimiento)."""
    from ytnlp.config import path

    v = normalize_videos(pd.read_csv(ROOT / "data/sample/videos-stats.csv")).drop_duplicates("video_id")
    c = normalize_comments(pd.read_csv(ROOT / "data/sample/comments.csv"))
    vv = v[v.video_id.isin(ids)].copy()
    if not with_keyword:
        vv = vv.drop(columns=["keyword"])
    vv["category_id"] = "22"
    vv["extracted_at"] = pd.Timestamp("2026-10-01", tz="UTC")
    cc = c[c.video_id.isin(ids)].drop(columns=["sentiment"])
    d = path("raw") / "api" / "2026-10-01"
    d.mkdir(parents=True, exist_ok=True)
    vv.to_parquet(d / "videos.parquet", index=False)
    cc.to_parquet(d / "comments.parquet", index=False)


def test_sample_ids_spread_across_keywords(sample):
    from ytnlp.data.kaggle_source import sample_ids

    v, _ = sample
    ids = sample_ids(v, 16)
    kws = v.drop_duplicates("video_id").set_index("video_id").loc[ids, "keyword"]
    assert len(ids) == 16 and kws.nunique() == v["keyword"].nunique()
    assert kws.value_counts().max() - kws.value_counts().min() <= 1


def test_api_extraction_without_keyword_runs(tmp_path, monkeypatch):
    """Caso reportado: extracción de la API sin columna keyword (Kruskal-Wallis fallaba)."""
    import shutil

    from ytnlp import config
    from ytnlp.pipeline import run

    monkeypatch.setenv("YTNLP_STORAGE", str(tmp_path))
    config.reload()
    raw = config.path("raw") / "kaggle"
    raw.mkdir(parents=True, exist_ok=True)
    for f in ("videos-stats.csv", "comments.csv"):
        shutil.copy(ROOT / "data/sample" / f, raw / f)
    ids = pd.read_csv(ROOT / "data/sample/videos-stats.csv")["Video ID"].unique()[:50]
    _fake_api_extraction(tmp_path, ids)
    out = run("api")
    assert out["videos"] == 50
    videos = pd.read_parquet(config.path("interim") / "videos.parquet")
    assert videos["keyword"].notna().all() and videos["keyword"].nunique() > 1
    monkeypatch.delenv("YTNLP_STORAGE")
    config.reload()


def test_small_or_single_keyword_extraction_does_not_crash(tmp_path, monkeypatch):
    from ytnlp import config
    from ytnlp.pipeline import run

    monkeypatch.setenv("YTNLP_STORAGE", str(tmp_path))
    config.reload()
    v = pd.read_csv(ROOT / "data/sample/videos-stats.csv").drop_duplicates("Video ID")
    ids = v[v.Keyword == v.Keyword.iloc[0]]["Video ID"].head(10)
    _fake_api_extraction(tmp_path, ids, with_keyword=True)
    out = run("api")
    assert "no_aplica" in out["baseline"]
    report = (config.path("reports") / "stats_report.md").read_text(encoding="utf-8")
    assert "No se pudo calcular" in report
    monkeypatch.delenv("YTNLP_STORAGE")
    config.reload()
