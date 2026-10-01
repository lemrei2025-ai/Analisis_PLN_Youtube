"""Baselines y curva de aprendizaje.

Responde dos preguntas de la etapa de datos:
1. ¿Los comentarios aportan algo frente a un modelo sin texto?  (comparación de baselines)
2. ¿Los datos son suficientes?  (curva de aprendizaje: si el error sigue bajando, faltan datos)

El split es por video (una fila = un video), nunca por comentario.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ytnlp.analysis.stats import COMMENT_FEATS
from ytnlp.config import load_config

MIN_VIDEOS = 30  # por debajo, el conjunto de prueba es demasiado pequeño para comparar modelos
CONTEXT_NUM = ["log_age_days", "pub_dow", "pub_hour", "title_len", "title_has_question",
               "title_upper_ratio", "log_duration", "log_subscribers"]


def _target(df: pd.DataFrame, name: str) -> tuple[pd.Series, bool]:
    col = f"target_{name}"
    y = df[col]
    if name == "engagement":
        y = np.log(y)  # log-engagement: distribución más simétrica
    return y, name == "perf_class"


def _make(cols_num: list[str], use_cat: bool, use_text: bool, clf: bool) -> Pipeline:
    parts = []
    if cols_num:
        parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cols_num))
    if use_cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["keyword"]))
    if use_text:
        parts.append(("txt", TfidfVectorizer(min_df=2, max_features=20000, ngram_range=(1, 2),
                                             sublinear_tf=True), "doc"))
    est = LogisticRegression(max_iter=2000, C=1.0) if clf else Ridge(alpha=1.0)
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", est)])


def _score(y_true, y_pred, clf: bool) -> dict:
    if clf:
        return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
                "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4)}
    y_pred = np.asarray(y_pred, dtype=float)
    constant = np.allclose(y_pred, y_pred[0])  # el dummy predice un valor constante
    rho = 0.0 if constant else stats.spearmanr(y_true, y_pred)[0]
    return {"MAE": round(mean_absolute_error(y_true, y_pred), 4),
            "R2": round(r2_score(y_true, y_pred), 4),
            "spearman": round(float(rho), 4)}


def run(features: pd.DataFrame, target: str | None = None) -> dict:
    cfg = load_config()
    target = target or cfg["target"]
    df = features.copy()
    y, clf = _target(df, target)
    mask = y.notna() & (np.isfinite(y) if not clf else True)
    df, y = df[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    if len(df) < MIN_VIDEOS:
        raise ValueError(
            f"se necesitan al menos {MIN_VIDEOS} videos con la variable objetivo; hay {len(df)}"
        )
    if clf and y.nunique() < 2:
        raise ValueError("la variable objetivo tiene una sola clase")
    ctx = [c for c in CONTEXT_NUM if c in df and df[c].notna().any()]

    can_stratify = clf and y.value_counts().min() >= 2
    X_tr, X_te, y_tr, y_te = train_test_split(
        df, y, test_size=cfg["model"]["test_size"], random_state=cfg["model"]["random_state"],
        stratify=y if can_stratify else None,
    )
    candidates = {
        "dummy": DummyClassifier(strategy="most_frequent") if clf else DummyRegressor(),
        "contexto (sin texto)": _make(ctx, True, False, clf),
        "contexto + features de comentarios": _make(ctx + COMMENT_FEATS, True, False, clf),
        "contexto + features + TF-IDF": _make(ctx + COMMENT_FEATS, True, True, clf),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        results[name] = _score(y_te, model.predict(X_te), clf)

    key = "f1_macro" if clf else "R2"
    best = max((k for k in results if k != "dummy"), key=lambda k: results[k][key])
    curve = []
    for frac in cfg["model"]["learning_curve_fractions"]:
        n = max(10, int(frac * len(X_tr)))
        Xs, ys = X_tr.iloc[:n], y_tr.iloc[:n]
        m = candidates[best]
        m.fit(Xs, ys)
        curve.append({"fraccion": frac, "n_train": n,
                      "train": _score(ys, m.predict(Xs), clf)[key],
                      "test": _score(y_te, m.predict(X_te), clf)[key]})
    gain = curve[-1]["test"] - curve[-2]["test"]
    return {
        "target": target,
        "tipo": "clasificación" if clf else "regresión (log)",
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "metrica_principal": key,
        "resultados": results,
        "mejor_modelo": best,
        "curva_aprendizaje": curve,
        "diagnostico_suficiencia": (
            "La métrica de test sigue mejorando en el último tramo: más datos probablemente ayudan."
            if gain > 0.01 else
            "La curva se aplana: más filas ayudan poco; priorizar mejores features o etiquetas."
        ),
    }


def write_skipped(reason: str, n_videos: int, out_dir: Path) -> Path:
    """Reporte cuando no hay datos suficientes para entrenar y evaluar los baselines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_report.json").write_text(
        json.dumps({"no_aplica": reason, "videos": n_videos}, indent=2, ensure_ascii=False)
    )
    md = out_dir / "baseline_report.md"
    md.write_text(
        "# Baselines y suficiencia de datos\n\n"
        f"No se calcularon con estos datos: {reason}.\n\n"
        f"Videos disponibles: {n_videos}. Este resultado ya responde la pregunta de suficiencia: "
        "con esta cantidad de videos no es posible entrenar y evaluar un modelo de forma confiable. "
        "Se recomienda extraer más videos o usar el dataset de Kaggle completo.\n",
        encoding="utf-8",
    )
    return md


def write_report(res: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_report.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    k = res["metrica_principal"]
    lines = [
        "# Baselines y suficiencia de datos\n",
        f"Objetivo: `{res['target']}` ({res['tipo']}). Train={res['n_train']}, test={res['n_test']}.\n",
        "| Modelo | " + " | ".join(next(iter(res["resultados"].values())).keys()) + " |",
        "| --- |" + " --- |" * len(next(iter(res["resultados"].values()))),
    ]
    for name, sc in res["resultados"].items():
        lines.append(f"| {name} | " + " | ".join(str(v) for v in sc.values()) + " |")
    lines += [
        f"\nMejor modelo: **{res['mejor_modelo']}**.\n",
        f"## Curva de aprendizaje ({k})\n",
        "| Fracción | n_train | Train | Test |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {c['fraccion']} | {c['n_train']} | {c['train']} | {c['test']} |"
              for c in res["curva_aprendizaje"]]
    lines.append(f"\n**Diagnóstico:** {res['diagnostico_suficiencia']}")
    md = out_dir / "baseline_report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md
