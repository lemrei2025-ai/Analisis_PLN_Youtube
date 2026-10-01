"""Análisis estadísticos no triviales sobre la tabla de features y los comentarios.

Cada análisis devuelve un dict con: pregunta, hipótesis, prueba, resultado y conclusión.
Se reportan tamaños de efecto e intervalos de confianza, no solo valores p.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ytnlp.config import load_config

RNG = np.random.default_rng(42)


def _r(x: float, n: int = 4) -> float:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# 1. Cola pesada en vistas ---------------------------------------------------------
def heavy_tail(views: pd.Series) -> dict:
    x = views.dropna()
    x = x[x > 0].to_numpy(dtype=float)
    xmin = np.median(x)
    tail = x[x >= xmin]
    # Pareto (ley de potencias) por máxima verosimilitud en la cola
    alpha = 1 + len(tail) / np.sum(np.log(tail / xmin))
    ll_pl = np.log(alpha - 1) - np.log(xmin) - alpha * np.log(tail / xmin)
    # Log-normal truncada en xmin
    mu, sigma = np.log(tail).mean(), np.log(tail).std(ddof=1)
    sf = stats.norm.sf((np.log(xmin) - mu) / sigma)
    ll_ln = stats.lognorm.logpdf(tail, s=sigma, scale=np.exp(mu)) - np.log(sf)
    diff = ll_ln - ll_pl
    R = diff.sum()
    z = R / (np.sqrt(len(diff)) * diff.std(ddof=1))  # prueba de Vuong
    p = 2 * stats.norm.sf(abs(z))
    top10 = np.sort(x)[::-1][: max(1, len(x) // 10)].sum() / x.sum()
    better = "log-normal" if R > 0 else "ley de potencias"
    return {
        "pregunta": "¿Las vistas siguen una distribución de cola pesada? ¿Log-normal o ley de potencias?",
        "prueba": "MLE de Pareto y log-normal en la cola (x ≥ mediana) + prueba de Vuong",
        "resultado": {
            "alpha_pareto": _r(alpha, 3),
            "LR_lognormal_vs_pareto": _r(R, 2),
            "vuong_z": _r(z, 3),
            "p_valor": _r(p),
            "pct_vistas_en_top10pct": _r(100 * top10, 1),
        },
        "conclusion": (
            f"El 10 % de videos con más vistas concentra {100 * top10:.1f} % de las vistas. "
            + (
                f"El ajuste favorece {better} (p={p:.3g}). "
                if p < load_config()["stats"]["alpha"]
                else f"Log-normal y ley de potencias no se distinguen con estos datos (p={p:.3g}). "
            )
            +
            "Implicación: modelar log(views) o métricas relativas, no views crudas."
        ),
    }


# 2. Correlación de rangos con bootstrap -------------------------------------------
def spearman_bootstrap(df: pd.DataFrame, x: str, y: str, n_boot: int) -> dict:
    d = df[[x, y]].dropna()
    rho, p = stats.spearmanr(d[x], d[y])
    tau, p_tau = stats.kendalltau(d[x], d[y])
    idx = RNG.integers(0, len(d), size=(n_boot, len(d)))
    xs, ys = d[x].to_numpy(), d[y].to_numpy()
    boots = [stats.spearmanr(xs[i], ys[i])[0] for i in idx]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    sig = "sí" if lo > 0 or hi < 0 else "no"
    return {
        "pregunta": f"¿{x} se asocia con {y}?",
        "prueba": f"Spearman y Kendall, IC 95 % por bootstrap ({n_boot} réplicas)",
        "resultado": {
            "n": len(d),
            "spearman_rho": _r(rho),
            "IC95": [_r(lo), _r(hi)],
            "p_valor": _r(p),
            "kendall_tau": _r(tau),
            "p_kendall": _r(p_tau),
        },
        "conclusion": f"ρ={rho:.3f} [{lo:.3f}, {hi:.3f}]; el IC {'excluye' if sig == 'sí' else 'incluye'} 0.",
    }


# 3. Diferencias entre grupos ------------------------------------------------------
def kruskal_dunn(df: pd.DataFrame, y: str, group: str) -> dict:
    d = df[[y, group]].dropna()
    groups = {k: v[y].to_numpy() for k, v in d.groupby(group) if len(v) >= 3}
    H, p = stats.kruskal(*groups.values())
    n, k = sum(len(v) for v in groups.values()), len(groups)
    eps2 = (H - k + 1) / (n - k)  # tamaño de efecto épsilon² (Tomczak & Tomczak 2014)
    # Dunn post-hoc con corrección de Bonferroni
    ranks = stats.rankdata(np.concatenate(list(groups.values())))
    pos, mean_rank = 0, {}
    for g, v in groups.items():
        mean_rank[g] = ranks[pos : pos + len(v)].mean()
        pos += len(v)
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie = np.sum(tie_counts**3 - tie_counts) / (12 * (n - 1))
    pairs = []
    m = k * (k - 1) / 2
    for a, b in itertools.combinations(groups, 2):
        se = np.sqrt((n * (n + 1) / 12 - tie) * (1 / len(groups[a]) + 1 / len(groups[b])))
        z = (mean_rank[a] - mean_rank[b]) / se
        p_adj = min(1.0, 2 * stats.norm.sf(abs(z)) * m)
        pairs.append({"a": a, "b": b, "z": _r(z, 3), "p_bonferroni": _r(p_adj)})
    sig_pairs = [p for p in pairs if p["p_bonferroni"] < load_config()["stats"]["alpha"]]
    medians = d.groupby(group)[y].median().sort_values(ascending=False)
    return {
        "pregunta": f"¿{y} difiere entre valores de {group}?",
        "prueba": "Kruskal-Wallis + post-hoc de Dunn (Bonferroni); efecto ε²",
        "resultado": {
            "H": _r(H, 3),
            "p_valor": _r(p),
            "epsilon2": _r(eps2),
            "grupos": k,
            "mediana_por_grupo_top5": {str(i): _r(v, 5) for i, v in medians.head(5).items()},
            "pares_significativos": len(sig_pairs),
            "ejemplos_pares": sig_pairs[:5],
        },
        "conclusion": (
            f"ε²={eps2:.3f} ({'grande' if eps2 >= 0.26 else 'moderado' if eps2 >= 0.08 else 'pequeño'}); "
            f"{len(sig_pairs)} de {len(pairs)} pares difieren tras Bonferroni. "
            "Si el efecto es relevante, controlar por keyword en el modelo."
        ),
    }


# 4. Regresión con controles (efectos fijos por keyword) ---------------------------
def ols_fixed_effects(df: pd.DataFrame, y: str, xs: list[str], fe: str) -> dict:
    d = df[[y, *xs, fe]].dropna()
    d = d[np.isfinite(d[y])]
    X = pd.concat([d[xs], pd.get_dummies(d[fe], prefix=fe, drop_first=True, dtype=float)], axis=1)
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)  # coeficientes estandarizados
    X.insert(0, "const", 1.0)
    Xm, yv = X.to_numpy(float), d[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    resid = yv - Xm @ beta
    n, k = Xm.shape
    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    meat = Xm.T @ (Xm * resid[:, None] ** 2)
    cov = XtX_inv @ meat @ XtX_inv * n / (n - k)  # errores robustos HC1
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=n - k)
    r2 = 1 - resid.var() / yv.var()
    coefs = {
        c: {"beta_std": _r(b), "se_HC1": _r(s), "p_valor": _r(pp)}
        for c, b, s, pp in zip(X.columns, beta, se, p)
        if c in xs
    }
    sig = [c for c, v in coefs.items() if v["p_valor"] < load_config()["stats"]["alpha"]]
    return {
        "pregunta": f"¿Las features de comentarios explican {y} una vez se controla por {fe} y edad?",
        "prueba": f"OLS con efectos fijos por {fe}, coeficientes estandarizados, errores robustos HC1",
        "resultado": {"n": n, "R2": _r(r2), "coeficientes": coefs},
        "conclusion": (
            f"R²={r2:.3f}. Significativas al 5 %: {', '.join(sig) if sig else 'ninguna'}. "
            "Es asociación, no causalidad."
        ),
    }


# 5. Zipf y Heaps ------------------------------------------------------------------
def zipf_heaps(texts: pd.Series) -> dict:
    toks = " ".join(texts.fillna("")).split()
    freq = np.array(sorted(Counter(toks).values(), reverse=True), dtype=float)
    r = np.arange(1, len(freq) + 1)
    cut = max(10, len(freq) // 2)  # ajuste en la zona media, evitando la cola de hapax
    s, _ = np.polyfit(np.log(r[:cut]), np.log(freq[:cut]), 1)
    seen, growth = set(), []
    step = max(1, len(toks) // 50)
    for i, tk in enumerate(toks, 1):
        seen.add(tk)
        if i % step == 0:
            growth.append((i, len(seen)))
    n_arr, v_arr = np.array(growth, dtype=float).T
    beta, logk = np.polyfit(np.log(n_arr), np.log(v_arr), 1)
    hapax = float((freq == 1).sum() / len(freq))
    return {
        "pregunta": "¿El vocabulario de los comentarios se comporta como lenguaje natural o hay mucho ruido?",
        "prueba": "Ajuste log-log de Zipf (frecuencia vs rango) y de Heaps (V = K·N^β)",
        "resultado": {
            "tokens": len(toks),
            "vocabulario": len(freq),
            "pendiente_zipf": _r(s, 3),
            "beta_heaps": _r(beta, 3),
            "pct_hapax": _r(100 * hapax, 1),
        },
        "conclusion": (
            f"Pendiente de Zipf {s:.2f} (≈ -1 en lenguaje natural) y β de Heaps {beta:.2f} "
            "(típico 0,4–0,6). β alto o muchos hapax indican ruido (typos, spam, varios idiomas) "
            "y justifican normalización o subword tokenization."
        ),
        "_freq": freq,
    }


# 6. Palabras distintivas: log-odds con prior de Dirichlet (Monroe et al., 2008) ----
def log_odds_dirichlet(df: pd.DataFrame, text: str, label: str, a: str, b: str, k: int) -> dict:
    ca = Counter(" ".join(df.loc[df[label] == a, text]).split())
    cb = Counter(" ".join(df.loc[df[label] == b, text]).split())
    prior = ca + cb
    a0 = sum(prior.values())
    na, nb = sum(ca.values()), sum(cb.values())
    rows = []
    for w, aw in prior.items():
        ya, yb = ca[w], cb[w]
        la = np.log((ya + aw) / (na + a0 - ya - aw))
        lb = np.log((yb + aw) / (nb + a0 - yb - aw))
        var = 1 / (ya + aw) + 1 / (yb + aw)
        rows.append((w, (la - lb) / np.sqrt(var), ya, yb))
    res = pd.DataFrame(rows, columns=["palabra", "z", f"n_{a}", f"n_{b}"]).sort_values("z")
    return {
        "pregunta": f"¿Qué palabras distinguen los comentarios de videos de desempeño {a} vs {b}?",
        "prueba": "Log-odds ratio ponderado con prior de Dirichlet informativo (z > 1,96 ≈ p < 0,05)",
        "resultado": {
            f"top_{a}": res.tail(k)[::-1][["palabra", "z"]].round(2).to_dict("records"),
            f"top_{b}": res.head(k)[["palabra", "z"]].round(2).to_dict("records"),
            "palabras_significativas": int((res["z"].abs() > 1.96).sum()),
        },
        "conclusion": "Insumo para features léxicas e interpretabilidad del modelo.",
    }


# 7. Comentario más votado vs promedio ---------------------------------------------
def top_vs_mean(df: pd.DataFrame) -> dict:
    d = df[["top_comment_sent", "sent_mean"]].dropna()
    diff = d["top_comment_sent"] - d["sent_mean"]
    stat, p = stats.wilcoxon(d["top_comment_sent"], d["sent_mean"], zero_method="zsplit")
    r_rb = 1 - 2 * stat / (len(d) * (len(d) + 1) / 2)  # correlación rango-biserial
    return {
        "pregunta": "¿El comentario con más likes es más positivo o negativo que el promedio?",
        "prueba": "Wilcoxon de rangos con signo (pareado por video); efecto r rango-biserial",
        "resultado": {
            "n": len(d),
            "diferencia_mediana": _r(diff.median()),
            "W": _r(stat, 1),
            "p_valor": _r(p),
            "r_rango_biserial": _r(r_rb),
        },
        "conclusion": (
            "Si difiere, la audiencia premia con likes un tono distinto al promedio: "
            "usar el sentimiento ponderado por likes como feature."
        ),
    }


# 8. Información mutua con prueba de permutación -----------------------------------
def mutual_information(df: pd.DataFrame, features: list[str], label: str, n_perm: int = 200) -> dict:
    from sklearn.feature_selection import mutual_info_classif

    d = df[[*features, label]].dropna()
    X, y = d[features].to_numpy(float), d[label].astype(str).to_numpy()
    mi = mutual_info_classif(X, y, random_state=42)
    null = np.array(
        [mutual_info_classif(X, RNG.permutation(y), random_state=42) for _ in range(n_perm)]
    )
    p = (1 + (null >= mi).sum(axis=0)) / (1 + n_perm)
    res = sorted(
        ({"feature": f, "MI": _r(m), "p_perm": _r(pp, 3)} for f, m, pp in zip(features, mi, p)),
        key=lambda r: -r["MI"],
    )
    return {
        "pregunta": f"¿Cuánta información aporta cada feature de comentarios sobre {label}?",
        "prueba": f"Información mutua (k-NN) + prueba de permutación ({n_perm} réplicas)",
        "resultado": {"n": len(d), "ranking": res},
        "conclusion": "Features con p_perm < 0,05 aportan información más allá del azar.",
    }


# Orquestación y reporte -----------------------------------------------------------
COMMENT_FEATS = [
    "sent_mean", "sent_weighted", "sent_std", "polarization", "pct_pos", "pct_neg",
    "tokens_mean", "ttr", "pct_question", "pct_exclaim", "emoji_per_comment",
]


def run_all(features: pd.DataFrame, videos: pd.DataFrame, comments: pd.DataFrame) -> dict:
    cfg = load_config()["stats"]
    f = features.copy()
    f["log_engagement"] = np.log(f["target_engagement"])
    out = {
        "1_cola_pesada_views": heavy_tail(videos["views"]),
        "2_sentimiento_vs_engagement": spearman_bootstrap(
            f, "sent_mean", "target_engagement", cfg["bootstrap_iterations"]
        ),
        "3_polarizacion_vs_engagement": spearman_bootstrap(
            f, "polarization", "target_engagement", cfg["bootstrap_iterations"]
        ),
        "4_engagement_por_keyword": kruskal_dunn(f, "target_engagement", "keyword"),
        "5_regresion_con_controles": ols_fixed_effects(
            f, "log_engagement", ["sent_mean", "polarization", "tokens_mean", "ttr", "log_age_days"],
            "keyword",
        ),
        "6_zipf_heaps": zipf_heaps(comments["comment"]),
        "7_palabras_distintivas": log_odds_dirichlet(
            f.dropna(subset=["target_perf_class"]), "doc", "target_perf_class", "alto", "bajo",
            cfg["top_k_words"],
        ),
        "8_comentario_top_vs_promedio": top_vs_mean(f),
        "9_informacion_mutua": mutual_information(
            f.dropna(subset=["target_perf_class"]), COMMENT_FEATS, "target_perf_class"
        ),
    }
    return out


def _figures(results: dict, videos: pd.DataFrame, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = []

    x = np.sort(videos["views"].dropna().to_numpy())[::-1]
    x = x[x > 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(x, np.arange(1, len(x) + 1) / len(x), ".", ms=3, color="#2b6cb0")
    ax.set_xlabel("Vistas")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title("CCDF de vistas (escala log-log)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fig_dir / "ccdf_views.png", dpi=130)
    plt.close(fig)
    names.append("figures/ccdf_views.png")

    freq = results["6_zipf_heaps"]["_freq"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.arange(1, len(freq) + 1), freq, color="#2b6cb0")
    ax.set_xlabel("Rango")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Ley de Zipf en comentarios")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fig_dir / "zipf.png", dpi=130)
    plt.close(fig)
    names.append("figures/zipf.png")
    return names


def write_report(results: dict, videos: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = _figures(results, videos, out_dir)
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    (out_dir / "stats_report.json").write_text(
        json.dumps(clean, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    lines = ["# Análisis estadísticos no triviales\n"]
    for key, r in clean.items():
        lines += [
            f"## {key.replace('_', ' ')}\n",
            f"**Pregunta:** {r['pregunta']}\n",
            f"**Prueba:** {r['prueba']}\n",
            "**Resultado:**\n",
            "```json",
            json.dumps(r["resultado"], indent=2, ensure_ascii=False, default=str),
            "```\n",
            f"**Conclusión:** {r['conclusion']}\n",
        ]
    lines += ["## Figuras\n"] + [f"![{f}]({f})" for f in figs]
    md = out_dir / "stats_report.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md
