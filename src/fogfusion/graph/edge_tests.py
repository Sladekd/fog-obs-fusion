"""Do graph edges carry physical signal beyond distance?

Compares edge pairs with non-edge pairs at the same distance, for the
correlation of hourly anomalies and for fog co-occurrence.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd

MIN_OVERLAP_HOURS = 500


def jaccard(a: pd.Series, b: pd.Series) -> float:
    """Shared-fog index: hours both foggy / hours either foggy (0 = never together)."""
    ab = pd.concat([a, b], axis=1).dropna()
    either = int(((ab.iloc[:, 0] + ab.iloc[:, 1]) > 0).sum())
    return np.nan if either == 0 else float((ab.iloc[:, 0] * ab.iloc[:, 1]).sum() / either)


def jaccard_matrix(fog: pd.DataFrame) -> pd.DataFrame:
    """station x station shared-fog index; diagonal = 1."""
    ids = list(fog.columns)
    J = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for a, b in itertools.combinations(ids, 2):
        J.loc[a, b] = J.loc[b, a] = jaccard(fog[a], fog[b])
    return J


def pair_stats(anom: pd.DataFrame, fog: pd.DataFrame | None, stations: pd.DataFrame,
               graph: nx.Graph, min_overlap: int = MIN_OVERLAP_HOURS) -> pd.DataFrame:
    """One row per station pair: distance, edge flag, anomaly correlation, fog co-occurrence."""
    P = stations.set_index("WMO_ID")
    ids = [s for s in P.index if s in anom.columns]
    rows = []
    for a, b in itertools.combinations(ids, 2):
        pr = anom[[a, b]].dropna()
        corr = pr[a].corr(pr[b]) if len(pr) >= min_overlap else np.nan
        cooc = np.nan
        if fog is not None and a in fog.columns and b in fog.columns:
            f = fog[[a, b]].dropna()
            fa, fb = f[a].sum(), f[b].sum()
            if fa > 0 and fb > 0:
                both = (f[a] * f[b]).sum()
                cooc = 0.5 * (both / fa + both / fb)   # mean of P(fog_b|fog_a), P(fog_a|fog_b)
        rows.append({
            "u": a, "v": b, "u_name": P.name[a], "v_name": P.name[b],
            "dist_km": float(np.hypot(P.x_km[a] - P.x_km[b], P.y_km[a] - P.y_km[b])),
            "abs_delev_m": abs(P.elev_m[a] - P.elev_m[b]),
            "is_edge": graph.has_edge(a, b), "corr": corr, "fog_cooc": cooc, "n_hours": len(pr),
        })
    return pd.DataFrame(rows)


def distance_matched(pairs: pd.DataFrame, col: str, bin_km: float = 10) -> tuple[float, pd.DataFrame]:
    """Edge minus non-edge mean of `col` within distance bins, weighted by the number of edges.

    Positive = edges carry more shared signal than equally distant non-edges.
    Returns (weighted difference, per-bin table); NaN if no bin holds both kinds.
    """
    p = pairs.dropna(subset=[col]).copy()
    p["bin"] = (p.dist_km // bin_km) * bin_km
    g = p.groupby(["bin", "is_edge"])[col].agg(["mean", "count"]).unstack("is_edge").dropna()
    if g.empty:
        return np.nan, g
    diff = g[("mean", True)] - g[("mean", False)]
    return float(np.average(diff, weights=g[("count", True)])), g


def elevation_effect(pairs: pd.DataFrame, col: str = "corr") -> float:
    """Spearman correlation of |elevation difference| with `col` along edges.

    Negative = edges across large height differences share less signal,
    which supports elevation-weighted edges.
    """
    e = pairs[pairs.is_edge].dropna(subset=[col])
    if len(e) <= 10:
        return np.nan
    return float(e[["abs_delev_m", col]].corr(method="spearman").iloc[0, 1])
