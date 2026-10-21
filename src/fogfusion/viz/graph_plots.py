"""Diagnostic plots for the station graph (light style, for analysis)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from ..graph.delaunay import StationGraph


def plot_station_graph(sg: StationGraph, mountain_m: float = 700, ax=None):
    """Map of kept and pruned edges, shaded by elevation difference."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7))
    st, e = sg.stations, sg.edges
    pos = {int(r.WMO_ID): (r.x_km, r.y_km) for r in st.itertuples()}

    for r in e[~e.kept].itertuples():
        ax.plot(*zip(pos[r.u], pos[r.v]), color="#bbbbbb", ls=(0, (3, 3)), lw=0.8, zorder=1)
    kept = e[e.kept]
    cmap, norm = plt.cm.Blues, plt.Normalize(0, max(kept.abs_delev_m.max(), 1))
    for r in kept.itertuples():
        ax.plot(*zip(pos[r.u], pos[r.v]), color=cmap(0.25 + 0.75 * norm(r.abs_delev_m)),
                lw=2, solid_capstyle="round", zorder=2)

    high = st.elev_m >= mountain_m
    for mask, marker, size in [(~high, "o", 40), (high, "^", 60)]:
        ax.scatter(st.x_km[mask], st.y_km[mask], s=size, marker=marker, color="#333333",
                   edgecolor="white", linewidth=1.2, zorder=3)
    for r in st.itertuples():
        ax.annotate(r.name, (r.x_km, r.y_km), xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="#444444")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, shrink=0.7, label="|Elevation difference| along edge (m)")
    ax.legend(handles=[
        Line2D([], [], color=cmap(0.6), lw=2, label=f"kept edge (<= {sg.cutoff_km:.0f} km)"),
        Line2D([], [], color="#bbbbbb", ls=(0, (3, 3)), lw=0.8, label="pruned edge"),
        Line2D([], [], marker="o", color="#333333", ls="", label=f"station < {mountain_m:.0f} m"),
        Line2D([], [], marker="^", color="#333333", ls="", label=f"station >= {mountain_m:.0f} m"),
    ], loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Delaunay graph of SYNOP stations", loc="left")
    ax.set_xlabel("UTM 33N easting (km)")
    ax.set_ylabel("UTM 33N northing (km)")
    ax.set_aspect("equal")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def plot_pair_signal(pairs: pd.DataFrame):
    """Correlation and fog co-occurrence vs distance, edges highlighted."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, col, label in [(axes[0], "corr", "Correlation of log-visibility anomalies"),
                           (axes[1], "fog_cooc", "Fog co-occurrence")]:
        p = pairs.dropna(subset=[col])
        ax.scatter(p.dist_km[~p.is_edge], p[col][~p.is_edge], s=12, color="#9aa5b1",
                   alpha=0.6, label="non-edge pair", edgecolor="none")
        ax.scatter(p.dist_km[p.is_edge], p[col][p.is_edge], s=28, color="#2a6fdb",
                   label="Delaunay edge", edgecolor="white", linewidth=0.8)
        ax.set_title(label, loc="left", fontsize=11)
        ax.set_xlabel("Distance (km)")
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
