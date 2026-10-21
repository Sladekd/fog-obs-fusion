"""Fog co-occurrence figures for the README / articles (dark style).

fog_web       hero map: stations glow by fog frequency, links by shared fog (Jaccard)
fog_regions   clustered station x station shared-fog matrix
fog_clock     polar heatmap: network fog fraction by hour x month
fog_night_gif hour-by-hour animation of the foggiest night
"""

from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from ..io.synop import FOG_VIS_M
from .style import BG, DIM_LINK, INK, MIST, QUIET, dark_axes, dark_colorbar

_MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def _stations(stations: pd.DataFrame, ids) -> pd.DataFrame:
    return stations.set_index("WMO_ID").loc[list(ids)]


def fog_web(fog: pd.DataFrame, J: pd.DataFrame, stations: pd.DataFrame, edges: pd.DataFrame,
            title: str = "Where fog travels together", region: str = "Czech SYNOP stations",
            n_labels: int = 6, threshold_m: float = FOG_VIS_M):
    M = _stations(stations, fog.columns)
    f = 100 * fog.mean()
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
    dark_axes(ax, M.lat.mean())

    ew = edges[edges.kept].copy() if "kept" in edges else edges.copy()
    ew = ew[ew.u.isin(J.index) & ew.v.isin(J.index)]
    ew["j"] = [J.loc[u, v] for u, v in zip(ew.u, ew.v)]
    ew = ew.dropna(subset=["j"]).sort_values("j")          # strongest links drawn on top
    jn = (ew.j - ew.j.min()) / max(ew.j.max() - ew.j.min(), 1e-9)
    for r, t in zip(ew.itertuples(), jn):
        xs, ys = [M.lon[r.u], M.lon[r.v]], [M.lat[r.u], M.lat[r.v]]
        c = MIST(0.15 + 0.85 * t)
        ax.plot(xs, ys, color=c, lw=1 + 9 * t, alpha=0.08 + 0.12 * t, solid_capstyle="round", zorder=1)
        ax.plot(xs, ys, color=c, lw=0.6 + 2.4 * t, alpha=0.35 + 0.6 * t, solid_capstyle="round", zorder=2)

    fn = (f - f.min()) / max(f.max() - f.min(), 1e-9)
    for scale, alpha in [(14, 0.05), (7, 0.10), (3.5, 0.20)]:
        ax.scatter(M.lon, M.lat, s=(40 + 260 * fn) * scale, color=MIST(0.5 + 0.5 * fn),
                   alpha=alpha, lw=0, zorder=3)
    ax.scatter(M.lon, M.lat, s=30 + 140 * fn, color=MIST(0.55 + 0.45 * fn),
               edgecolor=BG, lw=0.8, zorder=4)

    labelled = []                                  # foggiest stations, skipping crowded neighbours
    for wmo in f.sort_values(ascending=False).index:
        if len(labelled) == n_labels:
            break
        # labels extend ~1.6 deg to the right, so same-latitude neighbours need more room
        if any(np.hypot(M.lon[wmo] - M.lon[o], 1.5 * (M.lat[wmo] - M.lat[o])) < 0.6
               or (abs(M.lat[wmo] - M.lat[o]) < 0.2 and abs(M.lon[wmo] - M.lon[o]) < 1.6)
               for o in labelled):
            continue
        labelled.append(wmo)
        ax.annotate(f"{M.name[wmo]}  {f[wmo]:.1f}%", (M.lon[wmo], M.lat[wmo]), xytext=(10, 8),
                    textcoords="offset points", fontsize=10, fontweight="bold", color=INK, zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.5, foreground=BG)])

    fig.text(0.04, 0.93, title, fontsize=22, fontweight="bold", color=INK)
    y0, y1 = fog.index.min().year, fog.index.max().year
    years = f"{y0}" if y0 == y1 else f"{y0}-{y1}"
    fig.text(0.04, 0.885, f"{region}, {years}.  Glow = how often a "
             "station is in fog.  Links = how often neighbours share it (Jaccard).",
             fontsize=10.5, color=QUIET)
    fig.text(0.04, 0.04, f"Fog: visibility < {threshold_m:.0f} m.  Links: Delaunay neighbours.",
             fontsize=8.5, color=QUIET)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.07)
    return fig


def fog_regions(J: pd.DataFrame, stations: pd.DataFrame):
    Jf = J.fillna(0)
    order = leaves_list(linkage(squareform(1 - Jf.values, checks=False), method="average"))
    Jo = Jf.iloc[order, order]
    names = list(_stations(stations, Jo.index).name)

    fig, ax = plt.subplots(figsize=(10, 9), facecolor=BG)
    ax.set_facecolor(BG)
    show = Jo.values.copy()
    np.fill_diagonal(show, np.nan)                 # the diagonal (=1) would dominate the scale
    im = ax.imshow(show, cmap=MIST, vmin=0, vmax=np.nanpercentile(show, 99))
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7.5, color=QUIET)
    ax.set_yticks(range(len(names)), names, fontsize=7.5, color=QUIET)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    dark_colorbar(plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02), "Shared-fog index (Jaccard)")
    ax.set_title("Fog regions: stations that fog up together cluster into blocks",
                 loc="left", color=INK, fontsize=14, pad=14)
    fig.tight_layout()
    return fig


def fog_clock(fog: pd.DataFrame):
    net = 100 * fog.mean(axis=1)                   # % of reporting stations in fog, each hour
    clock = net.groupby([net.index.month, net.index.hour]).mean().unstack()
    clock = clock.reindex(index=range(1, 13), columns=range(24))

    theta = np.linspace(0, 2 * np.pi, 25)
    radii = np.arange(13)
    fig = plt.figure(figsize=(8, 8.6), facecolor=BG)
    ax = fig.add_subplot(projection="polar", facecolor=BG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    pm = ax.pcolormesh(theta, radii + 3, clock.values, cmap=MIST, shading="flat",
                       vmin=0, vmax=np.nanpercentile(clock.values, 99), edgecolors=BG, linewidth=0.6)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)) + np.pi / 24,
                  [f"{h:02d} UTC" for h in range(0, 24, 3)], color=QUIET, fontsize=9)
    ax.set_yticks(np.arange(12) + 3.5, _MONTHS, color=QUIET, fontsize=7.5)
    ax.set_rlabel_position(4)
    ax.set_ylim(0, 15.2)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    dark_colorbar(plt.colorbar(pm, ax=ax, orientation="horizontal", shrink=0.55, pad=0.06),
                  "% of stations in fog")
    fig.text(0.06, 0.95, "The fog clock", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.06, 0.915, "Share of the network in fog by hour (around) and month "
             "(Jan inside, Dec outside)", fontsize=10, color=QUIET)
    return fig


def foggiest_night(fog: pd.DataFrame, start_hour: int = 18, hours: int = 18) -> pd.DatetimeIndex:
    """Hourly frames of the night (start_hour UTC + `hours`) with the most station-hours in fog."""
    key = (fog.index - pd.Timedelta(hours=start_hour)).normalize()
    best = fog.sum(axis=1).groupby(key).sum().idxmax()
    t0 = best + pd.Timedelta(hours=start_hour)
    frames = pd.date_range(t0, t0 + pd.Timedelta(hours=hours), freq="h")
    return frames[frames.isin(fog.index)]


def fog_night_gif(vis: pd.DataFrame, stations: pd.DataFrame, edges: pd.DataFrame, path: str,
                  frames: pd.DatetimeIndex | None = None, threshold_m: float = FOG_VIS_M,
                  fps: int = 2, dpi: int = 90) -> str:
    """Animate visibility hour by hour; a link lights up when both ends are in fog."""
    if frames is None:
        frames = foggiest_night((vis < threshold_m).astype(float).where(vis.notna()))
    M = _stations(stations, vis.columns)
    ew = edges[edges.kept] if "kept" in edges else edges
    links = ew[ew.u.isin(M.index) & ew.v.isin(M.index)][["u", "v"]].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor=BG)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.06)
    xlim = (M.lon.min() - 0.25, M.lon.max() + 0.25)
    ylim = (M.lat.min() - 0.15, M.lat.max() + 0.15)
    t_title = fig.text(0.04, 0.92, "", fontsize=18, fontweight="bold", color=INK)
    t_sub = fig.text(0.04, 0.865, "", fontsize=11, color=QUIET)
    fig.text(0.96, 0.03, "brighter = lower visibility", ha="right", fontsize=9, color=QUIET)

    def draw(i):
        ax.clear()
        dark_axes(ax, M.lat.mean())
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        t = frames[i]
        v = vis.loc[t].reindex(M.index)
        foggy = v < threshold_m
        dens = np.clip((np.log10(5000) - np.log10(v.clip(lower=50))) / np.log10(5000 / 50), 0, 1).fillna(0)
        for u, w in links:
            both = bool(foggy.get(u, False) and foggy.get(w, False))
            ax.plot([M.lon[u], M.lon[w]], [M.lat[u], M.lat[w]],
                    color=MIST(0.9) if both else DIM_LINK, lw=2.2 if both else 0.6,
                    alpha=0.9 if both else 0.6, zorder=1)
        for scale, alpha in [(12, 0.06), (5, 0.12)]:
            ax.scatter(M.lon, M.lat, s=(30 + 300 * dens) * scale, color=MIST(0.4 + 0.6 * dens),
                       alpha=alpha * (dens > 0.05), lw=0, zorder=2)
        ax.scatter(M.lon, M.lat, s=20 + 120 * dens, color=MIST(0.2 + 0.8 * dens),
                   edgecolor=BG, lw=0.6, zorder=3)
        ax.scatter(M.lon[v.isna()], M.lat[v.isna()], s=18, facecolor="none", edgecolor="#3a4a5e",
                   lw=0.8, zorder=3)
        t_title.set_text(f"{t:%d %b %Y  %H:00} UTC")
        t_sub.set_text(f"{int(foggy.sum())} of {int(v.notna().sum())} stations in fog "
                       f"(visibility < {threshold_m:.0f} m)")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 // fps)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    return path
