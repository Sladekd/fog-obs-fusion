"""Shared dark "night fog" style for README and article figures."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BG = "#0b1320"        # night-sky background
INK = "#e8edf3"       # main text
QUIET = "#8a97a8"     # secondary text
DIM_LINK = "#22364d"  # inactive link
MIST = LinearSegmentedColormap.from_list("mist", ["#16324f", "#3f7cac", "#9cc9e8", "#f4f9ff"])


def dark_axes(ax, lat_mean=None):
    """Blank lon/lat map axes on the dark background, with an equal-distance aspect."""
    ax.set_facecolor(BG)
    if lat_mean is not None:
        ax.set_aspect(1 / np.cos(np.deg2rad(lat_mean)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def dark_colorbar(cb, label):
    cb.set_label(label, color=QUIET)
    cb.ax.tick_params(colors=QUIET)
    cb.outline.set_visible(False)
