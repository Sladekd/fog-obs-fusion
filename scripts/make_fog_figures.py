"""Render the fog co-occurrence figures (fog web, fog regions, fog clock, foggiest-night GIF).

    python scripts/make_fog_figures.py --obs data/processed/synop.parquet --out figures
"""

import argparse
from pathlib import Path

import matplotlib

from _common import read_obs
from fogfusion.graph import build_delaunay_graph, jaccard_matrix
from fogfusion.io import FOG_VIS_M, check_coverage, load_stations, wide
from fogfusion.viz import fog_clock, fog_night_gif, fog_regions, fog_web
from fogfusion.viz.style import BG


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", required=True, help="decoded SYNOP table (parquet or csv)")
    ap.add_argument("--stations", default=None, help="station csv; default: bundled Czech table")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--fog-vis-m", type=float, default=FOG_VIS_M)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    matplotlib.use("Agg")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    obs = read_obs(args.obs)
    stations = check_coverage(obs, load_stations(args.stations))
    sg = build_delaunay_graph(stations)

    vis = wide(obs, "visibility_m", agg="min")
    vis = vis[[c for c in vis.columns if c in set(sg.stations.WMO_ID)]]
    fog = (vis < args.fog_vis_m).astype(float).where(vis.notna())
    J = jaccard_matrix(fog)

    figs = {
        "fog_web.png": fog_web(fog, J, sg.stations, sg.edges, threshold_m=args.fog_vis_m),
        "fog_regions.png": fog_regions(J, sg.stations),
        "fog_clock.png": fog_clock(fog),
    }
    for name, fig in figs.items():
        fig.savefig(out / name, dpi=200, facecolor=BG)
        print(f"saved {out / name}")
    if not args.no_gif:
        print("saved", fog_night_gif(vis, sg.stations, sg.edges, str(out / "fog_night.gif"),
                                     threshold_m=args.fog_vis_m))


if __name__ == "__main__":
    main()
