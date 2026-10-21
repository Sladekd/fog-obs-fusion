"""Build the Delaunay station graph and test whether its edges carry signal.

    python scripts/build_graph.py --obs data/processed/synop.parquet --out outputs/graph
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from _common import read_obs
from fogfusion.graph import (build_delaunay_graph, distance_matched, elevation_effect,
                             pair_stats)
from fogfusion.io import anomalies, check_coverage, fog_matrix, load_stations
from fogfusion.viz import plot_pair_signal, plot_station_graph


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obs", required=True, help="decoded SYNOP table (parquet or csv)")
    ap.add_argument("--stations", default=None, help="station csv; default: bundled Czech table")
    ap.add_argument("--out", default="outputs/graph")
    ap.add_argument("--max-edge-km", type=float, default=None, help="default: 95th pct of edge length")
    ap.add_argument("--var", default="visibility_m", help="variable for the anomaly correlation test")
    args = ap.parse_args()

    matplotlib.use("Agg")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    obs = read_obs(args.obs)
    stations = check_coverage(obs, load_stations(args.stations))
    sg = build_delaunay_graph(stations, max_edge_km=args.max_edge_km)
    print(sg.summary())

    anom = anomalies(obs, args.var, log=args.var == "visibility_m")
    fog = fog_matrix(obs) if "visibility_m" in obs else None
    pairs = pair_stats(anom, fog, sg.stations, sg.graph)

    print("\nPositive = edges share more signal than non-edges at the same distance")
    for col in ["corr", "fog_cooc"]:
        diff, bins = distance_matched(pairs, col)
        print(f"  {col:9s} edge - non-edge: {diff:+.3f} over {len(bins)} distance bins")
    print(f"  Spearman(|elevation difference|, corr) along edges: {elevation_effect(pairs):+.2f}")

    sg.stations.to_csv(out / "stations.csv", index=False)
    sg.edges.to_csv(out / "edges.csv", index=False)
    pairs.to_csv(out / "pair_stats.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    plot_station_graph(sg, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "station_graph.png", dpi=150)
    plot_pair_signal(pairs).savefig(out / "pair_signal.png", dpi=150)
    print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
