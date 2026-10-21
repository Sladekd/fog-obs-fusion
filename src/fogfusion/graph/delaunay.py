"""Station graph from a Delaunay triangulation in projected coordinates."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Delaunay

UTM33N = "EPSG:32633"


def project(stations: pd.DataFrame, crs: str = UTM33N) -> pd.DataFrame:
    """Add planar x_km, y_km. Delaunay must run on planar coordinates, not degrees."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(stations.lon.values, stations.lat.values)
    return stations.assign(x_km=np.asarray(x) / 1000, y_km=np.asarray(y) / 1000)


@dataclass
class StationGraph:
    graph: nx.Graph          # pruned graph; nodes are WMO ids
    stations: pd.DataFrame   # metadata incl. x_km, y_km
    edges: pd.DataFrame      # every Delaunay edge, with `kept` flag
    cutoff_km: float

    def components(self) -> list[set[int]]:
        return sorted(nx.connected_components(self.graph), key=len, reverse=True)

    def summary(self) -> str:
        e, comps = self.edges, self.components()
        deg = [d for _, d in self.graph.degree()]
        lines = [
            f"stations: {self.graph.number_of_nodes()}, Delaunay edges: {len(e)}",
            f"edge length km: median {e.dist_km.median():.1f}, p95 {e.dist_km.quantile(.95):.1f}, "
            f"max {e.dist_km.max():.1f}",
            f"cutoff {self.cutoff_km:.1f} km -> kept {e.kept.sum()}, pruned {(~e.kept).sum()} "
            f"({e[~e.kept].on_hull.sum()} on the hull)",
            f"components: {len(comps)} {[len(c) for c in comps]}; "
            f"degree min/median/max {min(deg)}/{int(np.median(deg))}/{max(deg)}",
        ]
        for c in comps[1:]:
            lines.append("  separate: " + ", ".join(self.graph.nodes[n]["name"] for n in c))
        return "\n".join(lines)


def build_delaunay_graph(stations: pd.DataFrame, max_edge_km: float | None = None,
                         crs: str = UTM33N) -> StationGraph:
    """Triangulate stations and prune long edges.

    max_edge_km=None prunes at the 95th percentile of edge length, which removes
    the long convex-hull edges that link stations that are not real neighbours.
    """
    st = project(stations.reset_index(drop=True), crs)
    pts = st[["x_km", "y_km"]].to_numpy()
    tri = Delaunay(pts)
    hull = {tuple(sorted(e)) for e in tri.convex_hull}
    pairs = {tuple(sorted(p)) for s in tri.simplices for p in itertools.combinations(s, 2)}

    edges = pd.DataFrame([{
        "u": int(st.WMO_ID[a]), "v": int(st.WMO_ID[b]),
        "u_name": st.name[a], "v_name": st.name[b],
        "dist_km": float(np.hypot(*(pts[a] - pts[b]))),
        "delev_m": float(st.elev_m[a] - st.elev_m[b]),
        "on_hull": (a, b) in hull,
    } for a, b in sorted(pairs)])
    edges["abs_delev_m"] = edges.delev_m.abs()

    cutoff = float(edges.dist_km.quantile(0.95)) if max_edge_km is None else float(max_edge_km)
    edges["kept"] = edges.dist_km <= cutoff

    g = nx.Graph()
    for r in st.itertuples():
        g.add_node(int(r.WMO_ID), name=r.name, lat=r.lat, lon=r.lon, x=r.x_km, y=r.y_km, elev=r.elev_m)
    for r in edges[edges.kept].itertuples():
        g.add_edge(r.u, r.v, dist_km=r.dist_km, delev_m=r.delev_m)
    return StationGraph(g, st, edges, cutoff)
