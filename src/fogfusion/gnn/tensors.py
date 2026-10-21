"""Convert a StationGraph to plain arrays usable by PyTorch Geometric.

Kept free of torch so it runs anywhere; wrap with torch.as_tensor when training:

    arr = graph_arrays(sg)
    data = torch_geometric.data.Data(
        edge_index=torch.as_tensor(arr["edge_index"]),
        edge_attr=torch.as_tensor(arr["edge_attr"]),
        x_static=torch.as_tensor(arr["x_static"]))
"""

from __future__ import annotations

import numpy as np

from ..graph.delaunay import StationGraph

EDGE_FEATURES = ["dist_km", "delev_m", "bearing_sin", "bearing_cos"]
STATIC_FEATURES = ["lat", "lon", "elev_m"]


def graph_arrays(sg: StationGraph) -> dict:
    """Directed edges in both directions with features, plus static node features.

    Returns dict with:
        node_ids    (N,)    WMO ids; row i of every node array is node_ids[i]
        edge_index  (2, E)  int64 source/target node positions, both directions
        edge_attr   (E, 4)  float32 [dist_km, delev_m (target - source), sin/cos bearing]
        x_static    (N, 3)  float32 [lat, lon, elev_m]
    """
    st = sg.stations.reset_index(drop=True)
    node_ids = st.WMO_ID.astype(int).to_numpy()
    pos = {wmo: i for i, wmo in enumerate(node_ids)}
    x, y, z = st.x_km.to_numpy(), st.y_km.to_numpy(), st.elev_m.to_numpy()

    src, dst = [], []
    for u, v in sg.graph.edges:
        src += [pos[u], pos[v]]
        dst += [pos[v], pos[u]]
    src, dst = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)

    dx, dy = x[dst] - x[src], y[dst] - y[src]
    dist = np.hypot(dx, dy)
    bearing = np.arctan2(dx, dy)                   # 0 = north, clockwise positive
    edge_attr = np.column_stack([dist, z[dst] - z[src], np.sin(bearing), np.cos(bearing)])

    return {
        "node_ids": node_ids,
        "edge_index": np.vstack([src, dst]),
        "edge_attr": edge_attr.astype(np.float32),
        "x_static": st[STATIC_FEATURES].to_numpy(dtype=np.float32),
    }
