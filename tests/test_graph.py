import networkx as nx
import numpy as np
import pandas as pd
import pytest

from fogfusion.gnn import graph_arrays
from fogfusion.graph import (build_delaunay_graph, distance_matched, jaccard, jaccard_matrix,
                             pair_stats)
from fogfusion.io import anomalies, fog_matrix


@pytest.fixture(scope="module")
def sg(stations):
    return build_delaunay_graph(stations)


def test_delaunay_full_triangulation_is_connected(stations):
    sg = build_delaunay_graph(stations, max_edge_km=1e6)
    assert nx.is_connected(sg.graph)
    # planar triangulation: E <= 3N - 6
    assert sg.graph.number_of_edges() <= 3 * len(stations) - 6


def test_pruning_removes_only_long_edges(sg):
    assert sg.edges[sg.edges.kept].dist_km.max() <= sg.cutoff_km
    assert sg.edges[~sg.edges.kept].dist_km.min() > sg.cutoff_km
    assert sg.graph.number_of_edges() == int(sg.edges.kept.sum())


def test_graph_arrays_shapes(sg):
    arr = graph_arrays(sg)
    n, e = sg.graph.number_of_nodes(), sg.graph.number_of_edges()
    assert arr["edge_index"].shape == (2, 2 * e)
    assert arr["edge_attr"].shape == (2 * e, 4)
    assert arr["x_static"].shape == (n, 3)
    assert arr["edge_index"].max() < n
    # reverse edges have opposite elevation differences
    assert np.allclose(arr["edge_attr"][0::2, 1], -arr["edge_attr"][1::2, 1])


def test_jaccard():
    a = pd.Series([1, 1, 0, 0, np.nan])
    b = pd.Series([1, 0, 1, 0, 1])
    assert jaccard(a, b) == pytest.approx(1 / 3)
    assert np.isnan(jaccard(pd.Series([0, 0]), pd.Series([0, 0])))


def test_pair_stats_and_tests(sg, synthetic_obs):
    fog = fog_matrix(synthetic_obs)
    anom = anomalies(synthetic_obs, "visibility_m", log=True)
    pairs = pair_stats(anom, fog, sg.stations, sg.graph)
    n = len(sg.stations)
    assert len(pairs) == n * (n - 1) // 2
    assert pairs.is_edge.sum() == sg.graph.number_of_edges()
    diff, bins = distance_matched(pairs, "corr")
    assert np.isfinite(diff) and len(bins) > 0
    J = jaccard_matrix(fog)
    assert np.allclose(J.values, J.values.T, equal_nan=True)
