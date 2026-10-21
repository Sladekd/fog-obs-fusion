import matplotlib

matplotlib.use("Agg")

from fogfusion.graph import build_delaunay_graph, jaccard_matrix  # noqa: E402
from fogfusion.io import fog_matrix, wide  # noqa: E402
from fogfusion.viz import (fog_clock, fog_night_gif, fog_regions, fog_web,  # noqa: E402
                           foggiest_night)


def test_fog_figures_render(stations, synthetic_obs, tmp_path):
    sg = build_delaunay_graph(stations)
    fog = fog_matrix(synthetic_obs)
    J = jaccard_matrix(fog)
    for fig in [fog_web(fog, J, sg.stations, sg.edges), fog_regions(J, sg.stations), fog_clock(fog)]:
        fig.savefig(tmp_path / "f.png")

    vis = wide(synthetic_obs, "visibility_m", agg="min")
    frames = foggiest_night(fog, hours=3)
    path = fog_night_gif(vis, sg.stations, sg.edges, str(tmp_path / "n.gif"), frames=frames)
    assert (tmp_path / "n.gif").stat().st_size > 0 and path.endswith(".gif")
