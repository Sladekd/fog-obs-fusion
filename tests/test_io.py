import numpy as np
import pytest

from fogfusion.io import anomalies, fog_matrix, parse_ogimet_text
from fogfusion.io.ogimet import dms_to_deg


def test_dms_to_deg():
    assert dms_to_deg("50-04-05", "N") == pytest.approx(50.068056, abs=1e-5)
    assert dms_to_deg("017-14", "E") == pytest.approx(17.233333, abs=1e-5)
    assert dms_to_deg("10-30", "W") == pytest.approx(-10.5)


def test_parse_ogimet_headers(ogimet_sample):
    st = parse_ogimet_text(ogimet_sample)
    assert list(st.WMO_ID) == [11406, 11735]
    cheb = st.iloc[0]
    assert cheb["name"] == "Cheb"
    assert cheb.lat == pytest.approx(50.06806, abs=1e-4)
    assert cheb.lon == pytest.approx(12.39139, abs=1e-4)
    assert cheb.elev_m == 483
    assert st.iloc[1].elev_m == 1492          # minutes-only coordinates also parse


def test_bundled_stations(stations):
    assert len(stations) == 36
    assert stations.WMO_ID.is_unique
    assert stations.lat.between(48.5, 51.1).all()
    assert stations.lon.between(12.0, 18.9).all()


def test_fog_matrix_keeps_missing_as_nan(synthetic_obs):
    fog = fog_matrix(synthetic_obs)
    vals = set(np.unique(fog.values[~np.isnan(fog.values)]))
    assert vals <= {0.0, 1.0}
    assert fog.isna().values.any()            # 5% of reports were dropped


def test_anomalies_remove_climatology(synthetic_obs):
    a = anomalies(synthetic_obs, "visibility_m", log=True)
    clim = a.groupby([a.index.month, a.index.hour]).mean()
    assert np.nanmax(np.abs(clim.values)) < 1e-9
