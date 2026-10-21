import numpy as np
import pandas as pd
import pytest

from fogfusion.io import load_stations

OGIMET_SAMPLE = """\
##########################################################
# 11406, Cheb (Czech Republic)
# WIGOS Id: 0-20000-0-11406
# ICAO index: ----
# Latitude 50-04-05N. Longitude 012-23-29E. Altitude 483 m.
##########################################################

202610131300 0-20000-0-11406 11406 AAXX 13131 11406 22584 62103 10118 20041==

##########################################################
# 11735, Praded (Czech Republic)
# WIGOS Id: 0-20000-0-11735
# ICAO index: ----
# Latitude 50-04N. Longitude 017-14E. Altitude 1492 m.
##########################################################
"""


@pytest.fixture
def ogimet_sample():
    return OGIMET_SAMPLE


@pytest.fixture(scope="session")
def stations():
    return load_stations()


@pytest.fixture(scope="session")
def synthetic_obs(stations):
    """60 days of hourly visibility with regional night fog, at the real station positions."""
    rng = np.random.default_rng(0)
    t = pd.date_range("2017-11-01", periods=60 * 24, freq="h")
    night = np.where((t.hour >= 22) | (t.hour <= 8), 1.0, 0.0)
    regional = np.convolve(rng.normal(size=len(t)), np.ones(18) / 18, "same") * 4
    rows = []
    for r in stations.itertuples():
        w = np.exp(-((r.lon - 15) ** 2) / 4)
        local = np.convolve(rng.normal(size=len(t)), np.ones(4) / 4, "same")
        score = 1.4 * night * (r.elev_m < 600) + 0.8 * w * regional + 0.5 * local - 0.6
        vis = np.clip(10 ** (4.0 - 1.1 * np.clip(score, -1, 3)), 60, 40000)
        rows.append(pd.DataFrame({
            "WMO_ID": r.WMO_ID, "ANO": t.year.astype(float), "MES": t.month.astype(float),
            "DIA": t.day.astype(float), "HORA": t.hour.astype(float), "visibility_m": vis,
        }).sample(frac=0.95, random_state=int(r.WMO_ID)))
    return pd.concat(rows, ignore_index=True)
