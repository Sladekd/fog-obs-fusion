"""Decoded SYNOP tables -> station x time matrices.

Expected input (one row per report): WMO_ID, ANO (year), MES (month), DIA (day),
HORA (hour), plus decoded variables such as visibility_m.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

FOG_VIS_M = 1000


def load_stations(path: str | Path | None = None) -> pd.DataFrame:
    """Station metadata [WMO_ID, name, lat, lon, elev_m].

    Defaults to the bundled Czech SYNOP table (36 stations, parsed from OGIMET).
    """
    if path is None:
        path = files("fogfusion.resources").joinpath("stations_cz.csv")
        with path.open("r", encoding="utf-8") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `time` column built from ANO/MES/DIA/HORA, dropping rows missing any of them."""
    d = df.dropna(subset=["ANO", "MES", "DIA", "HORA"]).copy()
    d["time"] = pd.to_datetime(dict(year=d.ANO.astype(int), month=d.MES.astype(int),
                                    day=d.DIA.astype(int), hour=d.HORA.astype(int)))
    d["WMO_ID"] = d.WMO_ID.astype(int)
    return d


def wide(df: pd.DataFrame, var: str, agg: str = "mean") -> pd.DataFrame:
    """time x station matrix of one variable (NaN = no report)."""
    d = add_time(df).dropna(subset=[var])
    out = d.pivot_table(index="time", columns="WMO_ID", values=var, aggfunc=agg)
    out.columns = out.columns.astype(int)
    return out


def fog_matrix(df: pd.DataFrame, threshold_m: float = FOG_VIS_M) -> pd.DataFrame:
    """time x station: 1 = fog (visibility < threshold), 0 = no fog, NaN = missing."""
    vis = wide(df, "visibility_m", agg="min")
    return (vis < threshold_m).astype(float).where(vis.notna())


def anomalies(df: pd.DataFrame, var: str, log: bool = False) -> pd.DataFrame:
    """time x station anomalies vs each station's (month, hour) climatology."""
    w = wide(df, var)
    if log:
        w = np.log10(w.clip(lower=10))
    clim = w.groupby([w.index.month, w.index.hour]).transform("mean")
    return w - clim


def check_coverage(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Print stations missing metadata or observations; return metadata for observed stations."""
    obs_ids = set(pd.to_numeric(df.WMO_ID, errors="coerce").dropna().astype(int))
    no_meta = sorted(obs_ids - set(stations.WMO_ID))
    no_obs = sorted(set(stations.WMO_ID) - obs_ids)
    if no_meta:
        print(f"in observations but no coordinates (pass your own table to load_stations): {no_meta}")
    if no_obs:
        print(f"in metadata but no observations: {no_obs}")
    return stations[stations.WMO_ID.isin(obs_ids)].reset_index(drop=True)
