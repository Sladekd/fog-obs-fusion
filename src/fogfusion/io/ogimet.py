"""Station metadata from OGIMET SYNOP text pages.

OGIMET disallows automated scraping: open the query page in a browser, save it
as plain text, and pass the text (or the file path) to these functions.

Header format parsed::

    # 11406, Cheb (Czech Republic)
    # WIGOS Id: 0-20000-0-11406
    # ICAO index: ----
    # Latitude 50-04-05N. Longitude 012-23-29E. Altitude 483 m.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_HEADER = re.compile(
    r"#\s*(\d{5}),\s*(.+?)\s*\([^)]*\)\s*\n(?:#.*\n)*?"
    r"#\s*Latitude\s+([\d-]+)([NS])\.\s*Longitude\s+([\d-]+)([EW])\.\s*"
    r"Altitude\s+(-?\d+)\s*m"
)


def dms_to_deg(dms: str, hemisphere: str) -> float:
    """Convert '50-04-05' or '50-04' plus a hemisphere letter to decimal degrees."""
    parts = [float(p) for p in dms.split("-")]
    deg = parts[0] + (parts[1] if len(parts) > 1 else 0) / 60 + (parts[2] if len(parts) > 2 else 0) / 3600
    return -deg if hemisphere in "SW" else deg


def parse_ogimet_text(text: str) -> pd.DataFrame:
    """Return a DataFrame [WMO_ID, name, lat, lon, elev_m], one row per station."""
    rows = [
        {
            "WMO_ID": int(m[1]),
            "name": m[2],
            "lat": round(dms_to_deg(m[3], m[4]), 5),
            "lon": round(dms_to_deg(m[5], m[6]), 5),
            "elev_m": float(m[7]),
        }
        for m in _HEADER.finditer(text)
    ]
    cols = ["WMO_ID", "name", "lat", "lon", "elev_m"]
    return pd.DataFrame(rows, columns=cols).drop_duplicates("WMO_ID").reset_index(drop=True)


def parse_ogimet_file(path: str | Path) -> pd.DataFrame:
    return parse_ogimet_text(Path(path).read_text(encoding="utf-8", errors="replace"))
