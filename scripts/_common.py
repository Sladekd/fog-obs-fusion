from pathlib import Path

import pandas as pd


def read_obs(path: str) -> pd.DataFrame:
    """Read decoded SYNOP observations (the notebooks' `final_df`) from parquet or csv."""
    p = Path(path)
    if p.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)
