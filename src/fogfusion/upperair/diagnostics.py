"""Boundary-layer diagnostics from a single sounding. Interfaces only (Phase 1).

A profile is a DataFrame sorted by height with columns:
pressure_hpa, height_m, temp_c, dewpoint_c, wind_dir_deg, wind_speed_ms.
MetPy covers most of these computations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Inversion:
    base_m: float        # height above ground of the inversion base
    top_m: float
    strength_k: float    # temperature at top minus temperature at base

    @property
    def depth_m(self) -> float:
        return self.top_m - self.base_m


def lowest_inversion(profile: pd.DataFrame, max_height_m: float = 3000) -> Inversion | None:
    """Lowest layer where temperature increases with height, or None."""
    raise NotImplementedError


def mixing_height(profile: pd.DataFrame, surface_temp_c: float | None = None) -> float:
    """Parcel-method mixing height (m above ground)."""
    raise NotImplementedError


def transport_wind(profile: pd.DataFrame, mixing_height_m: float) -> float:
    """Mean wind speed (m/s) through the mixed layer."""
    raise NotImplementedError


def level_values(profile: pd.DataFrame, pressure_hpa: float = 850) -> dict:
    """T, Td and RH interpolated to a pressure level, e.g. T850/Td850/RH850."""
    raise NotImplementedError


def stability_indices(profile: pd.DataFrame) -> dict:
    """PWAT (mm), CAPE and CIN (J/kg) of the surface-based parcel."""
    raise NotImplementedError
