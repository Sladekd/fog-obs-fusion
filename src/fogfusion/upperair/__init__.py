"""Upper-air (radiosonde) features - Phase 1 of the roadmap. Not implemented yet.

Planned layout:

    upperair/
        sources.py      IGRA2 / University of Wyoming loaders -> tidy profile tables
        diagnostics.py  per-profile boundary-layer diagnostics (below)
        matching.py     assign each SYNOP station its nearest launch site and the
                        latest launch before forecast time; keep distance and age

Diagnostics to compute per profile (stubs in diagnostics.py):
inversion base/top/depth/strength, mixing height, transport wind,
T850, Td850, RH850, PWAT, CAPE, CIN.
"""
