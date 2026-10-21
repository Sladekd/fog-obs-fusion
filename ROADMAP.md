# Roadmap

The core question: **how much fog and visibility skill do soundings, neighbouring stations and synoptic context add beyond a station's own observations, and how much survives when NWP is removed at run time?**

The design principle is that NWP (ERA5, GFS, ALADIN) serves as a *training-only teacher*: allowed during training, never required at inference.

## Phase 0: Observation graph foundations (done)

- [x] Station metadata from OGIMET
- [x] Delaunay graph, pruning, connectivity
- [x] Edge-signal test against distance-matched non-edges
- [x] Fog co-occurrence figures
- [ ] Run on the 2016–2018 archive and commit figures

## Phase 1: Sounding-augmented baseline (`fogfusion.upperair`)

- [ ] Load IGRA2 / Wyoming soundings for nearby launch sites (Praha-Libuš 11520, Prostějov 11747, plus neighbouring countries)
- [ ] Diagnostics: inversion base/top/strength, mixing height, transport wind, T850, Td850, RH850, PWAT, CAPE, CIN
- [ ] Match each station to its nearest launch and the latest launch before forecast time; keep distance and age as features
- [ ] LightGBM: SYNOP only vs SYNOP + sounding, against persistence and climatology
- [ ] Skill as a function of distance from the launch site

## Phase 2: Graph neural network (`fogfusion.gnn`)

- [x] Graph → `edge_index` / `edge_attr` arrays
- [ ] Dataset: hourly node features + availability mask (stations drop out hour to hour)
- [ ] Rival baseline: LightGBM with hand-built neighbour features (k-nearest mean, upwind visibility)
- [ ] GraphSAGE for visibility and fog probability at t+1 h … t+6 h
- [ ] Distance-only vs elevation-weighted edges
- [ ] Evaluate on held-out *stations* as well as held-out times

## Phase 3: Synoptic embedding

- [ ] ERA5 MSLP, Z500, T850, RH700, U850, V850 around the region
- [ ] 16-D autoencoder vs PCA vs objective weather types, compared on the downstream fog task

## Phase 4: Obs-only system

- [ ] Student network that predicts the synoptic embedding from SYNOP + sounding alone
- [ ] Distillation from the NWP-informed teacher

## Later

- MTG-S1 IRS profiles once Level-2 products are released
- ALADIN as a terrain teacher for cold pools, foehn and valley fog

## Evaluation rules

- Time-blocked splits (never random), station holdouts from Phase 2 onwards
- Brier skill score, PR-AUC, reliability, CSI/ETS for fog; RMSE on log-visibility
- Always report persistence and climatology baselines; block-bootstrap 95% intervals
