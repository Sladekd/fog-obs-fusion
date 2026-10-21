"""Graph neural network models on the station graph - Phase 2 of the roadmap.

Ready:
    tensors.graph_arrays   StationGraph -> edge_index / edge_attr / node static features (numpy)

Planned:
    dataset.py   time-sliced node features + availability mask -> PyTorch Geometric Data
    models.py    GraphSAGE (PyTorch Geometric), fog probability and visibility heads
    train.py     time-blocked and station-holdout splits, evaluation vs persistence
"""

from .tensors import graph_arrays

__all__ = ["graph_arrays"]
