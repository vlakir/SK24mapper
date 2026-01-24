"""
Package initializer for contours.

Re-exports common API for contour generation and labeling.
"""

from __future__ import annotations

from .adaptive import ContourAdaptiveParams, compute_contour_adaptive_params
from .builder import (
    BuildOpts,
    ContourLevels,
    CoordMap,
    Sampling,
    SeedGrid,
)
from .helpers import (
    TileOverlapParams,
    tile_overlap_rect,
    tx_ty_from_index,
)
from .labels import draw_contour_labels
from .seeds import build_seed_polylines

__all__ = [
    'BuildOpts',
    'ContourAdaptiveParams',
    'ContourLevels',
    'CoordMap',
    'Sampling',
    'SeedGrid',
    'TileOverlapParams',
    'build_seed_polylines',
    'compute_contour_adaptive_params',
    'draw_contour_labels',
    'tile_overlap_rect',
    'tx_ty_from_index',
]
