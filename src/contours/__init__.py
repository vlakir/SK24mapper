"""Package initializer for contours.

Re-exports common API for contour generation and labeling.
"""
from __future__ import annotations

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
from .labels_overlay import draw_contour_labels_overlay
from .seeds import build_seed_polylines

__all__ = [
    'BuildOpts',
    'ContourLevels',
    'CoordMap',
    'Sampling',
    'SeedGrid',
    'TileOverlapParams',
    'build_seed_polylines',
    'draw_contour_labels',
    'draw_contour_labels_overlay',
    'tile_overlap_rect',
    'tx_ty_from_index',
]
