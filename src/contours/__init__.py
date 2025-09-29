from __future__ import annotations

"""
Package initializer for contours.
Re-exports common API and ensures `from contours import draw_contour_labels` works.
"""

# Public API re-exports
from contours_labels import draw_contour_labels  # uses shared implementation

from .labels_overlay import draw_contour_labels_overlay  # overlay drawer
from .seeds import build_seed_polylines  # marching-squares seeds
