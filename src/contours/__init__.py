from __future__ import annotations

from contours_labels import draw_contour_labels as draw_contour_labels

from .labels_overlay import draw_contour_labels_overlay as draw_contour_labels_overlay
from .seeds import build_seed_polylines as build_seed_polylines

"""
Package initializer for contours.
Re-exports common API and ensures `from contours import draw_contour_labels` works.
"""
