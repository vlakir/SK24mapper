from __future__ import annotations

from geometry import tile_overlap_rect_common as _tile_overlap_rect_common
from topography import compute_xyz_coverage as _compute_xyz_coverage


def compute_coverage(bounds, zoom):
    """
    Thin facade over topography.compute_xyz_coverage to standardize naming.

    Returns:
        tiles, (tiles_x, tiles_y), crop_rect, map_params

    """
    return _compute_xyz_coverage(bounds, zoom)


def iter_overlapping_tiles(
    tiles: list[tuple[int, tuple[int, int]]],
    tiles_x: int,
    crop_rect: tuple[int, int, int, int],
    *,
    tile_px: int,
):
    """Yield only tiles whose effective pixel rect overlaps with crop_rect."""
    for idx, (tile_x_world, tile_y_world) in tiles:
        tx = idx % tiles_x
        ty = idx // tiles_x
        if _tile_overlap_rect_common(tx, ty, crop_rect, tile_px) is not None:
            yield (idx, (tile_x_world, tile_y_world))
