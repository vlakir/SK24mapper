from __future__ import annotations

from typing import TYPE_CHECKING, Any

from geo.geometry import tile_overlap_rect_common as _tile_overlap_rect_common
from geo.topography import compute_xyz_coverage as _compute_xyz_coverage

if TYPE_CHECKING:
    from collections.abc import Generator


def compute_coverage(
    center_lat: float,
    center_lng: float,
    width_m: float,
    height_m: float,
    zoom: int,
    eff_scale: int,
    pad_px: int,
) -> tuple[
    list[tuple[int, tuple[int, int]]],
    tuple[int, int],
    tuple[int, int, int, int],
    dict[str, Any],
]:
    """
    Thin facade over topography.compute_xyz_coverage to standardize naming.

    Returns:
        tiles, (tiles_x, tiles_y), crop_rect, map_params

    """
    return _compute_xyz_coverage(
        center_lat,
        center_lng,
        width_m,
        height_m,
        zoom,
        eff_scale,
        pad_px,
    )


def iter_overlapping_tiles(
    tiles: list[tuple[int, tuple[int, int]]],
    tiles_x: int,
    crop_rect: tuple[int, int, int, int],
    *,
    tile_px: int,
) -> Generator[tuple[int, tuple[int, int]]]:
    """Yield only tiles whose effective pixel rect overlaps with crop_rect."""
    for idx, (tile_x_world, tile_y_world) in tiles:
        tx = idx % tiles_x
        ty = idx // tiles_x
        if _tile_overlap_rect_common(tx, ty, crop_rect, tile_px) is not None:
            yield (idx, (tile_x_world, tile_y_world))
