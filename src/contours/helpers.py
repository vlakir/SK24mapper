from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TileOverlapParams:
    """
    Parameters describing the effective tile size and working rectangle.

    This structure is generic and can be used across different processing phases
    that operate with the same tile grid concept.
    """

    full_eff_tile_px: int
    cx0: int
    cy0: int
    cx1: int
    cy1: int
    crop_rect: tuple[int, int, int, int] | None = None


def tile_overlap_rect(
    tx: int, ty: int, params: TileOverlapParams
) -> tuple[int, int, int, int] | None:
    """
    Compute overlap rectangle between a tile (tx, ty) and the working rect.

    Returns (x0, y0, x1, y1) in world pixels or None if no overlap.
    """
    base_x = tx * params.full_eff_tile_px
    base_y = ty * params.full_eff_tile_px
    x0 = max(params.cx0, base_x)
    y0 = max(params.cy0, base_y)
    x1 = min(params.cx1, base_x + params.full_eff_tile_px)
    y1 = min(params.cy1, base_y + params.full_eff_tile_px)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def tx_ty_from_index(idx: int, tile_cols: int) -> tuple[int, int]:
    """Convert a linear index into (tx, ty) using number of tile columns."""
    ty, tx = divmod(idx, tile_cols)
    return tx, ty
