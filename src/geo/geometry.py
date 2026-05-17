from __future__ import annotations


def tile_overlap_rect_common(
    tx: int,
    ty: int,
    crop_rect: tuple[int, int, int, int],
    full_eff_tile_px: int,
) -> tuple[int, int, int, int] | None:
    """
    Compute overlap rectangle between a tile (tx, ty) in pixel space and crop_rect.

    Returns (x0, y0, x1, y1) or None if there is no intersection.
    """
    base_x = tx * full_eff_tile_px
    base_y = ty * full_eff_tile_px
    cx, cy, cw, ch = crop_rect
    x0 = max(base_x, cx)
    y0 = max(base_y, cy)
    x1 = min(base_x + full_eff_tile_px, cx + cw)
    y1 = min(base_y + full_eff_tile_px, cy + ch)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1
