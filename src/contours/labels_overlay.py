from __future__ import annotations

from typing import TYPE_CHECKING

from contours.labels import draw_contour_labels

if TYPE_CHECKING:
    from PIL import Image


def draw_contour_labels_overlay(
    base_img: Image.Image,
    seeds_by_level: dict[int, list[list[tuple[float, float]]]],
    levels: list[float],
    mpp: float,
    *,
    seed_ds: int,
    dry_run: bool = False,
) -> list[tuple[int, int, int, int]]:
    """
    Draw contour labels overlay on the provided base image.

    This is a thin facade around contours_labels.draw_contour_labels that
    additionally returns the placed label bounding boxes to support gap carving.

    Args:
        base_img: Target image (RGBA recommended) to draw onto when dry_run=False.
        seeds_by_level: Seed polylines by level index (already in crop coordinate space/scale).
        levels: Ordered list of contour levels corresponding to seeds_by_level keys.
        mpp: Meters per pixel at current latitude and zoom.
        seed_ds: Downsample factor used when generating seeds (to scale into crop px).
        dry_run: If True, do not draw on the image, only compute and return label boxes.

    Returns:
        A list of bounding boxes (x0, y0, x1, y1) for all placed labels.

    """
    # contours_labels.draw_contour_labels already supports dry_run and returns
    # placements via an internal accumulator. We leverage its behavior and
    # simply propagate parameters.
    # The function signature there also accepts crop_rect but it is optional and
    # unused in typical overlay flow; we pass None to keep behavior consistent.
    return draw_contour_labels(
        base_img,
        seeds_by_level,
        levels,
        crop_rect=None,
        seed_ds=seed_ds,
        mpp=mpp,
        dry_run=dry_run,
    )
