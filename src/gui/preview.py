from __future__ import annotations

from shared.progress import publish_preview_image as _publish_preview_image

"""GUI bridge for publishing preview images.

This thin facade decouples UI integration from service.py and provides
stable API for preview publication.
"""


def publish_preview_image(
    img: object, metadata: object | None = None, dem_grid: object | None = None
) -> bool:
    return _publish_preview_image(img, metadata, dem_grid)
