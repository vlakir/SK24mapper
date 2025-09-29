from __future__ import annotations

"""GUI bridge for publishing preview images.

This thin facade decouples UI integration from service.py and provides
stable API for preview publication.
"""
from progress import publish_preview_image as _publish_preview_image


def publish_preview_image(img) -> bool:
    return _publish_preview_image(img)
