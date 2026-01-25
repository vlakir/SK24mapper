"""Base class for map processors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from services.map_context import MapDownloadContext


class BaseMapProcessor(ABC):
    """
    Base class for all map processors.

    Provides common functionality for tile fetching and processing.
    """

    def __init__(self, ctx: MapDownloadContext):
        """
        Initialize processor with context.

        Args:
            ctx: Map download context with all required state

        """
        self.ctx = ctx
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def process(self) -> Image.Image:
        """
        Process map and return result image.

        Returns:
            Processed PIL Image

        """

    def get_effective_tile_size(self) -> int:
        """
        Get effective tile size in pixels.

        Returns:
            Tile size accounting for retina scaling

        """
        return self.ctx.full_eff_tile_px

    def get_tile_count(self) -> int:
        """
        Get total number of tiles to process.

        Returns:
            Number of tiles

        """
        return len(self.ctx.tiles)
