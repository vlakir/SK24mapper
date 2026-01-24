"""Tests for services.processors.base module."""

import pytest

from services.map_context import MapDownloadContext
from services.processors.base import BaseMapProcessor


class DummyProcessor(BaseMapProcessor):
    """Minimal concrete processor for testing."""

    async def process(self):
        return None


@pytest.mark.asyncio
async def test_base_processor_helpers():
    """Helper methods should read values from context."""
    ctx = MapDownloadContext(
        center_x_sk42_gk=0.0,
        center_y_sk42_gk=0.0,
        width_m=1.0,
        height_m=1.0,
        api_key="key",
        output_path="out.png",
        max_zoom=1,
        settings=object(),
    )
    ctx.tiles = [(0, 0), (1, 1)]
    ctx.full_eff_tile_px = 512

    processor = DummyProcessor(ctx)

    assert processor.get_tile_count() == 2
    assert processor.get_effective_tile_size() == 512
    assert await processor.process() is None