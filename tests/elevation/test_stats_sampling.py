"""Tests for elevation.stats sampling helpers."""

import pytest
from PIL import Image

import elevation.stats as stats


@pytest.mark.asyncio
async def test_sample_elevation_percentiles_basic(monkeypatch):
    """Sampling should collect values and return cached tiles when enabled."""
    tile = Image.new("RGB", (2, 2), color=(0, 0, 0))
    progress_calls = []

    async def fake_get_tile_image(_x, _y):
        return tile

    def fake_decode(_img):
        return [[1.0, 2.0], [3.0, 4.0]]

    async def fake_progress(step):
        progress_calls.append(step)

    monkeypatch.setattr(stats, "decode_terrain_rgb_to_elevation_m", fake_decode)
    monkeypatch.setattr(stats, "_tile_overlap_rect_common", lambda *_args, **_kwargs: (0, 0, 2, 2))

    samples, seen_count, cache = await stats.sample_elevation_percentiles(
        tiles=[(0, (10, 20))],
        tiles_x=1,
        crop_rect=(0, 0, 2, 2),
        full_eff_tile_px=2,
        get_tile_image=fake_get_tile_image,
        max_samples=10,
        rng_seed=1,
        on_progress=fake_progress,
        cache_tiles=True,
    )

    assert seen_count == 4
    assert samples == [1.0, 2.0, 3.0, 4.0]
    assert progress_calls == [1]
    assert cache is not None
    assert cache[(10, 20)] is tile