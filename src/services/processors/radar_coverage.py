"""Radar coverage processor - compute and visualize radar detection zone."""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

from PIL import Image

from services.processors.radio_horizon import (
    _blend_coverage_with_topo,
    _load_dem_and_topo,
)
from services.radio_horizon import compute_and_colorize_coverage
from shared.constants import RADAR_COVERAGE_USE_RETINA
from shared.progress import LiveSpinner

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)


async def process_radar_coverage(ctx: MapDownloadContext) -> Image.Image:
    """
    Process radar coverage map (sector-limited detection zone).

    Args:
        ctx: Map download context with radar_* settings.

    Returns:
        Radar coverage visualization image.

    """
    (
        dem_full,
        topo_base,
        antenna_row,
        antenna_col,
        pixel_size_m,
        cp_elevation,
        ds_factor,
    ) = await _load_dem_and_topo(
        ctx,
        use_retina=RADAR_COVERAGE_USE_RETINA,
        label='зоны обнаружения РЛС',
    )

    sp = LiveSpinner('Вычисление зоны обнаружения РЛС')
    sp.start()

    result = compute_and_colorize_coverage(
        dem=dem_full,
        antenna_row=antenna_row,
        antenna_col=antenna_col,
        antenna_height_m=ctx.settings.antenna_height_m,
        pixel_size_m=pixel_size_m,
        max_height_m=ctx.settings.radar_target_height_max_m,
        uav_height_reference=ctx.settings.uav_height_reference,
        cp_elevation=cp_elevation,
        sector_enabled=True,
        radar_azimuth_deg=ctx.settings.radar_azimuth_deg,
        radar_sector_width_deg=ctx.settings.radar_sector_width_deg,
        elevation_min_deg=ctx.settings.radar_elevation_min_deg,
        elevation_max_deg=ctx.settings.radar_elevation_max_deg,
        max_range_m=ctx.settings.radar_max_range_km * 1000.0,
        target_height_min_m=ctx.settings.radar_target_height_min_m,
    )

    sp.stop('Зона обнаружения РЛС вычислена')

    del dem_full
    gc.collect()

    # Shared post-coverage block — numpy / cv2 replacement for the old
    # PIL .resize + .convert + .blend chain, plus DEM-size cache halving.
    # Same wins as radio_horizon: ~−0.6 s post-coverage, ~−0.5 s save.
    result = _blend_coverage_with_topo(
        ctx, result, topo_base, 'radar_coverage'
    )

    logger.info('Карта зоны обнаружения РЛС построена')
    return result
