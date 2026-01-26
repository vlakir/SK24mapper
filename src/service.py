"""
Map download service - backward compatibility wrapper.

This module provides the original download_satellite_rectangle function
as a wrapper around the new MapDownloadService for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from services.map_download_service import MapDownloadService
from shared.constants import MAX_ZOOM

if TYPE_CHECKING:
    from domain.models import MapMetadata, MapSettings

logger = logging.getLogger(__name__)


async def download_satellite_rectangle(
    center_x_sk42_gk: float,
    center_y_sk42_gk: float,
    width_m: float,
    height_m: float,
    api_key: str,
    output_path: str,
    max_zoom: int = MAX_ZOOM,
    settings: MapSettings | None = None,
) -> tuple[str, MapMetadata]:
    """
    Download satellite/map rectangle - backward compatibility wrapper.

    This function delegates to MapDownloadService for actual processing.

    Args:
        center_x_sk42_gk: Center X in SK-42 Gauss-Kruger (easting)
        center_y_sk42_gk: Center Y in SK-42 Gauss-Kruger (northing)
        width_m: Map width in meters
        height_m: Map height in meters
        api_key: Mapbox API key
        output_path: Output file path
        max_zoom: Maximum zoom level
        settings: Map settings

    Returns:
        Tuple of (output file path, map metadata)

    """
    service = MapDownloadService(api_key)
    return await service.download(
        center_x_sk42_gk=center_x_sk42_gk,
        center_y_sk42_gk=center_y_sk42_gk,
        width_m=width_m,
        height_m=height_m,
        output_path=output_path,
        max_zoom=max_zoom,
        settings=settings,
    )
