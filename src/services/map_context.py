"""Map download context - shared state for map generation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import aiohttp
from PIL import Image

if TYPE_CHECKING:
    from domain.models import MapSettings
    from elevation.provider import ElevationTileProvider

logger = logging.getLogger(__name__)


@dataclass
class MapDownloadContext:
    """Shared context for map download operations.
    
    Contains all the state needed by various map processors
    (elevation color, contours, radio horizon, XYZ tiles).
    """
    
    # Input parameters
    center_x_sk42_gk: float
    center_y_sk42_gk: float
    width_m: float
    height_m: float
    api_key: str
    output_path: str
    max_zoom: int
    settings: MapSettings
    
    # Computed coordinates
    center_lat_wgs: float = 0.0
    center_lng_wgs: float = 0.0
    rotation_deg: float = 0.0
    
    # Zoom and scale
    zoom: int = 0
    eff_scale: int = 1
    
    # Tile coverage
    tiles: list[tuple[int, int]] = field(default_factory=list)
    tiles_x: int = 0
    tiles_y: int = 0
    crop_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    map_params: dict[str, Any] = field(default_factory=dict)
    
    # Target dimensions
    target_w_px: int = 0
    target_h_px: int = 0
    
    # Transformers (set during initialization)
    t_sk42_to_wgs: Any = None
    t_sk42_from_gk: Any = None
    t_gk_from_sk42: Any = None
    
    # HTTP client (set during execution)
    client: aiohttp.ClientSession | None = None
    semaphore: asyncio.Semaphore | None = None
    
    # Elevation data (for legend)
    elev_min_m: float | None = None
    elev_max_m: float | None = None
    
    # Result image
    result: Image.Image | None = None
    
    # Map type flags
    style_id: str | None = None
    is_elev_color: bool = False
    is_elev_contours: bool = False
    is_radio_horizon: bool = False
    overlay_contours: bool = False
    
    # Tile size settings
    full_eff_tile_px: int = 512
    
    def get_meters_per_pixel(self) -> float:
        """Calculate meters per pixel at center latitude."""
        from geo.topography import meters_per_pixel
        return meters_per_pixel(self.center_lat_wgs, self.zoom, scale=self.eff_scale)
    
    def get_pixels_per_meter(self) -> float:
        """Calculate pixels per meter at center latitude."""
        mpp = self.get_meters_per_pixel()
        return 1.0 / mpp if mpp > 0 else 0.0
