"""Map download context - shared state for map generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from domain.models import MapMetadata, MapSettings
from geo import topography

if TYPE_CHECKING:
    import asyncio

    import aiohttp
    from PIL import Image


logger = logging.getLogger(__name__)


@dataclass
class MapDownloadContext:
    """
    Shared context for map download operations.

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

    # DEM grid for cursor elevation display (numpy array, same size as result image)
    dem_grid: Any | None = None
    # Raw DEM grid from processors (before rotation/crop, size = crop_rect)
    # Used to avoid re-downloading DEM in _load_dem_for_cursor
    raw_dem_for_cursor: Any | None = None

    # Result image
    result: Image.Image | None = None

    # Control point elevation (filled by processors if available)
    control_point_elevation: float | None = None

    # Radio horizon cache data (for interactive rebuilding)
    rh_cache_dem: Any | None = (
        None  # numpy array - DEM (даунсэмплированный для расчёта)
    )
    rh_cache_dem_full: Any | None = (
        None  # numpy array - DEM (полное разрешение для курсора)
    )
    rh_cache_topo_base: Image.Image | None = None  # PIL Image - топооснова
    rh_cache_antenna_row: int | None = None  # позиция антенны в DEM (строка)
    rh_cache_antenna_col: int | None = None  # позиция антенны в DEM (столбец)
    rh_cache_pixel_size_m: float | None = None  # размер пикселя в метрах
    rh_cache_crop_size: tuple[int, int] | None = (
        None  # (w, h) DEM до даунсэмплинга = crop_rect px
    )
    rh_cache_coverage: Image.Image | None = (
        None  # PIL Image - слой покрытия (для интерактивной альфы)
    )
    rh_cache_overlay: Image.Image | None = (
        None  # PIL Image - слой с сеткой/легендой/изолиниями
    )
    rh_cache_overlay_base: Image.Image | None = (
        None  # PIL Image - слой без легенды (сетка + изолинии)
    )
    # Temporary: contour layer (RGBA) after rotation/crop, for inclusion in overlay
    rh_contour_layer: Image.Image | None = None

    # Internal coordinate storage (set by service)
    coord_result: Any | None = None
    crs_sk42_gk: Any | None = None

    # Map type flags
    style_id: str | None = None
    is_elev_color: bool = False
    is_elev_contours: bool = False
    is_radio_horizon: bool = False
    is_radar_coverage: bool = False
    overlay_contours: bool = False

    # Tile size settings
    full_eff_tile_px: int = 512

    def to_metadata(self) -> MapMetadata:
        """Сборка метаданных для информера координат."""
        return MapMetadata(
            center_x_gk=self.center_x_sk42_gk,
            center_y_gk=self.center_y_sk42_gk,
            center_lat_wgs=self.center_lat_wgs,
            center_lng_wgs=self.center_lng_wgs,
            meters_per_pixel=self.get_meters_per_pixel(),
            rotation_deg=self.rotation_deg,
            width_px=self.target_w_px,
            height_px=self.target_h_px,
            zoom=self.zoom,
            scale=self.eff_scale,
            crop_x=self.crop_rect[0],
            crop_y=self.crop_rect[1],
            control_point_enabled=self.settings.control_point_enabled,
            original_cp_x_gk=self.settings.control_point_x_sk42_gk
            if self.settings.control_point_enabled
            else None,
            original_cp_y_gk=self.settings.control_point_y_sk42_gk
            if self.settings.control_point_enabled
            else None,
            helmert_params=self.settings.custom_helmert,
            map_type=self.settings.map_type,
        )

    def get_meters_per_pixel(self) -> float:
        """Calculate meters per pixel at center latitude."""
        return topography.meters_per_pixel(
            self.center_lat_wgs, self.zoom, scale=self.eff_scale
        )

    def get_pixels_per_meter(self) -> float:
        """Calculate pixels per meter at center latitude."""
        mpp = self.get_meters_per_pixel()
        return 1.0 / mpp if mpp > 0 else 0.0
