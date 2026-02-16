"""Main map download service - orchestrates map generation pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from domain.models import MapMetadata, MapSettings
from elevation.provider import ElevationTileProvider
from geo.topography import (
    ELEVATION_COLOR_RAMP,
    assemble_dem,
    choose_zoom_with_limit,
    effective_scale_for_xyz,
    latlng_to_pixel_xy,
    meters_per_pixel,
)
from gui.preview import publish_preview_image
from imaging import (
    center_crop,
    draw_axis_aligned_km_grid,
    draw_elevation_legend,
    draw_label_with_bg,
    draw_label_with_subscript_bg,
    load_grid_font,
    rotate_keep_size,
)
from imaging.io import build_save_kwargs as _build_save_kwargs
from imaging.io import save_jpeg as _save_jpeg
from imaging.transforms import _CV2_DIM_LIMIT, ROTATE_ANGLE_EPS
from infrastructure.http.client import cleanup_sqlite_cache as _cleanup_sqlite_cache
from infrastructure.http.client import make_http_session as _make_http_session
from infrastructure.http.client import resolve_cache_dir as _resolve_cache_dir
from infrastructure.http.client import (
    validate_style_api as _validate_api_and_connectivity,
)
from infrastructure.http.client import validate_terrain_api as _validate_terrain_api
from services.coordinate_transformer import (
    CoordinateTransformer,
    validate_control_point_bounds,
)
from services.map_context import MapDownloadContext
from services.map_postprocessing import (
    compute_control_point_image_coords,
    draw_center_cross_on_image,
    draw_control_point_triangle,
    draw_radar_marker,
)
from services.processors.elevation_contours import (
    apply_contours_to_image,
)
from services.radar_coverage import draw_sector_overlay
from services.tile_coverage import compute_tile_coverage
from shared.constants import (
    ASYNC_MAX_CONCURRENCY,
    CONTROL_POINT_LABEL_GAP_MIN_PX,
    CONTROL_POINT_LABEL_GAP_RATIO,
    DOWNLOAD_CONCURRENCY,
    ELEVATION_COLOR_USE_RETINA,
    ELEVATION_LEGEND_STEP_M,
    ELEVATION_USE_RETINA,
    MAX_OUTPUT_PIXELS,
    MAX_ZOOM,
    PIL_DISABLE_LIMIT,
    RADAR_COVERAGE_USE_RETINA,
    RADIO_HORIZON_COLOR_RAMP,
    RADIO_HORIZON_USE_RETINA,
    ROTATION_EPSILON,
    UAV_HEIGHT_REFERENCE_ABBR,
    XYZ_TILE_SIZE,
    XYZ_USE_RETINA,
    MapType,
    default_map_type,
    map_type_to_style_id,
)
from shared.diagnostics import log_memory_usage, log_thread_status
from shared.progress import LiveSpinner

logger = logging.getLogger(__name__)


class MapDownloadService:
    """Main service for downloading and generating maps."""

    def __init__(self, api_key: str):
        """
        Initialize service with API key.

        Args:
            api_key: Mapbox API key

        """
        self.api_key = api_key

    async def download(
        self,
        center_x_sk42_gk: float,
        center_y_sk42_gk: float,
        width_m: float,
        height_m: float,
        output_path: str,
        max_zoom: int = MAX_ZOOM,
        settings: MapSettings | None = None,
    ) -> tuple[str, MapMetadata]:
        """
        Download and generate map.

        Args:
            center_x_sk42_gk: Center X in SK-42 Gauss-Kruger (easting)
            center_y_sk42_gk: Center Y in SK-42 Gauss-Kruger (northing)
            width_m: Map width in meters
            height_m: Map height in meters
            output_path: Output file path
            max_zoom: Maximum zoom level
            settings: Map settings

        Returns:
            Tuple of (output file path, map metadata)

        """
        overall_start_time = time.monotonic()
        logger.info('=== ОБЩИЙ ТАЙМЕР: старт MapDownloadService.download ===')

        # Create context
        ctx = await self._create_context(
            center_x_sk42_gk=center_x_sk42_gk,
            center_y_sk42_gk=center_y_sk42_gk,
            width_m=width_m,
            height_m=height_m,
            output_path=output_path,
            max_zoom=max_zoom,
            settings=settings,
        )

        # Process map
        cache_dir_resolved = _resolve_cache_dir()
        session_ctx = _make_http_session(cache_dir_resolved)

        log_memory_usage('before tile download')
        log_thread_status('before tile download')

        try:
            async with session_ctx as client:
                ctx.client = client
                ctx.semaphore = asyncio.Semaphore(
                    DOWNLOAD_CONCURRENCY or ASYNC_MAX_CONCURRENCY
                )

                # Select and run processor
                ctx.result = await self._run_processor(ctx)

                # Post-processing (may require network for overlay contours)
                await self._postprocess(ctx)

        finally:
            self._cleanup_session(session_ctx, cache_dir_resolved)

        # Save result
        metadata = ctx.to_metadata()
        result_path = await self._save(ctx)

        overall_elapsed = time.monotonic() - overall_start_time
        logger.info(
            '=== ОБЩИЙ ТАЙМЕР: завершён MapDownloadService.download (%.2fs) ===',
            overall_elapsed,
        )

        return result_path, metadata

    async def _create_context(
        self,
        center_x_sk42_gk: float,
        center_y_sk42_gk: float,
        width_m: float,
        height_m: float,
        output_path: str,
        max_zoom: int,
        settings: MapSettings | None,
    ) -> MapDownloadContext:
        """Create map download context with all computed parameters."""
        # Default settings
        if settings is None:
            settings = MapSettings(
                from_x_high=54,
                from_y_high=74,
                to_x_high=54,
                to_y_high=74,
                from_x_low=14,
                from_y_low=43,
                to_x_low=23,
                to_y_low=49,
                output_path=output_path,
                grid_width_m=5.0,
                grid_font_size_m=100.0,
                grid_text_margin_m=50.0,
                grid_label_bg_padding_m=10.0,
                mask_opacity=0.35,
            )

        # Determine map type
        mt = getattr(settings, 'map_type', default_map_type())
        try:
            mt_enum = MapType(mt) if not isinstance(mt, MapType) else mt
        except Exception:
            mt_enum = default_map_type()

        # Determine style and flags
        (
            style_id,
            is_elev_color,
            is_elev_contours,
            is_radio_horizon,
            is_radar_coverage,
        ) = await self._determine_map_type(mt_enum, settings)

        # Determine scale
        if is_elev_color:
            eff_scale = effective_scale_for_xyz(
                256, use_retina=ELEVATION_COLOR_USE_RETINA
            )
        elif is_elev_contours:
            eff_scale = effective_scale_for_xyz(256, use_retina=ELEVATION_USE_RETINA)
        elif is_radio_horizon:
            eff_scale = effective_scale_for_xyz(
                256, use_retina=RADIO_HORIZON_USE_RETINA
            )
        elif is_radar_coverage:
            eff_scale = effective_scale_for_xyz(
                256, use_retina=RADAR_COVERAGE_USE_RETINA
            )
        else:
            eff_scale = effective_scale_for_xyz(
                XYZ_TILE_SIZE, use_retina=XYZ_USE_RETINA
            )

        # Coordinate transformation
        sp = LiveSpinner('Подготовка: создание трансформеров')
        sp.start()

        custom_helmert = getattr(settings, 'custom_helmert', None)
        coord_transformer = CoordinateTransformer(
            center_x_gk=center_x_sk42_gk,
            center_y_gk=center_y_sk42_gk,
            helmert_params=custom_helmert,
        )
        coord_result = coord_transformer.get_result()

        sp.stop('Подготовка: трансформеры готовы')

        # Store transformer for later rotation calculation
        coord_transformer_obj = coord_transformer

        # Validate control point
        if settings.control_point_enabled:
            validate_control_point_bounds(
                control_x_gk=settings.control_point_x_sk42_gk,
                control_y_gk=settings.control_point_y_sk42_gk,
                center_x_gk=center_x_sk42_gk,
                center_y_gk=center_y_sk42_gk,
                width_m=width_m,
                height_m=height_m,
            )

        # Choose zoom
        sp = LiveSpinner('Подготовка: подбор zoom')
        sp.start()
        zoom = choose_zoom_with_limit(
            center_lat=coord_result.center_lat_wgs,
            width_m=width_m,
            height_m=height_m,
            desired_zoom=max_zoom,
            scale=eff_scale,
            max_pixels=MAX_OUTPUT_PIXELS,
        )
        sp.stop('Подготовка: zoom выбран')

        if PIL_DISABLE_LIMIT:
            Image.MAX_IMAGE_PIXELS = None

        # Compute tile coverage
        sp = LiveSpinner('Подготовка: расчёт покрытия XYZ')
        sp.start()
        coverage = compute_tile_coverage(
            center_lat_wgs=coord_result.center_lat_wgs,
            center_lng_wgs=coord_result.center_lng_wgs,
            width_m=width_m,
            height_m=height_m,
            zoom=zoom,
            eff_scale=eff_scale,
        )
        sp.stop('Подготовка: покрытие рассчитано')

        # Now compute rotation with map_params available
        rotation_deg = coord_transformer_obj.compute_rotation_deg(coverage.map_params)

        # Determine effective tile size
        if is_elev_color:
            full_eff_tile_px = 256 * (2 if ELEVATION_COLOR_USE_RETINA else 1)
        elif is_elev_contours:
            full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)
        elif is_radio_horizon:
            full_eff_tile_px = 256 * (2 if RADIO_HORIZON_USE_RETINA else 1)
        elif is_radar_coverage:
            full_eff_tile_px = 256 * (2 if RADAR_COVERAGE_USE_RETINA else 1)
        else:
            full_eff_tile_px = XYZ_TILE_SIZE * (2 if XYZ_USE_RETINA else 1)

        # Create context
        ctx = MapDownloadContext(
            center_x_sk42_gk=center_x_sk42_gk,
            center_y_sk42_gk=center_y_sk42_gk,
            width_m=width_m,
            height_m=height_m,
            api_key=self.api_key,
            output_path=output_path,
            max_zoom=max_zoom,
            settings=settings,
            center_lat_wgs=coord_result.center_lat_wgs,
            center_lng_wgs=coord_result.center_lng_wgs,
            rotation_deg=rotation_deg,
            zoom=zoom,
            eff_scale=eff_scale,
            tiles=coverage.tiles,
            tiles_x=coverage.tiles_x,
            tiles_y=coverage.tiles_y,
            crop_rect=coverage.crop_rect,
            map_params=coverage.map_params,
            target_w_px=coverage.target_w_px,
            target_h_px=coverage.target_h_px,
            t_sk42_to_wgs=coord_result.t_sk42_to_wgs,
            t_sk42_from_gk=coord_result.t_sk42_from_gk,
            t_gk_from_sk42=coord_result.t_gk_from_sk42,
            style_id=style_id,
            is_elev_color=is_elev_color,
            is_elev_contours=is_elev_contours,
            is_radio_horizon=is_radio_horizon,
            is_radar_coverage=is_radar_coverage,
            overlay_contours=bool(getattr(settings, 'overlay_contours', False)),
            full_eff_tile_px=full_eff_tile_px,
        )

        # Store additional data for postprocessing
        ctx.coord_result = coord_result
        ctx.crs_sk42_gk = coord_result.crs_sk42_gk

        return ctx

    async def _determine_map_type(
        self, mt_enum: MapType, settings: MapSettings
    ) -> tuple[str | None, bool, bool, bool, bool]:
        """Determine map type and validate API access."""
        style_id = None
        is_elev_color = False
        is_elev_contours = False
        is_radio_horizon = False
        is_radar_coverage = False

        if mt_enum in (
            MapType.SATELLITE,
            MapType.HYBRID,
            MapType.STREETS,
            MapType.OUTDOORS,
        ):
            style_id = map_type_to_style_id(mt_enum)
            logger.info(
                'Тип карты: %s; style_id=%s; tile_size=%s; retina=%s',
                mt_enum,
                style_id,
                XYZ_TILE_SIZE,
                XYZ_USE_RETINA,
            )
            await _validate_api_and_connectivity(self.api_key, style_id)
        elif mt_enum == MapType.ELEVATION_COLOR:
            logger.info(
                'Тип карты: %s (Terrain-RGB, цветовая шкала); retina=%s',
                mt_enum,
                ELEVATION_USE_RETINA,
            )
            is_elev_color = True
            await _validate_terrain_api(self.api_key)
        elif mt_enum == MapType.ELEVATION_CONTOURS:
            logger.info(
                'Тип карты: %s (Terrain-RGB, контуры); retina=%s',
                mt_enum,
                ELEVATION_USE_RETINA,
            )
            is_elev_contours = True
            await _validate_terrain_api(self.api_key)
        elif mt_enum == MapType.RADIO_HORIZON:
            if not settings.control_point_enabled:
                msg = 'Для карты радиогоризонта необходимо включить контрольную точку'
                raise ValueError(msg)
            logger.info(
                'Тип карты: %s (радиогоризонт); высота антенны=%s м; retina=%s',
                mt_enum,
                settings.antenna_height_m,
                ELEVATION_USE_RETINA,
            )
            is_radio_horizon = True
            await _validate_terrain_api(self.api_key)
        elif mt_enum == MapType.RADAR_COVERAGE:
            if not settings.control_point_enabled:
                msg = (
                    'Для карты зоны обнаружения РЛС '
                    'необходимо включить контрольную точку'
                )
                raise ValueError(msg)
            logger.info(
                'Тип карты: %s (зона обнаружения РЛС); дальность=%s км; '
                'сектор=%s°; retina=%s',
                mt_enum,
                settings.radar_max_range_km,
                settings.radar_sector_width_deg,
                RADAR_COVERAGE_USE_RETINA,
            )
            is_radar_coverage = True
            await _validate_terrain_api(self.api_key)
        else:
            logger.warning(
                'Выбран режим высот (%s), пока не реализовано. Используется Спутник.',
                mt_enum,
            )
            style_id = map_type_to_style_id(default_map_type())
            await _validate_api_and_connectivity(self.api_key, style_id)

        return (
            style_id,
            is_elev_color,
            is_elev_contours,
            is_radio_horizon,
            is_radar_coverage,
        )

    async def _run_processor(self, ctx: MapDownloadContext) -> Image.Image:
        """Run appropriate processor based on map type."""
        if ctx.is_elev_color:
            module = importlib.import_module('services.processors.elevation_color')
            return await module.process_elevation_color(ctx)
        if ctx.is_elev_contours:
            module = importlib.import_module('services.processors.elevation_contours')
            return await module.process_elevation_contours(ctx)
        if ctx.is_radio_horizon:
            module = importlib.import_module('services.processors.radio_horizon')
            return await module.process_radio_horizon(ctx)
        if ctx.is_radar_coverage:
            module = importlib.import_module('services.processors.radar_coverage')
            return await module.process_radar_coverage(ctx)

        module = importlib.import_module('services.processors.xyz_tiles')
        return await module.process_xyz_tiles(ctx)

    async def _postprocess(self, ctx: MapDownloadContext) -> None:
        """Apply post-processing to the result image."""
        result = ctx.result
        if result is None:
            return

        # Load DEM for cursor elevation display if not already loaded
        if ctx.dem_grid is None:
            await self._load_dem_for_cursor(ctx)

        # Overlay contours if enabled
        if ctx.overlay_contours and not ctx.is_elev_contours:
            if ctx.is_radio_horizon or ctx.is_radar_coverage or ctx.is_elev_color:
                # For RH/Radar/ElevColor: draw contours on a separate transparent layer
                # for inclusion in the cached overlay (interactive alpha slider)
                contour_layer = Image.new('RGBA', result.size, (0, 0, 0, 0))
                contour_layer = await self._apply_overlay_contours(ctx, contour_layer)
                # Composite onto result
                result_rgba = result.convert('RGBA')
                result = Image.alpha_composite(result_rgba, contour_layer)
                result = result.convert('RGB')
                with contextlib.suppress(Exception):
                    result_rgba.close()
                ctx.rh_contour_layer = contour_layer
                logger.info('Contour layer created for RH overlay cache')
            else:
                result = await self._apply_overlay_contours(ctx, result)
            ctx.result = result

        # Rotation
        rotation_start_time = time.monotonic()
        logger.info('Поворот изображения — старт')
        sp = LiveSpinner('Поворот карты')
        sp.start()
        try:
            prev_result = result
            result = rotate_keep_size(
                prev_result, ctx.rotation_deg, fill=(255, 255, 255)
            )
            with contextlib.suppress(Exception):
                prev_result.close()
            # Also rotate contour layer for RH overlay cache
            if (
                ctx.rh_contour_layer is not None
                and abs(ctx.rotation_deg) > ROTATE_ANGLE_EPS
            ):
                ctx.rh_contour_layer = rotate_keep_size(
                    ctx.rh_contour_layer,
                    ctx.rotation_deg,
                    fill=(0, 0, 0, 0),
                )
        finally:
            sp.stop('Поворот карты завершён')
        rotation_elapsed = time.monotonic() - rotation_start_time
        logger.info('Поворот изображения — завершён (%.2fs)', rotation_elapsed)
        log_memory_usage('after rotation')

        # Cropping
        crop_start_time = time.monotonic()
        logger.info('Обрезка к целевому размеру — старт')
        prev_result = result
        result = center_crop(prev_result, ctx.target_w_px, ctx.target_h_px)
        with contextlib.suppress(Exception):
            prev_result.close()
        # Also crop contour layer for RH overlay cache
        if ctx.rh_contour_layer is not None:
            ctx.rh_contour_layer = center_crop(
                ctx.rh_contour_layer, ctx.target_w_px, ctx.target_h_px
            )
        crop_elapsed = time.monotonic() - crop_start_time
        logger.info('Обрезка к целевому размеру — завершена (%.2fs)', crop_elapsed)
        log_memory_usage('after cropping')

        # Grid
        grid_start_time = time.monotonic()
        logger.info('Рисование км-сетки — старт')
        self._draw_grid(ctx, result)
        grid_elapsed = time.monotonic() - grid_start_time
        logger.info('Рисование км-сетки — завершено (%.2fs)', grid_elapsed)
        log_memory_usage('after grid')

        # Legend
        if ctx.is_elev_color or ctx.is_radio_horizon or ctx.is_radar_coverage:
            self._draw_legend(ctx, result)

        # For RH / radar / elev_color: create and cache overlay
        if ctx.is_radio_horizon or ctx.is_radar_coverage or ctx.is_elev_color:
            self._create_rh_overlay_layer(ctx, result)

        # For radar coverage: draw sector overlay after grid/legend
        if ctx.is_radar_coverage:
            self._draw_radar_sector_overlay(ctx, result)

        # Center cross
        self._draw_center_cross(ctx, result)

        # Control point
        if ctx.settings.control_point_enabled:
            self._draw_control_point(ctx, result)

        # Clear raw DEM reference now that all processing is done
        ctx.raw_dem_for_cursor = None

        ctx.result = result

    async def _load_dem_for_cursor(self, ctx: MapDownloadContext) -> None:
        """Load or transform DEM for cursor elevation display."""
        dem_start_time = time.monotonic()

        try:
            # Check if raw DEM was already loaded by a processor
            if ctx.raw_dem_for_cursor is not None:
                logger.info(
                    'Используется DEM из процессора (размер %dx%d)',
                    ctx.raw_dem_for_cursor.shape[1],
                    ctx.raw_dem_for_cursor.shape[0],
                )
                dem_full = ctx.raw_dem_for_cursor
                # Note: Don't clear reference yet - apply_contours_to_image may need it
            else:
                # Need to load DEM from scratch (for XYZ maps without elevation)
                logger.info('Загрузка DEM для информера высоты — старт')

                provider = ElevationTileProvider(
                    client=ctx.client,
                    api_key=ctx.api_key,
                    use_retina=ELEVATION_USE_RETINA,
                    cache_root=_resolve_cache_dir(),
                )

                # DEM tiles have fixed size (256 or 512 with retina)
                dem_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)

                # Calculate scale factor between main image tiles and DEM tiles
                scale_factor = ctx.full_eff_tile_px / dem_tile_px

                # Adjust crop_rect for DEM tile size
                cx, cy, cw, ch = ctx.crop_rect
                dem_crop_rect = (
                    int(cx / scale_factor),
                    int(cy / scale_factor),
                    int(cw / scale_factor),
                    int(ch / scale_factor),
                )

                # Fetch DEM tiles
                async def fetch_dem_tile(
                    idx_xy: tuple[int, tuple[int, int]],
                ) -> tuple[int, list[list[float]]]:
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    async with ctx.semaphore:
                        dem_tile = await provider.get_tile_dem(
                            ctx.zoom, tile_x_world, tile_y_world
                        )
                        return idx, dem_tile

                tasks = [fetch_dem_tile(pair) for pair in enumerate(ctx.tiles)]
                results = await asyncio.gather(*tasks)
                results.sort(key=lambda t: t[0])
                dem_tiles_data = [dem for _, dem in results]

                # Assemble full DEM with adjusted crop_rect
                dem_full = assemble_dem(
                    tiles_data=dem_tiles_data,
                    tiles_x=ctx.tiles_x,
                    tiles_y=ctx.tiles_y,
                    eff_tile_px=dem_tile_px,
                    crop_rect=dem_crop_rect,
                )
                del dem_tiles_data
                gc.collect()

                # Scale DEM to match main image size (before rotation/crop)
                if scale_factor != 1.0:
                    target_h = int(dem_full.shape[0] * scale_factor)
                    target_w = int(dem_full.shape[1] * scale_factor)
                    dem_full = cv2.resize(
                        dem_full,
                        (target_w, target_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

            # Apply rotation (same as rotate_keep_size for main image)
            if abs(ctx.rotation_deg) > ROTATION_EPSILON:
                h, w = dem_full.shape
                if w >= _CV2_DIM_LIMIT or h >= _CV2_DIM_LIMIT:
                    # PIL fallback для больших DEM
                    dem_pil = Image.fromarray(dem_full, mode='F')
                    dem_pil = dem_pil.rotate(
                        ctx.rotation_deg,
                        resample=Image.Resampling.BICUBIC,
                        fillcolor=0.0,
                    )
                    dem_full = np.array(dem_pil)
                else:
                    center = (w / 2, h / 2)
                    rotation_matrix = cv2.getRotationMatrix2D(
                        center, ctx.rotation_deg, 1.0
                    )
                    dem_full = cv2.warpAffine(
                        dem_full,
                        rotation_matrix,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    )

            # Center crop DEM to target size (same as center_crop for main image)
            h, w = dem_full.shape
            tw, th = ctx.target_w_px, ctx.target_h_px
            if w != tw or h != th:
                left = (w - tw) // 2
                top = (h - th) // 2
                # Handle case where DEM might be slightly smaller due to rounding
                if left < 0 or top < 0:
                    # Pad DEM if needed
                    pad_left = max(0, -left)
                    pad_top = max(0, -top)
                    pad_right = max(0, tw - w + left)
                    pad_bottom = max(0, th - h + top)
                    dem_full = np.pad(
                        dem_full,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode='constant',
                        constant_values=0,
                    )
                    left = max(0, left)
                    top = max(0, top)
                dem_full = dem_full[top : top + th, left : left + tw]

            ctx.dem_grid = dem_full

            dem_elapsed = time.monotonic() - dem_start_time
            logger.info(
                'Загрузка DEM для информера высоты — завершена (%.2fs)', dem_elapsed
            )

        except Exception as e:
            logger.warning('Не удалось загрузить DEM для информера: %s', e)
            ctx.dem_grid = None

    async def _apply_overlay_contours(
        self, ctx: MapDownloadContext, result: Image.Image
    ) -> Image.Image:
        """Apply contour overlay to the result image."""
        logger.info('Наложение изолиний на карту — старт')
        try:
            result = await apply_contours_to_image(ctx, result)
            logger.info('Наложение изолиний на карту — завершено')
        except Exception as e:
            logger.warning('Не удалось наложить изолинии: %s', e)
        return result

    def _draw_grid(self, ctx: MapDownloadContext, result: Image.Image) -> None:
        """Draw kilometer grid on the result image."""
        try:
            coord_result = ctx.coord_result
            draw_axis_aligned_km_grid(
                img=result,
                center_lat_sk42=coord_result.center_lat_sk42,
                center_lng_sk42=coord_result.center_lng_sk42,
                center_lat_wgs=ctx.center_lat_wgs,
                center_lng_wgs=ctx.center_lng_wgs,
                zoom=ctx.zoom,
                crs_sk42_gk=coord_result.crs_sk42_gk,
                t_sk42_to_wgs=ctx.t_sk42_to_wgs,
                scale=ctx.eff_scale,
                width_m=ctx.settings.grid_width_m,
                grid_font_size_m=ctx.settings.grid_font_size_m,
                grid_text_margin_m=ctx.settings.grid_text_margin_m,
                grid_label_bg_padding_m=ctx.settings.grid_label_bg_padding_m,
                display_grid=ctx.settings.display_grid,
                rotation_deg=ctx.rotation_deg,
            )
        except Exception as e:
            logger.warning('Не удалось нарисовать км-сетку: %s', e)

    def _draw_legend(self, ctx: MapDownloadContext, result: Image.Image) -> None:
        """Draw elevation legend on the result image."""
        legend_start_time = time.monotonic()
        logger.info('Рисование легенды высот — старт')
        try:
            if ctx.is_elev_color:
                color_ramp = ELEVATION_COLOR_RAMP
                min_elev = ctx.elev_min_m or 0.0
                max_elev = ctx.elev_max_m or 1000.0
                title = 'Высота, м'
            elif ctx.is_radar_coverage:
                color_ramp = RADIO_HORIZON_COLOR_RAMP
                min_elev = ctx.settings.radar_target_height_min_m
                max_elev = ctx.settings.radar_target_height_max_m
                title = 'Мин. высота обнаружения РЛС, м'
            else:  # radio horizon
                color_ramp = RADIO_HORIZON_COLOR_RAMP
                min_elev = 0.0
                max_elev = ctx.settings.max_flight_height_m

                abbr = UAV_HEIGHT_REFERENCE_ABBR.get(
                    ctx.settings.uav_height_reference, ''
                )
                title = (
                    f'Минимальная высота БпЛА ({abbr}) для устойчивой радиосвязи'
                    if abbr
                    else 'Минимальная высота БпЛА'
                )

            draw_elevation_legend(
                img=result,
                color_ramp=color_ramp,
                min_elevation_m=min_elev,
                max_elevation_m=max_elev,
                center_lat_wgs=ctx.center_lat_wgs,
                zoom=ctx.zoom,
                scale=ctx.eff_scale,
                title=title,
                label_step_m=ELEVATION_LEGEND_STEP_M,
                grid_font_size_m=ctx.settings.grid_font_size_m,
            )
            legend_elapsed = time.monotonic() - legend_start_time
            logger.info('Рисование легенды высот — завершено (%.2fs)', legend_elapsed)
        except Exception as e:
            logger.warning('Не удалось нарисовать легенду высот: %s', e)

    def _draw_center_cross(self, ctx: MapDownloadContext, result: Image.Image) -> None:
        """Draw center cross on the result image."""
        try:
            mpp = meters_per_pixel(ctx.center_lat_wgs, ctx.zoom, scale=ctx.eff_scale)
            # Use red color for center cross as requested for consistency
            draw_center_cross_on_image(result, mpp)
        except Exception as e:
            logger.warning('Не удалось нарисовать центральный крест: %s', e)

    def _draw_control_point(self, ctx: MapDownloadContext, result: Image.Image) -> None:
        """Draw control point marker on the result image."""
        try:
            # Convert control point to WGS84
            cp_lng_sk42, cp_lat_sk42 = ctx.t_sk42_from_gk.transform(
                ctx.settings.control_point_x_sk42_gk,
                ctx.settings.control_point_y_sk42_gk,
            )
            cp_lng_wgs, cp_lat_wgs = ctx.t_sk42_to_wgs.transform(
                cp_lng_sk42, cp_lat_sk42
            )

            # Compute image coordinates
            cx_img, cy_img = compute_control_point_image_coords(
                cp_lat_wgs=cp_lat_wgs,
                cp_lng_wgs=cp_lng_wgs,
                center_lat_wgs=ctx.center_lat_wgs,
                center_lng_wgs=ctx.center_lng_wgs,
                zoom=ctx.zoom,
                eff_scale=ctx.eff_scale,
                img_width=result.width,
                img_height=result.height,
                rotation_deg=ctx.rotation_deg,
                latlng_to_pixel_xy_func=latlng_to_pixel_xy,
            )

            # Check if in bounds
            if 0 <= cx_img < result.width and 0 <= cy_img < result.height:
                mpp = meters_per_pixel(
                    ctx.center_lat_wgs, ctx.zoom, scale=ctx.eff_scale
                )
                draw_control_point_triangle(
                    result,
                    cx_img,
                    cy_img,
                    mpp,
                    ctx.rotation_deg,
                    size_m=ctx.settings.grid_font_size_m,
                )

                # Draw label
                self._draw_control_point_label(ctx, result, cx_img, cy_img, mpp)
            else:
                logger.warning(
                    'Контрольная точка вне кадра: (%.2f, %.2f) not in [0..%d]x[0..%d]',
                    cx_img,
                    cy_img,
                    result.width,
                    result.height,
                )
        except Exception as e:
            logger.warning('Не удалось нарисовать контрольную точку: %s', e)

    def _create_rh_overlay_layer(
        self,
        ctx: MapDownloadContext,
        result: Image.Image,
    ) -> None:
        """Create overlay layer with grid/legend/contours for radio horizon caching."""
        try:
            # Create transparent layer same size as result
            overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))

            # Include contours if available (rotated/cropped contour layer)
            has_contours = False
            if ctx.rh_contour_layer is not None:
                overlay = Image.alpha_composite(overlay, ctx.rh_contour_layer)
                has_contours = True
                # Free memory - no longer needed after inclusion in overlay
                ctx.rh_contour_layer = None

            # Draw grid on overlay
            self._draw_grid(ctx, overlay)

            # Save base overlay (contours + grid) for legend rebuild
            ctx.rh_cache_overlay_base = overlay.copy()

            # Draw legend on overlay
            if ctx.is_radio_horizon or ctx.is_radar_coverage or ctx.is_elev_color:
                self._draw_legend(ctx, overlay)

            # Save to cache
            ctx.rh_cache_overlay = overlay.copy()
            logger.info(
                'Created radio horizon overlay layer for caching (contours=%s)',
                has_contours,
            )

        except Exception as e:
            logger.warning('Failed to create radio horizon overlay layer: %s', e)

    def _draw_radar_sector_overlay(
        self, ctx: MapDownloadContext, result: Image.Image
    ) -> None:
        """Draw radar sector overlay (shadow, borders, ceiling arcs) on result."""
        try:
            mpp = ctx.get_meters_per_pixel()
            if mpp <= 0:
                return

            # Compute radar position in image coordinates
            cp_lng_sk42, cp_lat_sk42 = ctx.t_sk42_from_gk.transform(
                ctx.settings.control_point_x_sk42_gk,
                ctx.settings.control_point_y_sk42_gk,
            )
            cp_lng_wgs, cp_lat_wgs = ctx.t_sk42_to_wgs.transform(
                cp_lng_sk42, cp_lat_sk42
            )
            cx_img, cy_img = compute_control_point_image_coords(
                cp_lat_wgs=cp_lat_wgs,
                cp_lng_wgs=cp_lng_wgs,
                center_lat_wgs=ctx.center_lat_wgs,
                center_lng_wgs=ctx.center_lng_wgs,
                zoom=ctx.zoom,
                eff_scale=ctx.eff_scale,
                img_width=result.width,
                img_height=result.height,
                rotation_deg=ctx.rotation_deg,
                latlng_to_pixel_xy_func=latlng_to_pixel_xy,
            )

            max_range_px = (ctx.settings.radar_max_range_km * 1000.0) / mpp

            # Convert to RGBA for overlay drawing
            result_rgba = result.convert('RGBA') if result.mode != 'RGBA' else result

            ppm = 1.0 / mpp
            font_size_px = max(10, round(ctx.settings.grid_font_size_m * ppm * 0.4))
            try:
                arc_font = load_grid_font(font_size_px)
            except Exception:
                arc_font = None

            draw_sector_overlay(
                img=result_rgba,
                cx=cx_img,
                cy=cy_img,
                azimuth_deg=ctx.settings.radar_azimuth_deg,
                sector_width_deg=ctx.settings.radar_sector_width_deg,
                max_range_px=max_range_px,
                pixel_size_m=mpp,
                elevation_max_deg=ctx.settings.radar_elevation_max_deg,
                font=arc_font,
                rotation_deg=ctx.rotation_deg,
            )

            # Draw radar marker (diamond with direction)
            draw_radar_marker(
                result_rgba,
                cx_img,
                cy_img,
                mpp,
                azimuth_deg=ctx.settings.radar_azimuth_deg,
                rotation_deg=ctx.rotation_deg,
            )

            # Copy back to result if it was converted
            if result.mode != 'RGBA':
                result.paste(result_rgba.convert('RGB'))
            else:
                result.paste(result_rgba)

        except Exception as e:
            logger.warning('Не удалось нарисовать сектор РЛС: %s', e)

    def _draw_control_point_label(
        self,
        ctx: MapDownloadContext,
        result: Image.Image,
        cx_img: float,
        cy_img: float,
        mpp: float,
    ) -> None:
        """Draw control point label for maps."""
        # Always draw name if provided.
        # Detailed label (height) is for radio horizon / radar coverage maps.
        cp_name = getattr(ctx.settings, 'control_point_name', None)
        if not cp_name and not ctx.is_radio_horizon and not ctx.is_radar_coverage:
            return

        try:
            ppm = 1.0 / mpp if mpp > 0 else 0.0
            font_size_px = max(12, round(ctx.settings.grid_font_size_m * ppm))
            label_font = load_grid_font(font_size_px)
            subscript_font = load_grid_font(max(8, font_size_px * 2 // 3))
            bg_padding_px = max(2, round(ctx.settings.grid_label_bg_padding_m * ppm))

            draw = ImageDraw.Draw(result)
            antenna_h = ctx.settings.antenna_height_m

            # Position below triangle (triangle size matches font size)
            tri_size_px = font_size_px
            label_x = int(cx_img)
            label_gap_px = max(
                CONTROL_POINT_LABEL_GAP_MIN_PX,
                round(tri_size_px * CONTROL_POINT_LABEL_GAP_RATIO),
            )
            current_y = int(cy_img + tri_size_px / 2 + label_gap_px + bg_padding_px)

            # Name line
            if cp_name:
                draw_label_with_bg(
                    draw,
                    (label_x, current_y),
                    cp_name,
                    font=label_font,
                    anchor='mt',
                    img_size=result.size,
                    padding=bg_padding_px,
                )
                name_bbox = draw.textbbox((0, 0), cp_name, font=label_font, anchor='lt')
                name_height = name_bbox[3] - name_bbox[1]
                current_y += name_height + bg_padding_px * 2

            # Height line with subscript (Radio Horizon / Radar Coverage)
            if ctx.is_radio_horizon or ctx.is_radar_coverage:
                cp_elev = ctx.control_point_elevation
                if cp_elev is not None:
                    height_parts = [
                        ('h = ', False),
                        (f'{int(cp_elev)}', False),
                        (' + ', False),
                        (f'{int(antenna_h)} м', False),
                    ]
                else:
                    height_parts = [
                        ('h', False),
                        ('ант', True),
                        (f' = {int(antenna_h)} м', False),
                    ]
                draw_label_with_subscript_bg(
                    draw,
                    (label_x, current_y),
                    height_parts,
                    font=label_font,
                    subscript_font=subscript_font,
                    anchor='mt',
                    img_size=result.size,
                    padding=bg_padding_px,
                )
        except Exception as e:
            logger.warning('Не удалось нарисовать подпись контрольной точки: %s', e)

    def _cleanup_session(
        self, session_ctx: object, cache_dir_resolved: str | None
    ) -> None:
        """Clean up HTTP session resources."""
        try:
            # Note: This is sync cleanup, async close should be done in context
            cache_obj = getattr(session_ctx, '_cache', None)
            if cache_obj:
                inner_cache = getattr(cache_obj, '_cache', None)
                if inner_cache and hasattr(inner_cache, 'close'):
                    inner_cache.close()
        except Exception:
            logger.debug('Error during HTTP session cleanup')

        if cache_dir_resolved:
            _cleanup_sqlite_cache(cache_dir_resolved)

    async def _save(self, ctx: MapDownloadContext) -> str:
        """Save result image to file."""
        result = ctx.result
        if result is None:
            return ctx.output_path

        # Preview publishing
        preview_start_time = time.monotonic()
        logger.info('Публикация предпросмотра — старт')
        did_publish = False
        try:
            gui_image = None
            try:
                gui_image = result.copy()
            except Exception:
                gui_image = None
            metadata = ctx.to_metadata()
            # Log center resolution for diagnostics
            logger.info(
                'Map resolution at center: %.4f m/px (zoom %d, scale %d)',
                metadata.meters_per_pixel,
                metadata.zoom,
                metadata.scale,
            )

            # Collect radio horizon cache if available
            rh_cache = None
            if (
                ctx.is_radio_horizon or ctx.is_radar_coverage
            ) and ctx.rh_cache_dem is not None:
                rh_cache = {
                    'dem': ctx.rh_cache_dem,
                    'dem_full': ctx.rh_cache_dem_full,
                    'topo_base': ctx.rh_cache_topo_base,
                    'antenna_row': ctx.rh_cache_antenna_row,
                    'antenna_col': ctx.rh_cache_antenna_col,
                    'pixel_size_m': ctx.rh_cache_pixel_size_m,
                    'antenna_height_m': ctx.settings.antenna_height_m,
                    'overlay_alpha': ctx.settings.radio_horizon_overlay_alpha,
                    'max_height_m': (
                        ctx.settings.radar_target_height_max_m
                        if ctx.is_radar_coverage
                        else ctx.settings.max_flight_height_m
                    ),
                    'radar_target_height_min_m': ctx.settings.radar_target_height_min_m,
                    'radar_target_height_max_m': ctx.settings.radar_target_height_max_m,
                    'uav_height_reference': ctx.settings.uav_height_reference,
                    'final_size': (
                        ctx.target_w_px,
                        ctx.target_h_px,
                    ),  # Финальный размер для масштабирования
                    'crop_size': ctx.rh_cache_crop_size,
                    # Кэшированный слой с сеткой/легендой/изолиниями
                    'coverage_layer': ctx.rh_cache_coverage,
                    'overlay_layer': ctx.rh_cache_overlay,
                    'overlay_base': ctx.rh_cache_overlay_base,
                    # Параметры постобработки (fallback)
                    'settings': ctx.settings,
                    # Флаг типа карты
                    'is_radar_coverage': ctx.is_radar_coverage,
                    # Параметры РЛС (для пересчёта)
                    'radar_azimuth_deg': ctx.settings.radar_azimuth_deg,
                    'radar_sector_width_deg': ctx.settings.radar_sector_width_deg,
                    'radar_elevation_min_deg': ctx.settings.radar_elevation_min_deg,
                    'radar_elevation_max_deg': ctx.settings.radar_elevation_max_deg,
                    'radar_max_range_km': ctx.settings.radar_max_range_km,
                    # Угол поворота карты (для компенсации в overlay)
                    'rotation_deg': ctx.rotation_deg,
                }
            elif ctx.is_elev_color and ctx.rh_cache_coverage is not None:
                rh_cache = {
                    'topo_base': ctx.rh_cache_topo_base,
                    'overlay_alpha': ctx.settings.radio_horizon_overlay_alpha,
                    'coverage_layer': ctx.rh_cache_coverage,
                    'overlay_layer': ctx.rh_cache_overlay,
                    'overlay_base': ctx.rh_cache_overlay_base,
                    'settings': ctx.settings,
                    'is_elev_color': True,
                    'rotation_deg': ctx.rotation_deg,
                    'final_size': (ctx.target_w_px, ctx.target_h_px),
                    'crop_size': ctx.rh_cache_crop_size,
                }

            if gui_image is not None:
                did_publish = publish_preview_image(
                    gui_image, metadata, ctx.dem_grid, rh_cache
                )
            else:
                did_publish = publish_preview_image(
                    result, metadata, ctx.dem_grid, rh_cache
                )
        except Exception:
            did_publish = False
        preview_elapsed = time.monotonic() - preview_start_time
        logger.info(
            'Публикация предпросмотра — %s (%.2fs)',
            'успех' if did_publish else 'пропущено',
            preview_elapsed,
        )

        # Save if not published to GUI
        if not did_publish:
            save_start_time = time.monotonic()
            logger.info('Сохранение файла — старт')
            sp = LiveSpinner('Сохранение файла')
            sp.start()

            out_path = Path(ctx.output_path)
            if out_path.suffix.lower() not in ('.jpg', '.jpeg'):
                out_path = out_path.with_suffix('.jpg')
            out_path.resolve().parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = _build_save_kwargs(out_path, quality=95)

            _save_jpeg(result, out_path, save_kwargs)

            sp.stop('Сохранение файла: готово')
            save_elapsed = time.monotonic() - save_start_time
            logger.info('Сохранение файла — завершено (%.2fs)', save_elapsed)
            log_memory_usage('after file save')
            with contextlib.suppress(Exception):
                result.close()
        else:
            with contextlib.suppress(Exception):
                result.close()

        # Garbage collection
        gc_start_time = time.monotonic()
        logger.info('Сборка мусора — старт')
        try:
            gc.collect()
            gc_elapsed = time.monotonic() - gc_start_time
            logger.info('Сборка мусора — завершена (%.2fs)', gc_elapsed)
            log_memory_usage('after garbage collection')
            log_thread_status('final cleanup')
        except Exception as e:
            logger.debug(f'Garbage collection failed: {e}')

        return ctx.output_path


async def download_map(
    center_x_sk42_gk: float,
    center_y_sk42_gk: float,
    width_m: float,
    height_m: float,
    api_key: str,
    output_path: str,
    max_zoom: int = MAX_ZOOM,
    settings: MapSettings | None = None,
) -> str:
    """
    Convenience function for downloading maps.

    This is a wrapper around MapDownloadService for backward compatibility.
    """
    service = MapDownloadService(api_key)
    path, _ = await service.download(
        center_x_sk42_gk=center_x_sk42_gk,
        center_y_sk42_gk=center_y_sk42_gk,
        width_m=width_m,
        height_m=height_m,
        output_path=output_path,
        max_zoom=max_zoom,
        settings=settings,
    )
    return path
