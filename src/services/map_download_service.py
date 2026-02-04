"""Main map download service - orchestrates map generation pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from domain.models import MapMetadata, MapSettings
from geo.topography import (
    ELEVATION_COLOR_RAMP,
    choose_zoom_with_limit,
    effective_scale_for_xyz,
    latlng_to_pixel_xy,
    meters_per_pixel,
)
from gui.preview import publish_preview_image
from imaging import (
    draw_elevation_legend,
    draw_label_with_bg,
    draw_label_with_subscript_bg,
    load_grid_font,
)
from imaging.grid import compute_km_grid_elements
from imaging.grid_streaming import (
    draw_control_point_streaming,
    draw_cross_streaming,
    draw_labels_streaming,
    draw_polylines_streaming,
)
from imaging.pyramid import build_pyramid_from_streaming
from imaging.streaming import (
    StreamingImage,
    crop_streaming,
    rotate_streaming,
    save_streaming_image,
)
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
from tiles.cache import TileCache
from tiles.fetcher import TileFetcher
from tiles.writer import CacheWriter
from services.map_postprocessing import compute_control_point_image_coords
from services.processors.elevation_contours import (
    apply_contours_to_image,
)
from services.tile_coverage import compute_tile_coverage
from shared.constants import (
    ASYNC_MAX_CONCURRENCY,
    CONTROL_POINT_LABEL_GAP_MIN_PX,
    CONTROL_POINT_LABEL_GAP_RATIO,
    CONTROL_POINT_SIZE_M,
    DOWNLOAD_CONCURRENCY,
    ELEVATION_LEGEND_STEP_M,
    ELEVATION_USE_RETINA,
    MAX_OUTPUT_PIXELS,
    MAX_OVERLAY_PIXELS,
    MAX_SAVE_PIXELS,
    MAX_ZOOM,
    PIL_DISABLE_LIMIT,
    RADIO_HORIZON_COLOR_RAMP,
    RADIO_HORIZON_USE_RETINA,
    STREAMING_STRIP_HEIGHT,
    STREAMING_TEMP_DIR,
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
        logger.info('Создание HTTP сессии...')
        session_ctx = _make_http_session(cache_dir_resolved)
        logger.info('HTTP сессия создана')

        log_memory_usage('before tile download')
        log_thread_status('before tile download')

        try:
            logger.info('Вход в контекст HTTP сессии...')
            async with session_ctx as client:
                logger.info('HTTP сессия активна, запуск процессора...')
                ctx.client = client
                ctx.semaphore = asyncio.Semaphore(
                    DOWNLOAD_CONCURRENCY or ASYNC_MAX_CONCURRENCY
                )

                # Initialize tile cache and fetcher
                tile_cache = TileCache()
                cache_writer = CacheWriter(tile_cache)
                cache_writer.start()
                try:
                    # Check if offline mode is enabled in settings
                    offline_mode = getattr(ctx.settings, 'offline_mode', False)
                    ctx.tile_fetcher = TileFetcher(
                        cache=tile_cache,
                        writer=cache_writer,
                        api_key=self.api_key,
                        offline=offline_mode,
                    )
                    logger.info(
                        'Tile cache initialized: %s (offline=%s)',
                        tile_cache.cache_dir,
                        offline_mode,
                    )

                    # Select and run processor
                    logger.info(
                        'Запуск процессора для изображения %dx%d (%.1fM пикселей)',
                        ctx.crop_rect[2],
                        ctx.crop_rect[3],
                        ctx.crop_rect[2] * ctx.crop_rect[3] / 1_000_000,
                    )
                    ctx.result = await self._run_processor(ctx)

                    # Post-processing (may require network for overlay contours)
                    await self._postprocess(ctx)
                finally:
                    # Cleanup tile cache system and log statistics
                    cache_writer.stop()
                    if ctx.tile_fetcher:
                        stats = ctx.tile_fetcher.stats
                        total_requests = stats['cache_hits'] + stats['cache_misses']
                        hit_rate = (
                            stats['cache_hits'] / total_requests * 100
                            if total_requests > 0 else 0
                        )
                        logger.info(
                            'Tile cache: %d hits, %d misses (%.1f%% hit rate), '
                            '%d downloads, %d errors',
                            stats['cache_hits'],
                            stats['cache_misses'],
                            hit_rate,
                            stats['downloads'],
                            stats['errors'],
                        )
                    tile_cache.close()

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
        ) = await self._determine_map_type(mt_enum, settings)

        # Determine scale
        if is_elev_color or is_elev_contours:
            eff_scale = effective_scale_for_xyz(256, use_retina=ELEVATION_USE_RETINA)
        elif is_radio_horizon:
            eff_scale = effective_scale_for_xyz(
                256, use_retina=RADIO_HORIZON_USE_RETINA
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

        # Choose zoom with pixel limit to prevent excessive file sizes
        # Use settings.desired_zoom if explicitly set, otherwise use max_zoom
        effective_desired_zoom = (
            settings.desired_zoom if settings.desired_zoom is not None else max_zoom
        )
        zoom = choose_zoom_with_limit(
            center_lat=coord_result.center_lat_wgs,
            width_m=width_m,
            height_m=height_m,
            desired_zoom=effective_desired_zoom,
            scale=eff_scale,
            max_pixels=MAX_OUTPUT_PIXELS,
        )

        if zoom < max_zoom:
            logger.info(
                'Zoom понижен с %d до %d из-за ограничения %.0fM пикселей '
                '(файл был бы слишком большим)',
                max_zoom,
                zoom,
                MAX_OUTPUT_PIXELS / 1_000_000,
            )

        if PIL_DISABLE_LIMIT:
            Image.MAX_IMAGE_PIXELS = None

        # Compute rotation angle BEFORE tile coverage to determine if padding is needed
        # This avoids downloading extra tiles that would be cropped away
        from geo.topography import compute_rotation_deg_for_east_axis

        rotation_deg = compute_rotation_deg_for_east_axis(
            center_lat_sk42=coord_result.center_lat_sk42,
            center_lng_sk42=coord_result.center_lng_sk42,
            map_params=None,  # Not needed when zoom/scale provided
            crs_sk42_gk=coord_result.crs_sk42_gk,
            t_sk42_to_wgs=coord_result.t_sk42_to_wgs,
            zoom=zoom,
            scale=eff_scale,
        )
        logger.debug('Pre-computed rotation: %.4f°', rotation_deg)

        # Compute tile coverage (with or without rotation padding)
        sp = LiveSpinner('Подготовка: расчёт покрытия XYZ')
        sp.start()
        coverage = compute_tile_coverage(
            center_lat_wgs=coord_result.center_lat_wgs,
            center_lng_wgs=coord_result.center_lng_wgs,
            width_m=width_m,
            height_m=height_m,
            zoom=zoom,
            eff_scale=eff_scale,
            rotation_deg=rotation_deg,
        )
        sp.stop('Подготовка: покрытие рассчитано')

        # Determine effective tile size
        if is_elev_color or is_elev_contours:
            full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)
        elif is_radio_horizon:
            full_eff_tile_px = 256 * (2 if RADIO_HORIZON_USE_RETINA else 1)
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
            overlay_contours=bool(getattr(settings, 'overlay_contours', False)),
            full_eff_tile_px=full_eff_tile_px,
        )

        # Store additional data for postprocessing
        ctx.coord_result = coord_result
        ctx.crs_sk42_gk = coord_result.crs_sk42_gk

        return ctx

    async def _determine_map_type(
        self, mt_enum: MapType, settings: MapSettings
    ) -> tuple[str | None, bool, bool, bool]:
        """Determine map type and validate API access."""
        style_id = None
        is_elev_color = False
        is_elev_contours = False
        is_radio_horizon = False

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
        else:
            logger.warning(
                'Выбран режим высот (%s), пока не реализовано. Используется Спутник.',
                mt_enum,
            )
            style_id = map_type_to_style_id(default_map_type())
            await _validate_api_and_connectivity(self.api_key, style_id)

        return style_id, is_elev_color, is_elev_contours, is_radio_horizon

    async def _run_processor(self, ctx: MapDownloadContext) -> StreamingImage:
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

        module = importlib.import_module('services.processors.xyz_tiles')
        return await module.process_xyz_tiles(ctx)

    async def _postprocess(self, ctx: MapDownloadContext) -> None:
        """Apply post-processing to the result StreamingImage."""
        result = ctx.result
        if result is None:
            return

        temp_dir = getattr(ctx, 'temp_dir', STREAMING_TEMP_DIR)

        # Overlay contours if enabled
        if ctx.overlay_contours and not ctx.is_elev_contours:
            result = await self._apply_overlay_contours(ctx, result)
            ctx.result = result

        # Rotation - skip for small angles, rotate grid instead
        from shared.constants import ROTATE_GRID_INSTEAD_THRESHOLD

        rotation_start_time = time.monotonic()
        if abs(ctx.rotation_deg) < ROTATE_GRID_INSTEAD_THRESHOLD:
            # Пропускаем поворот изображения — сетка будет повёрнута вместо него
            logger.info(
                'Поворот изображения ПРОПУЩЕН (%.2f° < %.1f°) — сетка будет под углом',
                abs(ctx.rotation_deg),
                ROTATE_GRID_INSTEAD_THRESHOLD,
            )
            # Флаг: изображение не повёрнуто, сетка должна учитывать угол
            ctx.image_was_rotated = False
        else:
            # Большой угол — поворачиваем изображение как обычно
            logger.info('Поворот изображения — старт (%.2f°)', ctx.rotation_deg)
            sp = LiveSpinner('Поворот карты')
            sp.start()
            try:
                result = rotate_streaming(
                    result, ctx.rotation_deg, fill=(255, 255, 255), temp_dir=temp_dir
                )
            finally:
                sp.stop('Поворот карты завершён')
            ctx.image_was_rotated = True
        rotation_elapsed = time.monotonic() - rotation_start_time
        logger.info('Поворот изображения — завершён (%.2fs)', rotation_elapsed)
        log_memory_usage('after rotation')

        # Cropping
        crop_start_time = time.monotonic()
        logger.info('Обрезка к целевому размеру — старт')
        result = crop_streaming(result, ctx.target_w_px, ctx.target_h_px, temp_dir=temp_dir)
        crop_elapsed = time.monotonic() - crop_start_time
        logger.info('Обрезка к целевому размеру — завершена (%.2fs)', crop_elapsed)
        log_memory_usage('after cropping')

        # Grid, Legend, Cross, Control point - need to convert to PIL for drawing
        # Skip overlays for very large images to avoid memory issues
        total_pixels = result.width * result.height
        if total_pixels > MAX_OVERLAY_PIXELS:
            logger.warning(
                'Изображение слишком большое для оверлеев (%.1fM > %.1fM пикселей). '
                'Сетка, легенда и метки не будут нарисованы.',
                total_pixels / 1_000_000,
                MAX_OVERLAY_PIXELS / 1_000_000,
            )
        else:
            grid_start_time = time.monotonic()
            logger.info('Рисование км-сетки — старт')
            result = self._draw_overlays(ctx, result)
            grid_elapsed = time.monotonic() - grid_start_time
            logger.info('Рисование оверлеев — завершено (%.2fs)', grid_elapsed)
            log_memory_usage('after overlays')

        ctx.result = result

    def _draw_overlays(
        self, ctx: MapDownloadContext, result: StreamingImage
    ) -> StreamingImage:
        """Draw all overlays (grid, legend, cross, control point) on the image.

        Uses streaming approach to avoid loading entire image into memory.
        Memory usage is O(strip_height * width) instead of O(width * height).
        """
        # Draw grid using streaming approach
        self._draw_grid_streaming(ctx, result)

        # Legend - draw on local region only
        if ctx.is_elev_color or ctx.is_radio_horizon:
            self._draw_legend_streaming(ctx, result)

        # Center cross using streaming
        self._draw_center_cross_streaming(ctx, result)

        # Control point using streaming
        if ctx.settings.control_point_enabled:
            self._draw_control_point_streaming(ctx, result)

        result.flush()
        return result

    def _draw_grid_streaming(
        self, ctx: MapDownloadContext, result: StreamingImage
    ) -> None:
        """Draw kilometer grid using streaming approach."""
        try:
            coord_result = ctx.coord_result
            grid_rotation = (
                ctx.rotation_deg if getattr(ctx, 'image_was_rotated', True) else 0.0
            )

            # Compute grid elements without drawing
            grid_elements = compute_km_grid_elements(
                img_width=result.width,
                img_height=result.height,
                center_lat_sk42=coord_result.center_lat_sk42,
                center_lng_sk42=coord_result.center_lng_sk42,
                center_lat_wgs=ctx.center_lat_wgs,
                center_lng_wgs=ctx.center_lng_wgs,
                zoom=ctx.zoom,
                crs_sk42_gk=coord_result.crs_sk42_gk,
                t_sk42_to_wgs=ctx.t_sk42_to_wgs,
                scale=ctx.eff_scale,
                display_grid=ctx.settings.display_grid,
                rotation_deg=grid_rotation,
            )

            # Draw polylines (grid lines) in streaming mode
            if grid_elements.polylines:
                draw_polylines_streaming(
                    img=result,
                    polylines=grid_elements.polylines,
                    color=grid_elements.line_color,
                    width=grid_elements.line_width,
                )

            # Draw crosses (for display_grid=False mode)
            if grid_elements.crosses:
                cross_len = grid_elements.cross_length
                for cx, cy in grid_elements.crosses:
                    draw_cross_streaming(
                        img=result,
                        center_x=int(cx),
                        center_y=int(cy),
                        length=cross_len,
                        width=1,
                        color=grid_elements.line_color,
                    )

            # Draw labels in streaming mode
            if grid_elements.labels:
                draw_labels_streaming(result, grid_elements.labels)

        except Exception as e:
            logger.warning('Не удалось нарисовать км-сетку: %s', e)

    def _draw_legend_streaming(
        self, ctx: MapDownloadContext, result: StreamingImage
    ) -> None:
        """Draw elevation legend on a local region of the streaming image."""
        legend_start_time = time.monotonic()
        logger.info('Рисование легенды высот — старт')
        try:
            if ctx.is_elev_color:
                color_ramp = ELEVATION_COLOR_RAMP
                min_elev = ctx.elev_min_m or 0.0
                max_elev = ctx.elev_max_m or 1000.0
                title = 'Высота, м'
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

            # Compute exact legend bounds to load only the necessary region
            from imaging.legend import compute_legend_bounds

            legend_x1, legend_y1, legend_x2, legend_y2 = compute_legend_bounds(
                img_width=result.width,
                img_height=result.height,
                center_lat_wgs=ctx.center_lat_wgs,
                zoom=ctx.zoom,
                scale=ctx.eff_scale,
                title=title,
            )

            region_width = legend_x2 - legend_x1
            region_height = legend_y2 - legend_y1

            logger.info(
                'Легенда: область (%d,%d)-(%d,%d), размер %dx%d (%.1f МБ)',
                legend_x1,
                legend_y1,
                legend_x2,
                legend_y2,
                region_width,
                region_height,
                region_width * region_height * 3 / 1024 / 1024,
            )

            # Load only the legend region
            region_data = np.empty((region_height, region_width, 3), dtype=np.uint8)
            strip_h = STREAMING_STRIP_HEIGHT
            for y in range(legend_y1, legend_y2, strip_h):
                local_y = y - legend_y1
                actual_strip_h = min(strip_h, legend_y2 - y)
                strip = result.get_strip(y, actual_strip_h)
                # Extract only the horizontal region we need
                region_data[local_y : local_y + actual_strip_h] = strip[
                    :, legend_x1:legend_x2
                ]
                del strip

            # Create PIL image for the region
            pil_region = Image.fromarray(region_data)

            # Draw legend on the region
            # draw_elevation_legend expects full image size to compute positions,
            # so we need to create a wrapper that adjusts coordinates
            from imaging.legend import draw_elevation_legend_on_region

            draw_elevation_legend_on_region(
                img=pil_region,
                region_offset=(legend_x1, legend_y1),
                full_img_size=(result.width, result.height),
                color_ramp=color_ramp,
                min_elevation_m=min_elev,
                max_elevation_m=max_elev,
                center_lat_wgs=ctx.center_lat_wgs,
                zoom=ctx.zoom,
                scale=ctx.eff_scale,
                title=title,
                label_step_m=ELEVATION_LEGEND_STEP_M,
            )

            # Write back to streaming image
            region_arr = np.array(pil_region)
            for y in range(legend_y1, legend_y2, strip_h):
                local_y = y - legend_y1
                actual_strip_h = min(strip_h, legend_y2 - y)
                # Read full strip, update legend region, write back
                full_strip = result.get_strip(y, actual_strip_h)
                full_strip[:, legend_x1:legend_x2] = region_arr[
                    local_y : local_y + actual_strip_h
                ]
                result.set_strip(y, full_strip)
                del full_strip

            pil_region.close()
            del region_data, region_arr

            legend_elapsed = time.monotonic() - legend_start_time
            logger.info('Рисование легенды высот — завершено (%.2fs)', legend_elapsed)
        except Exception as e:
            logger.warning('Не удалось нарисовать легенду высот: %s', e)

    def _draw_center_cross_streaming(
        self, ctx: MapDownloadContext, result: StreamingImage
    ) -> None:
        """Draw center cross using streaming approach."""
        try:
            mpp = meters_per_pixel(ctx.center_lat_wgs, ctx.zoom, scale=ctx.eff_scale)
            ppm = 1.0 / mpp if mpp > 0 else 0.0

            # Cross parameters (from draw_center_cross_on_image)
            from shared.constants import CENTER_CROSS_LENGTH_M, CENTER_CROSS_LINE_WIDTH_M

            length_px = max(1, round(CENTER_CROSS_LENGTH_M * ppm))
            width_px = max(1, round(CENTER_CROSS_LINE_WIDTH_M * ppm))

            # Red color for center cross for consistency with informer and CP markers
            cross_color = (255, 0, 0)

            draw_cross_streaming(
                img=result,
                center_x=result.width // 2,
                center_y=result.height // 2,
                length=length_px,
                width=width_px,
                color=cross_color,
            )
        except Exception as e:
            logger.warning('Не удалось нарисовать центральный крест: %s', e)

    def _draw_control_point_streaming(
        self, ctx: MapDownloadContext, result: StreamingImage
    ) -> None:
        """Draw control point marker using streaming approach."""
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
            effective_rotation = (
                ctx.rotation_deg if getattr(ctx, 'image_was_rotated', True) else 0.0
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
                rotation_deg=effective_rotation,
                latlng_to_pixel_xy_func=latlng_to_pixel_xy,
            )

            # Check if in bounds
            if not (0 <= cx_img < result.width and 0 <= cy_img < result.height):
                logger.warning(
                    'Контрольная точка вне кадра: (%.2f, %.2f) not in [0..%d]x[0..%d]',
                    cx_img,
                    cy_img,
                    result.width,
                    result.height,
                )
                return

            mpp = meters_per_pixel(ctx.center_lat_wgs, ctx.zoom, scale=ctx.eff_scale)
            ppm = 1.0 / mpp if mpp > 0 else 0.0
            tri_size_px = max(5, round(CONTROL_POINT_SIZE_M * ppm))

            # Draw triangle using streaming
            from shared.constants import CONTROL_POINT_COLOR

            draw_control_point_streaming(
                img=result,
                x=int(cx_img),
                y=int(cy_img),
                size=tri_size_px,
                color=CONTROL_POINT_COLOR,
            )

            # Draw label - load local region around control point
            self._draw_control_point_label_streaming(
                ctx, result, cx_img, cy_img, mpp
            )

        except Exception as e:
            logger.warning('Не удалось нарисовать контрольную точку: %s', e)

    def _draw_control_point_label_streaming(
        self,
        ctx: MapDownloadContext,
        result: StreamingImage,
        cx_img: float,
        cy_img: float,
        mpp: float,
    ) -> None:
        """Draw control point label using streaming approach."""
        cp_name = getattr(ctx.settings, 'control_point_name', None)
        if not cp_name and not ctx.is_radio_horizon:
            return

        try:
            ppm = 1.0 / mpp if mpp > 0 else 0.0
            font_size_px = max(12, round(ctx.settings.grid_font_size_m * ppm))
            tri_size_px = max(5, round(CONTROL_POINT_SIZE_M * ppm))

            # Estimate label region (below control point, ~200px height)
            label_region_height = min(200, result.height - int(cy_img))
            if label_region_height < 50:
                return  # Not enough space for label

            label_region_y = int(cy_img)
            label_region_y_end = min(result.height, label_region_y + label_region_height)
            actual_height = label_region_y_end - label_region_y

            # Estimate label width region (centered on control point)
            label_half_width = min(300, result.width // 4)
            label_region_x = max(0, int(cx_img) - label_half_width)
            label_region_x_end = min(result.width, int(cx_img) + label_half_width)
            actual_width = label_region_x_end - label_region_x

            # Load region
            region_data = np.empty((actual_height, actual_width, 3), dtype=np.uint8)
            strip_h = STREAMING_STRIP_HEIGHT
            for y in range(label_region_y, label_region_y_end, strip_h):
                local_y = y - label_region_y
                actual_strip_h = min(strip_h, label_region_y_end - y)
                strip = result.get_strip(y, actual_strip_h)
                region_data[local_y : local_y + actual_strip_h] = strip[
                    :, label_region_x:label_region_x_end
                ]
                del strip

            # Create PIL image for region
            pil_region = Image.fromarray(region_data)
            draw = ImageDraw.Draw(pil_region)

            # Load fonts
            label_font = load_grid_font(font_size_px)
            subscript_font = load_grid_font(max(8, font_size_px * 2 // 3))
            bg_padding_px = max(2, round(ctx.settings.grid_label_bg_padding_m * ppm))

            # Local coordinates (relative to region)
            local_cx = int(cx_img) - label_region_x
            label_gap_px = max(
                CONTROL_POINT_LABEL_GAP_MIN_PX,
                round(tri_size_px * CONTROL_POINT_LABEL_GAP_RATIO),
            )
            current_y = int(tri_size_px / 2 + label_gap_px + bg_padding_px)

            # Name line
            if cp_name:
                draw_label_with_bg(
                    draw,
                    (local_cx, current_y),
                    cp_name,
                    font=label_font,
                    anchor='mt',
                    img_size=pil_region.size,
                    padding=bg_padding_px,
                )
                name_bbox = draw.textbbox((0, 0), cp_name, font=label_font, anchor='lt')
                name_height = name_bbox[3] - name_bbox[1]
                current_y += name_height + bg_padding_px * 2

            # Height line with subscript (Radio Horizon only)
            if ctx.is_radio_horizon:
                cp_elev = ctx.control_point_elevation
                if cp_elev is not None:
                    antenna_h = ctx.settings.antenna_height_m
                    height_parts = [
                        ('h = ', False),
                        (f'{int(cp_elev)}', False),
                        (' + ', False),
                        (f'{int(antenna_h)} м', False),
                    ]
                    draw_label_with_subscript_bg(
                        draw,
                        (local_cx, current_y),
                        height_parts,
                        main_font=label_font,
                        sub_font=subscript_font,
                        anchor='mt',
                        img_size=pil_region.size,
                        padding=bg_padding_px,
                    )

            # Write back to streaming image
            region_arr = np.array(pil_region)
            for y in range(label_region_y, label_region_y_end, strip_h):
                local_y = y - label_region_y
                actual_strip_h = min(strip_h, label_region_y_end - y)
                # Read full strip, update region, write back
                full_strip = result.get_strip(y, actual_strip_h)
                full_strip[:, label_region_x:label_region_x_end] = region_arr[
                    local_y : local_y + actual_strip_h
                ]
                result.set_strip(y, full_strip)
                del full_strip

            pil_region.close()
            del region_data, region_arr

        except Exception as e:
            logger.warning('Не удалось нарисовать подпись контрольной точки: %s', e)

    async def _apply_overlay_contours(
        self, ctx: MapDownloadContext, result: StreamingImage
    ) -> StreamingImage:
        """Apply contour overlay to the result image."""
        logger.info('Наложение изолиний на карту — старт')
        try:
            result = await apply_contours_to_image(ctx, result)
            logger.info('Наложение изолиний на карту — завершено')
        except Exception as e:
            logger.warning('Не удалось наложить изолинии: %s', e)
        return result

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
        """Save result StreamingImage to file."""
        result = ctx.result
        if result is None:
            return ctx.output_path

        # Preview publishing - build pyramid for efficient GUI display
        preview_start_time = time.monotonic()
        logger.info('Публикация предпросмотра — старт')
        did_publish = False
        pyramid = None
        try:
            # Build pyramid from StreamingImage (memory-efficient)
            pyramid = build_pyramid_from_streaming(result)

            metadata = ctx.to_metadata()
            # Log center resolution for diagnostics
            logger.info(
                'Map resolution at center: %.4f m/px (zoom %d, scale %d)',
                metadata.meters_per_pixel,
                metadata.zoom,
                metadata.scale,
            )
            # Publish pyramid instead of full image
            did_publish = publish_preview_image(pyramid, metadata)
            # Note: Do NOT close pyramid here - GUI keeps a reference to it
        except Exception as e:
            logger.warning('Failed to build preview pyramid: %s', e)
            did_publish = False
            # Only close pyramid on error
            if pyramid is not None:
                with contextlib.suppress(Exception):
                    pyramid.close()
        preview_elapsed = time.monotonic() - preview_start_time
        logger.info(
            'Публикация предпросмотра — %s (%.2fs)',
            'успех' if did_publish else 'пропущено',
            preview_elapsed,
        )

        # Always save the file (regardless of preview publication)
        save_start_time = time.monotonic()
        total_pixels = result.width * result.height
        logger.info(
            'Сохранение файла — старт (размер: %dx%d = %.1fM пикселей)',
            result.width,
            result.height,
            total_pixels / 1_000_000,
        )

        if total_pixels > MAX_SAVE_PIXELS:
            logger.warning(
                'Очень большое изображение (%.1fM пикселей). '
                'Сохранение может занять много времени и памяти.',
                total_pixels / 1_000_000,
            )

        sp = LiveSpinner('Сохранение файла')
        sp.start()

        out_path = Path(ctx.output_path)
        out_path.resolve().parent.mkdir(parents=True, exist_ok=True)

        # Use streaming save
        actual_path = save_streaming_image(result, str(out_path), quality=95)

        sp.stop('Сохранение файла: готово')
        save_elapsed = time.monotonic() - save_start_time
        logger.info('Сохранение файла — завершено (%.2fs)', save_elapsed)
        log_memory_usage('after file save')

        # Update output path if it changed (e.g., JPEG -> TIFF for large images)
        ctx.output_path = actual_path

        # Re-publish preview with updated output_path so GUI knows where the file is
        if did_publish and pyramid is not None:
            try:
                updated_metadata = ctx.to_metadata()
                publish_preview_image(pyramid, updated_metadata)
            except Exception as e:
                logger.debug('Failed to re-publish preview with updated path: %s', e)

        # Close StreamingImage (removes temp files)
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
