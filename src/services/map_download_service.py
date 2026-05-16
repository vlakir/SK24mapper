"""Main map download service - orchestrates map generation pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import math
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
    rotate_then_center_crop,
)
from imaging.io import build_save_kwargs as _build_save_kwargs
from imaging.io import save_jpeg as _save_jpeg
from imaging.text import draw_text_with_outline
from imaging.transforms import _CV2_DIM_LIMIT, ROTATE_ANGLE_EPS
from infrastructure.http.client import cleanup_sqlite_cache as _cleanup_sqlite_cache
from infrastructure.http.client import make_http_session as _make_http_session
from infrastructure.http.client import resolve_cache_dir as _resolve_cache_dir
from services.coordinate_transformer import (
    CoordinateTransformer,
    gk_to_sk42_raw,
    is_point_within_bounds,
    sk42_raw_to_gk,
)
from services.map_context import MapDownloadContext
from services.map_postprocessing import (
    compute_control_point_image_coords,
    draw_center_cross_on_image,
    draw_control_point_triangle,
    draw_radar_marker,
)
from services.processors import (
    elevation_color,
    elevation_contours,
    elevation_hillshade,
    link_profile,
    nsu_optimizer,
    radar_coverage,
    radio_horizon,
    xyz_tiles,
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
    PUBLISH_OVERLAY_MAX_DIM,
    NSU_OPTIMIZER_USE_RETINA,
    PIL_DISABLE_LIMIT,
    RADAR_COVERAGE_USE_RETINA,
    RADIO_HORIZON_COLOR_RAMP,
    RADIO_HORIZON_USE_RETINA,
    ROTATION_EPSILON,
    UAV_HEIGHT_REFERENCE_ABBR,
    XYZ_NON_RETINA_MAX_PIXELS,
    XYZ_TILE_SIZE,
    MapType,
    default_map_type,
    map_type_to_style_id,
    xyz_use_retina_for_style,
)
from shared.diagnostics import (
    crash_log,
    crash_log_reset,
)
from shared.memory_estimation import choose_safe_zoom
from shared.progress import LiveSpinner, emit_warning
from shared import contour_layer_disk_cache, dem_topo_cache
from shared.tile_memory_cache import get_global_cache
from shared.tile_profiling import collect_tile_stats, format_tile_stats

logger = logging.getLogger(__name__)


# Per-worker LRU cache of the post-rotation overlay (contour + grid +
# legend rotated to target size). Up to _OVERLAY_CACHE_MAX_ENTRIES entries
# — old ones evicted in insertion order (Python dict preserves it). Only
# used for ELEVATION_COLOR: a rebuild of the same area at the same zoom +
# identical settings reproduces the same overlay pixel-for-pixel, so we
# can skip ~0.35-0.45s of grid+legend drawing + .copy() work on warm
# rebuilds.
#
# Multi-entry sizing rationale: memory_estimation can auto-downgrade zoom
# based on Available memory, so a series of same-button-press rebuilds
# may bounce between z=17 / z=16 / z=15. With max_entries=3 we keep each
# observed zoom level's overlay cached so the bounce-back is a hit.
#
# Key includes everything that affects the cached image: zoom, eff_scale,
# center (lat/lng), area (width_m/height_m), final target size, rotation
# angle, grid settings, legend params (elev_min/max, map_type, font),
# overlay_contours flag. Anything that influences contour shape OR grid
# OR legend MUST be in the key — otherwise cache returns stale output.
#
# Memory: each cached 8404²RGBA image is ~282 MB at z=17, less at lower
# zooms (2395²RGBA ≈ 23 MB at z=15). Three entries max ≈ ~600 MB worst.
_OVERLAY_CACHE_MAX_ENTRIES = 3
_overlay_cache: dict[tuple, 'Image.Image'] = {}


# Per-worker LRU cache of the PRE-rotation contour_layer (RGBA at full
# source size, e.g. 9580² at z=17). Built by apply_contours_to_image
# (marching squares on the DEM + label placement + line drawing — costs
# ~500 ms on z=17). For same area + zoom + contour params the output is
# identical, so caching avoids the recomputation entirely.
#
# Note: this is the PRE-rotation, full-resolution version used to paste
# onto the base map. The post-rotation cropped+rotated version is part
# of _overlay_cache. Both caches live independently because they have
# different cache keys (rotation_deg + target size apply only to the
# post-rotation overlay).
#
# Memory: 9580²RGBA ≈ 367 MB at z=17 per entry. Default MAX=2 — covers
# the common case of same-zoom rebuilds plus one zoom downgrade (so
# ~730 MB worst).
_CONTOUR_LAYER_CACHE_MAX_ENTRIES = 2
_contour_layer_cache: dict[tuple, 'Image.Image'] = {}


def _downsample_overlay_for_publish(
    img: 'Image.Image | None',
    max_dim: int = PUBLISH_OVERLAY_MAX_DIM,
) -> 'Image.Image | None':
    """
    Downsample RGBA overlay перед публикацией через shared memory.

    На 8404² RGBA сериализация в shm = 282 MB raw + ~700 ms wallclock
    (memory-bandwidth bound). GUI всё равно делает cv2.resize обратно
    к result_image.size перед blend (view._on_rh_recompute_finished,
    _on_nsu_recompute_finished, _on_alpha_slider_released). Так что
    можно ужать слой здесь без потерь визуала — итоговое blend идёт
    INTER_LINEAR от уже-уменьшенного source.

    cv2.INTER_AREA для down-resize даёт лучшее качество на RGBA с
    тонкими grid-линиями, чем PIL.resize(LANCZOS): сглаживает по
    площади, без артефактов вокруг alpha-edge.
    """
    if img is None:
        return None
    w, h = img.size
    longer = max(w, h)
    if longer <= max_dim:
        return img
    scale = max_dim / longer
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    arr = np.asarray(img)
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized, img.mode)


def _make_contour_layer_cache_key(ctx: 'MapDownloadContext') -> tuple:
    """Key for the pre-rotation contour bitmap; same area+zoom+contour params."""
    s = ctx.settings
    return (
        'contour_layer_v1',
        ctx.zoom,
        ctx.eff_scale,
        round(ctx.center_lat_wgs, 6),
        round(ctx.center_lng_wgs, 6),
        round(ctx.width_m, 2),
        round(ctx.height_m, 2),
        # Font size affects rendered label bitmaps inside the contour layer.
        s.grid_font_size_m,
        s.overlay_contours,
    )


def _get_cached_contour_layer(ctx: 'MapDownloadContext') -> 'Image.Image | None':
    key = _make_contour_layer_cache_key(ctx)
    cached = _contour_layer_cache.get(key)
    if cached is not None:
        del _contour_layer_cache[key]
        _contour_layer_cache[key] = cached  # LRU bump
    return cached


def _set_cached_contour_layer(
    ctx: 'MapDownloadContext', contour_layer: 'Image.Image',
) -> None:
    """
    Store contour_layer in LRU cache.

    .copy() defensive — caller (RH/Radar/NSU branch) сохраняет
    contour_layer в ctx.rh_contour_layer и потом мутирует его
    через rotate_then_center_crop(close_input=True). Без независимой
    копии тут cache и disk-save thread смотрели бы на закрытый
    buffer; WebP encode валился с ValueError: encoding error 1, а
    следующий cache HIT возвращал бы closed image.
    """
    key = _make_contour_layer_cache_key(ctx)
    if key in _contour_layer_cache:
        old = _contour_layer_cache.pop(key)
        with contextlib.suppress(Exception):
            old.close()
    while len(_contour_layer_cache) >= _CONTOUR_LAYER_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_contour_layer_cache))
        old = _contour_layer_cache.pop(oldest_key)
        with contextlib.suppress(Exception):
            old.close()
    _contour_layer_cache[key] = contour_layer.copy()


def _make_overlay_cache_key(ctx: 'MapDownloadContext') -> tuple:
    """
    Build the cache key for the post-rotation rh_overlay image.

    Includes every input that affects the rendered overlay (contour layer
    composite + grid + legend). Cache covers elev_color / RH / Radar / NSU;
    legend text differs per map type, so map-type-specific legend bounds
    (max_flight_height_m, radar height range, uav_height_reference, etc.)
    are folded in — used as getattr(...) so the same key function works
    for every map type without per-type branches.
    """
    s = ctx.settings
    return (
        'rh_overlay_v3',
        ctx.zoom,
        ctx.eff_scale,
        round(ctx.center_lat_wgs, 6),
        round(ctx.center_lng_wgs, 6),
        round(ctx.width_m, 2),
        round(ctx.height_m, 2),
        ctx.target_w_px,
        ctx.target_h_px,
        round(ctx.rotation_deg, 4),
        getattr(s, 'grid_width_m', None),
        getattr(s, 'grid_font_size_m', None),
        getattr(s, 'grid_text_margin_m', None),
        getattr(s, 'grid_label_bg_padding_m', None),
        getattr(s, 'display_grid', None),
        getattr(s, 'overlay_contours', None),
        ctx.elev_min_m,
        ctx.elev_max_m,
        getattr(s, 'map_type', None),
        # Control point (RH/Radar use it as receiver; safe to fold for all).
        getattr(s, 'control_point_x', None),
        getattr(s, 'control_point_y', None),
        # RH computation + legend bounds.
        getattr(s, 'antenna_height_m', None),
        getattr(s, 'max_flight_height_m', None),
        getattr(s, 'uav_height_reference', None),
        # Radar computation.
        getattr(s, 'radar_azimuth_deg', None),
        getattr(s, 'radar_sector_width_deg', None),
        getattr(s, 'radar_elevation_min_deg', None),
        getattr(s, 'radar_elevation_max_deg', None),
        getattr(s, 'radar_max_range_km', None),
        getattr(s, 'radar_target_height_min_m', None),
        getattr(s, 'radar_target_height_max_m', None),
        # NSU computation.
        getattr(s, 'nsu_target_points_json', None),
        getattr(s, 'nsu_antenna_height_m', None),
        getattr(s, 'nsu_max_flight_height_m', None),
    )


def _get_cached_overlay(ctx: 'MapDownloadContext') -> 'Image.Image | None':
    key = _make_overlay_cache_key(ctx)
    cached = _overlay_cache.get(key)
    if cached is not None:
        # Re-insert to make this key the most-recent (LRU bump on hit).
        del _overlay_cache[key]
        _overlay_cache[key] = cached
    return cached


def _peek_cached_overlay(ctx: 'MapDownloadContext') -> bool:
    """
    Check whether the overlay cache will hit for ctx, without bumping LRU.

    Used at the top of postprocess to decide whether building contour_layer
    is worth it: if the overlay is already baked in cache, the contour layer
    is going to be discarded inside _create_rh_overlay_layer's cache-HIT
    branch anyway, so we can skip ~800 ms of contour build + paste + rotate.
    """
    return _make_overlay_cache_key(ctx) in _overlay_cache


def _set_cached_overlay(ctx: 'MapDownloadContext', overlay: 'Image.Image') -> None:
    """
    Store overlay in LRU cache, evicting oldest if at capacity.

    Caller передаёт overlay, который дальше может быть aliased в
    ctx.rh_cache_overlay / rh_cache_overlay_base — все три ссылки
    указывают на ОДИН объект. Это безопасно: cache HIT path при
    следующем build делает `cached_overlay.copy()` перед mutating
    использованием, а publish-side cv2.resize создаёт независимую
    downsampled-копию. Раньше здесь была defensive .copy() на 8404²
    RGBA = ~112 ms; на MISS-path вместе с двумя _create_rh_overlay
    _layer .copy()'ями накапливалось ~220 ms ненужного memcpy.
    """
    key = _make_overlay_cache_key(ctx)
    # If key already present, refresh it (close old, insert new at end).
    if key in _overlay_cache:
        old = _overlay_cache.pop(key)
        with contextlib.suppress(Exception):
            old.close()
    # LRU eviction: drop FIRST-inserted (= least-recently-used) entries
    # until we have room. Python dict preserves insertion order; cache
    # hits re-insert (in _get_cached_overlay) which bumps an entry to
    # the end, so iteration order matches LRU.
    while len(_overlay_cache) >= _OVERLAY_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(_overlay_cache))
        old = _overlay_cache.pop(oldest_key)
        with contextlib.suppress(Exception):
            old.close()
    _overlay_cache[key] = overlay


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

        crash_log_reset()
        crash_log(
            f'START map build: {width_m:.0f}x{height_m:.0f}m, '
            f'zoom={ctx.zoom}, tiles={len(ctx.tiles)}, '
            f'target={ctx.target_w_px}x{ctx.target_h_px}px, '
            f'type={settings.map_type if settings else "?"}'
        )

        run_processor_elapsed = 0.0
        postprocess_elapsed = 0.0
        try:
            with collect_tile_stats() as tile_stats:
                async with session_ctx as client:
                    ctx.client = client
                    ctx.semaphore = asyncio.Semaphore(
                        DOWNLOAD_CONCURRENCY or ASYNC_MAX_CONCURRENCY
                    )

                    # Run processor
                    t_run_start = time.monotonic()
                    ctx.result = await self._run_processor(ctx)
                    run_processor_elapsed = time.monotonic() - t_run_start

                    # Post-processing (may require network for overlay contours)
                    t_post_start = time.monotonic()
                    await self._postprocess(ctx)
                    postprocess_elapsed = time.monotonic() - t_post_start
                    crash_log('<<< _postprocess DONE')

        finally:
            self._cleanup_session(session_ctx, cache_dir_resolved)

        # Save result
        metadata = ctx.to_metadata()
        t_save_start = time.monotonic()
        result_path = await self._save(ctx)
        save_elapsed = time.monotonic() - t_save_start

        # Release heavy ctx fields — caller has metadata & result_path already
        ctx.result = None
        ctx.raw_dem_for_cursor = None
        ctx.dem_grid = None
        ctx.rh_cache_dem = None
        ctx.rh_cache_topo_base = None
        ctx.rh_cache_coverage = None
        ctx.rh_cache_overlay = None
        ctx.rh_cache_overlay_base = None
        ctx.rh_contour_layer = None
        ctx.nsu_cache_dem = None
        ctx.nsu_cache_topo_base = None
        ctx.nsu_cache_coverage = None
        ctx.link_profile_clean_base = None
        ctx.link_profile_data = None
        # Один финальный gc.collect() — собрать циклы и подготовить чистое
        # состояние для следующего билда в persistent worker.
        gc.collect()

        overall_elapsed = time.monotonic() - overall_start_time
        logger.info(
            '=== ОБЩИЙ ТАЙМЕР: завершён MapDownloadService.download (%.2fs) ===',
            overall_elapsed,
        )
        logger.info(
            'PROFILE breakdown: total=%.2fs run_processor=%.2fs '
            'postprocess=%.2fs save=%.2fs',
            overall_elapsed,
            run_processor_elapsed,
            postprocess_elapsed,
            save_elapsed,
        )
        logger.info('PROFILE %s', format_tile_stats(tile_stats))
        logger.info('PROFILE mem-cache: %s', get_global_cache().stats)

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
            is_link_profile,
            is_nsu_optimizer,
        ) = await self._determine_map_type(mt_enum, settings)

        # Determine scale
        if is_link_profile:
            from shared.constants import LINK_PROFILE_USE_RETINA

            eff_scale = effective_scale_for_xyz(256, use_retina=LINK_PROFILE_USE_RETINA)
        elif is_elev_color:
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
        elif is_nsu_optimizer:
            eff_scale = effective_scale_for_xyz(
                256, use_retina=NSU_OPTIMIZER_USE_RETINA
            )
        else:
            eff_scale = effective_scale_for_xyz(
                XYZ_TILE_SIZE,
                use_retina=xyz_use_retina_for_style(style_id),
            )

        # Coordinate transformation — fast (~3ms), no spinner needed. A
        # spinner here would just block 100ms on stop() waiting for its
        # thread to wake up.
        custom_helmert = getattr(settings, 'custom_helmert', None)
        coord_transformer = CoordinateTransformer(
            center_x_gk=center_x_sk42_gk,
            center_y_gk=center_y_sk42_gk,
            helmert_params=custom_helmert,
        )
        coord_result = coord_transformer.get_result()

        # Store transformer for later rotation calculation
        coord_transformer_obj = coord_transformer

        # Auto-correct control point if outside map bounds
        if settings.control_point_enabled and not is_point_within_bounds(
            settings.control_point_x_sk42_gk,
            settings.control_point_y_sk42_gk,
            center_x_sk42_gk,
            center_y_sk42_gk,
            width_m,
            height_m,
        ):
            raw_x, raw_y = gk_to_sk42_raw(center_x_sk42_gk, center_y_sk42_gk)
            settings.control_point_x = raw_x
            settings.control_point_y = raw_y
            emit_warning(
                'Контрольная точка за пределами карты — перемещена в центр.',
                {
                    'control_point_x': raw_x,
                    'control_point_y': raw_y,
                },
            )

        # Auto-correct link profile points A and B if outside map bounds
        if is_link_profile:
            a_ok = is_point_within_bounds(
                settings.link_point_a_x_sk42_gk,
                settings.link_point_a_y_sk42_gk,
                center_x_sk42_gk,
                center_y_sk42_gk,
                width_m,
                height_m,
            )
            b_ok = is_point_within_bounds(
                settings.link_point_b_x_sk42_gk,
                settings.link_point_b_y_sk42_gk,
                center_x_sk42_gk,
                center_y_sk42_gk,
                width_m,
                height_m,
            )
            if not a_ok or not b_ok:
                offset = min(1750.0, width_m * 0.25, height_m * 0.25)
                a_raw_x, a_raw_y = gk_to_sk42_raw(
                    center_x_sk42_gk - offset, center_y_sk42_gk
                )
                b_raw_x, b_raw_y = gk_to_sk42_raw(
                    center_x_sk42_gk + offset, center_y_sk42_gk
                )
                settings.link_point_a_x = a_raw_x
                settings.link_point_a_y = a_raw_y
                settings.link_point_b_x = b_raw_x
                settings.link_point_b_y = b_raw_y
                emit_warning(
                    'Точки профиля за пределами карты — перемещены к центру.',
                    {
                        'link_point_a_x': a_raw_x,
                        'link_point_a_y': a_raw_y,
                        'link_point_b_x': b_raw_x,
                        'link_point_b_y': b_raw_y,
                    },
                )

        # Filter out-of-bounds NSU target points
        if is_nsu_optimizer:
            import json as _json

            original_points = settings.nsu_target_points
            valid_pts: list[list[int]] = []
            removed_labels: list[str] = []
            for i, (x_sk42, y_sk42) in enumerate(original_points):
                gk_e, gk_n = sk42_raw_to_gk(x_sk42, y_sk42)
                if is_point_within_bounds(
                    gk_e,
                    gk_n,
                    center_x_sk42_gk,
                    center_y_sk42_gk,
                    width_m,
                    height_m,
                ):
                    valid_pts.append([x_sk42, y_sk42])
                else:
                    removed_labels.append(f'#{i + 1}: X={x_sk42}, Y={y_sk42}')
            if removed_labels:
                new_json = _json.dumps(valid_pts)
                settings.nsu_target_points_json = new_json
                emit_warning(
                    'Точки НСУ за пределами карты удалены:\n'
                    + '\n'.join(removed_labels),
                    {'nsu_target_points_json': new_json},
                )

        # Choose zoom (with memory-aware limit) — sub-ms, no spinner needed.
        is_dem_mode = (
            is_elev_color
            or is_elev_contours
            or is_radio_horizon
            or is_radar_coverage
            or is_link_profile
            or is_nsu_optimizer
        )
        # Tighten the pixel cap для всех non-retina XYZ-режимов, чтобы
        # choose_safe_zoom не поднимал zoom «на освободившийся бюджет»:
        # non-retina @ z=N + 1 загружает в 4× больше raw-тайлов, чем
        # retina @ z=N (тот же конечный mpp), и срывал RLIMIT_AS на
        # rotate. Cap удерживает image ≤ XYZ_NON_RETINA_MAX_PIXELS.
        max_pixels_eff = MAX_OUTPUT_PIXELS
        is_non_retina_xyz = mt_enum in (
            MapType.SATELLITE,
            MapType.HYBRID,
            MapType.STREETS,
            MapType.OUTDOORS,
        ) and not xyz_use_retina_for_style(style_id)
        if is_non_retina_xyz:
            max_pixels_eff = min(MAX_OUTPUT_PIXELS, XYZ_NON_RETINA_MAX_PIXELS)
        zoom, memory_estimate = choose_safe_zoom(
            center_lat=coord_result.center_lat_wgs,
            width_m=width_m,
            height_m=height_m,
            desired_zoom=max_zoom,
            eff_scale=eff_scale,
            max_pixels=max_pixels_eff,
            is_dem=is_dem_mode,
        )

        if PIL_DISABLE_LIMIT:
            Image.MAX_IMAGE_PIXELS = None

        # Compute tile coverage + rotation — sub-ms, no spinner needed.
        coverage = compute_tile_coverage(
            center_lat_wgs=coord_result.center_lat_wgs,
            center_lng_wgs=coord_result.center_lng_wgs,
            width_m=width_m,
            height_m=height_m,
            zoom=zoom,
            eff_scale=eff_scale,
        )
        rotation_deg = coord_transformer_obj.compute_rotation_deg(coverage.map_params)

        # Determine effective tile size
        if is_link_profile:
            from shared.constants import LINK_PROFILE_USE_RETINA

            full_eff_tile_px = 256 * (2 if LINK_PROFILE_USE_RETINA else 1)
        elif is_elev_color:
            full_eff_tile_px = 256 * (2 if ELEVATION_COLOR_USE_RETINA else 1)
        elif is_elev_contours:
            full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)
        elif is_radio_horizon:
            full_eff_tile_px = 256 * (2 if RADIO_HORIZON_USE_RETINA else 1)
        elif is_radar_coverage:
            full_eff_tile_px = 256 * (2 if RADAR_COVERAGE_USE_RETINA else 1)
        else:
            full_eff_tile_px = XYZ_TILE_SIZE * (
                2 if xyz_use_retina_for_style(style_id) else 1
            )

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
            is_link_profile=is_link_profile,
            is_nsu_optimizer=is_nsu_optimizer,
            overlay_contours=bool(getattr(settings, 'overlay_contours', False)),
            full_eff_tile_px=full_eff_tile_px,
            memory_estimate=memory_estimate,
        )

        # Store additional data for postprocessing
        ctx.coord_result = coord_result
        ctx.crs_sk42_gk = coord_result.crs_sk42_gk

        return ctx

    async def _determine_map_type(
        self, mt_enum: MapType, settings: MapSettings
    ) -> tuple[str | None, bool, bool, bool, bool, bool, bool]:
        """Determine map type and validate API access."""
        style_id = None
        is_elev_color = False
        is_elev_contours = False
        is_radio_horizon = False
        is_radar_coverage = False
        is_link_profile = False
        is_nsu_optimizer = False

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
                xyz_use_retina_for_style(style_id),
            )
            # Network probe removed: redundant with real tile-fetch error path.
        elif mt_enum == MapType.ELEVATION_COLOR:
            logger.info(
                'Тип карты: %s (Terrain-RGB, цветовая шкала); retina=%s',
                mt_enum,
                ELEVATION_USE_RETINA,
            )
            is_elev_color = True
            # Network probe removed: redundant with real tile-fetch error path.
        elif mt_enum == MapType.ELEVATION_CONTOURS:
            logger.info(
                'Тип карты: %s (Terrain-RGB, контуры); retina=%s',
                mt_enum,
                ELEVATION_USE_RETINA,
            )
            is_elev_contours = True
            # Network probe removed: redundant with real tile-fetch error path.
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
            # Network probe removed: redundant with real tile-fetch error path.
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
            # Network probe removed: redundant with real tile-fetch error path.
        elif mt_enum == MapType.LINK_PROFILE:
            logger.info(
                'Тип карты: %s (профиль радиолинии); частота=%s МГц',
                mt_enum,
                settings.link_freq_mhz,
            )
            is_link_profile = True
            # Network probe removed: redundant with real tile-fetch error path.
        elif mt_enum == MapType.NSU_OPTIMIZER:
            logger.info(
                'Тип карты: %s (оптимальное размещение НСУ); retina=%s',
                mt_enum,
                NSU_OPTIMIZER_USE_RETINA,
            )
            is_nsu_optimizer = True
            # Network probe removed: redundant with real tile-fetch error path.
        elif mt_enum == MapType.ELEVATION_HILLSHADE:
            logger.info(
                'Тип карты: %s (Terrain-RGB, теневая отмывка); retina=%s',
                mt_enum,
                ELEVATION_COLOR_USE_RETINA,
            )
            is_elev_color = True  # переиспользуем флаг для terrain-DEM pipeline
            # Network probe removed: redundant with real tile-fetch error path.
        else:
            logger.warning(
                'Неизвестный тип карты (%s). Используется Спутник.',
                mt_enum,
            )
            style_id = map_type_to_style_id(default_map_type())
            # Network probe removed: redundant with real tile-fetch error path.

        return (
            style_id,
            is_elev_color,
            is_elev_contours,
            is_radio_horizon,
            is_radar_coverage,
            is_link_profile,
            is_nsu_optimizer,
        )

    async def _run_processor(self, ctx: MapDownloadContext) -> Image.Image:  # noqa: PLR0911
        """Run appropriate processor based on map type."""
        if ctx.is_link_profile:
            return await link_profile.process_link_profile(ctx)
        if ctx.is_nsu_optimizer:
            return await nsu_optimizer.process_nsu_optimizer(ctx)
        if ctx.is_elev_color:
            if ctx.settings.map_type == MapType.ELEVATION_HILLSHADE:
                return await elevation_hillshade.process_elevation_hillshade(ctx)
            return await elevation_color.process_elevation_color(ctx)
        if ctx.is_elev_contours:
            return await elevation_contours.process_elevation_contours(ctx)
        if ctx.is_radio_horizon:
            return await radio_horizon.process_radio_horizon(ctx)
        if ctx.is_radar_coverage:
            return await radar_coverage.process_radar_coverage(ctx)
        return await xyz_tiles.process_xyz_tiles(ctx)

    async def _postprocess(self, ctx: MapDownloadContext) -> None:
        """Apply post-processing to the result image."""
        result = ctx.result
        if result is None:
            return

        # Per-stage timing accumulator — printed once at the end as a single
        # `postprocess timings:` line so we can see what's actually expensive
        # without scanning a dozen separate log lines.
        _t_segments: list[tuple[str, float]] = []
        _t_pp_start = time.monotonic()
        _t_seg = _t_pp_start

        def _mark(label: str) -> None:
            nonlocal _t_seg
            now = time.monotonic()
            _t_segments.append((label, now - _t_seg))
            _t_seg = now

        # Peek the post-rotation overlay cache early. If it'll hit, the
        # cached overlay already has contour_layer baked in — building +
        # pasting + pre-rotation-cropping + rotating contour_layer is pure
        # waste (it gets discarded inside _create_rh_overlay_layer's cache-
        # HIT branch). Skip the whole contour pipeline in that case (~800ms
        # warm win on elev_color / RH / Radar / NSU).
        needs_rh_overlay_early = (
            ctx.is_radio_horizon
            or ctx.is_radar_coverage
            or ctx.is_elev_color
            or ctx.is_nsu_optimizer
        )
        overlay_cache_will_hit = (
            needs_rh_overlay_early and _peek_cached_overlay(ctx)
        )

        # Overlay contours if enabled (ДО загрузки DEM для курсора,
        # чтобы не держать dem_grid и raw_dem одновременно с numpy-буферами)
        if ctx.overlay_contours and not ctx.is_elev_contours:
            if overlay_cache_will_hit:
                logger.info(
                    'contour_layer: skipped — overlay cache will HIT '
                    '(contour already baked in cached overlay)'
                )
                _mark('apply_contours_skipped')
            else:
                # Pre-load full DEM if no upstream processor populated it.
                # apply_contours_to_image will reuse ctx.raw_dem_for_cursor
                # instead of streaming-fetching DEM tiles separately, and
                # _load_dem_for_cursor below will reuse it as well — single
                # fetch instead of two.
                await self._ensure_raw_dem_loaded(ctx)
                _mark('ensure_raw_dem')
                if needs_rh_overlay_early:
                    # RH/Radar/ElevColor/NsuOpt: contours on separate
                    # transparent layer for the cached overlay (interactive
                    # alpha slider). Pre-rotation contour layer is identical
                    # for same area+zoom+contour params, so cache it — saves
                    # ~500 ms of marching-squares + label-place + line-draw
                    # work on warm rebuilds.
                    cached_contour = _get_cached_contour_layer(ctx)
                    if cached_contour is not None:
                        # .copy() because postprocess closes ctx.rh_contour
                        # _layer later in _create_rh_overlay_layer — would
                        # otherwise invalidate the cache entry.
                        contour_layer = cached_contour.copy()
                        logger.info(
                            'contour_layer: cache HIT — skipped marching '
                            'squares + label placement + line drawing'
                        )
                    else:
                        # In-memory miss → try disk before paying ~1.5 s of
                        # marching squares + label placement + line drawing.
                        # The disk cache persists across worker process
                        # restarts so the first build after a fresh app
                        # launch on a familiar area is no longer fully cold.
                        disk_key = _make_contour_layer_cache_key(ctx)
                        disk_layer = contour_layer_disk_cache.load(disk_key)
                        if disk_layer is not None:
                            contour_layer = disk_layer
                            # Populate the in-memory LRU so subsequent
                            # rebuilds in this worker hit RAM, not disk.
                            _set_cached_contour_layer(ctx, contour_layer)
                        else:
                            contour_layer = Image.new(
                                'RGBA', result.size, (0, 0, 0, 0)
                            )
                            contour_layer = (
                                await self._apply_overlay_contours(
                                    ctx, contour_layer,
                                )
                            )
                            _set_cached_contour_layer(ctx, contour_layer)
                            contour_layer_disk_cache.save(
                                disk_key, contour_layer,
                            )
                    _mark('apply_contours')
                    # Composite contour_layer onto result using PIL.paste
                    # with the RGBA contour_layer as mask. PIL drops contour
                    # .alpha as a mask and blends contour.rgb onto result.rgb
                    # in-place — exact same math as Image.alpha_composite(
                    # result.convert('RGBA'), contour_layer).convert('RGB'),
                    # but in ONE C-loop instead of three. Saves the
                    # 3-byte↔4-byte repack inside convert('RGBA') and
                    # convert('RGB') (~150ms each at 8488²) and skips holding
                    # two full-size RGBA copies during alpha_composite
                    # (~370 MB peak avoided).
                    result.paste(contour_layer, (0, 0), contour_layer)
                    ctx.rh_contour_layer = contour_layer
                    logger.info('Contour layer created for RH overlay cache')
                    _mark('paste_contour_onto_result')
                else:
                    # HYBRID / HILLSHADE / ELEV_CONTOURS / LINK_PROFILE:
                    # inline contour render onto result. For images
                    # at-or-below the cache-cost-effective size (~12k
                    # along the long edge) we route through the same
                    # cached transparent layer + paste-with-mask
                    # pattern as the rh_overlay map types — saves
                    # ~1.0-1.2 s on warm rebuilds (cache HIT skips
                    # marching squares + label placement + line draw).
                    # Above the cap, in-place draw is faster than
                    # encoding / decoding a 1.5+ GB transparent layer
                    # (HYBRID/SATELLITE z=16 retina hit this).
                    inplace_cache_max_dim = 12000
                    if max(result.size) <= inplace_cache_max_dim:
                        cached_contour = _get_cached_contour_layer(ctx)
                        if cached_contour is not None:
                            contour_layer = cached_contour.copy()
                            logger.info(
                                'contour_layer (inplace): cache HIT — '
                                'skipped marching squares + label '
                                'placement + line drawing'
                            )
                        else:
                            disk_key = _make_contour_layer_cache_key(ctx)
                            disk_layer = contour_layer_disk_cache.load(
                                disk_key,
                            )
                            if disk_layer is not None:
                                contour_layer = disk_layer
                                _set_cached_contour_layer(
                                    ctx, contour_layer,
                                )
                            else:
                                contour_layer = Image.new(
                                    'RGBA', result.size, (0, 0, 0, 0),
                                )
                                contour_layer = (
                                    await self._apply_overlay_contours(
                                        ctx, contour_layer,
                                    )
                                )
                                _set_cached_contour_layer(
                                    ctx, contour_layer,
                                )
                                contour_layer_disk_cache.save(
                                    disk_key, contour_layer,
                                )
                        # Composite onto result via PIL.paste with the
                        # RGBA contour_layer as mask. Same trick as the
                        # rh_overlay path; no transient ~370 MB RGBA
                        # buffer that Image.alpha_composite would build.
                        result.paste(
                            contour_layer, (0, 0), contour_layer,
                        )
                        with contextlib.suppress(Exception):
                            contour_layer.close()
                    else:
                        # Above the cap: inline draw wins.
                        result = await self._apply_overlay_contours(
                            ctx, result,
                        )
                    _mark('apply_contours_inplace')
                ctx.result = result

        # Load DEM for cursor elevation display if not already loaded.
        if ctx.dem_grid is None:
            await self._load_dem_for_cursor(ctx)
            _mark('load_dem_cursor')

        # Освобождаем полноразмерный DEM перед поворотом — dem_grid уже
        # создан, raw_dem больше не нужен, а поворот требует много памяти.
        # Setting to None drops the only refcount → numpy frees the 367MB
        # buffer immediately. No gc.collect() needed (no cycles).
        if ctx.raw_dem_for_cursor is not None:
            ctx.raw_dem_for_cursor = None

        # Pre-rotation crop: rotating 9580×9580 only to throw away the corners
        # in center_crop right after costs ~1s of PIL↔numpy conversion plus
        # warpAffine work. Compute the minimum centred bbox in the source that
        # still contains the rotated final crop, slice to it (PIL.crop is
        # essentially free), then rotate the smaller region.
        # Pre-rotation crop: rotating 9580×9580 only to throw away the corners
        # in center_crop right after costs ~1s of PIL↔numpy conversion plus
        # warpAffine work. Compute the minimum centred bbox in the source that
        # still contains the rotated final crop, slice to it (PIL.crop is
        # essentially free), then rotate the smaller region.
        # Pre-rotation crop: trim the empty rotation corners before
        # warpAffine to save its work on pixels that will be discarded.
        #
        # There are two ways to do this and the right one depends on
        # image size:
        #
        #  - Below ~14k along the long edge, PIL.crop is *lazy* (the
        #    crop is just a subimage handle; pixels materialise only
        #    when next consumed, which happens inside rotate_then_
        #    center_crop's _pil_to_numpy_low_peak anyway). Fusing the
        #    crop into the rotate call would force an extra
        #    np.ascontiguousarray memcpy that PIL was avoiding — net
        #    regression on LINK_PROFILE / RH / NSU 9580².
        #
        #  - At 16k+ (HYBRID / SATELLITE z=16 retina, 19k² source),
        #    PIL.crop is eager and materialises the cropped buffer
        #    (~1 s memcpy). Fusing into rotate via the precrop_box
        #    kwarg lets _pil_to_numpy_low_peak's tobytes serve both
        #    purposes — single memcpy on the full source then a
        #    contiguous slice — saves ~400-700 ms.
        #
        # Pick the path by image size.
        FUSION_PRECROP_MIN_DIM = 14000
        precrop_box: tuple[int, int, int, int] | None = None
        fuse_precrop = False
        if abs(ctx.rotation_deg) > ROTATE_ANGLE_EPS:
            precrop_box = self._precrop_bbox_for_rotation(
                result.size,
                (ctx.target_w_px, ctx.target_h_px),
                ctx.rotation_deg,
            )
            if precrop_box is not None:
                fuse_precrop = max(result.size) >= FUSION_PRECROP_MIN_DIM
                logger.info(
                    'Pre-rotation crop: %dx%d → %dx%d (angle=%.2f°, '
                    'fused=%s)',
                    result.size[0], result.size[1],
                    precrop_box[2] - precrop_box[0],
                    precrop_box[3] - precrop_box[1],
                    ctx.rotation_deg,
                    fuse_precrop,
                )
                if not fuse_precrop:
                    # Small-to-medium image: keep the lazy PIL.crop
                    # path. Apply the crop here and pass None to
                    # rotate_then_center_crop.
                    prev_result = result
                    result = prev_result.crop(precrop_box)
                    with contextlib.suppress(Exception):
                        prev_result.close()
                    if ctx.rh_contour_layer is not None:
                        old_contour = ctx.rh_contour_layer
                        ctx.rh_contour_layer = old_contour.crop(
                            precrop_box,
                        )
                        with contextlib.suppress(Exception):
                            old_contour.close()
                    _mark('pre_rotation_crop')

        # Rotation + center-crop in one cv2.warpAffine pass.
        #
        # Previously rotate (~360 ms per layer, full pre-crop size) and
        # center_crop (~100 ms PIL.crop) ran sequentially, with an extra
        # np→PIL→np cycle between them. rotate_then_center_crop folds
        # both into one warpAffine call with output sized to the final
        # crop — saves ~150 ms total across the two layers (result + contour)
        # and trims the intermediate buffer from 8488² to 8404² (~5%).
        rotation_start_time = time.monotonic()
        logger.info('Поворот + обрезка изображения — старт')
        sp = LiveSpinner('Поворот карты')
        sp.start()
        try:
            # close_input=True передаёт ownership PIL.Image в
            # rotate_then_center_crop — он закроет его СРАЗУ после
            # извлечения raw-байтов (до cv2.warpAffine). На больших
            # картах (z=16 retina SATELLITE даёт 16972×16972 RGB ≈ 864 МБ
            # на канал) это снимает 1× размер с пиковой памяти и спасает
            # от RLIMIT_AS на втором прогоне.
            # Pass precrop_box only if we elected fusion (large source).
            # Otherwise the crop was already applied via the lazy PIL.crop
            # branch above and `result` is already pre-cropped.
            rotate_precrop_box = precrop_box if fuse_precrop else None
            prev_result = result
            result = rotate_then_center_crop(
                prev_result,
                ctx.rotation_deg,
                ctx.target_w_px,
                ctx.target_h_px,
                fill=(255, 255, 255),
                close_input=True,
                precrop_box=rotate_precrop_box,
            )
            del prev_result  # уже закрыт внутри
            if ctx.rh_contour_layer is not None:
                old_contour = ctx.rh_contour_layer
                ctx.rh_contour_layer = rotate_then_center_crop(
                    old_contour,
                    ctx.rotation_deg,
                    ctx.target_w_px,
                    ctx.target_h_px,
                    fill=(0, 0, 0, 0),
                    close_input=True,
                    precrop_box=rotate_precrop_box,
                )
                del old_contour
        finally:
            sp.stop('Поворот карты завершён')
        rotation_elapsed = time.monotonic() - rotation_start_time
        logger.info(
            'Поворот + обрезка изображения — завершён (%.2fs)',
            rotation_elapsed,
        )
        _mark('rotate_and_crop')

        # Grid + legend + overlay cache layer.
        # For RH/Radar/elev_color/NSU we need a transparent RGBA overlay
        # cached separately (for interactive alpha-slider in GUI). Previously
        # grid+legend were drawn TWICE — once baked on `result`, then again on
        # the overlay. ~0.86s of duplicate work on elev_color. Now we draw
        # grid+legend ONCE on the overlay and alpha-composite it onto result
        # at the end — same visible output, no duplication.
        needs_rh_overlay = (
            ctx.is_radio_horizon
            or ctx.is_radar_coverage
            or ctx.is_elev_color
            or ctx.is_nsu_optimizer
        )

        if needs_rh_overlay:
            result = self._create_rh_overlay_layer(ctx, result)
            _mark('rh_overlay_create_bake')
        else:
            # Simple path (HYBRID etc.): grid baked directly on result, no
            # overlay cache, no legend.
            grid_start_time = time.monotonic()
            logger.info('Рисование км-сетки — старт')
            self._draw_grid(ctx, result)
            grid_elapsed = time.monotonic() - grid_start_time
            logger.info(
                'Рисование км-сетки — завершено (%.2fs)', grid_elapsed,
            )
            _mark('grid_only')

        # For radar coverage: draw sector overlay after grid/legend
        if ctx.is_radar_coverage:
            self._draw_radar_sector_overlay(ctx, result)
            _mark('radar_sector_overlay')

        # Center cross
        self._draw_center_cross(ctx, result)

        # Control point (не рисуем для Link Profile и NSU Optimizer)
        if (
            ctx.settings.control_point_enabled
            and not ctx.is_link_profile
            and not ctx.is_nsu_optimizer
        ):
            self._draw_control_point(ctx, result)
        _mark('center_cross+cp')

        # Link profile: save clean base for interactive drag, then draw overlay
        if ctx.is_link_profile and ctx.link_profile_data is not None:
            ctx.link_profile_clean_base = result.copy()
            result = self._draw_link_profile_overlay(ctx, result)
            _mark('link_profile_overlay')

        # Clear raw DEM reference now that all processing is done
        ctx.raw_dem_for_cursor = None

        ctx.result = result

        # Emit per-stage breakdown so we can see what's actually expensive
        # in postprocess (this matters for the cycle after every big PIL/cv2
        # op we've already optimised — the slack remaining is in glue code).
        total = sum(t for _, t in _t_segments)
        breakdown = ' '.join(f'{name}={t * 1000:.0f}ms' for name, t in _t_segments)
        logger.info(
            'postprocess timings (total=%.0fms): %s',
            total * 1000, breakdown,
        )

    @staticmethod
    def _precrop_bbox_for_rotation(
        src_size: tuple[int, int],
        target_size: tuple[int, int],
        angle_deg: float,
    ) -> tuple[int, int, int, int] | None:
        """
        Compute the centred crop box in the source that contains the rotated
        final target rectangle. Output the bbox of a target_w × target_h
        rectangle rotated by angle_deg around the source centre — anything
        outside this bbox will be discarded by the subsequent center_crop
        after rotation anyway.

        Returns None if the bbox is not strictly smaller than the source (no
        gain) or if target exceeds source (don't optimise unsafely).
        """
        src_w, src_h = src_size
        tw, th = target_size
        if tw <= 0 or th <= 0 or tw > src_w or th > src_h:
            return None
        rad = math.radians(angle_deg)
        c = abs(math.cos(rad))
        s = abs(math.sin(rad))
        # Bounding box of the rotated target rect (in source pixels). Add 2px
        # safety margin to absorb rounding from getRotationMatrix2D + INTER_LINEAR.
        bw = int(math.ceil(tw * c + th * s)) + 2
        bh = int(math.ceil(tw * s + th * c)) + 2
        if bw >= src_w and bh >= src_h:
            return None  # No win — current src is already at the minimum.
        bw = min(bw, src_w)
        bh = min(bh, src_h)
        left = (src_w - bw) // 2
        top = (src_h - bh) // 2
        return (left, top, left + bw, top + bh)

    async def _ensure_raw_dem_loaded(self, ctx: MapDownloadContext) -> None:
        """
        Ensure ctx.raw_dem_for_cursor is populated at native tile resolution.

        Previously the DEM was upscaled to ctx.full_eff_tile_px so that pixel
        coordinates matched the main image. Then the contour overlay (and
        elev_crop_rect-aware code) immediately resized it back down. That
        round-trip cost ~150ms per build. Now we keep the native resolution
        (typically 4790² for a 512px retina tile world) and downstream code
        scales coordinates dynamically:
          - contour overlay: scale_factor becomes 1.0, no resize.
          - _load_dem_for_cursor: rotates+crops at native res; the cropped
            result is sent to GUI as-is (already small enough for IPC).
          - GUI: handles dem_grid lookups by ratio of dem.shape to image
            metadata (done in step L).

        Retina decision is per-map-type:
          - elev_color / RH / Radar / NSU never reach this loader — their
            processors load the DEM themselves with the right retina flag.
          - elev_contours uses ELEVATION_USE_RETINA=True.
          - link_profile gets a non-retina DEM here. With HYBRID base
            (eff_scale=4), a retina DEM at z=17 would assemble to 19160²
            float32 = 1.37 GB and trip RLIMIT_AS. The contour seed pass
            downscales to 799² anyway, so the visual difference at non-
            retina (9580² source, 367 MB) is negligible.
        """
        if ctx.raw_dem_for_cursor is not None:
            return

        # link_profile: skip retina for the contour DEM (see docstring).
        use_retina = ELEVATION_USE_RETINA and not ctx.is_link_profile

        # Two-layer cache (same machinery as _load_dem / _load_topo): try
        # in-memory LRU and then disk .npy memmap before paying for the
        # tile fetch + decode + assembly (~1.0-1.3 s cold for elev_contours
        # / hillshade / link_profile / HYBRID). Saves the load entirely
        # on warm rebuilds of the same area and turns cold restarts on a
        # familiar area into a fast disk read.
        cache_key = dem_topo_cache.make_area_key(
            kind_tag='cursor_dem_v1',
            zoom=ctx.zoom,
            eff_scale=ctx.eff_scale,
            use_retina=use_retina,
            center_lat_wgs=ctx.center_lat_wgs,
            center_lng_wgs=ctx.center_lng_wgs,
            width_m=ctx.width_m,
            height_m=ctx.height_m,
            tile_size=256,
        )
        cached = dem_topo_cache.get('cursor_dem', cache_key)
        if cached is not None:
            ctx.raw_dem_for_cursor = cached
            logger.info(
                'DEM (общая): cache HIT, skipped %d tile fetches + decodes',
                len(ctx.tiles),
            )
            return

        logger.info(
            'Загрузка DEM (общая) — старт (loading %d tiles)', len(ctx.tiles),
        )
        t_start = time.monotonic()

        provider = ElevationTileProvider(
            client=ctx.client,
            api_key=ctx.api_key,
            use_retina=use_retina,
            cache_root=_resolve_cache_dir(),
        )
        dem_tile_px = 256 * (2 if use_retina else 1)
        scale_factor = ctx.full_eff_tile_px / dem_tile_px
        cx, cy, cw, ch = ctx.crop_rect
        dem_crop_rect = (
            int(cx / scale_factor),
            int(cy / scale_factor),
            int(cw / scale_factor),
            int(ch / scale_factor),
        )

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
        dem_full = assemble_dem(
            tiles_data=dem_tiles_data,
            tiles_x=ctx.tiles_x,
            tiles_y=ctx.tiles_y,
            eff_tile_px=dem_tile_px,
            crop_rect=dem_crop_rect,
        )
        del dem_tiles_data

        # No upscale to full image resolution — downstream code adapts to
        # the native dem resolution dynamically.
        ctx.raw_dem_for_cursor = dem_full
        # Populate both cache layers so subsequent builds in this worker
        # (in-mem) and after worker restart (disk) can skip the fetch.
        dem_topo_cache.put('cursor_dem', cache_key, dem_full)
        elapsed = time.monotonic() - t_start
        logger.info(
            'Загрузка DEM (общая) — завершена (%.2fs, native %dx%d, '
            'scale_to_image=%.2f)',
            elapsed,
            dem_full.shape[1],
            dem_full.shape[0],
            scale_factor,
        )

    async def _load_dem_for_cursor(self, ctx: MapDownloadContext) -> None:
        """Load or transform DEM for cursor elevation display."""
        dem_start_time = time.monotonic()

        try:
            had_preloaded = ctx.raw_dem_for_cursor is not None
            if not had_preloaded:
                logger.info('Загрузка DEM для информера высоты — старт')

            await self._ensure_raw_dem_loaded(ctx)

            logger.info(
                'DEM для курсора: использую raw_dem (размер %dx%d, preloaded=%s)',
                ctx.raw_dem_for_cursor.shape[1],
                ctx.raw_dem_for_cursor.shape[0],
                had_preloaded,
            )
            dem_full = ctx.raw_dem_for_cursor

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

            # Center crop DEM. raw_dem is now at native tile resolution
            # (typically 4790² for 512px retina tiles) rather than full image
            # resolution. Scale the target crop dimensions accordingly so the
            # cropped region maps to the same geographic area as the main
            # image's center_crop(target_w_px, target_h_px).
            h, w = dem_full.shape
            scale = w / max(1, ctx.crop_rect[2])  # dem px per image px
            tw = max(1, int(round(ctx.target_w_px * scale)))
            th = max(1, int(round(ctx.target_h_px * scale)))
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
            elif ctx.is_nsu_optimizer:
                color_ramp = RADIO_HORIZON_COLOR_RAMP
                min_elev = 0.0
                max_elev = ctx.settings.nsu_max_flight_height_m
                title = 'Минимальная высота полета для всех точек, м'
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

    def _draw_link_profile_overlay(
        self,
        ctx: MapDownloadContext,
        result: Image.Image,
    ) -> Image.Image:
        """
        Draw link profile overlays and append profile diagram below the map.

        Returns:
            New image with map on top and profile inset appended below.

        """
        from services.processors.link_profile import render_profile_inset
        from shared.constants import (
            LINK_PROFILE_MAP_TICK_FACTOR,
            LINK_PROFILE_MIN_TICK_DISTANCE_M,
            LINK_PROFILE_POINT_A_COLOR,
            LINK_PROFILE_POINT_B_COLOR,
            LINK_PROFILE_SHORT_TICK_DISTANCE_M,
        )

        try:
            link_data = ctx.link_profile_data
            mpp = meters_per_pixel(ctx.center_lat_wgs, ctx.zoom, scale=ctx.eff_scale)
            ppm = 1.0 / mpp if mpp > 0 else 0.0
            font_size_px = max(12, round(ctx.settings.grid_font_size_m * ppm))
            label_font = load_grid_font(font_size_px)
            bg_padding_px = max(
                2,
                round(ctx.settings.grid_label_bg_padding_m * ppm),
            )
            tri_size_px = font_size_px
            label_gap_px = max(
                CONTROL_POINT_LABEL_GAP_MIN_PX,
                round(tri_size_px * CONTROL_POINT_LABEL_GAP_RATIO),
            )

            def _draw_marker(
                lat: float, lng: float, name: str, color: tuple[int, ...]
            ) -> tuple[float, float]:
                """Draw triangle marker + label at given WGS84 coords."""
                px, py = compute_control_point_image_coords(
                    cp_lat_wgs=lat,
                    cp_lng_wgs=lng,
                    center_lat_wgs=ctx.center_lat_wgs,
                    center_lng_wgs=ctx.center_lng_wgs,
                    zoom=ctx.zoom,
                    eff_scale=ctx.eff_scale,
                    img_width=result.width,
                    img_height=result.height,
                    rotation_deg=ctx.rotation_deg,
                    latlng_to_pixel_xy_func=latlng_to_pixel_xy,
                )
                if 0 <= px < result.width and 0 <= py < result.height:
                    draw_control_point_triangle(
                        result,
                        px,
                        py,
                        mpp,
                        ctx.rotation_deg,
                        size_m=ctx.settings.grid_font_size_m,
                        color=color,
                    )
                    label_y = int(py + tri_size_px / 2 + label_gap_px + bg_padding_px)
                    draw_label_with_bg(
                        ImageDraw.Draw(result),
                        (int(px), label_y),
                        name,
                        font=label_font,
                        anchor='mt',
                        img_size=result.size,
                        padding=bg_padding_px,
                    )
                return px, py

            # Draw markers A and B
            ax_img, ay_img = _draw_marker(
                link_data['point_a_lat_wgs'],
                link_data['point_a_lng_wgs'],
                ctx.settings.link_point_a_name or 'A',
                LINK_PROFILE_POINT_A_COLOR,
            )
            bx_img, by_img = _draw_marker(
                link_data['point_b_lat_wgs'],
                link_data['point_b_lng_wgs'],
                ctx.settings.link_point_b_name or 'B',
                LINK_PROFILE_POINT_B_COLOR,
            )

            # Line A→B
            line_draw = ImageDraw.Draw(result)
            line_w = max(2, round(3.0 / mpp)) if mpp > 0 else 2
            line_draw.line(
                [(int(ax_img), int(ay_img)), (int(bx_img), int(by_img))],
                fill=(255, 255, 0, 200),
                width=line_w,
            )

            # Distance tick marks + labels on A→B line (same step as profile X axis)
            total_d = link_data['total_distance_m']
            dist_km = total_d / 1000.0
            if total_d < LINK_PROFILE_MIN_TICK_DISTANCE_M:
                tick_step_m = 0  # no ticks
            elif total_d < LINK_PROFILE_SHORT_TICK_DISTANCE_M:
                tick_step_m = LINK_PROFILE_MIN_TICK_DISTANCE_M
            elif dist_km > 1:
                tick_step_m = max(1, int(dist_km / 6)) * 1000
            else:
                tick_step_m = max(100, int(total_d / 5 / 100) * 100)

            if tick_step_m > 0 and total_d > 0 and mpp > 0:
                dx_line = bx_img - ax_img
                dy_line = by_img - ay_img
                line_len = math.sqrt(dx_line**2 + dy_line**2)
                if line_len > 0:
                    ux, uy = dx_line / line_len, dy_line / line_len
                    px_perp, py_perp = -uy, ux
                    tick_half = line_w * LINK_PROFILE_MAP_TICK_FACTOR
                    label_offset = tick_half + font_size_px

                    d_val = tick_step_m
                    while d_val < total_d:
                        frac = d_val / total_d
                        cx = ax_img + dx_line * frac
                        cy = ay_img + dy_line * frac
                        line_draw.line(
                            [
                                (
                                    int(cx - px_perp * tick_half),
                                    int(cy - py_perp * tick_half),
                                ),
                                (
                                    int(cx + px_perp * tick_half),
                                    int(cy + py_perp * tick_half),
                                ),
                            ],
                            fill=(255, 255, 0, 200),
                            width=line_w,
                        )
                        # Label
                        lbl = f'{d_val / 1000:g}'
                        lx = int(cx + px_perp * label_offset)
                        ly = int(cy + py_perp * label_offset)
                        draw_text_with_outline(
                            line_draw,
                            (lx, ly),
                            lbl,
                            font=label_font,
                            fill=(255, 255, 0),
                            outline=(0, 0, 0),
                            outline_width=2,
                            anchor='mm',
                        )
                        d_val += tick_step_m

            # Render profile inset with full map width
            inset = render_profile_inset(
                link_data,
                map_size=(result.width, result.height),
            )

            # Append inset below the map
            combined = Image.new(
                result.mode,
                (result.width, result.height + inset.height),
                (255, 255, 255),
            )
            combined.paste(result, (0, 0))
            combined.paste(
                inset.convert(result.mode),
                (0, result.height),
            )
            del inset

            logger.info('Link profile overlay drawn successfully')

        except Exception as e:
            logger.warning('Не удалось нарисовать overlay профиля радиолинии: %s', e)
            return result
        else:
            return combined

    def _create_rh_overlay_layer(
        self,
        ctx: MapDownloadContext,
        result: Image.Image,
    ) -> Image.Image:
        """
        Build the overlay layer (grid + legend + contours) and bake it onto
        result via alpha-composite. Returns the (possibly new) result image.

        Previously this method ALSO drew grid/legend on the overlay AFTER
        they had already been baked on `result` — duplicate drawing that
        cost ~0.86s on elev_color (grid 0.17s + legend 0.69s twice). Now
        grid/legend are drawn ONCE here, on the overlay, then alpha-composed
        onto result. Saves ~0.5s net (composite overhead is ~0.3s).

        Two snapshots of the overlay are cached for GUI interactive use:
          - rh_cache_overlay_base: contours + grid, no legend
            (used when user changes elevation range → legend redrawn)
          - rh_cache_overlay: contours + grid + legend (used as-is)
        """
        t_subs: list[tuple[str, float]] = []
        t_prev = time.monotonic()

        def _sub(label: str) -> None:
            nonlocal t_prev
            now = time.monotonic()
            t_subs.append((label, now - t_prev))
            t_prev = now

        try:
            # Cache fast-path (ELEVATION_COLOR only). Same area + zoom +
            # settings → pixel-identical overlay; reuse cached overlay to
            # skip ~0.35s of grid+legend drawing + 2× overlay.copy().
            #
            # Applied to elev_color / RH / Radar / NSU — every needs_rh_overlay
            # path. Cache key folds in map_type and legend-bound settings, so
            # different types never collide. rh_cache_overlay_base normally
            # means "contour+grid, no legend" (GUI uses it to redraw legend
            # when elev range changes), but on a cache hit the underlying
            # bitmap is identical to the previous build's, so the cached
            # legend in `overlay` is already correct — falling overlay_base
            # back to `overlay` is safe.
            cached_overlay = _get_cached_overlay(ctx)
            _sub('cache_lookup')
            if cached_overlay is not None:
                if ctx.rh_contour_layer is not None:
                    with contextlib.suppress(Exception):
                        ctx.rh_contour_layer.close()
                    ctx.rh_contour_layer = None
                # On the cache-HIT path, `overlay_base` (contours + grid)
                # and `overlay` (contours + grid + legend) are normally
                # different — but here the legend is already in the cached
                # bitmap, so they coincide. Make one private copy and alias
                # it into both ctx fields (vs. two ~80 ms RGBA copies). GUI
                # uses overlay_base only as a basis for `.copy()` before
                # mutating (`_apply_interactive_alpha` path) and uses
                # overlay_layer only via `Image.alpha_composite` (returns a
                # new image, doesn't mutate inputs), so the alias is safe.
                # Pasting onto result also doesn't mutate the source — PIL
                # treats overlay.alpha as a mask and writes into result.rgb
                # in C — so we can paste `cached_overlay` directly without
                # an extra defensive copy.
                shared_overlay = cached_overlay.copy()
                _sub('hit_copy')
                ctx.rh_cache_overlay_base = shared_overlay
                ctx.rh_cache_overlay = shared_overlay
                result.paste(cached_overlay, (0, 0), cached_overlay)
                _sub('hit_paste')
                logger.info(
                    'rh_overlay sub-timings HIT: %s',
                    ' '.join(f'{name}={t * 1000:.0f}ms' for name, t in t_subs),
                )
                return result

            # Start overlay from contour_layer if available, else fully empty.
            # Image.alpha_composite(empty_rgba, contour) is mathematically
            # identical to contour itself (Porter-Duff "over" with alpha=0
            # destination), so the old composite call cost ~200ms of pure
            # waste on 8404² RGBA. .copy() is ~50ms — saves ~150ms.
            has_contours = ctx.rh_contour_layer is not None
            if has_contours:
                # Transfer ownership: ctx.rh_contour_layer уже independent
                # — caller получает либо `cached_contour.copy()` (cache HIT),
                # либо свежий `disk_cache.load()` / `Image.new + draw`
                # (cache MISS), а `_set_cached_contour_layer` внутри сам
                # делает .copy() для своей LRU-entry. Дополнительный
                # .copy() здесь = ~87 ms лишнего memcpy на 8404² RGBA.
                overlay = ctx.rh_contour_layer
                ctx.rh_contour_layer = None
            else:
                overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))
            _sub('init_overlay')

            # Draw grid on overlay (status: «Рисование км-сетки»)
            grid_start_time = time.monotonic()
            logger.info('Рисование км-сетки — старт')
            self._draw_grid(ctx, overlay)
            grid_elapsed = time.monotonic() - grid_start_time
            logger.info(
                'Рисование км-сетки — завершено (%.2fs)', grid_elapsed,
            )
            _sub('draw_grid')

            # Draw legend on overlay (not for hillshade — grayscale, no elevation scale)
            if (
                ctx.is_radio_horizon
                or ctx.is_radar_coverage
                or ctx.is_elev_color
                or ctx.is_nsu_optimizer
            ) and ctx.settings.map_type != MapType.ELEVATION_HILLSHADE:
                self._draw_legend(ctx, overlay)
            _sub('draw_legend')

            # Alias all three references to the same overlay object — как
            # делает HIT-path. GUI consumers (`_on_rh_recompute_finished`
            # / `_on_alpha_slider_released` / NSU) трактуют overlay_layer
            # read-only (paste source + cv2.resize), а `overlay_base`
            # реально читается только когда `overlay_layer is None` —
            # путь dead-code в текущем flow. Раньше тут было 2× .copy()
            # 8404² RGBA = ~110 ms ненужного memcpy.
            ctx.rh_cache_overlay_base = overlay
            ctx.rh_cache_overlay = overlay

            # Store the full overlay in the per-worker cache so subsequent
            # rebuilds with identical params can skip grid+legend drawing.
            # Applies to every needs_rh_overlay map type (elev_color / RH /
            # Radar / NSU); cache key folds in map_type + legend-bound
            # settings so cross-type collisions can't happen.
            _set_cached_overlay(ctx, overlay)
            logger.info(
                'rh overlay layer: cache MISS — stored '
                '(LRU size=%d/%d)',
                len(_overlay_cache),
                _OVERLAY_CACHE_MAX_ENTRIES,
            )
            _sub('cache_store')

            # Bake overlay onto result via PIL.paste with RGBA mask. Math is
            # identical to alpha_composite(result.convert('RGBA'), overlay).
            # convert('RGB') (Porter-Duff "over" with alpha extracted from
            # the mask), but done in ONE C-loop — no RGB↔RGBA round-trip
            # repacks (~150ms each at 8404²) and no transient ~370 MB
            # baked-RGBA buffer. Same trick as _apply_overlay_contours.
            result.paste(overlay, (0, 0), overlay)
            _sub('paste_to_result')

            logger.info(
                'rh overlay layer built and baked (contours=%s)',
                has_contours,
            )
            logger.info(
                'rh_overlay sub-timings MISS: %s',
                ' '.join(f'{name}={t * 1000:.0f}ms' for name, t in t_subs),
            )
            return result

        except Exception as e:
            logger.exception(
                'Failed to create radio horizon overlay layer: %s', e
            )
            return result

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

            # Copy back to result if it was converted — free intermediate
            if result_rgba is not result:
                rgb_copy = result_rgba.convert('RGB')
                result_rgba.close()
                del result_rgba
                result.paste(rgb_copy)
                rgb_copy.close()
                del rgb_copy

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
            # No defensive copy: publish_preview_image only reads pixels
            # (PIL.Image.save/tobytes), and ctx.result is closed below after
            # publication completes. The copy was a 200+MB memcpy for nothing.
            metadata = ctx.to_metadata()
            # Log center resolution for diagnostics
            logger.info(
                'Map resolution at center: %.4f m/px (zoom %d, scale %d)',
                metadata.meters_per_pixel,
                metadata.zoom,
                metadata.scale,
            )

            # Heavy 8404² RGBA overlay layers — ужимаем к PUBLISH_OVERLAY
            # _MAX_DIM перед сериализацией в shm. GUI всё равно
            # cv2.resize-ит обратно к full image size перед blend, так
            # что net-визуал не меняется, но publish экономит ~0.5s
            # wallclock + ~210 MB shm payload на каждый RH/Radar/Elev
            # /NSU build. Identity-aliased dedupe сохраняется через
            # явный `is`-чек.
            _publish_overlay = _downsample_overlay_for_publish(ctx.rh_cache_overlay)
            if ctx.rh_cache_overlay_base is ctx.rh_cache_overlay:
                _publish_overlay_base = _publish_overlay
            else:
                _publish_overlay_base = _downsample_overlay_for_publish(
                    ctx.rh_cache_overlay_base,
                )

            # Collect radio horizon cache if available
            rh_cache = None
            if (
                ctx.is_radio_horizon or ctx.is_radar_coverage
            ) and ctx.rh_cache_dem is not None:
                rh_cache = {
                    'dem': ctx.rh_cache_dem,
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
                    'overlay_layer': _publish_overlay,
                    'overlay_base': _publish_overlay_base,
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
                    'overlay_layer': _publish_overlay,
                    'overlay_base': _publish_overlay_base,
                    'settings': ctx.settings,
                    'is_elev_color': True,
                    'rotation_deg': ctx.rotation_deg,
                    'final_size': (ctx.target_w_px, ctx.target_h_px),
                    'crop_size': ctx.rh_cache_crop_size,
                }
            elif ctx.is_nsu_optimizer and ctx.nsu_cache_dem is not None:
                rh_cache = {
                    'is_nsu_optimizer': True,
                    'dem': ctx.nsu_cache_dem,
                    'topo_base': ctx.nsu_cache_topo_base,
                    'pixel_size_m': ctx.nsu_cache_pixel_size_m,
                    'antenna_height_m': ctx.settings.nsu_antenna_height_m,
                    'overlay_alpha': ctx.settings.nsu_overlay_alpha,
                    'max_height_m': ctx.settings.nsu_max_flight_height_m,
                    'coverage_layer': ctx.nsu_cache_coverage,
                    'overlay_layer': _publish_overlay,
                    'overlay_base': _publish_overlay_base,
                    'settings': ctx.settings,
                    'rotation_deg': ctx.rotation_deg,
                    'final_size': (ctx.target_w_px, ctx.target_h_px),
                    'crop_size': ctx.nsu_cache_crop_size,
                }
            elif ctx.is_link_profile and ctx.link_profile_clean_base is not None:
                rh_cache = {
                    'is_link_profile': True,
                    'clean_base': ctx.link_profile_clean_base,
                    'link_profile_data': ctx.link_profile_data,
                    'settings': ctx.settings,
                    'rotation_deg': ctx.rotation_deg,
                    'final_size': (ctx.target_w_px, ctx.target_h_px),
                }

            did_publish = publish_preview_image(
                result, metadata, ctx.dem_grid, rh_cache
            )
        except Exception:
            logger.exception('publish_preview_image wrapper failed')
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
            with contextlib.suppress(Exception):
                result.close()
        else:
            with contextlib.suppress(Exception):
                result.close()

        # Финальный gc.collect() из этого блока убран — дублирует
        # gc.collect() в конце download() сразу после _save returns.

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
