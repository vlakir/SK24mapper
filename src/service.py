import asyncio
import contextlib
import gc
import logging
import os
import sqlite3
from datetime import timedelta
from pathlib import Path

import aiohttp
from aiohttp_client_cache import CachedSession, SQLiteBackend
from PIL import Image
from pyproj import CRS, Transformer

from constants import (
    ASYNC_MAX_CONCURRENCY,
    DOWNLOAD_CONCURRENCY,
    ENABLE_WHITE_MASK,
    EPSG_SK42_GK_BASE,
    GK_FALSE_EASTING,
    GK_ZONE_CM_OFFSET_DEG,
    GK_ZONE_WIDTH_DEG,
    GK_ZONE_X_PREFIX_DIV,
    GRID_COLOR,
    GRID_STEP_M,
    HTTP_CACHE_DIR,
    HTTP_CACHE_ENABLED,
    HTTP_CACHE_EXPIRE_HOURS,
    HTTP_CACHE_RESPECT_HEADERS,
    HTTP_CACHE_STALE_IF_ERROR_HOURS,
    MAPBOX_STYLE_ID,
    MAX_GK_ZONE,
    MAX_OUTPUT_PIXELS,
    MAX_ZOOM,
    PIL_DISABLE_LIMIT,
    ROTATION_PAD_MIN_PX,
    ROTATION_PAD_RATIO,
    SK42_VALID_LAT_MAX,
    SK42_VALID_LAT_MIN,
    SK42_VALID_LON_MAX,
    SK42_VALID_LON_MIN,
    XYZ_TILE_SIZE,
    XYZ_USE_RETINA,
)
from diagnostics import log_memory_usage, log_thread_status
from domen import MapSettings
from image import (
    apply_white_mask,
    assemble_and_crop,
    center_crop,
    draw_axis_aligned_km_grid,
    rotate_keep_size,
)
from progress import ConsoleProgress, LiveSpinner, publish_preview_image
from topography import (
    async_fetch_xyz_tile,
    build_transformers_sk42,
    choose_zoom_with_limit,
    compute_rotation_deg_for_east_axis,
    compute_xyz_coverage,
    crs_sk42_geog,
    effective_scale_for_xyz,
    estimate_crop_size_px,
)

logger = logging.getLogger(__name__)


def _determine_zone(center_x_sk42_gk: float) -> int:
    zone = int(center_x_sk42_gk // GK_ZONE_X_PREFIX_DIV)
    if zone < 1 or zone > MAX_GK_ZONE:
        zone = max(
            1,
            min(
                MAX_GK_ZONE,
                int((center_x_sk42_gk - GK_FALSE_EASTING) // GK_ZONE_X_PREFIX_DIV) + 1,
            ),
        )
    return zone


def _build_sk42_gk_crs(zone: int) -> CRS:
    try:
        return CRS.from_epsg(EPSG_SK42_GK_BASE + zone)
    except Exception:
        lon0 = zone * GK_ZONE_WIDTH_DEG - GK_ZONE_CM_OFFSET_DEG
        proj4 = (
            f'+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 '
            f'+x_0={GK_FALSE_EASTING} +y_0=0 +ellps=krass +units=m +no_defs +type=crs'
        )
        return CRS.from_proj4(proj4)


def _validate_sk42_bounds(lng: float, lat: float) -> None:
    if not (
        SK42_VALID_LON_MIN <= lng <= SK42_VALID_LON_MAX
        and SK42_VALID_LAT_MIN <= lat <= SK42_VALID_LAT_MAX
    ):
        msg = (
            'Выбранная область вне зоны применимости СК-42. '
            'Карта не будет сформирована.'
        )
        raise SystemExit(msg)


def _resolve_cache_dir() -> Path | None:
    raw_dir = Path(HTTP_CACHE_DIR)
    if raw_dir.is_absolute():
        return raw_dir
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / raw_dir).resolve()


def _cleanup_sqlite_cache(cache_dir: Path) -> None:
    """Force cleanup of SQLite cache connections."""
    try:
        cache_file = cache_dir / 'http_cache.sqlite'
        if cache_file.exists():
            # Close any remaining SQLite connections
            conn = sqlite3.connect(cache_file)
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE);')
            conn.close()
            # Give some time for file system operations
            import time

            time.sleep(0.1)
    except Exception as e:
        # Log cleanup errors but don't raise
        logger.debug(f'SQLite cache cleanup failed: {e}')


def _make_http_session(cache_dir: Path | None) -> aiohttp.ClientSession:
    use_cache = HTTP_CACHE_ENABLED
    if use_cache and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if use_cache and cache_dir is not None:
        cache_path = cache_dir / 'http_cache.sqlite'
        with contextlib.suppress(Exception):
            if not cache_path.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(cache_path) as _conn:
                    _conn.execute('PRAGMA journal_mode=WAL;')
        expire_td = timedelta(hours=max(0, int(HTTP_CACHE_EXPIRE_HOURS)))
        stale_hours = int(HTTP_CACHE_STALE_IF_ERROR_HOURS)
        stale_param: bool | timedelta
        stale_param = timedelta(hours=stale_hours) if stale_hours > 0 else False
        backend = SQLiteBackend(
            str(cache_path),
            expire_after=expire_td,
        )
        return CachedSession(
            cache=backend,
            expire_after=expire_td,
            cache_control=bool(HTTP_CACHE_RESPECT_HEADERS),
            stale_if_error=stale_param,
        )
    return aiohttp.ClientSession()


def _build_save_kwargs(out_path: Path, settings_obj: object) -> dict[str, object]:
    # Всегда сохраняем в JPEG; читаем качество из настроек
    try:
        q = int(getattr(settings_obj, 'jpeg_quality', 95))
    except Exception:
        q = 95
    q = max(10, min(100, q))
    return {
        'format': 'JPEG',
        'quality': q,
        'subsampling': 0,
        'optimize': True,
        'progressive': True,
        'exif': b'',
    }


async def download_satellite_rectangle(  # noqa: PLR0913
    center_x_sk42_gk: float,
    center_y_sk42_gk: float,
    width_m: float,
    height_m: float,
    api_key: str,
    output_path: str,
    max_zoom: int = MAX_ZOOM,
    settings: MapSettings | None = None,
) -> str:
    """Полный конвейер."""
    # Handle default settings if not provided
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
            grid_width_px=4,
            grid_font_size=86,
            grid_text_margin=43,
            grid_label_bg_padding=6,
            mask_opacity=0.35,
        )

    # Переопределяем параметры из профиля
    eff_scale = effective_scale_for_xyz(XYZ_TILE_SIZE, use_retina=XYZ_USE_RETINA)
    # Подготовка — конвертация из Гаусса-Крюгера в географические координаты СК-42
    sp = LiveSpinner('Подготовка: определение зоны')
    sp.start()

    # Зона и система координат
    zone = _determine_zone(center_x_sk42_gk)
    crs_sk42_gk = _build_sk42_gk_crs(zone)
    sp.stop('Подготовка: зона определена')

    sp = LiveSpinner('Подготовка: конвертация из ГК в СК-42')
    sp.start()
    # Конвертируем из Гаусса-Крюгера в географические СК-42
    t_sk42_from_gk = Transformer.from_crs(crs_sk42_gk, crs_sk42_geog, always_xy=True)
    center_lng_sk42, center_lat_sk42 = t_sk42_from_gk.transform(
        center_x_sk42_gk,
        center_y_sk42_gk,
    )
    sp.stop('Подготовка: координаты СК-42 готовы')

    # Проверка зоны применимости СК-42 (без формирования карты вне зоны)
    _validate_sk42_bounds(center_lng_sk42, center_lat_sk42)

    sp = LiveSpinner('Подготовка: создание трансформеров')
    sp.start()
    # Создаем трансформеры для работы с полученными координатами
    t_sk42_to_wgs, t_wgs_to_sk42, _ = build_transformers_sk42(center_lng_sk42)
    sp.stop('Подготовка: трансформеры готовы')

    sp = LiveSpinner('Подготовка: конвертация центра в WGS84')
    sp.start()
    center_lng_wgs, center_lat_wgs = t_sk42_to_wgs.transform(
        center_lng_sk42,
        center_lat_sk42,
    )
    sp.stop('Подготовка: центр WGS84 готов')

    sp = LiveSpinner('Подготовка: подбор zoom')
    sp.start()
    zoom = choose_zoom_with_limit(
        center_lat=center_lat_wgs,
        width_m=width_m,
        height_m=height_m,
        desired_zoom=max_zoom,
        scale=eff_scale,
        max_pixels=MAX_OUTPUT_PIXELS,
    )
    sp.stop('Подготовка: zoom выбран')

    if PIL_DISABLE_LIMIT:
        Image.MAX_IMAGE_PIXELS = None

    sp = LiveSpinner('Подготовка: оценка размера')
    sp.start()
    target_w_px, target_h_px, _ = estimate_crop_size_px(
        center_lat_wgs,
        width_m,
        height_m,
        zoom,
        eff_scale,
    )
    sp.stop('Подготовка: размер оценён')

    sp = LiveSpinner('Подготовка: расчёт покрытия XYZ')
    sp.start()
    base_pad = round(min(target_w_px, target_h_px) * ROTATION_PAD_RATIO)
    pad_px = max(base_pad, ROTATION_PAD_MIN_PX)
    tiles, (tiles_x, tiles_y), crop_rect, map_params = compute_xyz_coverage(
        center_lat=center_lat_wgs,
        center_lng=center_lng_wgs,
        width_m=width_m,
        height_m=height_m,
        zoom=zoom,
        eff_scale=eff_scale,
        pad_px=pad_px,
    )
    sp.stop('Подготовка: покрытие рассчитано')

    tile_progress = ConsoleProgress(total=len(tiles), label='Загрузка XYZ-тайлов')
    semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY or ASYNC_MAX_CONCURRENCY)

    cache_dir_resolved = _resolve_cache_dir()
    session_ctx = _make_http_session(cache_dir_resolved)

    log_memory_usage('before tile download')
    log_thread_status('before tile download')

    try:
        async with session_ctx as client:
            tile_count = 0

            async def bound_fetch(
                idx_xy: tuple[int, tuple[int, int]],
            ) -> tuple[int, Image.Image]:
                nonlocal tile_count
                idx, (tx, ty) = idx_xy
                async with semaphore:
                    img = await async_fetch_xyz_tile(
                        client=client,
                        api_key=api_key,
                        style_id=MAPBOX_STYLE_ID,
                        tile_size=XYZ_TILE_SIZE,
                        z=zoom,
                        x=tx,
                        y=ty,
                        use_retina=XYZ_USE_RETINA,
                    )
                    await tile_progress.step(1)

                    # Log memory usage every 50 tiles
                    tile_count += 1
                    if tile_count % 50 == 0:
                        log_memory_usage(f'after {tile_count} tiles')

                    return idx, img

            tasks = [bound_fetch(pair) for pair in enumerate(tiles)]
            results = await asyncio.gather(*tasks)
            tile_progress.close()

            log_memory_usage('after all tiles downloaded')
            log_thread_status('after all tiles downloaded')

            results.sort(key=lambda t: t[0])
            images: list[Image.Image] = [img for _, img in results]
    finally:
        # Explicit cleanup of HTTP session resources
        try:
            if hasattr(session_ctx, '_cache') and session_ctx._cache:
                # Close SQLite cache backend if it exists
                if hasattr(session_ctx._cache, '_cache') and hasattr(
                    session_ctx._cache._cache, 'close'
                ):
                    await session_ctx._cache._cache.close()
        except Exception:
            # Ignore cleanup errors but log them
            logging.getLogger(__name__).debug(
                'Error during HTTP session cleanup', exc_info=True
            )

        # Force cleanup of SQLite connections in cache directory
        if cache_dir_resolved:
            _cleanup_sqlite_cache(cache_dir_resolved)

    eff_tile_px = XYZ_TILE_SIZE * (2 if XYZ_USE_RETINA else 1)
    result = assemble_and_crop(
        images=images,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        eff_tile_px=eff_tile_px,
        crop_rect=crop_rect,
    )
    # The assemble function already clears tile images; ensure local refs are dropped
    with contextlib.suppress(Exception):
        images.clear()

    angle_deg = compute_rotation_deg_for_east_axis(
        center_lat_sk42=center_lat_sk42,
        center_lng_sk42=center_lng_sk42,
        map_params=map_params,
        crs_sk42_gk=crs_sk42_gk,
        t_sk42_to_wgs=t_sk42_to_wgs,
    )
    result = rotate_keep_size(result, angle_deg, fill=(255, 255, 255))

    result = center_crop(result, target_w_px, target_h_px)

    if ENABLE_WHITE_MASK and settings.mask_opacity > 0:
        result = apply_white_mask(result, settings.mask_opacity)

    draw_axis_aligned_km_grid(
        img=result,
        center_lat_sk42=center_lat_sk42,
        center_lng_sk42=center_lng_sk42,
        center_lat_wgs=center_lat_wgs,
        zoom=zoom,
        crs_sk42_gk=crs_sk42_gk,
        t_sk42_to_wgs=t_sk42_to_wgs,
        step_m=GRID_STEP_M,
        color=GRID_COLOR,
        width_px=settings.grid_width_px,
        scale=eff_scale,
        grid_font_size=settings.grid_font_size,
        grid_text_margin=settings.grid_text_margin,
        grid_label_bg_padding=settings.grid_label_bg_padding,
    )

    # Показать предпросмотр до сохранения файла
    did_publish = False
    with contextlib.suppress(Exception):
        did_publish = publish_preview_image(result)

    # Если GUI подписан на предпросмотр (did_publish=True), не сохраняем автоматически.
    if not did_publish:
        sp = LiveSpinner('Сохранение файла')
        sp.start()

        out_path = Path(output_path)
        # Принудительно используем расширение .jpg
        if out_path.suffix.lower() not in ('.jpg', '.jpeg'):
            out_path = out_path.with_suffix('.jpg')
        out_path.resolve().parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = _build_save_kwargs(out_path, settings)
        result.convert('RGB').save(out_path, **save_kwargs)

        fd = os.open(out_path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)

        sp.stop('Сохранение файла: готово')

    # Encourage Python to reclaim memory of large temporaries
    try:
        gc.collect()
    except Exception as e:
        logger.debug(f'Garbage collection failed: {e}')

    return output_path
