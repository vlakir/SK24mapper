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
    HTTP_FORBIDDEN,
    HTTP_OK,
    HTTP_UNAUTHORIZED,
    MAPBOX_STATIC_BASE,
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
    MapType,
    default_map_type,
    map_type_to_style_id,
)
from diagnostics import log_memory_usage, log_thread_status
from domen import MapSettings
from image import (
    assemble_and_crop,
    center_crop,
    draw_axis_aligned_km_grid,
    rotate_keep_size,
)
from progress import ConsoleProgress, LiveSpinner, publish_preview_image
from topography import (
    async_fetch_terrain_rgb_tile,
    async_fetch_xyz_tile,
    build_transformers_sk42,
    choose_zoom_with_limit,
    compute_rotation_deg_for_east_axis,
    compute_xyz_coverage,
    crs_sk42_geog,
    decode_terrain_rgb_to_elevation_m,
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
    # Prefer user LOCALAPPDATA for writable cache dir
    local = os.getenv('LOCALAPPDATA')
    if local:
        return (Path(local) / 'SK42mapper' / '.cache' / 'tiles').resolve()
    # Fallback: user's home directory
    return (Path.home() / '.sk42mapper_cache' / 'tiles').resolve()


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


async def _validate_api_and_connectivity(api_key: str, style_id: str) -> None:
    """Проверяет доступность стилей Mapbox (Styles API tiles endpoint)."""
    """
    Проверяет доступность интернета и валидность API-ключа перед началом тяжёлой обработки.

    Делает быстрый запрос к одному тайлу (z=0/x=0/y=0). В случае проблем бросает RuntimeError
    с понятным для пользователя сообщением.
    """
    test_path = f'{MAPBOX_STATIC_BASE}/{style_id}/tiles/256/0/0/0'
    test_url = f'{test_path}?access_token={api_key}'
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with (
            aiohttp.ClientSession() as client,
            client.get(test_url, timeout=timeout) as resp,
        ):
            sc = resp.status
            if sc == HTTP_OK:
                # прочитать тело, чтобы гарантированно освободить соединение в пул
                with contextlib.suppress(Exception):
                    await resp.read()
                return
            if sc in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                msg = 'Неверный или недействительный API-ключ. Проверьте ключ и попробуйте снова.'
                raise RuntimeError(
                    msg,
                )
            msg = f'Ошибка доступа к серверу карт (HTTP {sc}). Повторите попытку позже.'
            raise RuntimeError(
                msg,
            )
    except (TimeoutError, aiohttp.ClientConnectorError, aiohttp.ClientOSError):
        msg = 'Нет соединения с интернетом или сервер недоступен. Проверьте подключение к сети.'
        raise RuntimeError(msg) from None


async def _validate_terrain_api(api_key: str) -> None:
    """Быстрая проверка доступности Terrain-RGB источника."""
    from constants import MAPBOX_TERRAIN_RGB_PATH

    test_path = f'{MAPBOX_TERRAIN_RGB_PATH}/0/0/0.pngraw'
    test_url = f'{test_path}?access_token={api_key}'
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with (
            aiohttp.ClientSession() as client,
            client.get(test_url, timeout=timeout) as resp,
        ):
            sc = resp.status
            if sc == HTTP_OK:
                with contextlib.suppress(Exception):
                    await resp.read()
                return
            if sc in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                msg = 'Неверный или недействительный API-ключ. Проверьте ключ и попробуйте снова.'
                raise RuntimeError(msg)
            msg = f'Ошибка доступа к серверу карт (HTTP {sc}). Повторите попытку позже.'
            raise RuntimeError(msg)
    except (TimeoutError, aiohttp.ClientConnectorError, aiohttp.ClientOSError):
        msg = 'Нет соединения с интернетом или сервер недоступен. Проверьте подключение к сети.'
        raise RuntimeError(msg) from None


async def download_satellite_rectangle(  # noqa: PLR0913, PLR0912
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
    import time
    overall_start_time = time.monotonic()
    logger.info('=== ОБЩИЙ ТАЙМЕР: старт download_satellite_rectangle ===')
    
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

    # Определяем тип карты и стратегию
    mt = getattr(settings, 'map_type', default_map_type())
    try:
        mt_enum = MapType(mt) if not isinstance(mt, MapType) else mt
    except Exception:
        mt_enum = default_map_type()

    style_id: str | None = None
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
        await _validate_api_and_connectivity(api_key, style_id)
    elif mt_enum == MapType.ELEVATION_COLOR:
        from constants import ELEVATION_USE_RETINA

        logger.info(
            'Тип карты: %s (Terrain-RGB, цветовая шкала); retina=%s',
            mt_enum,
            ELEVATION_USE_RETINA,
        )
    elif mt_enum == MapType.ELEVATION_CONTOURS:
        from constants import ELEVATION_USE_RETINA

        logger.info(
            'Тип карты: %s (Terrain-RGB, контуры); retina=%s',
            mt_enum,
            ELEVATION_USE_RETINA,
        )
    else:
        # Нереализованные режимы пока откатываются к Спутнику
        logger.warning(
            'Выбран режим высот (%s), пока не реализовано. Используется Спутник.',
            mt_enum,
        )
        style_id = map_type_to_style_id(default_map_type())
        await _validate_api_and_connectivity(api_key, style_id)

    # Выбор масштаба под тип карты
    is_elev_color = False
    is_elev_contours = False
    try:
        mt_val = MapType(mt) if not isinstance(mt, MapType) else mt
        is_elev_color = mt_val == MapType.ELEVATION_COLOR
        is_elev_contours = mt_val == MapType.ELEVATION_CONTOURS
    except Exception:
        is_elev_color = False
        is_elev_contours = False

    # Флаг оверлея изолиний поверх выбранного типа карты
    overlay_contours = bool(getattr(settings, 'overlay_contours', False))

    if is_elev_color or is_elev_contours:
        # Для Terrain-RGB базовый тайл 256px; @2x даёт 512
        from constants import ELEVATION_USE_RETINA

        eff_scale = effective_scale_for_xyz(256, use_retina=ELEVATION_USE_RETINA)
        # Ранняя проверка доступности Terrain-RGB
        await _validate_terrain_api(api_key)
    else:
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

    tile_label = (
        'Загрузка Terrain-RGB тайлов' if (is_elev_color or is_elev_contours) else 'Загрузка XYZ-тайлов'
    )
    tile_progress = ConsoleProgress(total=len(tiles), label=tile_label)
    semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY or ASYNC_MAX_CONCURRENCY)

    cache_dir_resolved = _resolve_cache_dir()
    session_ctx = _make_http_session(cache_dir_resolved)

    log_memory_usage('before tile download')
    log_thread_status('before tile download')

    try:
        async with session_ctx as client:
            tile_count = 0

            if is_elev_color:
                # Two-pass streaming without storing full DEM (color ramp)
                from topography import (
                    ELEV_MIN_RANGE_M,
                    ELEV_PCTL_HI,
                    ELEV_PCTL_LO,
                    ELEVATION_COLOR_RAMP,
                    compute_percentiles,
                )

                # Helper: build LUT from color ramp for fast palette lookup
                def _lerp(a: float, b: float, t: float) -> float:
                    return a + (b - a) * t

                lut_size = 2048
                _lut: list[tuple[int, int, int]] = []
                ramp = ELEVATION_COLOR_RAMP
                # Precompute cumulative ramp into fixed-size LUT
                for i in range(lut_size):
                    t = i / (lut_size - 1)
                    # find segment
                    for j in range(1, len(ramp)):
                        t0, c0 = ramp[j - 1]
                        t1, c1 = ramp[j]
                        if t <= t1 or j == len(ramp) - 1:
                            local = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                            r = round(_lerp(c0[0], c1[0], local))
                            g = round(_lerp(c0[1], c1[1], local))
                            b = round(_lerp(c0[2], c1[2], local))
                            _lut.append((r, g, b))
                            break

                def _color_at(t: float) -> tuple[int, int, int]:
                    # Clamp and map to LUT index
                    if t <= 0.0:
                        return _lut[0]
                    if t >= 1.0:
                        return _lut[-1]
                    idx = int(t * (lut_size - 1))
                    return _lut[idx]

                from constants import ELEVATION_USE_RETINA

                full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)

                # Fast overlap check to skip tiles outside crop
                def _tile_overlap_rect(
                    tx: int, ty: int
                ) -> tuple[int, int, int, int] | None:
                    base_x = tx * full_eff_tile_px
                    base_y = ty * full_eff_tile_px
                    cx, cy, cw, ch = crop_rect
                    x0 = max(base_x, cx)
                    y0 = max(base_y, cy)
                    x1 = min(base_x + full_eff_tile_px, cx + cw)
                    y1 = min(base_y + full_eff_tile_px, cy + ch)
                    if x1 <= x0 or y1 <= y0:
                        return None
                    return x0, y0, x1, y1

                # Map tiles list index -> (tile grid coords)
                # tiles list is ordered row-major corresponding to tx in [0..tiles_x), ty in [0..tiles_y)
                # We reconstruct tx,ty from enumerate index
                def _tx_ty_from_index(idx: int) -> tuple[int, int]:
                    ty = idx // tiles_x
                    tx = idx % tiles_x
                    return tx, ty

                # Pass A: sample elevations for percentiles (reservoir sampling)
                max_samples = 50000
                samples: list[float] = []  # reservoir
                seen_count = 0
                import random

                rng = random.Random(42)  # noqa: S311
                tile_progress.label = 'Проверка диапазона высот (проход 1/2)'

                async def fetch_and_sample(idx_xy: tuple[int, tuple[int, int]]) -> None:
                    nonlocal tile_count, samples, seen_count
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = _tx_ty_from_index(idx)
                    if _tile_overlap_rect(tx, ty) is None:
                        await tile_progress.step(1)
                        return
                    async with semaphore:
                        from constants import ELEVATION_USE_RETINA

                        img = await async_fetch_terrain_rgb_tile(
                            client=client,
                            api_key=api_key,
                            z=zoom,
                            x=tile_x_world,
                            y=tile_y_world,
                            use_retina=ELEVATION_USE_RETINA,
                        )
                        dem_tile = decode_terrain_rgb_to_elevation_m(img)
                        # Iterate coarse grid within tile to limit CPU, but feed values into reservoir
                        h = len(dem_tile)
                        w = len(dem_tile[0]) if h else 0
                        if h and w:
                            step_y = max(1, h // 32)
                            step_x = max(1, w // 32)
                            # jitter to avoid regular sampling artifacts
                            off_y = (
                                rng.randrange(0, min(step_y, h)) if step_y > 1 else 0
                            )
                            off_x = (
                                rng.randrange(0, min(step_x, w)) if step_x > 1 else 0
                            )
                            for ry in range(off_y, h, step_y):
                                row = dem_tile[ry]
                                for rx in range(off_x, w, step_x):
                                    v = row[rx]
                                    seen_count += 1
                                    if len(samples) < max_samples:
                                        samples.append(v)
                                    else:
                                        j = rng.randrange(0, seen_count)
                                        if j < max_samples:
                                            samples[j] = v
                        await tile_progress.step(1)
                        tile_count += 1
                        if tile_count % 50 == 0:
                            log_memory_usage(f'pass1 after {tile_count} tiles')

                # launch tasks
                await asyncio.gather(
                    *[fetch_and_sample(pair) for pair in enumerate(tiles)]
                )
                tile_progress.close()
                # Compute percentiles from reservoir
                logger.info(
                    'DEM sampling reservoir: kept=%s seen~=%s', len(samples), seen_count
                )
                lo, hi = compute_percentiles(samples, ELEV_PCTL_LO, ELEV_PCTL_HI)
                if hi - lo < ELEV_MIN_RANGE_M:
                    mid = (lo + hi) / 2.0
                    lo = mid - ELEV_MIN_RANGE_M / 2.0
                    hi = mid + ELEV_MIN_RANGE_M / 2.0
                inv = 1.0 / (hi - lo) if hi > lo else 0.0

                # Pass B: render directly to output image using producer–consumer (I/O vs CPU)
                result = Image.new('RGB', (crop_rect[2], crop_rect[3]))
                tile_progress = ConsoleProgress(
                    total=len(tiles), label='Окрашивание DEM (проход 2/2)'
                )
                tile_count = 0

                queue: asyncio.Queue[
                    tuple[int, int, int, int, int, int, Image.Image]
                ] = asyncio.Queue(maxsize=4)
                paste_lock = asyncio.Lock()

                async def producer(idx_xy: tuple[int, tuple[int, int]]) -> None:
                    nonlocal tile_count
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = _tx_ty_from_index(idx)
                    ov = _tile_overlap_rect(tx, ty)
                    if ov is None:
                        await tile_progress.step(1)
                        return
                    async with semaphore:
                        from constants import ELEVATION_USE_RETINA

                        img = await async_fetch_terrain_rgb_tile(
                            client=client,
                            api_key=api_key,
                            z=zoom,
                            x=tile_x_world,
                            y=tile_y_world,
                            use_retina=ELEVATION_USE_RETINA,
                        )
                    # Enqueue with metadata for consumers
                    x0, y0, x1, y1 = ov
                    await queue.put((tx, ty, x0, y0, x1, y1, img))
                    # Progress is stepped when consumer finishes to reflect actual painting

                # Precompute normalization coeffs for consumers (used with DEM values)
                async def consumer() -> None:
                    nonlocal tile_count
                    # Precompute linear coefficients for t = ar*R + ag*G + ab*B + a0
                    ar = 0.1 * 65536.0 * inv
                    ag = 0.1 * 256.0 * inv
                    ab = 0.1 * 1.0 * inv
                    a0 = (-10000.0 - lo) * inv
                    from constants import USE_NUMPY_FASTPATH as _NP_FLAG

                    try:
                        import numpy as np  # optional fast-path

                        _numpy_ok = bool(_NP_FLAG)
                    except Exception:
                        _numpy_ok = False
                    while True:
                        item = await queue.get()
                        if item is None:  # type: ignore[comparison-overlap]
                            queue.task_done()
                            break
                        tx, ty, x0, y0, x1, y1, img = item
                        try:
                            cx, cy, _, _ = crop_rect
                            dx0 = x0 - cx
                            dy0 = y0 - cy
                            base_x = tx * full_eff_tile_px
                            base_y = ty * full_eff_tile_px
                            # Build raw RGB buffer for the overlap block (scanline writing)
                            block_w = x1 - x0
                            block_h = y1 - y0
                            if _numpy_ok:
                                # NumPy fast-path: vectorize RGB -> t -> LUT index
                                import numpy as np

                                arr = np.asarray(img, dtype=np.uint8)  # HxWx3
                                # Slice overlap
                                sub = arr[
                                    y0 - base_y : y1 - base_y,
                                    x0 - base_x : x1 - base_x,
                                    :3,
                                ]
                                # Compute t linear combination
                                t = (
                                    ar * sub[..., 0].astype(np.float32)
                                    + ag * sub[..., 1].astype(np.float32)
                                    + ab * sub[..., 2].astype(np.float32)
                                    + a0
                                )
                                # Clamp and map to LUT
                                _l = np.clip(
                                    (t * (lut_size - 1)).astype(np.int32),
                                    0,
                                    lut_size - 1,
                                )
                                lut = np.asarray(_lut, dtype=np.uint8)
                                rgb = lut[_l]
                                block_img = Image.fromarray(rgb, mode='RGB')
                            else:
                                # Fallback: per-pixel loop without NumPy
                                buf = bytearray(block_w * block_h * 3)
                                out_idx = 0
                                pix = img.load()
                                assert pix is not None
                                for yy in range(y0, y1):
                                    for xx in range(x0, x1):
                                        from typing import cast

                                        r0, g0, b0 = cast(
                                            'tuple[int, int, int]',
                                            pix[xx - base_x, yy - base_y],
                                        )[:3]
                                        t = ar * r0 + ag * g0 + ab * b0 + a0
                                        cr, cg, cb = _color_at(t)
                                        buf[out_idx] = cr
                                        buf[out_idx + 1] = cg
                                        buf[out_idx + 2] = cb
                                        out_idx += 3
                                block_img = Image.frombytes(
                                    'RGB', (block_w, block_h), bytes(buf)
                                )
                            # Paste into result under lock (PIL isn't thread-safe)
                            async with paste_lock:
                                result.paste(block_img, (dx0, dy0))
                        finally:
                            with contextlib.suppress(Exception):
                                img.close()
                            queue.task_done()
                            tile_count += 1
                            if tile_count % 50 == 0:
                                log_memory_usage(f'pass2 after {tile_count} tiles')
                            await tile_progress.step(1)

                # Launch producers
                producers = [
                    asyncio.create_task(producer(pair)) for pair in enumerate(tiles)
                ]
                # Launch a few consumers (CPU workers)
                cpu_workers = max(1, min(os.cpu_count() or 2, 4))
                consumers = [
                    asyncio.create_task(consumer()) for _ in range(cpu_workers)
                ]

                # Wait for all producers to finish, then send sentinels
                await asyncio.gather(*producers)
                for _ in consumers:
                    await queue.put(None)  # type: ignore[arg-type]
                await queue.join()
                # Wait consumers to exit
                await asyncio.gather(*consumers)

                tile_progress.close()
            elif is_elev_contours:
                # Two-pass streaming without storing full DEM (contours)
                from constants import (
                    CONTOUR_COLOR,
                    CONTOUR_INDEX_COLOR,
                    CONTOUR_INDEX_EVERY,
                    CONTOUR_INDEX_WIDTH,
                    CONTOUR_INTERVAL_M,
                    CONTOUR_WIDTH,
                    ELEVATION_USE_RETINA,
                )

                full_eff_tile_px = 256 * (2 if ELEVATION_USE_RETINA else 1)

                def _tile_overlap_rect(tx: int, ty: int) -> tuple[int, int, int, int] | None:
                    base_x = tx * full_eff_tile_px
                    base_y = ty * full_eff_tile_px
                    cx, cy, cw, ch = crop_rect
                    x0 = max(base_x, cx)
                    y0 = max(base_y, cy)
                    x1 = min(base_x + full_eff_tile_px, cx + cw)
                    y1 = min(base_y + full_eff_tile_px, cy + ch)
                    if x1 <= x0 or y1 <= y0:
                        return None
                    return x0, y0, x1, y1

                def _tx_ty_from_index(idx: int) -> tuple[int, int]:
                    ty = idx // tiles_x
                    tx = idx % tiles_x
                    return tx, ty

                # Pass A: sample min/max elevations and build global low-res DEM seed
                max_samples = 50000
                samples: list[float] = []
                seen = 0
                import random

                rng = random.Random(42)  # noqa: S311
                tile_progress.label = 'Проверка диапазона высот (проход 1/2)'

                # Prepare low-res DEM seed canvas in crop coordinates
                from constants import CONTOUR_SEED_DOWNSAMPLE as _SEED_DS
                seed_ds = max(2, int(_SEED_DS))
                cx, cy, cw, ch = crop_rect
                seed_w = max(1, (cw + seed_ds - 1) // seed_ds)
                seed_h = max(1, (ch + seed_ds - 1) // seed_ds)
                # initialize with None to track unfilled cells; will average contributions
                seed_sum: list[list[float]] = [[0.0] * seed_w for _ in range(seed_h)]
                seed_cnt: list[list[int]] = [[0] * seed_w for _ in range(seed_h)]

                async def fetch_and_sample2(idx_xy: tuple[int, tuple[int, int]]) -> None:
                    nonlocal seen, samples, tile_count
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = _tx_ty_from_index(idx)
                    if _tile_overlap_rect(tx, ty) is None:
                        await tile_progress.step(1)
                        return
                    async with semaphore:
                        img = await async_fetch_terrain_rgb_tile(
                            client=client,
                            api_key=api_key,
                            z=zoom,
                            x=tile_x_world,
                            y=tile_y_world,
                            use_retina=ELEVATION_USE_RETINA,
                        )
                        dem_tile = decode_terrain_rgb_to_elevation_m(img)
                        h = len(dem_tile)
                        w = len(dem_tile[0]) if h else 0
                        if h and w:
                            # reservoir sampling for min/max
                            step_y = max(1, h // 32)
                            step_x = max(1, w // 32)
                            off_y = rng.randrange(0, min(step_y, h)) if step_y > 1 else 0
                            off_x = rng.randrange(0, min(step_x, w)) if step_x > 1 else 0
                            for ry in range(off_y, h, step_y):
                                row = dem_tile[ry]
                                for rx in range(off_x, w, step_x):
                                    v = row[rx]
                                    seen += 1
                                    if len(samples) < max_samples:
                                        samples.append(v)
                                    else:
                                        j = rng.randrange(0, seen)
                                        if j < max_samples:
                                            samples[j] = v
                            # accumulate into low-res seed for this tile overlap
                            ov = _tile_overlap_rect(tx, ty)
                            if ov is not None:
                                x0, y0, x1, y1 = ov
                                base_x = tx * full_eff_tile_px
                                base_y = ty * full_eff_tile_px
                                # iterate cropped region at stride seed_ds
                                for yy in range(y0, y1, seed_ds):
                                    sy = (yy - cy) // seed_ds
                                    if sy < 0 or sy >= seed_h:
                                        continue
                                    src_y = yy - base_y
                                    row_src = dem_tile[src_y]
                                    for xx in range(x0, x1, seed_ds):
                                        sx = (xx - cx) // seed_ds
                                        if sx < 0 or sx >= seed_w:
                                            continue
                                        src_x = xx - base_x
                                        v = row_src[src_x]
                                        seed_sum[sy][sx] += v
                                        seed_cnt[sy][sx] += 1
                    await tile_progress.step(1)
                    tile_count += 1
                    if tile_count % 50 == 0:
                        log_memory_usage(f'pass1(contours) after {tile_count} tiles')

                await asyncio.gather(*[fetch_and_sample2(pair) for pair in enumerate(tiles)])
                tile_progress.close()
                if samples:
                    mn = min(samples)
                    mx = max(samples)
                else:
                    mn, mx = 0.0, 1.0
                if mx < mn:
                    mn, mx = mx, mn

                # Finalize low-res DEM seed (average accumulated values)
                seed_dem: list[list[float]] = [[0.0] * seed_w for _ in range(seed_h)]
                filled = 0
                for sy in range(seed_h):
                    row_s = seed_sum[sy]
                    row_c = seed_cnt[sy]
                    out = seed_dem[sy]
                    for sx in range(seed_w):
                        c = row_c[sx]
                        if c > 0:
                            out[sx] = row_s[sx] / float(c)
                            filled += 1
                        else:
                            out[sx] = (mn + mx) * 0.5  # fallback to mid
                logger.info('Seed DEM low-res size=%sx%s, filled=%s/%s', seed_w, seed_h, filled, seed_w*seed_h)

                # Build contour levels
                import math as _math

                if CONTOUR_INTERVAL_M <= 0:
                    interval = 25.0
                else:
                    interval = float(CONTOUR_INTERVAL_M)
                start = _math.floor(mn / interval) * interval
                end = _math.ceil(mx / interval) * interval
                levels: list[float] = []
                k = 0
                v = start
                while v <= end:
                    levels.append(v)
                    k += 1
                    v = start + k * interval

                # Pass B: draw contours into result image using global low-res seed polylines
                result = Image.new('RGB', (crop_rect[2], crop_rect[3]), color=(255, 255, 255))
                tile_progress = ConsoleProgress(total=len(tiles), label='Построение изогипс (проход 2/2)')
                tile_count = 0

                # Build global polylines from seed_dem via marching squares once (in crop pixels / seed ds)
                def build_seed_polylines() -> dict[int, list[list[tuple[float, float]]]]:
                    Hs = seed_h
                    Ws = seed_w
                    polylines_by_level: dict[int, list[list[tuple[float, float]]]] = {}
                    # Simple marching squares on seed grid; coordinates will be in seed-cell space
                    def interp(p0: float, p1: float, v0: float, v1: float, level: float) -> float:
                        if v1 == v0:
                            return p0
                        t = (level - v0) / (v1 - v0)
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        return p0 + (p1 - p0) * t

                    # For each level index (use indexes into levels)
                    for li, level in enumerate(levels):
                        segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
                        for j in range(Hs - 1):
                            row0 = seed_dem[j]
                            row1 = seed_dem[j + 1]
                            for i in range(Ws - 1):
                                v00 = row0[i]
                                v10 = row0[i + 1]
                                v11 = row1[i + 1]
                                v01 = row1[i]
                                mask = (1 if v00 >= level else 0) | ((1 if v10 >= level else 0) << 1) | ((1 if v11 >= level else 0) << 2) | ((1 if v01 >= level else 0) << 3)
                                if mask == 0 or mask == 15:
                                    continue
                                x = i
                                y = j
                                yl = interp(y, y + 1, v00, v01, level)
                                yr = interp(y, y + 1, v10, v11, level)
                                xt = interp(x, x + 1, v00, v10, level)
                                xb = interp(x, x + 1, v01, v11, level)
                                def add(a: tuple[float,float], b: tuple[float,float]) -> None:
                                    segs.append((a, b))
                                if mask in (1, 14):
                                    add((xt, y), (x, yl))
                                elif mask in (2, 13):
                                    add((xt, y), (x + 1, yr))
                                elif mask in (3, 12):
                                    add((x, yl), (x + 1, yr))
                                elif mask in (4, 11):
                                    add((x + 1, yr), (xb, y + 1))
                                elif mask in (5, 10):
                                    center = (v00 + v10 + v11 + v01) * 0.25
                                    choose_diag = center >= level
                                    if choose_diag:
                                        if mask == 5:
                                            add((xt, y), (xb, y + 1))
                                        else:
                                            add((x, yl), (x + 1, yr))
                                    else:
                                        if mask == 5:
                                            add((x, yl), (x + 1, yr))
                                        else:
                                            add((xt, y), (xb, y + 1))
                                elif mask in (6, 9):
                                    add((xt, y), (xb, y + 1))
                                elif mask in (7, 8):
                                    add((x, yl), (xb, y + 1))
                        # Chain segments into polylines (simple greedy)
                        polylines: list[list[tuple[float, float]]] = []
                        if segs:
                            from collections import defaultdict
                            buckets = defaultdict(list)
                            def key(p: tuple[float,float]) -> tuple[int,int]:
                                # quantize in seed grid to reduce splits
                                qx = int(round(p[0]*8))
                                qy = int(round(p[1]*8))
                                return qx, qy
                            unused = {}
                            for idx,(a,b) in enumerate(segs):
                                unused[idx]=True
                                buckets[key(a)].append((idx,0))
                                buckets[key(b)].append((idx,1))
                            for si in range(len(segs)):
                                if si not in unused:
                                    continue
                                # start a polyline from seg si
                                stack = [si]
                                unused.pop(si, None)
                                a,b = segs[si]
                                poly = [a,b]
                                end = b
                                # extend forward
                                while True:
                                    k = key(end)
                                    found = None
                                    for idx, endpos in buckets.get(k,[]):
                                        if idx in unused:
                                            aa,bb = segs[idx]
                                            if endpos==0:
                                                end = bb
                                                poly.append(bb)
                                            else:
                                                end = aa
                                                poly.append(aa)
                                            unused.pop(idx,None)
                                            found = True
                                            break
                                    if not found:
                                        break
                                polylines.append(poly)
                        polylines_by_level[li] = polylines
                    return polylines_by_level

                seed_polylines = build_seed_polylines()

                queue: asyncio.Queue[tuple[int, int, int, int, int, int]] = asyncio.Queue(maxsize=4)
                paste_lock = asyncio.Lock()

                async def producer2(idx_xy: tuple[int, tuple[int, int]]) -> None:
                    nonlocal tile_count
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = _tx_ty_from_index(idx)
                    ov = _tile_overlap_rect(tx, ty)
                    if ov is None:
                        await tile_progress.step(1)
                        return
                    # We don't need image in pass B anymore; only metadata for block
                    await queue.put(((tx, ty, *ov)))

                from PIL import ImageDraw

                async def consumer2() -> None:
                    nonlocal tile_count
                    while True:
                        item = await queue.get()
                        if item is None:  # type: ignore[comparison-overlap]
                            queue.task_done()
                            break
                        tx, ty, x0, y0, x1, y1 = item
                        try:
                            cx, cy, _, _ = crop_rect
                            dx0 = x0 - cx
                            dy0 = y0 - cy
                            block_w = x1 - x0
                            block_h = y1 - y0

                            # local RGBA buffer with 1px edge pad
                            EDGE_PAD = 1
                            pad_w = block_w + 2 * EDGE_PAD
                            pad_h = block_h + 2 * EDGE_PAD
                            tmp = Image.new('RGBA', (pad_w, pad_h), (0,0,0,0))
                            draw = ImageDraw.Draw(tmp)

                            # Block bbox in crop coords
                            bx0, by0, bx1, by1 = x0, y0, x1, y1

                            # iterate levels and draw clipped polylines
                            for li, level in enumerate(levels):
                                is_index = ((li % max(1, int(CONTOUR_INDEX_EVERY))) == 0)
                                color = CONTOUR_INDEX_COLOR if is_index else CONTOUR_COLOR
                                width = int(CONTOUR_INDEX_WIDTH if is_index else CONTOUR_WIDTH)
                                for poly in seed_polylines.get(li, []):
                                    # map seed coords to crop pixel coords by multiplying by seed_ds and adding crop origin
                                    pts_crop: list[tuple[float,float]] = [
                                        (cx + p[0]*seed_ds, cy + p[1]*seed_ds) for p in poly
                                    ]
                                    # clip to block bbox (simple bbox clip by skipping if all outside and no crossing)
                                    # quick bbox test
                                    minx = min(p[0] for p in pts_crop)
                                    maxx = max(p[0] for p in pts_crop)
                                    miny = min(p[1] for p in pts_crop)
                                    maxy = max(p[1] for p in pts_crop)
                                    if maxx < bx0 or minx > bx1 or maxy < by0 or miny > by1:
                                        continue
                                    # render as polyline in block-local coords
                                    prev = None
                                    for px,py in pts_crop:
                                        lx = px - bx0 + EDGE_PAD
                                        ly = py - by0 + EDGE_PAD
                                        if prev is not None:
                                            draw.line((prev[0], prev[1], lx, ly), fill=tuple(list(color)+[255]), width=width)
                                        prev = (lx, ly)

                            composed = tmp.crop((1,1,1+block_w,1+block_h))
                            async with paste_lock:
                                result.paste(composed, (dx0, dy0), composed)
                        finally:
                            queue.task_done()
                            tile_count += 1
                            if tile_count % 50 == 0:
                                log_memory_usage(f'pass2(contours) after {tile_count} tiles')
                            await tile_progress.step(1)

                # Launch producers and consumers
                producers = [asyncio.create_task(producer2(pair)) for pair in enumerate(tiles)]
                cpu_workers = max(1, min(os.cpu_count() or 2, 4))
                consumers = [asyncio.create_task(consumer2()) for _ in range(cpu_workers)]
                await asyncio.gather(*producers)
                for _ in consumers:
                    await queue.put(None)  # type: ignore[arg-type]
                await queue.join()
                await asyncio.gather(*consumers)

                tile_progress.close()

                # Draw contour labels (global pass) before rotation
                try:
                    from constants import CONTOUR_LABELS_ENABLED as _CL_EN
                except Exception:
                    _CL_EN = False
                if is_elev_contours and _CL_EN:
                    def _draw_contour_labels(img, seed_polylines, levels, crop_rect, seed_ds, mpp, dry_run=False):
                        from PIL import Image, ImageDraw, ImageFont
                        from constants import (
                            CONTOUR_INDEX_EVERY,
                            CONTOUR_LABELS_ENABLED,
                            CONTOUR_LABEL_BG_PADDING,
                            CONTOUR_LABEL_BG_RGBA,
                            CONTOUR_LABEL_EDGE_MARGIN_PX,
                            CONTOUR_LABEL_FONT_BOLD,
                            CONTOUR_LABEL_FONT_PATH,
                            CONTOUR_LABEL_FONT_SIZE,
                            CONTOUR_LABEL_FONT_KM,
                            CONTOUR_LABEL_FONT_MIN_PX,
                            CONTOUR_LABEL_FONT_MAX_PX,
                            CONTOUR_LABEL_FORMAT,
                            CONTOUR_LABEL_INDEX_ONLY,
                            CONTOUR_LABEL_MIN_SEG_LEN_PX,
                            CONTOUR_LABEL_OUTLINE_COLOR,
                            CONTOUR_LABEL_OUTLINE_WIDTH,
                            CONTOUR_LABEL_SPACING_PX,
                            CONTOUR_LABEL_TEXT_COLOR,
                            GRID_FONT_PATH,
                            GRID_FONT_PATH_BOLD,
                        )
                        if not CONTOUR_LABELS_ENABLED:
                            return []
                        W, H = img.size
                        # font (will be chosen dynamically based on mpp)
                        fp = CONTOUR_LABEL_FONT_PATH or (GRID_FONT_PATH_BOLD if CONTOUR_LABEL_FONT_BOLD else GRID_FONT_PATH)
                        name = 'DejaVuSans-Bold.ttf' if CONTOUR_LABEL_FONT_BOLD else 'DejaVuSans.ttf'
                        font_cache: dict[int, ImageFont.FreeTypeFont] = {}
                        def get_font_px() -> int:
                            try:
                                px = int(round((CONTOUR_LABEL_FONT_KM * 1000.0) / max(1e-9, mpp)))
                            except Exception:
                                px = CONTOUR_LABEL_FONT_SIZE
                            
                            # Применяем ограничения
                            if px < CONTOUR_LABEL_FONT_MIN_PX:
                                px = int(CONTOUR_LABEL_FONT_MIN_PX)
                            if px > CONTOUR_LABEL_FONT_MAX_PX:
                                px = int(CONTOUR_LABEL_FONT_MAX_PX)
                            
                            return px
                        def get_font(is_index: bool):
                            size = get_font_px()
                            if size in font_cache:
                                return font_cache[size]
                            try:
                                if fp:
                                    f = ImageFont.truetype(fp, size)
                                else:
                                    f = ImageFont.truetype(name, size)
                            except Exception:
                                f = ImageFont.load_default()
                            font_cache[size] = f
                            return f
                        placed = []  # bboxes
                        import math
                        def poly_to_crop(poly):
                            # seed_polylines are in seed grid coordinates starting at (0,0) of the crop
                            # Convert to image pixel coords by scaling only
                            return [(x*seed_ds, y*seed_ds) for (x, y) in poly]
                        def intersects(a, b):
                            ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
                            return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)
                        for li, level in enumerate(levels):
                            if CONTOUR_LABEL_INDEX_ONLY and (li % max(1, int(CONTOUR_INDEX_EVERY)) != 0):
                                continue
                            is_index_line = (li % max(1, int(CONTOUR_INDEX_EVERY)) == 0)
                            text = CONTOUR_LABEL_FORMAT.format(level)
                            for poly in seed_polylines.get(li, []):
                                pts = poly_to_crop(poly)
                                if len(pts) < 2:
                                    continue
                                # lengths
                                segL = []
                                total = 0.0
                                for i in range(1, len(pts)):
                                    dx = pts[i][0]-pts[i-1][0]; dy = pts[i][1]-pts[i-1][1]
                                    L = math.hypot(dx, dy)
                                    segL.append(L); total += L
                                if total < max(CONTOUR_LABEL_MIN_SEG_LEN_PX, CONTOUR_LABEL_SPACING_PX*0.8):
                                    continue
                                target = CONTOUR_LABEL_SPACING_PX
                                while target < total:
                                    acc = 0.0; idx = -1
                                    for i, L in enumerate(segL):
                                        if acc + L >= target:
                                            idx = i; break
                                        acc += L
                                    if idx < 0:
                                        break
                                    t = (target - acc) / max(1e-9, segL[idx])
                                    x0p, y0p = pts[idx]; x1p, y1p = pts[idx+1]
                                    px = x0p + (x1p-x0p)*t; py = y0p + (y1p-y0p)*t
                                    # Compute angle for image coordinates (y increases downward) and normalize to keep text upright
                                    dx = x1p - x0p
                                    dy = y1p - y0p
                                    ang = math.degrees(math.atan2(-dy, dx))
                                    if ang < -90:
                                        ang += 180
                                    elif ang > 90:
                                        ang -= 180
                                    if not (CONTOUR_LABEL_EDGE_MARGIN_PX <= px <= W-CONTOUR_LABEL_EDGE_MARGIN_PX and CONTOUR_LABEL_EDGE_MARGIN_PX <= py <= H-CONTOUR_LABEL_EDGE_MARGIN_PX):
                                        target += CONTOUR_LABEL_SPACING_PX; continue
                                    tmp = Image.new('RGBA', (1,1), (0,0,0,0))
                                    td = ImageDraw.Draw(tmp)
                                    font = get_font(is_index_line)
                                    tw, th = td.textbbox((0,0), text, font=font)[2:]
                                    pad = int(CONTOUR_LABEL_BG_PADDING)
                                    bw, bh = tw + 2*pad, th + 2*pad
                                    box = Image.new('RGBA', (bw, bh), (0,0,0,0))
                                    bd = ImageDraw.Draw(box)
                                    if CONTOUR_LABEL_BG_RGBA:
                                        bd.rectangle((0,0,bw-1,bh-1), fill=CONTOUR_LABEL_BG_RGBA)
                                    ow = max(0, int(CONTOUR_LABEL_OUTLINE_WIDTH))
                                    if ow > 0:
                                        for ox in (-ow, 0, ow):
                                            for oy in (-ow, 0, ow):
                                                if ox == 0 and oy == 0: continue
                                                bd.text((pad+ox, pad+oy), text, font=font, fill=CONTOUR_LABEL_OUTLINE_COLOR)
                                    bd.text((pad, pad), text, font=font, fill=CONTOUR_LABEL_TEXT_COLOR)
                                    rot = box.rotate(ang, expand=True, resample=Image.BICUBIC)
                                    rw, rh = rot.size
                                    x0b = int(round(px - rw/2)); y0b = int(round(py - rh/2))
                                    x1b = x0b + rw; y1b = y0b + rh
                                    bbox = (x0b, y0b, x1b, y1b)
                                    if x0b < 0 or y0b < 0 or x1b > W or y1b > H or any(intersects(bbox, bb) for bb in placed):
                                        target += CONTOUR_LABEL_SPACING_PX; continue
                                    if not dry_run:
                                        if img.mode == 'RGBA':
                                            img.alpha_composite(rot, dest=(x0b, y0b))
                                        else:
                                            img.paste(rot, (x0b, y0b), rot)
                                    placed.append(bbox)
                                    target += CONTOUR_LABEL_SPACING_PX
                        return placed
                    try:
                        import math
                        from constants import EARTH_RADIUS_M, TILE_SIZE
                        lat_rad = math.radians(center_lat_wgs)
                        # Web Mercator meters-per-pixel at given zoom and latitude
                        mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (TILE_SIZE * (2 ** zoom))
                        # Первый проход: получаем позиции подписей БЕЗ размещения на изображении
                        label_bboxes = _draw_contour_labels(result, seed_polylines, levels, crop_rect, seed_ds, mpp, dry_run=True)
                        
                        # Создаем разрывы линий контуров в местах подписей
                        from constants import CONTOUR_LABEL_GAP_ENABLED, CONTOUR_LABEL_GAP_PADDING
                        if CONTOUR_LABEL_GAP_ENABLED and label_bboxes:
                            from PIL import ImageDraw
                            draw = ImageDraw.Draw(result)
                            gap_padding = int(CONTOUR_LABEL_GAP_PADDING)
                            for bbox in label_bboxes:
                                x0, y0, x1, y1 = bbox
                                # Расширяем область на gap_padding
                                gap_area = (
                                    max(0, x0 - gap_padding),
                                    max(0, y0 - gap_padding),
                                    min(result.width, x1 + gap_padding),
                                    min(result.height, y1 + gap_padding)
                                )
                                # "Стираем" контуры в этой области (заливаем белым)
                                draw.rectangle(gap_area, fill=(255, 255, 255))
                        
                        # Второй проход: размещаем подписи поверх созданных разрывов
                        _draw_contour_labels(result, seed_polylines, levels, crop_rect, seed_ds, mpp, dry_run=False)
                    except Exception as e:
                        logger.warning('Не удалось нанести подписи изогипс: %s', e)
            else:

                async def bound_fetch(
                    idx_xy: tuple[int, tuple[int, int]],
                ) -> tuple[int, Image.Image]:
                    nonlocal tile_count
                    idx, (tx, ty) = idx_xy
                    async with semaphore:
                        img = await async_fetch_xyz_tile(
                            client=client,
                            api_key=api_key,
                            style_id=style_id,  # type: ignore[arg-type]
                            tile_size=XYZ_TILE_SIZE,
                            z=zoom,
                            x=tx,
                            y=ty,
                            use_retina=XYZ_USE_RETINA,
                        )
                        await tile_progress.step(1)
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
                eff_tile_px = XYZ_TILE_SIZE * (2 if XYZ_USE_RETINA else 1)
                result = assemble_and_crop(
                    images=images,
                    tiles_x=tiles_x,
                    tiles_y=tiles_y,
                    eff_tile_px=eff_tile_px,
                    crop_rect=crop_rect,
                )
                with contextlib.suppress(Exception):
                    images.clear()
    finally:
        # Explicit cleanup of HTTP session resources
        try:
            if hasattr(session_ctx, '_cache') and session_ctx._cache:
                # Close SQLite cache backend if it exists
                if hasattr(session_ctx._cache, '_cache') and hasattr(
                    session_ctx._cache._cache,
                    'close',
                ):
                    await session_ctx._cache._cache.close()
        except Exception:
            logging.getLogger(__name__).debug('Error during HTTP session cleanup')

        if cache_dir_resolved:
            _cleanup_sqlite_cache(cache_dir_resolved)

    # Перед поворотом: при необходимости наложим изолинии поверх основы
    if overlay_contours and not is_elev_contours:
        try:
            # Быстрая проверка доступности Terrain-RGB перед началом фазы оверлея
            await _validate_terrain_api(api_key)
            from constants import ELEVATION_USE_RETINA as _ELEV_RETINA
            # Строим Terrain-RGB оверлей изолиний независимо, затем масштабируем к размеру основы
            eff_scale_cont = effective_scale_for_xyz(256, use_retina=_ELEV_RETINA)
            base_pad = round(min(target_w_px, target_h_px) * ROTATION_PAD_RATIO)
            pad_px_cont = max(base_pad, ROTATION_PAD_MIN_PX)
            tiles_c, (tiles_x_c, tiles_y_c), crop_rect_c, _ = compute_xyz_coverage(
                center_lat=center_lat_wgs,
                center_lng=center_lng_wgs,
                width_m=width_m,
                height_m=height_m,
                zoom=zoom,
                eff_scale=eff_scale_cont,
                pad_px=pad_px_cont,
            )

            # Создаём собственную HTTP-сессию для Terrain-RGB (одну на фазу) и ограничитель параллелизма
            cache_dir_resolved2 = _resolve_cache_dir()
            session_ctx2 = _make_http_session(cache_dir_resolved2)
            from constants import (
                CONTOUR_COLOR as _C_COLOR,
                CONTOUR_INDEX_COLOR as _C_IDX_COLOR,
                CONTOUR_INDEX_EVERY as _C_IDX_EVERY,
                CONTOUR_INDEX_WIDTH as _C_IDX_WIDTH,
                CONTOUR_INTERVAL_M as _C_INTERVAL,
                CONTOUR_WIDTH as _C_WIDTH,
            )
            full_eff_tile_px = 256 * (2 if _ELEV_RETINA else 1)

            def _tile_overlap_rect(tx: int, ty: int) -> tuple[int, int, int, int] | None:
                base_x = tx * full_eff_tile_px
                base_y = ty * full_eff_tile_px
                cx, cy, cw, ch = crop_rect_c
                x0 = max(base_x, cx)
                y0 = max(base_y, cy)
                x1 = min(base_x + full_eff_tile_px, cx + cw)
                y1 = min(base_y + full_eff_tile_px, cy + ch)
                if x1 <= x0 or y1 <= y0:
                    return None
                return x0, y0, x1, y1

            def _tx_ty_from_index(idx: int) -> tuple[int, int]:
                ty = idx // tiles_x_c
                tx = idx % tiles_x_c
                return tx, ty

            # Pass A: gather samples and build low-res seed
            max_samples = 50000
            samples: list[float] = []
            seen = 0
            import random
            rng = random.Random(42)  # noqa: S311

            from constants import CONTOUR_SEED_DOWNSAMPLE as _SEED_DS
            seed_ds = max(2, int(_SEED_DS))
            cx_c, cy_c, cw_c, ch_c = crop_rect_c
            seed_w = max(1, (cw_c + seed_ds - 1) // seed_ds)
            seed_h = max(1, (ch_c + seed_ds - 1) // seed_ds)
            seed_sum: list[list[float]] = [[0.0] * seed_w for _ in range(seed_h)]
            seed_cnt: list[list[int]] = [[0] * seed_w for _ in range(seed_h)]

            # Ограничитель параллелизма для overlay-запросов
            overlay_semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY or ASYNC_MAX_CONCURRENCY)

            async def fetch_and_sample_overlay(idx_xy: tuple[int, tuple[int, int]], client2: aiohttp.ClientSession) -> None:
                nonlocal seen, samples
                try:
                    idx, (tile_x_world, tile_y_world) = idx_xy
                    tx, ty = _tx_ty_from_index(idx)
                    if _tile_overlap_rect(tx, ty) is None:
                        return
                    async with overlay_semaphore:
                        img = await async_fetch_terrain_rgb_tile(
                            client=client2,
                            api_key=api_key,
                            z=zoom,
                            x=tile_x_world,
                            y=tile_y_world,
                            use_retina=_ELEV_RETINA,
                        )
                    dem_tile = decode_terrain_rgb_to_elevation_m(img)
                    h = len(dem_tile)
                    w = len(dem_tile[0]) if h else 0
                    if h and w:
                        step_y = max(1, h // 32)
                        step_x = max(1, w // 32)
                        off_y = rng.randrange(0, min(step_y, h)) if step_y > 1 else 0
                        off_x = rng.randrange(0, min(step_x, w)) if step_x > 1 else 0
                        for ry in range(off_y, h, step_y):
                            row = dem_tile[ry]
                            for rx in range(off_x, w, step_x):
                                v = row[rx]
                                seen += 1
                                if len(samples) < max_samples:
                                    samples.append(v)
                                else:
                                    j = rng.randrange(0, seen)
                                    if j < max_samples:
                                        samples[j] = v
                        ov = _tile_overlap_rect(tx, ty)
                        if ov is not None:
                            x0, y0, x1, y1 = ov
                            base_x = tx * full_eff_tile_px
                            base_y = ty * full_eff_tile_px
                            for yy in range(y0, y1, seed_ds):
                                sy = (yy - cy_c) // seed_ds
                                if sy < 0 or sy >= seed_h:
                                    continue
                                src_y = yy - base_y
                                row_src = dem_tile[src_y]
                                for xx in range(x0, x1, seed_ds):
                                    sx = (xx - cx_c) // seed_ds
                                    if sx < 0 or sx >= seed_w:
                                        continue
                                    src_x = xx - base_x
                                    v = row_src[src_x]
                                    seed_sum[sy][sx] += v
                                    seed_cnt[sy][sx] += 1
                except Exception:
                    # Ignore individual tile errors during overlay pass A
                    pass
                finally:
                    with contextlib.suppress(Exception):
                        await overlay_progress_a.step(1)

            # Прогресс-индикатор для overlay-прохода A
            logger.info('Изолинии: старт прохода 1/2')
            overlay_progress_a = ConsoleProgress(total=len(tiles_c), label='Изолинии: загрузка Terrain-RGB (проход 1/2)')
            async with session_ctx2 as client2:
                await asyncio.gather(
                    *[fetch_and_sample_overlay(pair, client2) for pair in enumerate(tiles_c)],
                    return_exceptions=True,
                )
            overlay_progress_a.close()
            # Логи по завершении прохода A
            try:
                tiles_len = len(tiles_c)
                samples_len = len(samples)
                logger.info('Изолинии: завершён проход 1/2: tiles=%d, samples=%d', tiles_len, samples_len)
            except Exception:
                logger.info('Изолинии: завершён проход 1/2')
            
            if samples:
                mn = min(samples); mx = max(samples)
            else:
                mn, mx = 0.0, 1.0
            if mx < mn:
                mn, mx = mx, mn

            seed_dem: list[list[float]] = [[0.0] * seed_w for _ in range(seed_h)]
            for sy in range(seed_h):
                row_s = seed_sum[sy]; row_c = seed_cnt[sy]; out = seed_dem[sy]
                for sx in range(seed_w):
                    c = row_c[sx]
                    out[sx] = (row_s[sx] / float(c)) if c > 0 else (mn + mx) * 0.5

            import math as _math
            interval = float(_C_INTERVAL) if _C_INTERVAL > 0 else 25.0
            start = _math.floor(mn / interval) * interval
            end = _math.ceil(mx / interval) * interval
            levels: list[float] = []
            k = 0; v = start
            while v <= end:
                levels.append(v); k += 1; v = start + k * interval

            def build_seed_polylines() -> dict[int, list[list[tuple[float, float]]]]:
                Hs = seed_h; Ws = seed_w
                polylines_by_level: dict[int, list[list[tuple[float, float]]]] = {}
                def interp(p0: float, p1: float, v0: float, v1: float, level: float) -> float:
                    if v1 == v0: return p0
                    t = (level - v0) / (v1 - v0)
                    if t < 0.0: t = 0.0
                    elif t > 1.0: t = 1.0
                    return p0 + (p1 - p0) * t
                for li, level in enumerate(levels):
                    segs: list[tuple[tuple[float, float], tuple[float, float]]]=[]
                    for j in range(Hs - 1):
                        row0 = seed_dem[j]; row1 = seed_dem[j + 1]
                        for i in range(Ws - 1):
                            v00 = row0[i]; v10 = row0[i + 1]; v11 = row1[i + 1]; v01 = row1[i]
                            mask = (1 if v00 >= level else 0) | ((1 if v10 >= level else 0) << 1) | ((1 if v11 >= level else 0) << 2) | ((1 if v01 >= level else 0) << 3)
                            if mask == 0 or mask == 15: continue
                            x = i; y = j
                            yl = interp(y, y + 1, v00, v01, level)
                            yr = interp(y, y + 1, v10, v11, level)
                            xt = interp(x, x + 1, v00, v10, level)
                            xb = interp(x, x + 1, v01, v11, level)
                            def add(a: tuple[float,float], b: tuple[float,float]) -> None:
                                segs.append((a, b))
                            if mask in (1, 14): add((xt, y), (x, yl))
                            elif mask in (2, 13): add((xt, y), (x + 1, yr))
                            elif mask in (3, 12): add((x, yl), (x + 1, yr))
                            elif mask in (4, 11): add((x + 1, yr), (xb, y + 1))
                            elif mask in (5, 10):
                                center = (v00 + v10 + v11 + v01) * 0.25
                                choose_diag = center >= level
                                if choose_diag:
                                    if mask == 5: add((xt, y), (xb, y + 1))
                                    else: add((x, yl), (x + 1, yr))
                                else:
                                    if mask == 5: add((x, yl), (x + 1, yr))
                                    else: add((xt, y), (xb, y + 1))
                            elif mask in (6, 9): add((xt, y), (xb, y + 1))
                            elif mask in (7, 8): add((x, yl), (xb, y + 1))
                    polylines: list[list[tuple[float, float]]]=[]
                    if segs:
                        from collections import defaultdict
                        buckets = defaultdict(list)
                        def key(p: tuple[float,float]) -> tuple[int,int]:
                            qx = int(round(p[0]*8)); qy = int(round(p[1]*8)); return qx, qy
                        unused = {}
                        for idx,(a,b) in enumerate(segs):
                            unused[idx]=True
                            buckets[key(a)].append((idx,0))
                            buckets[key(b)].append((idx,1))
                        for si in range(len(segs)):
                            if si not in unused: continue
                            a,b = segs[si]; poly=[a,b]; end=b
                            while True:
                                k = key(end); found=None
                                for idx,endpos in buckets.get(k,[]):
                                    if idx in unused:
                                        aa,bb = segs[idx]
                                        if endpos==0:
                                            end=bb; poly.append(bb)
                                        else:
                                            end=aa; poly.append(aa)
                                        unused.pop(idx,None); found=True; break
                                if not found: break
                            polylines.append(poly)
                    polylines_by_level[li] = polylines
                return polylines_by_level

            seed_polylines = build_seed_polylines()

            # Draw overlay RGBA with transparent background
            from PIL import ImageDraw
            overlay = Image.new('RGBA', (crop_rect_c[2], crop_rect_c[3]), (0,0,0,0))
            overlay_progress_b = ConsoleProgress(total=len(levels), label='Изолинии: построение и рисование (проход 2/2)')
            logger.info('Изолинии: старт прохода 2/2: levels=%d', len(levels))
            for li, level in enumerate(levels):
                is_index = ((li % max(1, int(_C_IDX_EVERY))) == 0)
                color = _C_IDX_COLOR if is_index else _C_COLOR
                width = int(_C_IDX_WIDTH if is_index else _C_WIDTH)
                for poly in seed_polylines.get(li, []):
                    # draw full polylines; block optimization skipped for simplicity
                    pts_crop = [(p[0]*seed_ds, p[1]*seed_ds) for p in poly]
                    if len(pts_crop) < 2:
                        continue
                    draw = ImageDraw.Draw(overlay)
                    for i in range(1, len(pts_crop)):
                        x0, y0 = pts_crop[i-1]; x1, y1 = pts_crop[i]
                        draw.line((x0, y0, x1, y1), fill=tuple(list(color)+[255]), width=width)
                # step progress per level
                try:
                    overlay_progress_b.step_sync(1)
                except Exception:
                    pass
                # periodic diagnostics every 5 levels
                try:
                    if (li + 1) % 5 == 0 or (li + 1) == len(levels):
                        log_memory_usage(f'Contours pass 2/2 after level {li+1}/{len(levels)}')
                        log_thread_status(f'Contours pass 2/2 after level {li+1}/{len(levels)}')
                except Exception:
                    pass
            logger.info('Изолинии: завершён проход 2/2')

            # Labels on overlay
            import time
            labels_start_time = time.monotonic()
            logger.info('Изолинии: подписи (оверлей) — старт')
            log_memory_usage('before overlay labels')
            try:
                from constants import CONTOUR_LABELS_ENABLED as _CL_EN
            except Exception:
                _CL_EN = False
            if _CL_EN:
                try:
                    # прогресс для подписей можно отразить тем же лейблом
                    from constants import EARTH_RADIUS_M, TILE_SIZE
                    import math
                    lat_rad = math.radians(center_lat_wgs)
                    mpp = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (TILE_SIZE * (2 ** zoom))
                    # Reuse simplified overlay label drawer similar to above
                    def _draw_contour_labels_overlay(img, seed_polylines, levels, seed_ds, mpp, dry_run=False):
                        from PIL import Image, ImageDraw, ImageFont
                        from constants import (
                            CONTOUR_INDEX_EVERY,
                            CONTOUR_LABELS_ENABLED,
                            CONTOUR_LABEL_BG_PADDING,
                            CONTOUR_LABEL_BG_RGBA,
                            CONTOUR_LABEL_EDGE_MARGIN_PX,
                            CONTOUR_LABEL_FONT_BOLD,
                            CONTOUR_LABEL_FONT_PATH,
                            CONTOUR_LABEL_FONT_SIZE,
                            CONTOUR_LABEL_FONT_KM,
                            CONTOUR_LABEL_FONT_MIN_PX,
                            CONTOUR_LABEL_FONT_MAX_PX,
                            CONTOUR_LABEL_FORMAT,
                            CONTOUR_LABEL_INDEX_ONLY,
                            CONTOUR_LABEL_MIN_SEG_LEN_PX,
                            CONTOUR_LABEL_OUTLINE_COLOR,
                            CONTOUR_LABEL_OUTLINE_WIDTH,
                            CONTOUR_LABEL_SPACING_PX,
                            CONTOUR_LABEL_TEXT_COLOR,
                            GRID_FONT_PATH,
                            GRID_FONT_PATH_BOLD,
                        )
                        if not CONTOUR_LABELS_ENABLED:
                            return []
                        logger.info('Подписи overlay: старт обработки %d уровней', len(levels))
                        W, H = img.size
                        fp = CONTOUR_LABEL_FONT_PATH or (GRID_FONT_PATH_BOLD if CONTOUR_LABEL_FONT_BOLD else GRID_FONT_PATH)
                        name = 'DejaVuSans-Bold.ttf' if CONTOUR_LABEL_FONT_BOLD else 'DejaVuSans.ttf'
                        font_cache = {}
                        total_labels = 0
                        def get_font_px() -> int:
                            try:
                                px = int(round((CONTOUR_LABEL_FONT_KM * 1000.0) / max(1e-9, mpp)))
                            except Exception:
                                px = CONTOUR_LABEL_FONT_SIZE
                            final_px = int(max(CONTOUR_LABEL_FONT_MIN_PX, min(CONTOUR_LABEL_FONT_MAX_PX, px)))
                            return final_px
                        def get_font(is_index: bool):
                            size = get_font_px()
                            if size in font_cache:
                                return font_cache[size]
                            try:
                                if fp:
                                    f = ImageFont.truetype(fp, size)
                                else:
                                    f = ImageFont.truetype(name, size)
                            except Exception:
                                f = ImageFont.load_default()
                            font_cache[size] = f
                            return f
                        placed = []
                        import math
                        def poly_to_crop(poly):
                            return [(x*seed_ds, y*seed_ds) for (x, y) in poly]
                        def intersects(a,b):
                            ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
                            return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)
                        for li, level in enumerate(levels):
                            if CONTOUR_LABEL_INDEX_ONLY and (li % max(1, int(CONTOUR_INDEX_EVERY)) != 0):
                                continue
                            is_index_line = (li % max(1, int(CONTOUR_INDEX_EVERY)) == 0)
                            text = CONTOUR_LABEL_FORMAT.format(level)
                            level_polys = seed_polylines.get(li, [])
                            logger.info('Подписи overlay: уровень %d/%d (%.1fm) - полилиний: %d', 
                                       li+1, len(levels), level, len(level_polys))
                            level_labels = 0
                            for poly in level_polys:
                                pts = poly_to_crop(poly)
                                if len(pts) < 2:
                                    continue
                                segL = []; total = 0.0
                                for i in range(1, len(pts)):
                                    dx = pts[i][0]-pts[i-1][0]; dy = pts[i][1]-pts[i-1][1]
                                    L = math.hypot(dx, dy); segL.append(L); total += L
                                if total < max(CONTOUR_LABEL_MIN_SEG_LEN_PX, CONTOUR_LABEL_SPACING_PX*0.8):
                                    continue
                                target = CONTOUR_LABEL_SPACING_PX
                                while target < total:
                                    acc = 0.0; idx = -1
                                    for i, L in enumerate(segL):
                                        if acc + L >= target:
                                            idx = i; break
                                        acc += L
                                    if idx < 0:
                                        break
                                    t = (target - acc) / max(1e-9, segL[idx])
                                    x0p, y0p = pts[idx]; x1p, y1p = pts[idx+1]
                                    px = x0p + (x1p-x0p)*t; py = y0p + (y1p-y0p)*t
                                    dx = x1p - x0p; dy = y1p - y0p
                                    ang = math.degrees(math.atan2(-dy, dx))
                                    if ang < -90: ang += 180
                                    elif ang > 90: ang -= 180
                                    from PIL import Image, ImageDraw
                                    font = get_font(is_index_line)
                                    tmp = Image.new('RGBA', (1,1), (0,0,0,0))
                                    dd = ImageDraw.Draw(tmp)
                                    tw, th = dd.textbbox((0,0), text, font=font)[2:]
                                    pad = int(CONTOUR_LABEL_BG_PADDING)
                                    bw, bh = tw + 2*pad, th + 2*pad
                                    box = Image.new('RGBA', (bw, bh), (0,0,0,0))
                                    bdraw = ImageDraw.Draw(box)
                                    if CONTOUR_LABEL_BG_RGBA:
                                        bdraw.rectangle((0,0,bw-1,bh-1), fill=CONTOUR_LABEL_BG_RGBA)
                                    ow = max(0, int(CONTOUR_LABEL_OUTLINE_WIDTH))
                                    if ow > 0:
                                        for ox in (-ow, 0, ow):
                                            for oy in (-ow, 0, ow):
                                                if ox == 0 and oy == 0: continue
                                                bdraw.text((pad+ox, pad+oy), text, font=font, fill=CONTOUR_LABEL_OUTLINE_COLOR)
                                    bdraw.text((pad, pad), text, font=font, fill=CONTOUR_LABEL_TEXT_COLOR)
                                    logger.debug('Поворот подписи "%s" на %.1f°', text, ang)
                                    rot = box.rotate(ang, expand=True, resample=Image.BICUBIC)
                                    rw, rh = rot.size
                                    x0b = int(round(px - rw/2)); y0b = int(round(py - rh/2))
                                    if 0 <= x0b < W and 0 <= y0b < H:
                                        if not dry_run:
                                            img.alpha_composite(rot, dest=(x0b, y0b))
                                    placed.append((x0b, y0b, x0b+rw, y0b+rh))
                                    level_labels += 1
                                    total_labels += 1
                                    target += CONTOUR_LABEL_SPACING_PX
                            logger.info('Подписи overlay: уровень %d завершен, размещено %d подписей', 
                                       li+1, level_labels)
                            
                            # Периодический отчет о памяти
                            if (li + 1) % 5 == 0:
                                log_memory_usage(f'after overlay labels level {li+1}/{len(levels)}')
                        
                        logger.info('Подписи overlay: завершено, всего размещено %d подписей', total_labels)
                        return placed
                    # Первый проход: получаем позиции подписей БЕЗ размещения на изображении
                    overlay_label_bboxes = _draw_contour_labels_overlay(overlay, seed_polylines, levels, seed_ds, mpp, dry_run=True)
                    
                    # Создаем разрывы линий контуров в местах подписей для overlay
                    from constants import CONTOUR_LABEL_GAP_ENABLED, CONTOUR_LABEL_GAP_PADDING
                    if CONTOUR_LABEL_GAP_ENABLED and overlay_label_bboxes:
                        from PIL import ImageDraw
                        draw_overlay = ImageDraw.Draw(overlay)
                        gap_padding = int(CONTOUR_LABEL_GAP_PADDING)
                        for bbox in overlay_label_bboxes:
                            x0, y0, x1, y1 = bbox
                            # Расширяем область на gap_padding
                            gap_area = (
                                max(0, x0 - gap_padding),
                                max(0, y0 - gap_padding),
                                min(overlay.width, x1 + gap_padding),
                                min(overlay.height, y1 + gap_padding)
                            )
                            # "Стираем" контуры в этой области (делаем прозрачным для RGBA)
                            draw_overlay.rectangle(gap_area, fill=(0, 0, 0, 0))
                    
                    # Второй проход: размещаем подписи поверх созданных разрывов
                    _draw_contour_labels_overlay(overlay, seed_polylines, levels, seed_ds, mpp, dry_run=False)
                except Exception as e:
                    logger.warning('Не удалось нанести подписи изолиний (оверлей): %s', e)
            
            labels_elapsed = time.monotonic() - labels_start_time
            logger.info('Изолинии: подписи (оверлей) — завершены (%.2fs)', labels_elapsed)
            log_memory_usage('after overlay labels')

            # Composite overlay onto base (pre-rotation)
            composite_start_time = time.monotonic()
            logger.info('Изолинии: компоновка overlay на базу — старт')
            try:
                if overlay.size != (crop_rect[2], crop_rect[3]):
                    overlay = overlay.resize((crop_rect[2], crop_rect[3]), Image.BICUBIC)
                base_rgba = result.convert('RGBA')
                base_rgba.alpha_composite(overlay)
                result = base_rgba.convert('RGB')
            except Exception as e:
                logger.warning('Не удалось наложить изолинии: %s', e)
            composite_elapsed = time.monotonic() - composite_start_time
            logger.info('Изолинии: компоновка overlay на базу — завершена (%.2fs)', composite_elapsed)
            log_memory_usage('after overlay composite')
        except Exception as e:
            logger.warning('Построение оверлея изолиний не удалось: %s', e)

    # Image rotation
    rotation_start_time = time.monotonic()
    logger.info('Поворот изображения — старт')
    angle_deg = compute_rotation_deg_for_east_axis(
        center_lat_sk42=center_lat_sk42,
        center_lng_sk42=center_lng_sk42,
        map_params=map_params,
        crs_sk42_gk=crs_sk42_gk,
        t_sk42_to_wgs=t_sk42_to_wgs,
    )
    result = rotate_keep_size(result, angle_deg, fill=(255, 255, 255))
    rotation_elapsed = time.monotonic() - rotation_start_time
    logger.info('Поворот изображения — завершён (%.2fs)', rotation_elapsed)
    log_memory_usage('after rotation')

    # Cropping
    crop_start_time = time.monotonic()
    logger.info('Обрезка к целевому размеру — старт')
    result = center_crop(result, target_w_px, target_h_px)
    crop_elapsed = time.monotonic() - crop_start_time
    logger.info('Обрезка к целевому размеру — завершена (%.2fs)', crop_elapsed)
    log_memory_usage('after cropping')

    # Grid drawing
    grid_start_time = time.monotonic()
    logger.info('Рисование км-сетки — старт')
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
    grid_elapsed = time.monotonic() - grid_start_time
    logger.info('Рисование км-сетки — завершено (%.2fs)', grid_elapsed)
    log_memory_usage('after grid drawing')
    log_thread_status('after grid drawing')

    # Preview publishing
    preview_start_time = time.monotonic()
    logger.info('Публикация предпросмотра — старт')
    did_publish = False
    with contextlib.suppress(Exception):
        did_publish = publish_preview_image(result)
    preview_elapsed = time.monotonic() - preview_start_time
    logger.info('Публикация предпросмотра — %s (%.2fs)', 'успех' if did_publish else 'пропущено', preview_elapsed)

    # Если GUI подписан на предпросмотр (did_publish=True), не сохраняем автоматически.
    if not did_publish:
        save_start_time = time.monotonic()
        logger.info('Сохранение файла — старт')
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
        save_elapsed = time.monotonic() - save_start_time
        logger.info('Сохранение файла — завершено (%.2fs)', save_elapsed)
        log_memory_usage('after file save')

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

    overall_elapsed = time.monotonic() - overall_start_time
    logger.info('=== ОБЩИЙ ТАЙМЕР: завершён download_satellite_rectangle (%.2fs) ===', overall_elapsed)
    return output_path
