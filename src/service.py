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
    default_map_type,
    map_type_to_style_id,
    MapType,
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
    async_fetch_xyz_tile,
    async_fetch_terrain_rgb_tile,
    assemble_dem,
    build_transformers_sk42,
    choose_zoom_with_limit,
    colorize_dem_to_image,
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
    """
    Проверяет доступность стилей Mapbox (Styles API tiles endpoint).
    """
    """
    Проверяет доступность интернета и валидность API-ключа перед началом тяжёлой обработки.

    Делает быстрый запрос к одному тайлу (z=0/x=0/y=0). В случае проблем бросает RuntimeError
    с понятным для пользователя сообщением.
    """
    test_path = f'{MAPBOX_STATIC_BASE}/{style_id}/tiles/256/0/0/0'
    test_url = f'{test_path}?access_token={api_key}'
    timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession() as client:  # noqa: SIM117
            async with client.get(test_url, timeout=timeout) as resp:
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
        async with aiohttp.ClientSession() as client:
            async with client.get(test_url, timeout=timeout) as resp:
                sc = resp.status
                if sc == HTTP_OK:
                    with contextlib.suppress(Exception):
                        await resp.read()
                    return
                if sc in (HTTP_UNAUTHORIZED, HTTP_FORBIDDEN):
                    raise RuntimeError('Неверный или недействительный API-ключ. Проверьте ключ и попробуйте снова.')
                raise RuntimeError(f'Ошибка доступа к серверу карт (HTTP {sc}). Повторите попытку позже.')
    except (TimeoutError, aiohttp.ClientConnectorError, aiohttp.ClientOSError):
        raise RuntimeError('Нет соединения с интернетом или сервер недоступен. Проверьте подключение к сети.') from None


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

    # Определяем тип карты и стратегию
    mt = getattr(settings, 'map_type', default_map_type())
    try:
        mt_enum = MapType(mt) if not isinstance(mt, MapType) else mt
    except Exception:
        mt_enum = default_map_type()

    style_id: str | None = None
    if mt_enum in (MapType.SATELLITE, MapType.HYBRID, MapType.STREETS, MapType.OUTDOORS):
        style_id = map_type_to_style_id(mt_enum)
        logger.info('Тип карты: %s; style_id=%s; tile_size=%s; retina=%s', mt_enum, style_id, XYZ_TILE_SIZE, XYZ_USE_RETINA)
        await _validate_api_and_connectivity(api_key, style_id)
    elif mt_enum == MapType.ELEVATION_COLOR:
        logger.info('Тип карты: %s (Terrain-RGB, цветовая шкала); retina=%s', mt_enum, False)
    else:
        # Нереализованные режимы пока откатываются к Спутнику
        logger.warning('Выбран режим высот (%s), пока не реализовано. Используется Спутник.', mt_enum)
        style_id = map_type_to_style_id(default_map_type())
        await _validate_api_and_connectivity(api_key, style_id)

    # Выбор масштаба под тип карты
    is_elev_color = False
    try:
        is_elev_color = MapType(mt) == MapType.ELEVATION_COLOR if not isinstance(mt, MapType) else mt == MapType.ELEVATION_COLOR
    except Exception:
        is_elev_color = False

    if is_elev_color:
        # Для Terrain-RGB базовый тайл 256px; @2x даёт 512
        eff_scale = effective_scale_for_xyz(256, use_retina=False)
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

    tile_label = 'Загрузка Terrain-RGB тайлов' if is_elev_color else 'Загрузка XYZ-тайлов'
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
                # Two-pass streaming without storing full DEM
                from topography import compute_percentiles, ELEV_PCTL_LO, ELEV_PCTL_HI, ELEV_MIN_RANGE_M, ELEVATION_COLOR_RAMP
                # Helper: build LUT from color ramp for fast palette lookup
                def _lerp(a: float, b: float, t: float) -> float:
                    return a + (b - a) * t
                LUT_SIZE = 2048
                _LUT: list[tuple[int, int, int]] = []
                ramp = ELEVATION_COLOR_RAMP
                # Precompute cumulative ramp into fixed-size LUT
                for i in range(LUT_SIZE):
                    t = i / (LUT_SIZE - 1)
                    # find segment
                    for j in range(1, len(ramp)):
                        t0, c0 = ramp[j - 1]
                        t1, c1 = ramp[j]
                        if t <= t1 or j == len(ramp) - 1:
                            local = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                            r = int(round(_lerp(c0[0], c1[0], local)))
                            g = int(round(_lerp(c0[1], c1[1], local)))
                            b = int(round(_lerp(c0[2], c1[2], local)))
                            _LUT.append((r, g, b))
                            break
                def _color_at(t: float) -> tuple[int, int, int]:
                    # Clamp and map to LUT index
                    if t <= 0.0:
                        return _LUT[0]
                    if t >= 1.0:
                        return _LUT[-1]
                    idx = int(t * (LUT_SIZE - 1))
                    return _LUT[idx]

                full_eff_tile_px = 256  # retina disabled for elevation

                # Fast overlap check to skip tiles outside crop
                def _tile_overlap_rect(tx: int, ty: int) -> tuple[int,int,int,int] | None:
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
                def _tx_ty_from_index(idx: int) -> tuple[int,int]:
                    ty = idx // tiles_x
                    tx = idx % tiles_x
                    return tx, ty

                # Pass A: sample elevations for percentiles (reservoir sampling)
                MAX_SAMPLES = 50000
                samples: list[float] = []  # reservoir
                seen_count = 0
                import random
                rng = random.Random(42)  # deterministic for reproducibility
                tile_progress.label = 'Проверка диапазона высот (проход 1/2)'

                async def fetch_and_sample(idx_xy: tuple[int, tuple[int, int]]) -> None:
                    nonlocal tile_count, samples, seen_count
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
                            use_retina=False,
                        )
                        dem_tile = decode_terrain_rgb_to_elevation_m(img)
                        # Iterate coarse grid within tile to limit CPU, but feed values into reservoir
                        h = len(dem_tile)
                        w = len(dem_tile[0]) if h else 0
                        if h and w:
                            step_y = max(1, h // 32)
                            step_x = max(1, w // 32)
                            # jitter to avoid regular sampling artifacts
                            off_y = rng.randrange(0, min(step_y, h)) if step_y > 1 else 0
                            off_x = rng.randrange(0, min(step_x, w)) if step_x > 1 else 0
                            for ry in range(off_y, h, step_y):
                                row = dem_tile[ry]
                                for rx in range(off_x, w, step_x):
                                    v = row[rx]
                                    seen_count += 1
                                    if len(samples) < MAX_SAMPLES:
                                        samples.append(v)
                                    else:
                                        j = rng.randrange(0, seen_count)
                                        if j < MAX_SAMPLES:
                                            samples[j] = v
                        await tile_progress.step(1)
                        tile_count += 1
                        if tile_count % 50 == 0:
                            log_memory_usage(f'pass1 after {tile_count} tiles')
                # launch tasks
                await asyncio.gather(*[fetch_and_sample(pair) for pair in enumerate(tiles)])
                tile_progress.close()
                # Compute percentiles from reservoir
                logger.info('DEM sampling reservoir: kept=%s seen~=%s', len(samples), seen_count)
                lo, hi = compute_percentiles(samples, ELEV_PCTL_LO, ELEV_PCTL_HI)
                if hi - lo < ELEV_MIN_RANGE_M:
                    mid = (lo + hi) / 2.0
                    lo = mid - ELEV_MIN_RANGE_M / 2.0
                    hi = mid + ELEV_MIN_RANGE_M / 2.0
                inv = 1.0 / (hi - lo) if hi > lo else 0.0

                # Pass B: render directly to output image using producer–consumer (I/O vs CPU)
                result = Image.new('RGB', (crop_rect[2], crop_rect[3]))
                tile_progress = ConsoleProgress(total=len(tiles), label='Окрашивание DEM (проход 2/2)')
                tile_count = 0

                queue: asyncio.Queue[tuple[int,int,int,int,int,int,Image.Image]] = asyncio.Queue(maxsize=4)
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
                        img = await async_fetch_terrain_rgb_tile(
                            client=client,
                            api_key=api_key,
                            z=zoom,
                            x=tile_x_world,
                            y=tile_y_world,
                            use_retina=False,
                        )
                    # Enqueue with metadata for consumers
                    x0, y0, x1, y1 = ov
                    await queue.put((tx, ty, x0, y0, x1, y1, img))
                    # Progress is stepped when consumer finishes to reflect actual painting

                # Precompute normalization coeffs for consumers (used with DEM values)
                async def consumer() -> None:
                    nonlocal tile_count
                    while True:
                        item = await queue.get()
                        if item is None:  # type: ignore[comparison-overlap]
                            queue.task_done()
                            break
                        tx, ty, x0, y0, x1, y1, img = item
                        try:
                            dem_tile = decode_terrain_rgb_to_elevation_m(img)
                            cx, cy, _, _ = crop_rect
                            dx0 = x0 - cx
                            dy0 = y0 - cy
                            base_x = tx * full_eff_tile_px
                            base_y = ty * full_eff_tile_px
                            # Build raw RGB buffer for the overlap block (scanline writing)
                            block_w = x1 - x0
                            block_h = y1 - y0
                            buf = bytearray(block_w * block_h * 3)
                            out_idx = 0
                            for yy in range(y0, y1):
                                src_row = dem_tile[yy - base_y]
                                for xx in range(x0, x1):
                                    val = src_row[xx - base_x]
                                    t = (val - lo) * inv
                                    r, g, b = _color_at(t)
                                    buf[out_idx] = r
                                    buf[out_idx + 1] = g
                                    buf[out_idx + 2] = b
                                    out_idx += 3
                            # Create image from bytes and paste under lock (PIL isn't thread-safe)
                            block_img = Image.frombytes('RGB', (block_w, block_h), bytes(buf))
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
                producers = [asyncio.create_task(producer(pair)) for pair in enumerate(tiles)]
                # Launch a few consumers (CPU workers)
                cpu_workers = max(1, min(os.cpu_count() or 2, 4))
                consumers = [asyncio.create_task(consumer()) for _ in range(cpu_workers)]

                # Wait for all producers to finish, then send sentinels
                await asyncio.gather(*producers)
                for _ in consumers:
                    await queue.put(None)  # type: ignore[arg-type]
                await queue.join()
                # Wait consumers to exit
                await asyncio.gather(*consumers)

                tile_progress.close()
            else:
                async def bound_fetch(idx_xy: tuple[int, tuple[int, int]]) -> tuple[int, Image.Image]:
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

    angle_deg = compute_rotation_deg_for_east_axis(
        center_lat_sk42=center_lat_sk42,
        center_lng_sk42=center_lng_sk42,
        map_params=map_params,
        crs_sk42_gk=crs_sk42_gk,
        t_sk42_to_wgs=t_sk42_to_wgs,
    )
    result = rotate_keep_size(result, angle_deg, fill=(255, 255, 255))

    result = center_crop(result, target_w_px, target_h_px)

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
