# ------------------------------
# Основной асинхронный процесс
# ------------------------------
import asyncio
import contextlib
import os
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import aiohttp
from PIL import Image
from pyproj import CRS, Transformer

try:
    from aiohttp_client_cache import CachedSession as _CachedSession
    from aiohttp_client_cache import SQLiteBackend as _SQLiteBackend

    CachedSession: object | None = _CachedSession
    SQLiteBackend: object | None = _SQLiteBackend
    _CACHE_IMPORT_ERROR: Exception | None = None
except Exception as e:
    CachedSession = None
    SQLiteBackend = None
    _CACHE_IMPORT_ERROR = e

from constants import (
    ASYNC_MAX_CONCURRENCY,
    CURRENT_PROFILE,
    ENABLE_WHITE_MASK,
    GRID_COLOR,
    GRID_STEP_M,
    MAX_GK_ZONE,
    MAX_OUTPUT_PIXELS,
    MAX_ZOOM,
    PIL_DISABLE_LIMIT,
    ROTATION_PAD_MIN_PX,
    ROTATION_PAD_RATIO,
)
from image import (
    apply_white_mask,
    assemble_and_crop,
    center_crop,
    draw_axis_aligned_km_grid,
    rotate_keep_size,
)
from profiles import load_profile
from progress import ConsoleProgress, LiveSpinner
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

settings = load_profile(CURRENT_PROFILE)


async def download_satellite_rectangle(  # noqa: PLR0915, PLR0913, C901
    center_x_sk42_gk: float,
    center_y_sk42_gk: float,
    width_m: float,
    height_m: float,
    api_key: str,
    output_path: str,
    max_zoom: int = MAX_ZOOM,
) -> str:
    """Полный конвейер."""
    # Переопределяем параметры из профиля
    eff_scale = effective_scale_for_xyz(
        settings.tile_size, use_retina=settings.use_retina
    )
    # Подготовка — конвертация из Гаусса-Крюгера в географические координаты СК-42
    sp = LiveSpinner('Подготовка: определение зоны')
    sp.start()

    zone = int(center_x_sk42_gk // 1000000)  # Зона из первых цифр X координаты
    if zone < 1 or zone > MAX_GK_ZONE:
        # Fallback: пытаемся определить зону из координаты
        zone = max(1, min(MAX_GK_ZONE, int((center_x_sk42_gk - 500000) // 1000000) + 1))
    crs_sk42_gk = CRS.from_epsg(28400 + zone)
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
    semaphore = asyncio.Semaphore(settings.concurrency or ASYNC_MAX_CONCURRENCY)

    # HTTP session with optional persistent cache
    use_cache = getattr(settings, 'cache', None) is not None and getattr(
        settings.cache, 'enabled', True
    )

    # Resolve cache directory to an absolute path anchored at project root if relative
    cache_dir_resolved: Path | None = None
    if getattr(settings, 'cache', None) is not None:
        raw_dir = Path(settings.cache.dir)
        if raw_dir.is_absolute():
            cache_dir_resolved = raw_dir
        else:
            # Project root is one level above src/
            repo_root = Path(__file__).resolve().parent.parent
            cache_dir_resolved = (repo_root / raw_dir).resolve()
        # If caching is enabled, ensure the directory exists
        # regardless of backend availability
        if use_cache:
            cache_dir_resolved.mkdir(parents=True, exist_ok=True)

    if (
        use_cache
        and CachedSession is not None
        and SQLiteBackend is not None
        and cache_dir_resolved is not None
    ):
        cache_path = cache_dir_resolved / 'http_cache.sqlite'
        # Ensure SQLite DB file exists even before first cached response
        with contextlib.suppress(Exception):
            if not cache_path.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(cache_path) as _conn:
                    _conn.execute('PRAGMA journal_mode=WAL;')

        expire_td = timedelta(
            hours=max(0, int(getattr(settings.cache, 'expire_hours', 0)))
        )
        stale_hours = int(getattr(settings.cache, 'stale_if_error_hours', 0))
        stale_param: bool | timedelta
        stale_param = timedelta(hours=stale_hours) if stale_hours > 0 else False

        backend = cast('Any', SQLiteBackend)(
            str(cache_path),
            expire_after=expire_td,
        )
        session_ctx = cast('Any', CachedSession)(
            cache=backend,
            expire_after=expire_td,
            cache_control=bool(getattr(settings.cache, 'respect_cache_control', True)),
            stale_if_error=stale_param,
        )
    else:
        # Fallback to regular session if cache backend isn't available or disabled
        if use_cache and (CachedSession is None or SQLiteBackend is None):
            # Provide a hint if cache lib is not available
            pass
        elif not use_cache:
            pass
        session_ctx = aiohttp.ClientSession()

    async with session_ctx as client:

        async def bound_fetch(
            idx_xy: tuple[int, tuple[int, int]],
        ) -> tuple[int, Image.Image]:
            idx, (tx, ty) = idx_xy
            async with semaphore:
                img = await async_fetch_xyz_tile(
                    client=client,
                    api_key=api_key,
                    style_id=settings.style_id,
                    tile_size=settings.tile_size,
                    z=zoom,
                    x=tx,
                    y=ty,
                    use_retina=settings.use_retina,
                )
                await tile_progress.step(1)
                return idx, img

        tasks = [bound_fetch(pair) for pair in enumerate(tiles)]
        results = await asyncio.gather(*tasks)
        tile_progress.close()
        results.sort(key=lambda t: t[0])
        images: list[Image.Image] = [img for _, img in results]

    eff_tile_px = settings.tile_size * (2 if settings.use_retina else 1)
    result = assemble_and_crop(
        images=images,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        eff_tile_px=eff_tile_px,
        crop_rect=crop_rect,
    )

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
    )

    sp = LiveSpinner('Сохранение файла')
    sp.start()

    out_path = Path(output_path)
    out_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    fd = os.open(out_path, os.O_RDONLY)

    os.fsync(fd)

    os.close(fd)

    sp.stop('Сохранение файла: готово')

    return output_path
