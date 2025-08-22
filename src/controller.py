# ------------------------------
# Основной асинхронный процесс
# ------------------------------
import asyncio
import os
from pathlib import Path

import httpx
from PIL import Image
from pyproj import CRS, Transformer

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
    STATIC_SCALE,
    STATIC_SIZE_PX,
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
    async_fetch_static_map,
    build_transformers_sk42,
    choose_zoom_with_limit,
    compute_grid,
    compute_rotation_deg_for_east_axis,
    crs_sk42_geog,
    estimate_crop_size_px,
)

settings = load_profile(CURRENT_PROFILE)


async def download_satellite_rectangle(  # noqa: PLR0915, PLR0913
    center_x_sk42_gk: float,
    center_y_sk42_gk: float,
    width_m: float,
    height_m: float,
    api_key: str,
    output_path: str,
    max_zoom: int = MAX_ZOOM,
    scale: int = STATIC_SCALE,
    static_size_px: int = STATIC_SIZE_PX,
) -> str:
    """Полный конвейер."""
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
        scale=scale,
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
        scale,
    )
    sp.stop('Подготовка: размер оценён')

    sp = LiveSpinner('Подготовка: расчёт сетки')
    sp.start()
    base_pad = round(min(target_w_px, target_h_px) * ROTATION_PAD_RATIO)
    pad_px = max(base_pad, ROTATION_PAD_MIN_PX)
    centers, (tiles_x, tiles_y), (_gw, _gh), crop_rect, map_params = compute_grid(
        center_lat=center_lat_wgs,
        center_lng=center_lng_wgs,
        width_m=width_m,
        height_m=height_m,
        zoom=zoom,
        scale=scale,
        tile_size_px=static_size_px,
        pad_px=pad_px,
    )
    sp.stop('Подготовка: сетка рассчитана')

    tile_progress = ConsoleProgress(total=len(centers), label='Загрузка тайлов')
    semaphore = asyncio.Semaphore(ASYNC_MAX_CONCURRENCY)
    async with httpx.AsyncClient(http2=True) as client:

        async def bound_fetch(
            idx_lat_lng: tuple[int, tuple[float, float]],
        ) -> tuple[int, Image.Image]:
            idx, (lt, ln) = idx_lat_lng
            async with semaphore:
                img = await async_fetch_static_map(
                    client=client,
                    lat=lt,
                    lng=ln,
                    zoom=zoom,
                    api_key=api_key,
                    size_px=static_size_px,
                    scale=scale,
                    maptype='satellite',
                )
                await tile_progress.step(1)
                return idx, img

        tasks = [bound_fetch(pair) for pair in enumerate(centers)]
        results = await asyncio.gather(*tasks)
        tile_progress.close()
        results.sort(key=lambda t: t[0])
        images: list[Image.Image] = [img for _, img in results]

    eff_tile_px = static_size_px * scale
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
