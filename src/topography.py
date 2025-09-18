import asyncio
import math
from http import HTTPStatus
from io import BytesIO
from typing import Sequence

import aiohttp
from PIL import Image
from pyproj import CRS, Transformer

from constants import (
    EARTH_RADIUS_M,
    ELEV_MIN_RANGE_M,
    ELEV_PCTL_HI,
    ELEV_PCTL_LO,
    ELEVATION_COLOR_RAMP,
    MAPBOX_TERRAIN_RGB_PATH,
    EAST_VECTOR_SAMPLE_M,
    EPSG_SK42_GK_BASE,
    GK_FALSE_EASTING,
    GK_ZONE_CM_OFFSET_DEG,
    GK_ZONE_WIDTH_DEG,
    HTTP_5XX_MAX,
    HTTP_5XX_MIN,
    HTTP_BACKOFF_FACTOR,
    HTTP_RETRIES_DEFAULT,
    HTTP_TIMEOUT_DEFAULT,
    MAPBOX_STATIC_BASE,
    MERCATOR_MAX_SIN,
    RETINA_FACTOR,
    SK42_CODE,
    STATIC_SCALE,
    STATIC_SIZE_PX,
    TILE_SIZE,
    TILE_SIZE_512,
    WGS84_CODE,
    WORLD_LAT_MAX_DEG,
    WORLD_LNG_HALF_SPAN_DEG,
    WORLD_LNG_SPAN_DEG,
    XY_EPSILON,
)

# Географическая СК-42 (Pulkovo 1942)
crs_sk42_geog = CRS.from_epsg(SK42_CODE)
# Географическая WGS84
crs_wgs84 = CRS.from_epsg(WGS84_CODE)


def build_transformers_sk42(
    zone_from_lon: float,
    custom_helmert: tuple[float, float, float, float, float, float, float]
    | None = None,
) -> tuple[Transformer, Transformer, CRS]:
    """
    Собирает трансформеры.

    - СК‑42 географические <-> WGS84 (с учётом доступных параметров);
    - СК‑42 / Гаусса–Крюгера (EPSG:284xx) для выбранной 6-градусной зоны.
    """
    zone = int(
        math.floor((zone_from_lon + GK_ZONE_CM_OFFSET_DEG) / float(GK_ZONE_WIDTH_DEG))
        + 1,
    )
    zone = max(1, min(60, zone))
    try:
        # EPSG:284xx
        crs_sk42_gk = CRS.from_epsg(EPSG_SK42_GK_BASE + zone)
    except Exception:
        # Резервный вариант: построить СК‑42 / Гаусса–Крюгера (6°)
        # вручную, если EPSG недоступен
        # Центральный меридиан 6-градусной зоны
        lon0 = zone * GK_ZONE_WIDTH_DEG - GK_ZONE_CM_OFFSET_DEG
        proj4 = (
            f'+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 '
            f'+x_0={GK_FALSE_EASTING} +y_0=0 +ellps=krass +units=m +no_defs +type=crs'
        )
        crs_sk42_gk = CRS.from_proj4(proj4)

    if custom_helmert:
        dx, dy, dz, rx_as, ry_as, rz_as, ds_ppm = custom_helmert
        proj4_sk42_custom = CRS.from_proj4(
            (
                f'+proj=longlat +a=6378245.0 +rf=298.3 '
                f'+towgs84={dx},{dy},{dz},{rx_as},{ry_as},{rz_as},{ds_ppm} '
                '+no_defs'
            ),
        )
        t_sk42_to_wgs = Transformer.from_crs(
            proj4_sk42_custom,
            crs_wgs84,
            always_xy=True,
        )
        t_wgs_to_sk42 = Transformer.from_crs(
            crs_wgs84,
            proj4_sk42_custom,
            always_xy=True,
        )
    else:
        t_sk42_to_wgs = Transformer.from_crs(crs_sk42_geog, crs_wgs84, always_xy=True)
        t_wgs_to_sk42 = Transformer.from_crs(crs_wgs84, crs_sk42_geog, always_xy=True)

    return t_sk42_to_wgs, t_wgs_to_sk42, crs_sk42_gk


def meters_per_pixel(lat_deg: float, zoom: int, scale: int = STATIC_SCALE) -> float:
    """Возвращает метров на пиксель в проекции Mercator на заданной широте и зуме."""
    lat_rad = math.radians(lat_deg)
    return (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (
        TILE_SIZE * (2**zoom) * scale
    )


def latlng_to_pixel_xy(
    lat_deg: float,
    lng_deg: float,
    zoom: int,
) -> tuple[float, float]:
    """Преобразует WGS84 (lat, lng) в координаты «мира» (пиксели) Web Mercator."""
    siny = math.sin(math.radians(lat_deg))
    siny = min(max(siny, -MERCATOR_MAX_SIN), MERCATOR_MAX_SIN)
    world_size = TILE_SIZE * (2**zoom)
    x = (lng_deg + WORLD_LNG_HALF_SPAN_DEG) / WORLD_LNG_SPAN_DEG * world_size
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world_size
    return x, y


def pixel_xy_to_latlng(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Обратное преобразование: «мировые» пиксели -> WGS84 (lat, lng)."""
    world_size = TILE_SIZE * (2**zoom)
    lng = (x / world_size) * WORLD_LNG_SPAN_DEG - WORLD_LNG_HALF_SPAN_DEG
    merc_y = 0.5 - (y / world_size)
    lat = (
        WORLD_LAT_MAX_DEG
        - WORLD_LNG_SPAN_DEG * math.atan(math.exp(-merc_y * 2 * math.pi)) / math.pi
    )
    return lat, lng


def estimate_crop_size_px(
    center_lat: float,
    width_m: float,
    height_m: float,
    zoom: int,
    scale: int = STATIC_SCALE,
) -> tuple[int, int, int]:
    """Оценка итоговых размеров кадра (после crop, без припусков) в пикселях."""
    mpp = meters_per_pixel(center_lat, zoom, scale=scale)
    req_w_px = round(width_m / mpp)
    req_h_px = round(height_m / mpp)
    return req_w_px, req_h_px, req_w_px * req_h_px


def choose_zoom_with_limit(  # noqa: PLR0913
    center_lat: float,
    width_m: float,
    height_m: float,
    desired_zoom: int,
    scale: int,
    max_pixels: int,
) -> int:
    """Выбирает максимально возможный zoom, не превышающий max_pixels по площади."""
    zoom = desired_zoom
    while zoom >= 0:
        _, _, total = estimate_crop_size_px(center_lat, width_m, height_m, zoom, scale)
        if total <= max_pixels:
            return zoom
        zoom -= 1
    return 0


def compute_grid(  # noqa: PLR0913
    center_lat: float,
    center_lng: float,
    width_m: float,
    height_m: float,
    zoom: int,
    scale: int = STATIC_SCALE,
    tile_size_px: int = STATIC_SIZE_PX,
    pad_px: int = 0,
) -> tuple[
    list[tuple[float, float]],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int, int, int],
    tuple[float, float, float, int, int, int, int],
]:
    """
    Строит сетку центров статик-квадратов, и возвращает параметры для сборки.

    pad_px — припуск на сторону в пикселях холста для компенсации поворота/кропа.
    """
    eff_tile_px = tile_size_px * scale

    # Центр области в «мировых» пикселях Web Mercator
    cx, cy = latlng_to_pixel_xy(center_lat, center_lng, zoom)

    # Требуемые размеры (без припуска) в пикселях холста
    mpp = meters_per_pixel(center_lat, zoom, scale=scale)
    req_w_px = width_m / mpp
    req_h_px = height_m / mpp

    # Добавляем припуск
    pad_px = max(int(pad_px), 0)
    padded_w_px = req_w_px + 2 * pad_px
    padded_h_px = req_h_px + 2 * pad_px

    # Количество статик-тайлов
    tiles_x = max(1, math.ceil(padded_w_px / eff_tile_px))
    tiles_y = max(1, math.ceil(padded_h_px / eff_tile_px))

    # Размер холста и кроп расширенной области
    grid_w_px = tiles_x * eff_tile_px
    grid_h_px = tiles_y * eff_tile_px
    crop_x = round((grid_w_px - padded_w_px) / 2.0)
    crop_y = round((grid_h_px - padded_h_px) / 2.0)
    crop_w = round(padded_w_px)
    crop_h = round(padded_h_px)

    # Позиции тайлов в «мировых» координатах
    base_step_world = float(tile_size_px)
    grid_w_world = tiles_x * base_step_world
    grid_h_world = tiles_y * base_step_world
    origin_x_world = cx - (grid_w_world / 2.0) + (base_step_world / 2.0)
    origin_y_world = cy - (grid_h_world / 2.0) + (base_step_world / 2.0)

    centers: list[tuple[float, float]] = []
    for j in range(tiles_y):
        for i in range(tiles_x):
            center_x_world = origin_x_world + i * base_step_world
            center_y_world = origin_y_world + j * base_step_world
            lat, lng = pixel_xy_to_latlng(center_x_world, center_y_world, zoom)
            centers.append((lat, lng))

    map_params = (
        origin_x_world,
        origin_y_world,
        base_step_world,
        scale,
        crop_x,
        crop_y,
        zoom,
    )
    return (
        centers,
        (tiles_x, tiles_y),
        (grid_w_px, grid_h_px),
        (crop_x, crop_y, crop_w, crop_h),
        map_params,
    )


def effective_scale_for_xyz(tile_size: int, *, use_retina: bool) -> int:
    """Эффективный масштаб (мозаичных пикселей на 256 мировой пиксель)."""
    base = TILE_SIZE_512 if tile_size >= TILE_SIZE_512 else TILE_SIZE
    return (base // TILE_SIZE) * (RETINA_FACTOR if use_retina else 1)


def compute_xyz_coverage(  # noqa: PLR0913
    center_lat: float,
    center_lng: float,
    width_m: float,
    height_m: float,
    zoom: int,
    eff_scale: int,
    pad_px: int,
) -> tuple[
    list[tuple[int, int]],
    tuple[int, int],
    tuple[int, int, int, int],
    tuple[float, float, float, int, int, int, int],
]:
    """
    Вычисляет покрытие XYZ-тайлами для расширенной области и параметры сборки.

    Возвращает:
      - список тайлов (x, y) в порядке строк (y) и столбцов (x) с учётом wrap по x;
      - (count_x, count_y);
      - crop_rect = (crop_x, crop_y, crop_w, crop_h) в пикселях мозаики;
      - map_params совместимый с latlng_to_final_pixel.
    """
    # Центр в мировых пикселях (256 * 2^z)
    cx, cy = latlng_to_pixel_xy(center_lat, center_lng, zoom)

    # Требуемые размеры в пикселях мозаики
    mpp = meters_per_pixel(center_lat, zoom, scale=eff_scale)
    req_w_px = width_m / mpp
    req_h_px = height_m / mpp

    # Перевод в мировые пиксели (без учёта масштаба)
    req_w_world = req_w_px / eff_scale
    req_h_world = req_h_px / eff_scale
    pad_world = max(0, round(pad_px / eff_scale))
    padded_w_world = req_w_world + 2 * pad_world
    padded_h_world = req_h_world + 2 * pad_world

    x_min_world = cx - padded_w_world / 2.0
    y_min_world = cy - padded_h_world / 2.0
    x_max_world = cx + padded_w_world / 2.0
    y_max_world = cy + padded_h_world / 2.0

    # Диапазон тайлов XYZ (размер тайла = 256 мировых пикселей)
    t = 2**zoom
    x_min = math.floor(x_min_world / float(TILE_SIZE))
    y_min = math.floor(y_min_world / float(TILE_SIZE))
    x_max = math.floor((x_max_world - XY_EPSILON) / float(TILE_SIZE))
    y_max = math.floor((y_max_world - XY_EPSILON) / float(TILE_SIZE))

    # Ограничиваем y (clamp) и замыкаем x по модулю (wrap)
    y_min_clamped = max(0, min(t - 1, y_min))
    y_max_clamped = max(0, min(t - 1, y_max))

    # Количество тайлов по осям
    count_x = x_max - x_min + 1
    count_y = y_max_clamped - y_min_clamped + 1

    # Нормализованный x для построения списка (wrap по модулю t)
    tiles: list[tuple[int, int]] = []
    for y in range(y_min_clamped, y_max_clamped + 1):
        for i in range(count_x):
            x = (x_min + i) % t
            tiles.append((x, y))

    # Эффективный размер тайла в пикселях мозаики
    base_step_world = float(TILE_SIZE)
    int(base_step_world * eff_scale)

    # Смещения кропа в пикселях мозаики от (0,0) верхнего-левого тайла
    # Верхний-левый тайл в нашей сетке имеет индекс x_min (без мод), y_min_clamped
    crop_x = round((x_min_world - x_min * base_step_world) * eff_scale)
    crop_y = round((y_min_world - y_min_clamped * base_step_world) * eff_scale)
    crop_w = round(padded_w_world * eff_scale)
    crop_h = round(padded_h_world * eff_scale)

    # Параметры для latlng_to_final_pixel
    origin_x_world = x_min * base_step_world + base_step_world / 2.0
    origin_y_world = y_min_clamped * base_step_world + base_step_world / 2.0
    map_params = (
        origin_x_world,
        origin_y_world,
        base_step_world,
        eff_scale,
        crop_x,
        crop_y,
        zoom,
    )

    return tiles, (count_x, count_y), (crop_x, crop_y, crop_w, crop_h), map_params


async def async_fetch_xyz_tile(  # noqa: PLR0913
    client: aiohttp.ClientSession,
    api_key: str,
    style_id: str,
    tile_size: int,
    z: int,
    x: int,
    y: int,
    *,
    use_retina: bool,
    async_timeout: float = HTTP_TIMEOUT_DEFAULT,
    retries: int = HTTP_RETRIES_DEFAULT,
    backoff: float = HTTP_BACKOFF_FACTOR,
) -> Image.Image:
    """
    Загружает один XYZ тайл стиля Mapbox и возвращает PIL.Image (RGB).

    - URL: /styles/v1/{style_id}/tiles/{tileSize}/{z}/{x}/{y}{@2x}
    - Не добавляем токен в лог/ошибки; логируем путь без токена.
    - Обрабатываем 401/403/404 явно; 429/5xx с ретраями и экспоненциальной задержкой.
    """
    ts = TILE_SIZE_512 if tile_size >= TILE_SIZE_512 else TILE_SIZE
    scale_suffix = '@2x' if use_retina else ''
    path = f'{MAPBOX_STATIC_BASE}/{style_id}/tiles/{ts}/{z}/{x}/{y}{scale_suffix}'
    url = f'{path}?access_token={api_key}'

    def _fail(msg: str) -> None:
        raise RuntimeError(msg)

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            timeout = aiohttp.ClientTimeout(total=async_timeout)
            resp = await client.get(url, timeout=timeout)
            try:
                sc = resp.status
                if sc == HTTPStatus.OK:
                    data = await resp.read()
                    # Контент может быть png/jpg/webp — PIL откроет всё; конвертируем в RGB
                    return Image.open(BytesIO(data)).convert('RGB')
                if sc in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
                    msg = (
                        f'Доступ запрещён (HTTP {sc}). Проверьте токен и права. '
                        f'z/x/y={z}/{x}/{y} path={path}'
                    )
                    _fail(msg)
                if sc == HTTPStatus.NOT_FOUND:
                    msg = f'Ресурс не найден (404) для тайла z/x/y={z}/{x}/{y} path={path}'
                    _fail(msg)
                is_rate_or_5xx = (sc == HTTPStatus.TOO_MANY_REQUESTS) or (
                    HTTP_5XX_MIN <= sc < HTTP_5XX_MAX
                )
                if is_rate_or_5xx:
                    last_exc = RuntimeError(
                        f'HTTP {sc} при загрузке тайла z/x/y={z}/{x}/{y} path={path}',
                    )
                else:
                    last_exc = RuntimeError(
                        f'Неожиданный ответ HTTP {sc} для z/x/y={z}/{x}/{y} path={path}',
                    )
            finally:
                # Освобождение ресурсов ответа для обоих типов (aiohttp и CachedResponse)
                try:
                    close = getattr(resp, 'close', None)
                    if callable(close):
                        close()
                    release = getattr(resp, 'release', None)
                    if callable(release):
                        release()
                except Exception as e:
                    # Log and move on; we don't want cleanup issues to mask the real HTTP error
                    import logging

                    logging.getLogger(__name__).debug(
                        'Failed to cleanup HTTP response: %s', e, exc_info=True
                    )
        except Exception as e:
            last_exc = e
        await asyncio.sleep(backoff**attempt)
    msg = f'Не удалось загрузить тайл z/x/y={z}/{x}/{y}: {last_exc}'
    raise RuntimeError(msg)


async def async_fetch_terrain_rgb_tile(  # noqa: PLR0913
    client: aiohttp.ClientSession,
    api_key: str,
    z: int,
    x: int,
    y: int,
    *,
    use_retina: bool,
    async_timeout: float = HTTP_TIMEOUT_DEFAULT,
    retries: int = HTTP_RETRIES_DEFAULT,
    backoff: float = HTTP_BACKOFF_FACTOR,
) -> Image.Image:
    """
    Загружает один тайл Terrain-RGB (pngraw) и возвращает PIL.Image in RGB.

    URL: https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}{@2x}.pngraw?access_token=...
    Токен не логируется; в сообщениях используем путь без query.
    Поведение ошибок/ретраев аналогично async_fetch_xyz_tile.
    """
    scale_suffix = '@2x' if use_retina else ''
    path = f'{MAPBOX_TERRAIN_RGB_PATH}/{z}/{x}/{y}{scale_suffix}.pngraw'
    url = f'{path}?access_token={api_key}'

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            timeout = aiohttp.ClientTimeout(total=async_timeout)
            resp = await client.get(url, timeout=timeout)
            try:
                sc = resp.status
                if sc == HTTPStatus.OK:
                    data = await resp.read()
                    return Image.open(BytesIO(data)).convert('RGB')
                if sc in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
                    raise RuntimeError(
                        f'Доступ запрещён (HTTP {sc}) для terrain тайла z/x/y={z}/{x}/{y} path={path}',
                    )
                if sc == HTTPStatus.NOT_FOUND:
                    raise RuntimeError(
                        f'Ресурс не найден (404) для terrain тайла z/x/y={z}/{x}/{y} path={path}',
                    )
                is_rate_or_5xx = (sc == HTTPStatus.TOO_MANY_REQUESTS) or (
                    HTTP_5XX_MIN <= sc < HTTP_5XX_MAX
                )
                if is_rate_or_5xx:
                    last_exc = RuntimeError(
                        f'HTTP {sc} при загрузке terrain тайла z/x/y={z}/{x}/{y} path={path}',
                    )
                else:
                    last_exc = RuntimeError(
                        f'Неожиданный ответ HTTP {sc} для terrain z/x/y={z}/{x}/{y} path={path}',
                    )
            finally:
                try:
                    close = getattr(resp, 'close', None)
                    if callable(close):
                        close()
                    release = getattr(resp, 'release', None)
                    if callable(release):
                        release()
                except Exception:
                    pass
        except Exception as e:
            last_exc = e
        await asyncio.sleep(backoff**attempt)
    raise RuntimeError(f'Не удалось загрузить terrain тайл z/x/y={z}/{x}/{y}: {last_exc}')


def decode_terrain_rgb_to_elevation_m(img: Image.Image) -> list[list[float]]:
    """Декодирует Terrain-RGB картинку в двумерный массив высот (метры).

    elevation = -10000 + (R*256*256 + G*256 + B) * 0.1

    Возвращает список строк для экономии зависимостей (без numpy).
    """
    w, h = img.size
    pix = img.load()
    rows: list[list[float]] = []
    for y in range(h):
        row: list[float] = []
        for x in range(w):
            r, g, b = pix[x, y][:3]
            elev = -10000.0 + (r * 256 * 256 + g * 256 + b) * 0.1
            row.append(elev)
        rows.append(row)
    return rows


def assemble_dem(  # noqa: PLR0913
    tiles_data: list[list[list[float]]],
    tiles_x: int,
    tiles_y: int,
    eff_tile_px: int,
    crop_rect: tuple[int, int, int, int],
) -> list[list[float]]:
    """
    Сшивает список DEM тайлов (в порядке строк) в единый DEM и применяет crop_rect.

    Возвращает 2D список float метров.
    """
    full_w = tiles_x * eff_tile_px
    full_h = tiles_y * eff_tile_px
    # Инициализация полотна
    canvas: list[list[float]] = [[0.0] * full_w for _ in range(full_h)]

    # Размещение тайлов
    idx = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = tiles_data[idx]
            idx += 1
            # Копируем по строкам
            base_y = ty * eff_tile_px
            base_x = tx * eff_tile_px
            for y in range(eff_tile_px):
                row_src = tile[y]
                row_dst = canvas[base_y + y]
                row_dst[base_x : base_x + eff_tile_px] = row_src[:eff_tile_px]

    cx, cy, cw, ch = crop_rect
    # Обрезка
    cropped: list[list[float]] = [row[cx : cx + cw] for row in canvas[cy : cy + ch]]
    # Освободить полотно
    canvas.clear()
    return cropped


def compute_percentiles(values: Sequence[float], p_lo: float, p_hi: float) -> tuple[float, float]:
    """Простая перцентильная оценка без numpy (O(n log n))."""
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0, 1.0
    def idx_of(p: float) -> int:
        p = max(0.0, min(100.0, p))
        k = int(round((p / 100.0) * (n - 1)))
        return max(0, min(n - 1, k))
    return vals[idx_of(p_lo)], vals[idx_of(p_hi)]


def colorize_dem_to_image(
    dem: list[list[float]],
    p_lo: float = ELEV_PCTL_LO,
    p_hi: float = ELEV_PCTL_HI,
    min_range_m: float = ELEV_MIN_RANGE_M,
) -> Image.Image:
    """Преобразует DEM в цветное изображение по заданной палитре с перцентильной нормализацией."""
    h = len(dem)
    w = len(dem[0]) if h else 0
    # Выборка для перцентилей: используем равномерную подвыборку, чтобы не хранить всё
    samples: list[float] = []
    if h and w:
        step_y = max(1, h // 200)
        step_x = max(1, w // 200)
        for y in range(0, h, step_y):
            row = dem[y]
            for x in range(0, w, step_x):
                samples.append(row[x])
    lo, hi = compute_percentiles(samples or [v for row in dem for v in row][:10000], p_lo, p_hi)
    if hi - lo < min_range_m:
        mid = (lo + hi) / 2.0
        lo = mid - min_range_m / 2.0
        hi = mid + min_range_m / 2.0
    inv = 1.0 / (hi - lo) if hi > lo else 0.0

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    # Build LUT once per image for fast palette mapping
    LUT_SIZE = 2048
    _LUT: list[tuple[int, int, int]] = []
    ramp = ELEVATION_COLOR_RAMP
    for i in range(LUT_SIZE):
        tt = i / (LUT_SIZE - 1)
        for j in range(1, len(ramp)):
            t0, c0 = ramp[j - 1]
            t1, c1 = ramp[j]
            if tt <= t1 or j == len(ramp) - 1:
                local = 0.0 if t1 == t0 else (tt - t0) / (t1 - t0)
                r = int(round(lerp(c0[0], c1[0], local)))
                g = int(round(lerp(c0[1], c1[1], local)))
                b = int(round(lerp(c0[2], c1[2], local)))
                _LUT.append((r, g, b))
                break

    def color_at(t: float) -> tuple[int, int, int]:
        if t <= 0.0:
            return _LUT[0]
        if t >= 1.0:
            return _LUT[-1]
        idx = int(t * (LUT_SIZE - 1))
        return _LUT[idx]

    # Build entire image buffer as bytes (scanlines) and create Image from bytes
    buf = bytearray(w * h * 3)
    k = 0
    for y in range(h):
        row = dem[y]
        for x in range(w):
            t = (row[x] - lo) * inv
            r, g, b = color_at(t)
            buf[k] = r
            buf[k + 1] = g
            buf[k + 2] = b
            k += 3
    return Image.frombytes('RGB', (w, h), bytes(buf))


def latlng_to_final_pixel(
    lat: float,
    lng: float,
    map_params: tuple[float, float, float, int, int, int, int],
) -> tuple[float, float]:
    """Преобразует точку WGS84 в координаты пикселей итогового изображения."""
    origin_x_world, origin_y_world, base_step_world, scale, crop_x, crop_y, zoom = (
        map_params
    )
    wx, wy = latlng_to_pixel_xy(lat, lng, zoom)
    canvas_x = (wx - (origin_x_world - base_step_world / 2.0)) * scale
    canvas_y = (wy - (origin_y_world - base_step_world / 2.0)) * scale
    final_x = canvas_x - crop_x
    final_y = canvas_y - crop_y
    return final_x, final_y


def compute_rotation_deg_for_east_axis(
    center_lat_sk42: float,
    center_lng_sk42: float,
    map_params: tuple[float, float, float, int, int, int, int],
    crs_sk42_gk: CRS,
    t_sk42_to_wgs: Transformer,
) -> float:
    """
    Вычисляет угол поворота оси «восток» СК‑42/ГК.

    Поворачивает исходное изображение так, чтобы ось «восток» (X в СК‑42/ГК)
    стала строго горизонтальной.
    """
    t_sk42gk_from_sk42 = Transformer.from_crs(
        crs_sk42_geog,
        crs_sk42_gk,
        always_xy=True,
    )
    t_sk42_from_sk42gk = Transformer.from_crs(
        crs_sk42_gk,
        crs_sk42_geog,
        always_xy=True,
    )

    x0_gk, y0_gk = t_sk42gk_from_sk42.transform(center_lng_sk42, center_lat_sk42)
    # Точка на EAST_VECTOR_SAMPLE_M метров восточнее
    x1_gk, y1_gk = x0_gk + EAST_VECTOR_SAMPLE_M, y0_gk

    lon_s0, lat_s0 = t_sk42_from_sk42gk.transform(x0_gk, y0_gk)
    lon_w0, lat_w0 = t_sk42_to_wgs.transform(lon_s0, lat_s0)
    lon_s1, lat_s1 = t_sk42_from_sk42gk.transform(x1_gk, y1_gk)
    lon_w1, lat_w1 = t_sk42_to_wgs.transform(lon_s1, lat_s1)

    p0x, p0y = latlng_to_final_pixel(lat_w0, lon_w0, map_params)
    p1x, p1y = latlng_to_final_pixel(lat_w1, lon_w1, map_params)
    vx, vy = (p1x - p0x), (p1y - p0y)

    angle_rad = math.atan2(vy, vx)
    return -math.degrees(angle_rad)
