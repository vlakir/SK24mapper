import asyncio
import math
from http import HTTPStatus
from io import BytesIO

import aiohttp
from PIL import Image
from pyproj import CRS, Transformer

from constants import (
    CURRENT_PROFILE,
    EARTH_RADIUS_M,
    HTTP_5XX_MAX,
    HTTP_5XX_MIN,
    MAPBOX_STATIC_BASE,
    RETINA_FACTOR,
    SK42_CODE,
    STATIC_SCALE,
    STATIC_SIZE_PX,
    TILE_SIZE,
    TILE_SIZE_512,
    WGS84_CODE,
)
from profiles import load_profile

settings = load_profile(CURRENT_PROFILE)

# ------------------------------
# СК‑42 <-> WGS84 (pyproj)
# ------------------------------
crs_sk42_geog = CRS.from_epsg(SK42_CODE)  # Географическая СК-42 (Pulkovo 1942)
crs_wgs84 = CRS.from_epsg(WGS84_CODE)  # Географическая WGS84

# Центр и размеры области в системе GK:
# ширина по ΔX (горизонталь), высота по ΔY (вертикаль)
center_x_sk42_gk = (settings.bottom_left_x_sk42_gk + settings.top_right_x_sk42_gk) / 2
center_y_sk42_gk = (settings.bottom_left_y_sk42_gk + settings.top_right_y_sk42_gk) / 2
width_m = settings.top_right_x_sk42_gk - settings.bottom_left_x_sk42_gk
height_m = settings.top_right_y_sk42_gk - settings.bottom_left_y_sk42_gk


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
    zone = int(math.floor((zone_from_lon + 3) / 6.0) + 1)
    zone = max(1, min(60, zone))
    crs_sk42_gk = CRS.from_epsg(28400 + zone)  # EPSG:284xx

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


# ------------------------------
# Geo utils (Web Mercator на WGS84)
# ------------------------------
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
    siny = min(max(siny, -0.9999), 0.9999)
    world_size = TILE_SIZE * (2**zoom)
    x = (lng_deg + 180.0) / 360.0 * world_size
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world_size
    return x, y


def pixel_xy_to_latlng(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Обратное преобразование: «мировые» пиксели -> WGS84 (lat, lng)."""
    world_size = TILE_SIZE * (2**zoom)
    lng = (x / world_size) * 360.0 - 180.0
    merc_y = 0.5 - (y / world_size)
    lat = 90.0 - 360.0 * math.atan(math.exp(-merc_y * 2 * math.pi)) / math.pi
    return lat, lng


# ------------------------------
# Вспомогательные оценки размера
# ------------------------------
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


# ------------------------------
# Расчёт сетки тайлов (с припуском под поворот)
# ------------------------------
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


# ------------------------------
# XYZ покрытие и преобразования
# ------------------------------


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
    x_min = math.floor(x_min_world / 256.0)
    y_min = math.floor(y_min_world / 256.0)
    x_max = math.floor((x_max_world - 1e-9) / 256.0)
    y_max = math.floor((y_max_world - 1e-9) / 256.0)

    # Кламп y, wrap x
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
    base_step_world = 256.0
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


# ------------------------------
# Асинхронная загрузка XYZ тайла (Mapbox Styles)
# ------------------------------
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
    async_timeout: float = 20.0,
    retries: int = 4,
    backoff: float = 1.6,
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
            sc = resp.status
            if sc == HTTPStatus.OK:
                data = await resp.read()
                # Content may be png/jpg/webp — PIL opens all; convert to RGB
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
                    f'HTTP {sc} при загрузке тайла z/x/y={z}/{x}/{y} path={path}'
                )
            else:
                last_exc = RuntimeError(
                    f'Неожиданный ответ HTTP {sc} для z/x/y={z}/{x}/{y} path={path}'
                )
        except Exception as e:
            last_exc = e
        await asyncio.sleep(backoff**attempt)
    msg = f'Не удалось загрузить тайл z/x/y={z}/{x}/{y}: {last_exc}'
    raise RuntimeError(msg)


# ------------------------------
# Преобразование lat/lng -> пиксели итогового кадра
# ------------------------------
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


# ------------------------------
# Угол поворота по «востоку» СК‑42/ГК
# ------------------------------
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
    x1_gk, y1_gk = x0_gk + 200.0, y0_gk  # точка на 200 м восточнее

    lon_s0, lat_s0 = t_sk42_from_sk42gk.transform(x0_gk, y0_gk)
    lon_w0, lat_w0 = t_sk42_to_wgs.transform(lon_s0, lat_s0)
    lon_s1, lat_s1 = t_sk42_from_sk42gk.transform(x1_gk, y1_gk)
    lon_w1, lat_w1 = t_sk42_to_wgs.transform(lon_s1, lat_s1)

    p0x, p0y = latlng_to_final_pixel(lat_w0, lon_w0, map_params)
    p1x, p1y = latlng_to_final_pixel(lat_w1, lon_w1, map_params)
    vx, vy = (p1x - p0x), (p1y - p0y)

    angle_rad = math.atan2(vy, vx)
    return -math.degrees(angle_rad)
