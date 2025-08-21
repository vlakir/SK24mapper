# Python

import math
import os
import sys
import time
import asyncio
import threading
from string import Formatter
from typing import Tuple, List, Optional

import httpx
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pyproj import CRS, Transformer


# ------------------------------
# Утилита: загрузка переменных окружения из .env (без внешних зависимостей)
# ------------------------------
def load_dotenv(path: str = ".env") -> None:
    """
    Простейший загрузчик .env:
    - поддерживает строки вида KEY=VALUE
    - игнорирует пустые строки и комментарии (#...)
    - не экранирует и не обрезает кавычки; используется «как есть»
    """
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Убираем обрамляющие кавычки, если есть
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                os.environ[key] = val
    except Exception:
        # В случае ошибок парсинга просто продолжаем без падения
        pass


# ------------------------------
# Константы и настройки
# ------------------------------
GOOGLE_STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"  # Базовый URL Google Static Maps API
STATIC_SIZE_PX = 640              # Базовый размер тайла по одной стороне (px) для statics (максимум: обычно 640)
STATIC_SCALE = 2                  # Масштаб 2 => фактический тайл ~1280x1280 пикселей
MAX_ZOOM = 18                     # Максимальный уровень приближения (переменная, подбирается автоматически вниз при больших размерах)
EARTH_RADIUS_M = 6378137.0        # Радиус Земли для Web Mercator (метры)
TILE_SIZE = 256                   # Базовый размер тайла Web Mercator (пикселей)

ASYNC_MAX_CONCURRENCY = 20        # Максимальное число параллельных HTTP-запросов
MAX_OUTPUT_PIXELS = 150_000_000   # Ограничение на итоговое число пикселей результирующего кадра (без припуска)
PIL_DISABLE_LIMIT = False         # Отключить защиту Pillow от «бомб декомпрессии» (используйте с осторожностью)

ROTATION_PAD_RATIO = 0.07         # Доля припуска на сторону под поворот (7% от меньшей стороны)
ROTATION_PAD_MIN_PX = 128         # Минимальный припуск на сторону (px)

# ------------------------------
# Входные параметры области съемки (в СК‑42)
# ------------------------------
API_KEY_ENV_VAR = "API_KEY"       # Имя переменной окружения с API ключом


# Координаты области по углам (СК-42 Гаусса-Крюгера, метры)

# Старшие разряды
FROM_X_HIGH = 54
FROM_Y_HIGH = 74
TO_X_HIGH = 54
TO_Y_HIGH = 74

# Младшие разряды
FROM_X_LOW = 14
FROM_Y_LOW = 43
TO_X_LOW = 18
TO_Y_LOW = 49

ADDITIVE_RATIO = 0.3

# X координата левого нижнего угла
BOTTOM_LEFT_X_SK42_GK = 1e3 * (FROM_X_LOW - ADDITIVE_RATIO) + 1e5 * FROM_X_HIGH
# Y координата левого нижнего угла
BOTTOM_LEFT_Y_SK42_GK = 1e3 * (FROM_Y_LOW - ADDITIVE_RATIO) + 1e5 * FROM_Y_HIGH
# X координата правого верхнего угла
TOP_RIGHT_X_SK42_GK = 1e3 * (TO_X_LOW + ADDITIVE_RATIO) + 1e5 * TO_X_HIGH
# Y координата правого верхнего угла
TOP_RIGHT_Y_SK42_GK = 1e3 * (TO_Y_LOW + ADDITIVE_RATIO) + 1e5 * TO_Y_HIGH


# # Координаты области по углам (СК-42 Гаусса-Крюгера, метры)
# BOTTOM_LEFT_Y_SK42_GK = 74_43_391   # X координата левого нижнего угла
# BOTTOM_LEFT_X_SK42_GK = 54_14_265   # Y координата левого нижнего угла
# TOP_RIGHT_Y_SK42_GK = 74_49_391     # X координата правого верхнего угла
# TOP_RIGHT_X_SK42_GK = 54_18_265     # Y координата правого верхнего угла

OUTPUT_PATH = "./moscow_sat.png"  # Путь к итоговому файлу
ZOOM = MAX_ZOOM                   # Желательный зум (будет снижен автоматически при превышении лимитов)

# ------------------------------
# Параметры сетки и подписей
# ------------------------------
GRID_STEP_M = 1000                     # Шаг километровой сетки (метры)
GRID_COLOR = (0, 0, 0)                 # Цвет линий сетки (RGB)
GRID_WIDTH_PX = 20                     # Толщина линий сетки (px)

GRID_FONT_SIZE = 86                    # Размер шрифта подписей (px)
GRID_FONT_PATH: Optional[str] = None   # Путь к TTF/OTF обычного шрифта (если None — использовать DejaVu из Pillow)
GRID_TEXT_COLOR = (0, 0, 0)            # Цвет текста подписи (RGB)
GRID_TEXT_OUTLINE_COLOR = (255, 255, 255)  # Цвет обводки текста (RGB)
GRID_TEXT_OUTLINE_WIDTH = 2            # Толщина обводки текста (px)
GRID_TEXT_MARGIN = 43                   # Отступ подписи от края изображения (px)

GRID_LABEL_BG_COLOR = (255, 255, 0)    # Цвет жёлтой подложки под подписью (RGB)
GRID_LABEL_BG_PADDING = 6              # Внутренний отступ подложки вокруг текста (px)

GRID_FONT_BOLD: bool = True                 # Использовать жирный шрифт для подписей (если доступен)
GRID_FONT_PATH_BOLD: Optional[str] = None   # Путь к жирному шрифту TTF/OTF (если None — DejaVuSans-Bold.ttf)

# ------------------------------
# Полупрозрачная белая маска на карту (не влияет на сетку/подписи)
# ------------------------------
ENABLE_WHITE_MASK = True              # Включить наложение белой маски поверх карты
MASK_OPACITY = 0.35                   # Прозрачность белой маски (0.0 — прозрачная, 1.0 — непрозрачная)


# ------------------------------
# СК‑42 <-> WGS84 (pyproj)
# ------------------------------
CRS_SK42_GEOG = CRS.from_epsg(4284)   # Географическая СК-42 (Pulkovo 1942)
CRS_WGS84 = CRS.from_epsg(4326)       # Географическая WGS84

# Для обратной совместимости - вычисляем центр и размеры из углов
center_y_sk42_gk = (BOTTOM_LEFT_Y_SK42_GK + TOP_RIGHT_Y_SK42_GK) / 2
center_x_sk42_gk = (BOTTOM_LEFT_X_SK42_GK + TOP_RIGHT_X_SK42_GK) / 2
WIDTH_M = TOP_RIGHT_Y_SK42_GK - BOTTOM_LEFT_Y_SK42_GK
HEIGHT_M = TOP_RIGHT_X_SK42_GK - BOTTOM_LEFT_X_SK42_GK


def build_transformers_sk42(
    zone_from_lon: float,
    custom_helmert: Tuple[float, float, float, float, float, float, float] | None = None
) -> Tuple[Transformer, Transformer, CRS]:
    """
    Собирает трансформеры:
    - СК‑42 географические <-> WGS84 (с учётом доступных параметров);
    - СК‑42 / Гаусса–Крюгера (EPSG:284xx) для выбранной 6-градусной зоны.
    """
    zone = int(math.floor((zone_from_lon + 3) / 6.0) + 1)
    zone = max(1, min(60, zone))
    crs_sk42_gk = CRS.from_epsg(28400 + zone)  # EPSG:284xx

    if custom_helmert:
        dx, dy, dz, rx_as, ry_as, rz_as, ds_ppm = custom_helmert
        proj4_sk42_custom = CRS.from_proj4(
            f"+proj=longlat +a=6378245.0 +rf=298.3 +towgs84={dx},{dy},{dz},{rx_as},{ry_as},{rz_as},{ds_ppm} +no_defs"
        )
        t_sk42_to_wgs = Transformer.from_crs(proj4_sk42_custom, CRS_WGS84, always_xy=True)
        t_wgs_to_sk42 = Transformer.from_crs(CRS_WGS84, proj4_sk42_custom, always_xy=True)
    else:
        t_sk42_to_wgs = Transformer.from_crs(CRS_SK42_GEOG, CRS_WGS84, always_xy=True)
        t_wgs_to_sk42 = Transformer.from_crs(CRS_WGS84, CRS_SK42_GEOG, always_xy=True)

    return t_sk42_to_wgs, t_wgs_to_sk42, crs_sk42_gk


# ------------------------------
# Geo utils (Web Mercator на WGS84)
# ------------------------------
def meters_per_pixel(lat_deg: float, zoom: int, scale: int = STATIC_SCALE) -> float:
    """
    Возвращает метров на пиксель в проекции Web Mercator на заданной широте и уровне зума.
    """
    lat_rad = math.radians(lat_deg)
    return (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS_M) / (TILE_SIZE * (2 ** zoom) * scale)

def latlng_to_pixel_xy(lat_deg: float, lng_deg: float, zoom: int) -> Tuple[float, float]:
    """
    Преобразует WGS84 (lat, lng) в координаты «мира» (пиксели) Web Mercator.
    """
    siny = math.sin(math.radians(lat_deg))
    siny = min(max(siny, -0.9999), 0.9999)
    world_size = TILE_SIZE * (2 ** zoom)
    x = (lng_deg + 180.0) / 360.0 * world_size
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world_size
    return x, y

def pixel_xy_to_latlng(x: float, y: float, zoom: int) -> Tuple[float, float]:
    """
    Обратное преобразование: «мировые» пиксели -> WGS84 (lat, lng).
    """
    world_size = TILE_SIZE * (2 ** zoom)
    lng = (x / world_size) * 360.0 - 180.0
    merc_y = 0.5 - (y / world_size)
    lat = 90.0 - 360.0 * math.atan(math.exp(-merc_y * 2 * math.pi)) / math.pi
    return lat, lng


# ------------------------------
# Единая строка прогресса + спиннер (для «монолитных» шагов)
# ------------------------------
_PROGRESS_SINGLE_LINE = True                 # Рисовать всё в одной строке
_LAST_PROGRESS_LINE_LEN = 0                  # Длина последней строки прогресса
_SINGLE_LINE_LOCK = threading.Lock()         # Блокировка на вывод в единую строку

def _clear_line():
    """
    Полностью очистить текущую строку прогресса.
    """
    global _LAST_PROGRESS_LINE_LEN
    with _SINGLE_LINE_LOCK:
        if _PROGRESS_SINGLE_LINE and _LAST_PROGRESS_LINE_LEN > 0:
            sys.stdout.write("\r" + " " * _LAST_PROGRESS_LINE_LEN + "\r")
            sys.stdout.flush()
            _LAST_PROGRESS_LINE_LEN = 0

def _write_line(msg: str):
    """
    Перерисовать текущую строку прогресса.
    """
    global _LAST_PROGRESS_LINE_LEN
    with _SINGLE_LINE_LOCK:
        if _PROGRESS_SINGLE_LINE:
            pad = max(0, _LAST_PROGRESS_LINE_LEN - len(msg))
            sys.stdout.write("\r" + msg + (" " * pad))
        else:
            sys.stdout.write("\r" + msg)
        sys.stdout.flush()
        _LAST_PROGRESS_LINE_LEN = len(msg)

class LiveSpinner:
    """
    «Крутилка» для операций, у которых нет естественных шагов (поворот, сохранение и т.п.).
    """
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    def __init__(self, label: str = "Выполнение", interval: float = 0.1):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        _clear_line()
        i = 0
        while not self._stop.is_set():
            msg = f"{self.label}: {self.frames[i % len(self.frames)]}"
            _write_line(msg)
            time.sleep(self.interval)
            i += 1
    def start(self):
        self._th.start()
    def stop(self, final_message: str | None = None):
        self._stop.set()
        self._th.join()
        if final_message is not None:
            _write_line(final_message)

class ConsoleProgress:
    """
    Прогресс-бар для пошаговых операций (загрузка тайлов, склейка, рисование сетки).
    """
    def __init__(self, total: int, label: str = "Прогресс"):
        self.total = max(1, int(total))
        self.done = 0
        self.start = time.monotonic()
        self.label = label
        try:
            self._lock = asyncio.Lock()
        except Exception:
            self._lock = None
        _clear_line()
        self._render()  # показать 0%
    def _format_eta(self, remaining: float) -> str:
        if remaining is None or remaining == float("inf"):
            return "--:--"
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
    def _render(self):
        elapsed = max(1e-6, time.monotonic() - self.start)
        rps = self.done / elapsed
        remaining = (self.total - self.done) / rps if rps > 0 else float("inf")
        bar_len = 30
        filled = int(bar_len * self.done / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)
        msg = f"{self.label}: [{bar}] {self.done}/{self.total} | {rps:4.1f}/s | ETA {self._format_eta(remaining)}"
        _write_line(msg)
    def step_sync(self, n: int = 1):
        self.done = min(self.total, self.done + n)
        self._render()
    async def step(self, n: int = 1):
        if self._lock is not None:
            async with self._lock:
                self.done = min(self.total, self.done + n)
                self._render()
        else:
            self.step_sync(n)
    def close(self):
        sys.stdout.flush()


# ------------------------------
# Вспомогательные оценки размера
# ------------------------------
def estimate_crop_size_px(
    center_lat: float,
    width_m: float,
    height_m: float,
    zoom: int,
    scale: int = STATIC_SCALE,
) -> Tuple[int, int, int]:
    """
    Оценка итоговых размеров кадра (после crop, без припусков) в пикселях.
    """
    mpp = meters_per_pixel(center_lat, zoom, scale=scale)
    req_w_px = int(round(width_m / mpp))
    req_h_px = int(round(height_m / mpp))
    return req_w_px, req_h_px, req_w_px * req_h_px

def choose_zoom_with_limit(
    center_lat: float,
    width_m: float,
    height_m: float,
    desired_zoom: int,
    scale: int,
    max_pixels: int,
) -> int:
    """
    Выбирает максимально возможный zoom, не превышающий max_pixels по площади кадра.
    """
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
def compute_grid(
    center_lat: float,
    center_lng: float,
    width_m: float,
    height_m: float,
    zoom: int,
    scale: int = STATIC_SCALE,
    tile_size_px: int = STATIC_SIZE_PX,
    pad_px: int = 0,
) -> Tuple[
    List[Tuple[float, float]],
    Tuple[int, int],
    Tuple[int, int],
    Tuple[int, int],
    Tuple[float, float, float, int, int, int, int]
]:
    """
    Строит сетку центров статик-квадратов, и возвращает параметры для последующей сборки.
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
    crop_x = int(round((grid_w_px - padded_w_px) / 2.0))
    crop_y = int(round((grid_h_px - padded_h_px) / 2.0))
    crop_w = int(round(padded_w_px))
    crop_h = int(round(padded_h_px))

    # Позиции тайлов в «мировых» координатах
    base_step_world = tile_size_px
    grid_w_world = tiles_x * base_step_world
    grid_h_world = tiles_y * base_step_world
    origin_x_world = cx - (grid_w_world / 2.0) + (base_step_world / 2.0)
    origin_y_world = cy - (grid_h_world / 2.0) + (base_step_world / 2.0)

    centers: List[Tuple[float, float]] = []
    for j in range(tiles_y):
        for i in range(tiles_x):
            center_x_world = origin_x_world + i * base_step_world
            center_y_world = origin_y_world + j * base_step_world
            lat, lng = pixel_xy_to_latlng(center_x_world, center_y_world, zoom)
            centers.append((lat, lng))

    map_params = (origin_x_world, origin_y_world, base_step_world, scale, crop_x, crop_y, zoom)
    return centers, (tiles_x, tiles_y), (grid_w_px, grid_h_px), (crop_x, crop_y, crop_w, crop_h), map_params


# ------------------------------
# Асинхронная загрузка тайла
# ------------------------------
async def async_fetch_static_map(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    zoom: int,
    api_key: str,
    size_px: int = STATIC_SIZE_PX,
    scale: int = STATIC_SCALE,
    maptype: str = "satellite",
    timeout: float = 20.0,
    retries: int = 3,
    backoff: float = 1.5,
) -> Image.Image:
    """
    Загружает один статик-тайл Google Static Maps и возвращает PIL.Image (RGB).
    """
    params = {
        "center": f"{lat:.8f},{lng:.8f}",
        "zoom": str(zoom),
        "size": f"{size_px}x{size_px}",
        "scale": str(scale),
        "maptype": maptype,
        "format": "png",
        "key": api_key,
    }
    last_exc = None
    for attempt in range(retries):
        try:
            resp = await client.get(GOOGLE_STATIC_MAPS_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}...")
        except Exception as e:
            last_exc = e
        await asyncio.sleep(backoff ** attempt)
    raise RuntimeError(f"Не удалось загрузить карту: {last_exc}")


# ------------------------------
# Склейка и обрезка (пошаговый прогресс)
# ------------------------------
def assemble_and_crop(
    images: List[Image.Image],
    tiles_x: int,
    tiles_y: int,
    eff_tile_px: int,
    crop_rect: Tuple[int, int, int, int],
) -> Image.Image:
    """
    Склеивает все тайлы в общий холст и обрезает «расширенную» область (с припуском).
    """
    grid_w_px = tiles_x * eff_tile_px
    grid_h_px = tiles_y * eff_tile_px

    paste_progress = ConsoleProgress(total=tiles_x * tiles_y, label="Склейка тайлов")
    canvas = Image.new("RGB", (grid_w_px, grid_h_px))
    idx = 0
    for j in range(tiles_y):
        for i in range(tiles_x):
            img = images[idx]
            if img.size != (eff_tile_px, eff_tile_px):
                img = img.resize((eff_tile_px, eff_tile_px), Image.Resampling.LANCZOS)
            canvas.paste(img, (i * eff_tile_px, j * eff_tile_px))
            idx += 1
            paste_progress.step_sync(1)
    paste_progress.close()

    crop_x, crop_y, crop_w, crop_h = crop_rect
    crop_progress = ConsoleProgress(total=1, label="Обрезка (расширенная область)")
    out = canvas.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    crop_progress.step_sync(1)
    crop_progress.close()
    return out


# ------------------------------
# Преобразование lat/lng -> пиксели итогового кадра
# ------------------------------
def latlng_to_final_pixel(lat: float, lng: float, map_params: Tuple[float, float, float, int, int, int, int]) -> Tuple[float, float]:
    """
    Преобразует точку WGS84 в координаты пикселей итогового изображения.
    """
    origin_x_world, origin_y_world, base_step_world, scale, crop_x, crop_y, zoom = map_params
    wx, wy = latlng_to_pixel_xy(lat, lng, zoom)
    canvas_x = (wx - (origin_x_world - base_step_world / 2.0)) * scale
    canvas_y = (wy - (origin_y_world - base_step_world / 2.0)) * scale
    final_x = canvas_x - crop_x
    final_y = canvas_y - crop_y
    return final_x, final_y


# ------------------------------
# Поворот (спиннер) и центр‑кроп (спиннер)
# ------------------------------
def rotate_keep_size(img: Image.Image, angle_deg: float, fill=(255, 255, 255)) -> Image.Image:
    """
    Поворачивает изображение с expand=True и центр-кропает к исходному размеру,
    чтобы избежать «срезанных» углов.
    """
    spinner = LiveSpinner("Поворот карты")
    spinner.start()
    try:
        w, h = img.size
        rotated = img.rotate(
            angle=angle_deg,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=fill
        )
        rw, rh = rotated.size
        cx, cy = rw // 2, rh // 2
        left = int(cx - w / 2)
        top = int(cy - h / 2)
        cropped = rotated.crop((left, top, left + w, top + h))
        return cropped
    finally:
        spinner.stop("Поворот карты: готово")

def center_crop(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """
    Центрированный кроп изображения до размеров (out_w x out_h).
    """
    spinner = LiveSpinner("Финальный центр-кроп")
    spinner.start()
    try:
        w, h = img.size
        left = max(0, (w - out_w) // 2)
        top = max(0, (h - out_h) // 2)
        return img.crop((left, top, left + out_w, top + out_h))
    finally:
        spinner.stop("Финальный центр-кроп: готово")


# ------------------------------
# Угол поворота по «востоку» СК‑42/ГК
# ------------------------------
def compute_rotation_deg_for_east_axis(
    center_lat_sk42: float,
    center_lng_sk42: float,
    center_lat_wgs: float,
    center_lng_wgs: float,
    map_params: Tuple[float, float, float, int, int, int, int],
    crs_sk42_gk: CRS,
    t_sk42_to_wgs: Transformer,
) -> float:
    """
    Вычисляет угол, на который нужно повернуть исходное изображение,
    чтобы ось «восток» (X в СК‑42/ГК) стала строго горизонтальной.
    """
    t_sk42gk_from_sk42 = Transformer.from_crs(CRS_SK42_GEOG, crs_sk42_gk, always_xy=True)
    t_sk42_from_sk42gk = Transformer.from_crs(crs_sk42_gk, CRS_SK42_GEOG, always_xy=True)

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


# ------------------------------
# Рисование текста с обводкой и жёлтым фоном
# ------------------------------
def draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: Tuple[int, int, int] = GRID_TEXT_COLOR,
    outline: Tuple[int, int, int] = GRID_TEXT_OUTLINE_COLOR,
    outline_width: int = GRID_TEXT_OUTLINE_WIDTH,
    anchor: str = "lt",
):
    """
    Рисует текст с «обводкой» для лучшей читаемости.
    """
    x, y = xy
    if outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline, anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)

def draw_label_with_bg(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    anchor: str,
    img_size: Tuple[int, int],
    bg_color: Tuple[int, int, int] = GRID_LABEL_BG_COLOR,
    padding: int = GRID_LABEL_BG_PADDING,
):
    """
    Рисует жёлтую подложку под подписью, затем сам текст (с обводкой).
    Подложка рисуется только если после обрезки по границам изображения прямоугольник не вырожден.
    """
    x, y = xy
    w, h = img_size
    bbox = draw.textbbox((x, y), text, font=font, anchor=anchor)
    left = max(0, int(math.floor(bbox[0] - padding)))
    top = max(0, int(math.floor(bbox[1] - padding)))
    right = min(w, int(math.ceil(bbox[2] + padding)))
    bottom = min(h, int(math.ceil(bbox[3] + padding)))
    if right > left and bottom > top:
        draw.rectangle([left, top, right, bottom], fill=bg_color)
    draw_text_with_outline(draw, xy, text, font=font, anchor=anchor)


def load_grid_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Подгружает масштабируемый шрифт для подписей (предпочтительно — жирный).
    Порядок:
      1) GRID_FONT_PATH_BOLD (если задан) — жирный шрифт.
      2) GRID_FONT_PATH (если задан) — обычный шрифт.
      3) DejaVuSans-Bold.ttf, если GRID_FONT_BOLD=True, иначе DejaVuSans.ttf.
      4) Резерв: встроенный маленький шрифт PIL.
    """
    if GRID_FONT_PATH_BOLD:
        try:
            return ImageFont.truetype(GRID_FONT_PATH_BOLD, GRID_FONT_SIZE)
        except Exception:
            pass
    if GRID_FONT_PATH:
        try:
            return ImageFont.truetype(GRID_FONT_PATH, GRID_FONT_SIZE)
        except Exception:
            pass
    if GRID_FONT_BOLD:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", GRID_FONT_SIZE)
        except Exception:
            pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", GRID_FONT_SIZE)
    except Exception:
        pass
    return ImageFont.load_default()


# ------------------------------
# Рисование сетки (пошаговый прогресс) + подписи (две последние цифры тысяч)
# ------------------------------
def draw_axis_aligned_km_grid(
    img: Image.Image,
    center_lat_sk42: float,
    center_lng_sk42: float,
    center_lat_wgs: float,
    zoom: int,
    crs_sk42_gk: CRS,
    t_sk42_to_wgs: Transformer,
    step_m: int = GRID_STEP_M,
    color: Tuple[int, int, int] = GRID_COLOR,
    width_px: int = GRID_WIDTH_PX,
) -> None:
    """
    Рисует сетку 1 км (СК‑42/Гаусса–Крюгера) строго по осям экрана и подписывает
    только 4-й и 5-й младшие знаки (две последние цифры тысяч метров).
    """
    draw = ImageDraw.Draw(img)
    font = load_grid_font()
    w, h = img.size

    mpp = meters_per_pixel(center_lat_wgs, zoom, scale=STATIC_SCALE)
    ppm = 1.0 / mpp

    cx, cy = w / 2.0, h / 2.0

    t_sk42gk_from_sk42 = Transformer.from_crs(CRS_SK42_GEOG, crs_sk42_gk, always_xy=True)
    x0_gk, y0_gk = t_sk42gk_from_sk42.transform(center_lng_sk42, center_lat_sk42)

    def floor_to_step(v: float, step: float) -> int: return math.floor(v / step) * step

    half_w_m = (w / 2.0) / ppm
    half_h_m = (h / 2.0) / ppm

    gx_left_m = floor_to_step(x0_gk - half_w_m, step_m)
    gx_right_m = floor_to_step(x0_gk + half_w_m, step_m) + step_m
    gy_down_m = floor_to_step(y0_gk - half_h_m, step_m)
    gy_up_m = floor_to_step(y0_gk + half_h_m, step_m) + step_m

    # Прогресс: линии + подписи (верх+низ на вертикалях, лево+право на горизонталях)
    n_vert = int((gx_right_m - gx_left_m) / step_m) + 1
    n_horz = int((gy_up_m - gy_down_m) / step_m) + 1
    total_units = n_vert + n_horz + n_vert*2 + n_horz*2
    grid_progress = ConsoleProgress(total=max(1, total_units), label="Сетка и подписи")

    # Вертикальные линии и подписи X
    x_m = gx_left_m
    while x_m <= gx_right_m:
        dx_m = x_m - x0_gk
        x_px = cx + dx_m * ppm
        draw.line([(x_px, 0), (x_px, h)], fill=color, width=width_px)
        grid_progress.step_sync(1)

        x_digits = (int(round(x_m)) // 1000) % 100
        x_label = f"{x_digits:02d}"
        
        # Сдвиг подписей вправо на половину шага сетки (к середине стороны квадрата)
        half_step_px = (step_m / 2) * ppm
        
        # Верх - сдвигаем вправо
        draw_label_with_bg(
            draw, (x_px + half_step_px, GRID_TEXT_MARGIN), x_label, font=font,
            anchor="ma", img_size=(w, h), bg_color=GRID_LABEL_BG_COLOR, padding=GRID_LABEL_BG_PADDING
        )
        grid_progress.step_sync(1)
        # Низ - сдвигаем вправо
        draw_label_with_bg(
            draw, (x_px + half_step_px, h - GRID_TEXT_MARGIN), x_label, font=font,
            anchor="ms", img_size=(w, h), bg_color=GRID_LABEL_BG_COLOR, padding=GRID_LABEL_BG_PADDING
        )
        grid_progress.step_sync(1)
        x_m += step_m

    # Горизонтальные линии и подписи Y
    y_m = gy_down_m
    while y_m <= gy_up_m:
        dy_m = y_m - y0_gk
        y_px = cy - dy_m * ppm
        draw.line([(0, y_px), (w, y_px)], fill=color, width=width_px)
        grid_progress.step_sync(1)

        y_digits = (int(round(y_m)) // 1000) % 100
        y_label = f"{y_digits:02d}"
        
        # Сдвиг подписей вверх на половину шага сетки (к середине стороны квадрата)
        half_step_px = (step_m / 2) * ppm
        
        # Лево - сдвигаем вверх
        draw_label_with_bg(
            draw, (GRID_TEXT_MARGIN, y_px + half_step_px), y_label, font=font,
            anchor="lm", img_size=(w, h), bg_color=GRID_LABEL_BG_COLOR, padding=GRID_LABEL_BG_PADDING
        )
        grid_progress.step_sync(1)
        # Право - сдвигаем вверх
        draw_label_with_bg(
            draw, (w - GRID_TEXT_MARGIN, y_px + half_step_px), y_label, font=font,
            anchor="rm", img_size=(w, h), bg_color=GRID_LABEL_BG_COLOR, padding=GRID_LABEL_BG_PADDING
        )
        grid_progress.step_sync(1)
        y_m += step_m

    grid_progress.close()


# ------------------------------
# Маска (спиннер)
# ------------------------------
def apply_white_mask(img: Image.Image, opacity: float) -> Image.Image:
    """
    Накладывает поверх изображения белую полупрозрачную маску (только для карты).
    """
    opacity = max(0.0, min(1.0, opacity))
    if opacity == 0.0:
        return img
    spinner = LiveSpinner("Наложение маски")
    spinner.start()
    try:
        rgba = img.convert("RGBA")
        overlay = Image.new("RGBA", rgba.size, (255, 255, 255, int(round(255 * opacity))))
        composited = Image.alpha_composite(rgba, overlay)
        return composited.convert("RGB")
    finally:
        spinner.stop("Наложение маски: готово")


# ------------------------------
# Основной асинхронный процесс
# ------------------------------
async def download_satellite_rectangle(
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
    """
    Полный конвейер: подготовка -> загрузка тайлов -> склейка -> поворот -> кроп -> маска -> сетка -> сохранение.
    """
    # Подготовка — конвертация из Гаусса-Крюгера в географические координаты СК-42
    sp = LiveSpinner("Подготовка: определение зоны")
    sp.start()
    # Определяем зону Гаусса-Крюгера из X координаты (номер зоны содержится в старших разрядах)
    zone = int(center_x_sk42_gk // 1000000)  # Зона из первых цифр X координаты
    if zone < 1 or zone > 60:
        # Fallback: пытаемся определить зону из координаты
        zone = max(1, min(60, int((center_x_sk42_gk - 500000) // 1000000) + 1))
    crs_sk42_gk = CRS.from_epsg(28400 + zone)
    sp.stop("Подготовка: зона определена")

    sp = LiveSpinner("Подготовка: конвертация из ГК в СК-42")
    sp.start()
    # Конвертируем из Гаусса-Крюгера в географические СК-42
    t_sk42_from_gk = Transformer.from_crs(crs_sk42_gk, CRS_SK42_GEOG, always_xy=True)
    center_lng_sk42, center_lat_sk42 = t_sk42_from_gk.transform(center_x_sk42_gk, center_y_sk42_gk)
    sp.stop("Подготовка: координаты СК-42 готовы")

    sp = LiveSpinner("Подготовка: создание трансформеров")
    sp.start()
    # Создаем трансформеры для работы с полученными координатами
    t_sk42_to_wgs, t_wgs_to_sk42, _ = build_transformers_sk42(center_lng_sk42)
    sp.stop("Подготовка: трансформеры готовы")

    sp = LiveSpinner("Подготовка: конвертация центра в WGS84")
    sp.start()
    center_lng_wgs, center_lat_wgs = t_sk42_to_wgs.transform(center_lng_sk42, center_lat_sk42)
    sp.stop("Подготовка: центр WGS84 готов")

    sp = LiveSpinner("Подготовка: подбор zoom")
    sp.start()
    zoom = choose_zoom_with_limit(
        center_lat=center_lat_wgs,
        width_m=width_m,
        height_m=height_m,
        desired_zoom=max_zoom,
        scale=scale,
        max_pixels=MAX_OUTPUT_PIXELS,
    )
    sp.stop("Подготовка: zoom выбран")

    if PIL_DISABLE_LIMIT:
        Image.MAX_IMAGE_PIXELS = None

    sp = LiveSpinner("Подготовка: оценка размера")
    sp.start()
    target_w_px, target_h_px, _ = estimate_crop_size_px(center_lat_wgs, width_m, height_m, zoom, scale)
    sp.stop("Подготовка: размер оценён")

    sp = LiveSpinner("Подготовка: расчёт сетки")
    sp.start()
    base_pad = int(round(min(target_w_px, target_h_px) * ROTATION_PAD_RATIO))
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
    sp.stop("Подготовка: сетка рассчитана")

    # Загрузка тайлов — прогресс
    tile_progress = ConsoleProgress(total=len(centers), label="Загрузка тайлов")
    semaphore = asyncio.Semaphore(ASYNC_MAX_CONCURRENCY)
    async with httpx.AsyncClient(http2=True) as client:
        async def bound_fetch(idx_lat_lng):
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
                    maptype="satellite",
                )
                await tile_progress.step(1)
                return idx, img

        tasks = [bound_fetch(pair) for pair in enumerate(centers)]
        results = await asyncio.gather(*tasks)
        tile_progress.close()
        results.sort(key=lambda t: t[0])
        images: List[Image.Image] = [img for _, img in results]

    # Склейка + обрезка — прогресс
    eff_tile_px = static_size_px * scale
    result = assemble_and_crop(
        images=images,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        eff_tile_px=eff_tile_px,
        crop_rect=crop_rect,
    )

    # Поворот — спиннер
    angle_deg = compute_rotation_deg_for_east_axis(
        center_lat_sk42=center_lat_sk42,
        center_lng_sk42=center_lng_sk42,
        center_lat_wgs=center_lat_wgs,
        center_lng_wgs=center_lng_wgs,
        map_params=map_params,
        crs_sk42_gk=crs_sk42_gk,
        t_sk42_to_wgs=t_sk42_to_wgs,
    )
    result = rotate_keep_size(result, angle_deg, fill=(255, 255, 255))

    # Центр-кроп — спиннер
    result = center_crop(result, target_w_px, target_h_px)

    # Маска — спиннер
    if ENABLE_WHITE_MASK and MASK_OPACITY > 0:
        result = apply_white_mask(result, MASK_OPACITY)

    # Сетка + подписи — прогресс
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
        width_px=GRID_WIDTH_PX,
    )

    # Сохранение — спиннер
    sp = LiveSpinner("Сохранение файла")
    sp.start()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        result.save(output_path)
        try:
            fd = os.open(output_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            pass
    finally:
        sp.stop("Сохранение файла: готово")

    # Завершение — спиннер
    sp = LiveSpinner("Завершение")
    sp.start()
    try:
        images.clear()
        sys.stdout.flush()
    finally:
        sp.stop("Завершение: готово")

    return output_path


def main():
    """
    Точка входа.
    - Загружает переменные окружения из .env
    - Проверяет наличие API ключа
    - Запускает основной асинхронный конвейер
    """
    # 1) Загрузка переменных окружения из .env
    load_dotenv(".env")

    # 2) Чтение API ключа из окружения
    api_key = os.getenv(API_KEY_ENV_VAR, "").strip()
    if not api_key:
        raise SystemExit(
            "Не найден API ключ. Создайте файл .env с содержимым вида:\n"
            "API_KEY=ваш_ключ\n"
            "Либо задайте переменную окружения API_KEY перед запуском."
        )

    # 3) Запуск конвейера
    out = asyncio.run(
        download_satellite_rectangle(
            center_x_sk42_gk=center_y_sk42_gk,
            center_y_sk42_gk=center_x_sk42_gk,
            width_m=WIDTH_M,
            height_m=HEIGHT_M,
            api_key=api_key,
            output_path=OUTPUT_PATH,
            max_zoom=ZOOM,
        )
    )

    # Разорвать строку прогресса перед финальным сообщением
    sys.stdout.write("\n")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()