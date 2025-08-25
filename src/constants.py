# Базовый URL Mapbox Static Images API
MAPBOX_STATIC_BASE = 'https://api.mapbox.com/styles/v1'

# Базовый размер тайла по одной стороне (px) для статичных изображений
# По умолчанию используем 1024 (вмещается в лимит 1280 и хорошо сочетается с @2x)
STATIC_SIZE_PX = 1024

# Масштаб 2 => HiDPI (@2x) — фактический тайл в 2 раза больше
STATIC_SCALE = 2

# Максимальный уровень приближения
# (переменная, подбирается автоматически вниз при больших размерах)
MAX_ZOOM = 18

# Радиус Земли для Web Mercator (метры)
EARTH_RADIUS_M = 6378137.0

# Базовый размер тайла Web Mercator (пикселей)
TILE_SIZE = 256
# Дополнительные размеры тайла
TILE_SIZE_512 = 512

# Множитель для HiDPI (ретина)
RETINA_FACTOR = 2

# Максимальная 6-градусная зона Гаусса-Крюгера
MAX_GK_ZONE = 60

# Максимальное число параллельных HTTP-запросов
ASYNC_MAX_CONCURRENCY = 20

# Ограничение на итоговое число пикселей результирующего кадра (без припуска)
MAX_OUTPUT_PIXELS = 150_000_000

# Отключить защиту Pillow от «бомб декомпрессии» (используйте с осторожностью)
PIL_DISABLE_LIMIT = True

# Доля припуска на сторону под поворот (7% от меньшей стороны)
ROTATION_PAD_RATIO = 0.07

# Минимальный припуск на сторону (px)
ROTATION_PAD_MIN_PX = 128

# припуск по осям в дополнение к заданным квадратам
ADDITIVE_RATIO = 0.3

# Шаг километровой сетки (метры)
GRID_STEP_M = 1000

# Цвет линий сетки (RGB)
GRID_COLOR = (0, 0, 0)

# Цвет текста подписи (RGB)
GRID_TEXT_COLOR = (0, 0, 0)

# Цвет обводки текста (RGB)
GRID_TEXT_OUTLINE_COLOR = (255, 255, 255)

# Цвет жёлтой подложки под подписью (RGB)
GRID_LABEL_BG_COLOR = (255, 255, 0)

# Использовать жирный шрифт для подписей (если доступен)
GRID_FONT_BOLD = True

# Путь к шрифту TTF/OTF (если None — DejaVuSans.ttf)
GRID_FONT_PATH = None

# Путь к жирному шрифту TTF/OTF (если None — DejaVuSans-Bold.ttf)
GRID_FONT_PATH_BOLD = None

# Включить наложение белой маски поверх карты
ENABLE_WHITE_MASK = True

WGS84_CODE = 4326
SK42_CODE = 4284

CURRENT_PROFILE = 'default'

PROFILES_DIR = '../configs/profiles'

# HTTP диапазоны ошибок сервера
HTTP_5XX_MIN = 500
HTTP_5XX_MAX = 600

# --- Options previously in profile (moved to constants)
# Marked with '# -' in default.toml
# Desired zoom level (will be reduced automatically if limits are exceeded)
DESIRED_ZOOM = 22
# Mapbox style identifier
MAPBOX_STYLE_ID = 'mapbox/satellite-v9'
# XYZ tile size preference (256 or 512)
XYZ_TILE_SIZE = 512
# Use @2x retina tiles
XYZ_USE_RETINA = True
# Parallel HTTP concurrency
DOWNLOAD_CONCURRENCY = 20
# Static source compatibility values
STATIC_TILE_WIDTH_PX_PROFILE = 1024
STATIC_TILE_HEIGHT_PX_PROFILE = 1024
IMAGE_FORMAT_PROFILE = 'png'

# --- Cache options (moved from profile [cache] section into constants)
HTTP_CACHE_ENABLED = True
# Cache directory (relative paths are resolved from project root)
HTTP_CACHE_DIR = '.cache/tiles'
# TTL in hours
HTTP_CACHE_EXPIRE_HOURS = 168
# Respect HTTP Cache-Control/ETag/Last-Modified headers
HTTP_CACHE_RESPECT_HEADERS = True
# Allow using stale cache when network errors occur (hours); 0 to disable
HTTP_CACHE_STALE_IF_ERROR_HOURS = 72

# --- Grid text outline width (moved from profile to constants)
GRID_TEXT_OUTLINE_WIDTH = 2

# --- GUI limits
# Максимальный размер стороны участка (км)
MAX_SIDE_SIZE = 60

# --- SK-42 applicability (approximate area of use)
# Longitudes (degrees East) roughly covering the former USSR area
SK42_VALID_LON_MIN = 19.0
SK42_VALID_LON_MAX = 190.0
# Latitudes (degrees North) where SK-42 is typically applicable
SK42_VALID_LAT_MIN = 35.0
SK42_VALID_LAT_MAX = 85.0
