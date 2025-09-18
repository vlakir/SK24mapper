from enum import Enum

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

PROFILES_DIR = 'configs/profiles'

# HTTP диапазоны ошибок сервера
HTTP_5XX_MIN = 500
HTTP_5XX_MAX = 600

# --- Опции ранее находились в профиле (перенесены в constants)
# В default.toml были помечены как '# -'
# Желаемый уровень масштаба (при превышении лимитов будет уменьшен автоматически)
DESIRED_ZOOM = 22

# Тип карты и резолвер стилей Mapbox (Этап 1)


class MapType(str, Enum):
    SATELLITE = 'SATELLITE'
    HYBRID = 'HYBRID'
    STREETS = 'STREETS'
    OUTDOORS = 'OUTDOORS'
    ELEVATION_COLOR = 'ELEVATION_COLOR'
    ELEVATION_CONTOURS = 'ELEVATION_CONTOURS'
    ELEVATION_HILLSHADE = 'ELEVATION_HILLSHADE'


# Человекочитаемые названия для GUI
MAP_TYPE_LABELS_RU: dict[MapType, str] = {
    MapType.SATELLITE: 'Спутник',
    MapType.HYBRID: 'Гибрид',
    MapType.STREETS: 'Улицы',
    MapType.OUTDOORS: 'Топографический',
    MapType.ELEVATION_COLOR: 'Карта высот (цветовая шкала)',
    MapType.ELEVATION_CONTOURS: 'Карта высот (контуры)',
    MapType.ELEVATION_HILLSHADE: 'Карта высот (hillshade)',
}

# Резолвер для стилевых карт (этап 1)
MAPBOX_STYLE_BY_TYPE: dict[MapType, str] = {
    MapType.SATELLITE: 'mapbox/satellite-v9',
    MapType.HYBRID: 'mapbox/satellite-streets-v12',
    MapType.STREETS: 'mapbox/streets-v12',
    MapType.OUTDOORS: 'mapbox/outdoors-v12',
}


def default_map_type() -> MapType:
    return MapType.SATELLITE


def map_type_to_style_id(map_type: MapType | str) -> str | None:
    """Возвращает style_id для стилевых карт. Для режимов высот возвращает None."""
    try:
        mt = MapType(map_type) if not isinstance(map_type, MapType) else map_type
    except Exception:
        mt = MapType.SATELLITE
    return MAPBOX_STYLE_BY_TYPE.get(mt)


# Предпочтительный размер тайла XYZ (256 или 512)
XYZ_TILE_SIZE = 512
# Использовать ретина-тайлы @2x
XYZ_USE_RETINA = True
# Использовать ретину для карт высот (Terrain-RGB)
ELEVATION_USE_RETINA = True
# Параллелизм загрузки HTTP
DOWNLOAD_CONCURRENCY = 20
# Значения совместимости для «статичного» источника
STATIC_TILE_WIDTH_PX_PROFILE = 1024
STATIC_TILE_HEIGHT_PX_PROFILE = 1024
IMAGE_FORMAT_PROFILE = 'jpg'

# --- Опции кэша (были в секции профиля [cache], перенесены в constants)
HTTP_CACHE_ENABLED = True
# Каталог кэша (относительные пути считаются от корня проекта)
HTTP_CACHE_DIR = '.cache/tiles'
# Время жизни (TTL) в часах
HTTP_CACHE_EXPIRE_HOURS = 168
# Учитывать заголовки Cache-Control/ETag/Last-Modified
HTTP_CACHE_RESPECT_HEADERS = True
# Разрешить использовать устаревший кэш при сетевых ошибках (часы); 0 — запретить
HTTP_CACHE_STALE_IF_ERROR_HOURS = 72

# --- Толщина обводки текста сетки (перенесено из профиля в constants)
GRID_TEXT_OUTLINE_WIDTH = 2

# --- GUI limits
# Максимальный размер стороны участка (км)
MAX_SIDE_SIZE = 60

# --- Применимость СК-42 (примерные границы области применения)
# Долготы (в градусах восточной долготы), примерно охватывающие территорию бывшего СССР
SK42_VALID_LON_MIN = 19.0
SK42_VALID_LON_MAX = 190.0
# Широты (в градусах северной широты), где обычно применима СК-42
SK42_VALID_LAT_MIN = 35.0
SK42_VALID_LAT_MAX = 85.0

# --- Временные и размерные константы GUI
# Ширина предпросмотра по умолчанию (px), когда измерить ширину невозможно
PREVIEW_FALLBACK_WIDTH = 600
# Задержка восстановления режима расширения после первичной оценки размеров (мс)
PREVIEW_RESTORE_EXPAND_DELAY_MS = 200
# Нормализованная величина шага колеса мыши
MOUSEWHEEL_DELTA = 120
RENDER_DEBOUNCE_MS = 12
HQ_RENDER_DELAY_MS = 160
RESIZE_DEBOUNCE_MS = 60
MIN_FIT_SCALE = 0.01

# --- Зона Гаусса–Крюгера и константы EPSG
# Делитель для извлечения номера зоны из координаты X (метры)
GK_ZONE_X_PREFIX_DIV = 1_000_000
# Ложное восточное смещение (метры)
GK_FALSE_EASTING = 500_000
# Ширина зоны (градусы)
GK_ZONE_WIDTH_DEG = 6
# Смещение до центрального меридиана зоны (градусы)
GK_ZONE_CM_OFFSET_DEG = 3
# База EPSG для СК‑42 / зоны Гаусса–Крюгера (к базе добавляется номер зоны)
EPSG_SK42_GK_BASE = 28400

# --- Константы Web Mercator и XYZ
# Ограничение синуса для избежания бесконечностей у полюсов
MERCATOR_MAX_SIN = 0.9999
WORLD_LNG_SPAN_DEG = 360.0
WORLD_LNG_HALF_SPAN_DEG = 180.0
WORLD_LAT_MAX_DEG = 90.0
# Небольшой эпсилон для расчётов на границах тайлов
XY_EPSILON = 1e-9

# --- Параметры сетевых запросов по умолчанию
HTTP_TIMEOUT_DEFAULT = 20.0
HTTP_RETRIES_DEFAULT = 4
HTTP_BACKOFF_FACTOR = 1.6

# --- Terrain-RGB (Этапы 2–4)
MAPBOX_TERRAIN_RGB_PATH = 'https://api.mapbox.com/v4/mapbox.terrain-rgb'
# Нормализация по перцентилям (в процентах)
ELEV_PCTL_LO = 2.0
ELEV_PCTL_HI = 98.0
# Запас для защиты от плоских регионов (минимальная дельта высот в метрах)
ELEV_MIN_RANGE_M = 10.0
# Опциональный быстрый путь на NumPy (если установлен)
USE_NUMPY_FASTPATH = True
# Палитра: список контрольных точек (t in [0,1], (R,G,B))
ELEVATION_COLOR_RAMP = [
    (0.00, (0, 0, 130)),  # deep blue
    (0.15, (0, 100, 200)),  # blue
    (0.30, (0, 160, 100)),  # green
    (0.45, (180, 200, 0)),  # yellowish
    (0.60, (200, 140, 0)),  # orange
    (0.75, (160, 80, 30)),  # brown
    (0.90, (220, 220, 220)),  # light gray
    (1.00, (255, 255, 255)),  # white
]
# --- Контуры (Этап 3)
# Интервал между изогипсами (метры)
CONTOUR_INTERVAL_M = 10.0
# Downsample factor for global low-res DEM seed (integer >= 2)
CONTOUR_SEED_DOWNSAMPLE = 4
# Optional spline smoothing for seed polylines (not mandatory for MVP)
CONTOUR_SEED_SMOOTHING = False
# Цвет обычных изогипс (RGB)
CONTOUR_COLOR = (60, 60, 60)
# Толщина обычных изогипс в пикселях (задаётся здесь)
CONTOUR_WIDTH = 2
# Каждая N‑я изогипса считается «индексной» (выделенной)
CONTOUR_INDEX_EVERY = 5
# Цвет индексных изогипс (RGB)
CONTOUR_INDEX_COLOR = (30, 30, 30)
# Толщина индексных изогипс в пикселях (задаётся здесь)
CONTOUR_INDEX_WIDTH = 4
# --- Подписи изогипс
CONTOUR_LABELS_ENABLED = True
CONTOUR_LABEL_INDEX_ONLY = False
CONTOUR_LABEL_SPACING_PX = 300
CONTOUR_LABEL_MIN_SEG_LEN_PX = 40
CONTOUR_LABEL_EDGE_MARGIN_PX = 8
CONTOUR_LABEL_TEXT_COLOR = (30, 30, 30)
CONTOUR_LABEL_OUTLINE_COLOR = (255, 255, 255)
CONTOUR_LABEL_OUTLINE_WIDTH = 2
CONTOUR_LABEL_BG_RGBA = (255, 255, 255, 200)
CONTOUR_LABEL_BG_PADDING = 3
# Старый фиксированный размер (для бэкапа/совместимости)
CONTOUR_LABEL_FONT_SIZE = 18
# Новый масштабируемый размер шрифта в долях километра
# По умолчанию 0.03 км (30 м) даёт хороший размер.
CONTOUR_LABEL_FONT_KM = 0.03
# Отдельный коэффициент для индексных изогипс (чуть крупнее)
CONTOUR_LABEL_FONT_KM_INDEX = 0.06
# Диапазон клампа в пикселях, чтобы подписи оставались читаемыми
CONTOUR_LABEL_FONT_MIN_PX = 12
CONTOUR_LABEL_FONT_MAX_PX = 48
CONTOUR_LABEL_FONT_PATH = None
CONTOUR_LABEL_FONT_BOLD = True
CONTOUR_LABEL_FORMAT = '{:.0f}'

# --- Оценка поворота
# Отсчёт вдоль восточной оси для оценки угла (метры)
EAST_VECTOR_SAMPLE_M = 200.0

# --- Вспомогательные константы для подписей сетки
GRID_LABEL_THOUSAND_DIV = 1000
GRID_LABEL_MOD = 100

# --- Общие константы для валидаторов/обработчиков GUI
# Минимальное число аргументов команды скролла (tk Scrollbar callback)
SCROLL_CMD_MIN_ARGS = 3

# --- Константы для окна предпросмотра
# Фиксированный угол поворота изображения в градусах для улучшения
# видимости тонких линий
PREVIEW_ROTATION_ANGLE = 0.0

# --- Константы интерфейса пользователя (GUI)
# Допуск для сравнения чисел с плавающей точкой
FLOAT_COMPARISON_TOLERANCE = 0.0001
# Пороговое значение байтов для перевода в килобайты (1024 байта = 1 КБ)
BYTES_TO_KB_THRESHOLD = 1024
# Коэффициент конвертации байтов в килобайты (делитель)
BYTES_CONVERSION_FACTOR = 1024.0

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
# Availability flags for optional libs
PSUTIL_AVAILABLE = True
