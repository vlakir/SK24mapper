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
MAX_ZOOM = 20

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

# Количество видимых символов API-ключа при маскировке
API_KEY_VISIBLE_PREFIX_LEN = 4

# Максимальное число параллельных HTTP-запросов
ASYNC_MAX_CONCURRENCY = 10

# Ограничение на итоговое число пикселей результирующего кадра (без припуска)
MAX_OUTPUT_PIXELS = 500_000_000

# Отключить защиту Pillow от «бомб декомпрессии» (используйте с осторожностью)
PIL_DISABLE_LIMIT = True

# Доля припуска на сторону под поворот (7% от меньшей стороны)
ROTATION_PAD_RATIO = 0.07

# Минимальный припуск на сторону (px)
ROTATION_PAD_MIN_PX = 128

# Эпсилон для проверки, что угол поворота не нулевой
ROTATION_EPSILON = 1e-6

# Порог угла поворота для обратной трансформации клика (градусы)
ROTATION_INVERSE_THRESHOLD_DEG = 0.01

# Максимальное число сэмплов для расчёта диапазона высот изолиний
CONTOUR_MAX_ELEVATION_SAMPLES = 50000

# припуск по осям в дополнение к заданным квадратам
ADDITIVE_RATIO = 0.3

# Шаг километровой сетки (метры)
GRID_STEP_M = 1000

# Длина линий крестиков при отключенной сетке (метры)
GRID_CROSS_LENGTH_M = 50

# Минимальное количество точек для рисования линии (draw.line требует >= 2)
MIN_POINTS_FOR_LINE = 2

# Минимальное количество точек для валидного сегмента полилинии
MIN_POINTS_FOR_SEGMENT = 2

# Цвет линий сетки (RGB)
GRID_COLOR = (0, 0, 0)

# Цвет текста подписи (RGB)
GRID_TEXT_COLOR = (0, 0, 0)

# Цвет обводки текста (RGB)
GRID_TEXT_OUTLINE_COLOR = (255, 255, 255)

# Цвет жёлтой подложки под подписью (RGB)
GRID_LABEL_BG_COLOR = (255, 255, 0)

# Путь к шрифту TTF/OTF (если None — системный Arial Bold)
GRID_FONT_PATH = None

# Путь к жирному шрифту TTF/OTF (если None — системный Arial Bold)
GRID_FONT_PATH_BOLD = None


# Включить наложение белой маски поверх карты

WGS84_CODE = 4326
SK42_CODE = 4284


PROFILES_DIR = 'configs/profiles'

# HTTP диапазоны ошибок сервера
HTTP_5XX_MIN = 500
HTTP_5XX_MAX = 600

# --- Опции ранее находились в профиле (перенесены в constants)
# В default.toml были помечены как '# -'

# Тип карты и резолвер стилей Mapbox (Этап 1)


class MapType(str, Enum):
    SATELLITE = 'SATELLITE'
    HYBRID = 'HYBRID'
    STREETS = 'STREETS'
    OUTDOORS = 'OUTDOORS'
    ELEVATION_COLOR = 'ELEVATION_COLOR'
    ELEVATION_CONTOURS = 'ELEVATION_CONTOURS'
    ELEVATION_HILLSHADE = 'ELEVATION_HILLSHADE'
    RADIO_HORIZON = 'RADIO_HORIZON'
    RADAR_COVERAGE = 'RADAR_COVERAGE'
    LINK_PROFILE = 'LINK_PROFILE'


class UavHeightReference(str, Enum):
    """Режим отсчёта высоты БпЛА для карты радиогоризонта."""

    CONTROL_POINT = 'control_point'  # От уровня контрольной точки
    GROUND = 'ground'  # От уровня земной поверхности
    SEA_LEVEL = 'sea_level'  # От уровня моря


# Человекочитаемые названия для GUI
MAP_TYPE_LABELS_RU: dict[MapType, str] = {
    MapType.SATELLITE: 'Спутник',
    MapType.HYBRID: 'Гибрид',
    MapType.STREETS: 'Улицы',
    MapType.OUTDOORS: 'Топографический',
    MapType.ELEVATION_COLOR: 'Карта высот (цветовая шкала)',
    MapType.ELEVATION_CONTOURS: 'Карта высот (контуры)',
    MapType.ELEVATION_HILLSHADE: 'Карта высот (hillshade)',
    MapType.RADIO_HORIZON: 'Радиогоризонт НСУ БпЛА',
    MapType.RADAR_COVERAGE: 'Зона обнаружения РЛС',
    MapType.LINK_PROFILE: 'Профиль радиолинии',
}

UAV_HEIGHT_REFERENCE_LABELS_RU: dict[UavHeightReference, str] = {
    UavHeightReference.CONTROL_POINT: 'От уровня КТ',
    UavHeightReference.GROUND: 'От земли',
    UavHeightReference.SEA_LEVEL: 'От уровня моря',
}

UAV_HEIGHT_REFERENCE_ABBR: dict[UavHeightReference, str] = {
    UavHeightReference.CONTROL_POINT: 'AGL',
    UavHeightReference.GROUND: 'RA',
    UavHeightReference.SEA_LEVEL: 'AMSL',
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
# Использовать ретину для карт высот (Terrain-RGB) — контуры и overlay-DEM для курсора
ELEVATION_USE_RETINA = True
# Использовать ретину для ELEVATION_COLOR (False = 256px, быстрее в 2-4 раза)
ELEVATION_COLOR_USE_RETINA = False
# --- Hillshade (теневая отмывка рельефа)
# Азимут источника света (0=север, по часовой), градусы
HILLSHADE_AZIMUTH_DEG: float = 315.0  # Северо-запад (классика топографии)
# Угол источника света над горизонтом, градусы
HILLSHADE_ALTITUDE_DEG: float = 45.0
# Использовать ретина-тайлы для hillshade (False = 256px, как ELEVATION_COLOR)
HILLSHADE_USE_RETINA: bool = False
# Коэффициент вертикального преувеличения рельефа (z-factor).
# 1.0 = реальный масштаб, 5-10 = хорошо на равнинах,
# 20+ = для очень плоских районов
HILLSHADE_Z_FACTOR: float = 8.0
# Zoom-уровень DEM для hillshade (14 = нативное разрешение Mapbox Terrain-RGB).
# При zoom > 14 данные Mapbox — просто интерполяция, создающая ступенчатые артефакты.
HILLSHADE_DEM_ZOOM: int = 14
# Sigma Гауссова сглаживания DEM перед вычислением градиентов (пиксели DEM).
# Убирает остаточные ступенчатые артефакты дискретизации. 0 = без сглаживания.
HILLSHADE_SMOOTH_SIGMA: float = 1.5
# Параллелизм загрузки HTTP (увеличено для ускорения)
DOWNLOAD_CONCURRENCY = 40
# Значения совместимости для «статичного» источника

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

# --- Применимость СК-42 (примерные границы области применения)
# Долготы (в градусах восточной долготы), примерно охватывающие территорию бывшего СССР
SK42_VALID_LON_MIN = 19.0
SK42_VALID_LON_MAX = 190.0
# Широты (в градусах северной широты), где обычно применима СК-42
SK42_VALID_LAT_MIN = 35.0
SK42_VALID_LAT_MAX = 85.0

# --- Временные и размерные константы GUI
# Ширина предпросмотра по умолчанию (px), когда измерить ширину невозможно
# Задержка восстановления режима расширения после первичной оценки размеров (мс)
# Нормализованная величина шага колеса мыши

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

# --- Параметры форматирования координат
# Количество символов в строке координаты, после которого вставляется пробел-разделитель
COORDINATE_FORMAT_SPLIT_LENGTH = 5

# --- Terrain-RGB (Этапы 2–4)
MAPBOX_TERRAIN_RGB_PATH = 'https://api.mapbox.com/v4/mapbox.terrain-rgb'
# Нормализация по перцентилям (в процентах)
ELEV_PCTL_LO = 2.0
ELEV_PCTL_HI = 98.0
# Запас для защиты от плоских регионов (минимальная дельта высот в метрах)
ELEV_MIN_RANGE_M = 10.0
# Шаг округления высот для легенды и палитры (метры)
ELEVATION_LEGEND_STEP_M = 10.0
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
# Базовый размер карты для адаптации параметров изолиний (метры)
CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M = 15000.0
# Степень влияния размера карты на масштабирование параметров изолиний
CONTOUR_ADAPTIVE_ALPHA = 0.6
# Клампы коэффициента масштабирования
CONTOUR_ADAPTIVE_MIN_SCALE = 0.7
CONTOUR_ADAPTIVE_MAX_SCALE = 2.5
# Степень влияния масштабирования на размер шрифта подписей
CONTOUR_LABEL_FONT_SCALE_ALPHA = 0.5
# Downsample factor for global low-res DEM seed (integer >= 2)
CONTOUR_SEED_DOWNSAMPLE = 12
# Optional spline smoothing for seed polylines (not mandatory for MVP)
CONTOUR_SEED_SMOOTHING = True
# Параметры сглаживания изолиний
CONTOUR_SMOOTHING_FACTOR = 3  # Множитель точек (2-7, больше = плавнее)
CONTOUR_SMOOTHING_STRENGTH = (
    1.5  # Параметр s для splprep (0.5-3.0, больше = агрессивнее)
)
CONTOUR_SMOOTHING_ITERATIONS = 2  # Итерации для fallback-метода (1-3)
# Минимальное количество точек для применения сглаживания
MIN_POINTS_FOR_SMOOTHING = 3
# Количество параллельных воркеров для построения изолиний
CONTOUR_PARALLEL_WORKERS = 4
# Цвет обычных изогипс (RGB)
CONTOUR_COLOR = (30, 30, 30)
# Толщина обычных изогипс в метрах на местности (переводится в пиксели по mpp)
CONTOUR_WIDTH = 4
# Каждая N‑я изогипса считается «индексной» (выделенной)
CONTOUR_INDEX_EVERY = 5
# Цвет индексных изогипс (RGB)
CONTOUR_INDEX_COLOR = (20, 20, 20)
# Толщина индексных изогипс в метрах на местности (переводится в пиксели по mpp)
CONTOUR_INDEX_WIDTH = 6
# --- Подписи изогипс
CONTOUR_LABELS_ENABLED = True
CONTOUR_LABEL_INDEX_ONLY = False
# Интервал между подписями в метрах (пересчитывается в пиксели по текущему масштабу)
CONTOUR_LABEL_SPACING_M = 2000
# Минимальная длина сегмента для размещения подписи (метры)
CONTOUR_LABEL_MIN_SEG_LEN_M = 40
# Отступ подписей от границ изображения (метры)
CONTOUR_LABEL_EDGE_MARGIN_M = 8
CONTOUR_LABEL_TEXT_COLOR = (20, 20, 20)
CONTOUR_LABEL_OUTLINE_COLOR = (255, 255, 255)
CONTOUR_LABEL_OUTLINE_WIDTH = 0
# Подложка для подписи изолиний отключена по умолчанию; для включения задайте RGBA-цвет
CONTOUR_LABEL_BG_RGBA = None

CONTOUR_LABEL_BG_PADDING = 3

# --- Разрывы линий контуров для подписей
# Включить разрывы линий в местах размещения подписей
CONTOUR_LABEL_GAP_ENABLED = True
# Дополнительный отступ вокруг подписей для разрыва линии (метры)
CONTOUR_LABEL_GAP_PADDING_M = 5

# Старый фиксированный размер (для бэкапа/совместимости)
CONTOUR_LABEL_FONT_SIZE = 18
# Множитель размера шрифта изолиний относительно grid_font_size_m
CONTOUR_FONT_SIZE_RATIO = 0.5
# Диапазон клампа в пикселях, чтобы подписи оставались читаемыми
CONTOUR_LABEL_FONT_MIN_PX = 16
CONTOUR_LABEL_FONT_MAX_PX = 128
CONTOUR_LABEL_FORMAT = '{:.0f}'
# Минимальная длина полилинии для размещения подписи
MIN_POLYLINE_POINTS = 10

# --- Оценка поворота
# Отсчёт вдоль восточной оси для оценки угла (метры)
EAST_VECTOR_SAMPLE_M = 200.0

# --- Вспомогательные константы для подписей сетки
GRID_LABEL_THOUSAND_DIV = 1000
GRID_LABEL_MOD = 100
# --- Общие константы для валидаторов/обработчиков GUI
# Минимальное число аргументов команды скролла (tk Scrollbar callback)
MIN_DECIMALS_FOR_SMALL_STEP = 2
COORD_DIMENSIONS = 3
MIN_POINTS_FOR_HELMERT = 3
MIN_POINT_PAIRS = 2
CSV_COLUMNS_REQUIRED = 4
MAX_TRANSLATION_M = 10000
MAX_ROTATION_ARCSEC = 3600
MAX_SCALE_PPM = 1000

# --- Константы для окна предпросмотра
# Минимальная длина линии в единицах сцены для отображения подписи
PREVIEW_MIN_LINE_LENGTH_FOR_LABEL = 10.0
# Порог угла (градусы), при котором текст считается перевернутым и требует коррекции
PREVIEW_UPRIGHT_TEXT_ANGLE_LIMIT = 90.0

# Фиксированный угол поворота изображения в градусах для улучшения
# видимости тонких линий
PREVIEW_ROTATION_ANGLE = 0.0

# --- События модели (GUI)
MODEL_EVENT_SETTINGS_CHANGED = 'SETTINGS_CHANGED'
MODEL_EVENT_PROFILE_LOADED = 'PROFILE_LOADED'
MODEL_EVENT_PROFILE_SAVED = 'PROFILE_SAVED'
MODEL_EVENT_DOWNLOAD_STARTED = 'DOWNLOAD_STARTED'
MODEL_EVENT_DOWNLOAD_PROGRESS = 'DOWNLOAD_PROGRESS'
MODEL_EVENT_DOWNLOAD_COMPLETED = 'DOWNLOAD_COMPLETED'
MODEL_EVENT_DOWNLOAD_FAILED = 'DOWNLOAD_FAILED'
MODEL_EVENT_PREVIEW_UPDATED = 'PREVIEW_UPDATED'
MODEL_EVENT_WARNING_OCCURRED = 'WARNING_OCCURRED'
MODEL_EVENT_ERROR_OCCURRED = 'ERROR_OCCURRED'

HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
# Availability flags for optional libs
PSUTIL_AVAILABLE = True

# --- Центрированный крест на карте
# Цвет креста (RGB)
CENTER_CROSS_COLOR = (255, 255, 255)
# Толщина линий креста (метры)
CENTER_CROSS_LINE_WIDTH_M = 1
# Полная длина линии креста (метры)
CENTER_CROSS_LENGTH_M = 40

# --- Контрольная точка: единый стиль — красный треугольник с подписью
# Размер контрольной точки (треугольника) в метрах на местности
CONTROL_POINT_SIZE_M = 100.0
# Цвет контрольной точки (треугольника) — ярко-красный
CONTROL_POINT_COLOR = (255, 0, 0)
# Порог допустимой неточности для определения совпадения точек (метры)
CONTROL_POINT_PRECISION_TOLERANCE_M = 0.1
# Минимальный отступ подписи от треугольника контрольной точки (пиксели)
CONTROL_POINT_LABEL_GAP_MIN_PX = 8
# Коэффициент отступа подписи от треугольника (доля от размера треугольника)
CONTROL_POINT_LABEL_GAP_RATIO = 0.3

# --- Вспомогательные константы для генерации контуров в service.py
# Квантование координат при сцеплении отрезков (чем больше — тем грубее)
SEED_POLYLINE_QUANT_FACTOR = 8
# Вес для усреднения четырёх значений в ячейке (1/4) в алгоритме marching squares
MARCHING_SQUARES_CENTER_WEIGHT = 0.25
# Минимальный размер DEM-сетки для построения seed-полилиний
MIN_GRID_SIZE = 2

# Marching Squares — именованные маски и группы случаев
# Битовая раскладка (по часовой стрелке, начиная с верхнего левого):
# b0: TL, b1: TR, b2: BR, b3: BL
# Пустая и полная маски (все углы ниже/выше уровня соответственно)
MS_MASK_EMPTY = 0  # 0b0000 — все ниже уровня
MS_MASK_FULL = 15  # 0b1111 — все выше уровня

# Одиночные углы
MS_MASK_TL = 1  # 0b0001 — только верхний левый
MS_MASK_TR = 2  # 0b0010 — только верхний правый
MS_MASK_BR = 4  # 0b0100 — только нижний правый
MS_MASK_BL = 8  # 0b1000 — только нижний левый

# Две вершины — стороны клетки
MS_MASK_TOP = 3  # 0b0011 — TL+TR (верх)
MS_MASK_RIGHT = 6  # 0b0110 — TR+BR (право)
MS_MASK_BOTTOM = 12  # 0b1100 — BL+BR (низ)
MS_MASK_LEFT = 9  # 0b1001 — TL+BL (лево)

# Диагональные (седловые) случаи — неоднозначны без разрешения диагонали
MS_MASK_TL_BR = 5  # 0b0101 — TL+BR
MS_MASK_TR_BL = 10  # 0b1010 — TR+BL

# Три вершины — «всё кроме …»
MS_MASK_NOT_TL = 14  # 0b1110 — все кроме TL
MS_MASK_NOT_TR = 13  # 0b1101 — все кроме TR
MS_MASK_NOT_BR = 11  # 0b1011 — все кроме BR
MS_MASK_NOT_BL = 7  # 0b0111 — все кроме BL

# Случаи, когда изолиния в клетке отсутствует
MS_NO_CONTOUR_CASES = {MS_MASK_EMPTY, MS_MASK_FULL}

# Комплементарные группы масок: какие рёбра клетки соединяет изолиния
# (именование указывает пары рёбер; порядок в кортеже — комплементарные случаи)
MS_CONNECT_TOP_LEFT = (MS_MASK_TL, MS_MASK_NOT_TL)  # (1, 14) верх ↔ лево
MS_CONNECT_TOP_RIGHT = (MS_MASK_TR, MS_MASK_NOT_TR)  # (2, 13) верх ↔ право
MS_CONNECT_LEFT_RIGHT = (
    MS_MASK_TOP,
    MS_MASK_BOTTOM,
)  # (3, 12) лево ↔ право (горизонталь)
MS_CONNECT_RIGHT_BOTTOM = (MS_MASK_BR, MS_MASK_NOT_BR)  # (4, 11) право ↔ низ
MS_AMBIGUOUS_CASES = (MS_MASK_TL_BR, MS_MASK_TR_BL)  # (5, 10) седловые
MS_CONNECT_TOP_BOTTOM = (MS_MASK_RIGHT, MS_MASK_LEFT)  # (6, 9)  верх ↔ низ (вертикаль)
MS_CONNECT_LEFT_BOTTOM = (MS_MASK_NOT_BL, MS_MASK_BL)  # (7, 8)  лево ↔ низ

# Периодичность логирования использования памяти (каждые N тайлов)
CONTOUR_LOG_MEMORY_EVERY_TILES = 50

# --- Легенда высот на карте (elevation legend)
# Желаемая высота легенды как доля от высоты карты (10%)
LEGEND_HEIGHT_RATIO = 0.10
# Минимальная/максимальная высота легенды как доля от высоты карты
LEGEND_HEIGHT_MIN_RATIO = 0.10
LEGEND_HEIGHT_MAX_RATIO = 0.50
# Минимальная высота легенды в километровых квадратах (для карт ниже порога высоты)
LEGEND_MIN_HEIGHT_GRID_SQUARES = 1.0
# Порог высоты карты (в метрах), ниже которого легенда занимает минимум 1 км-квадрат
# Используется вместо «магического» числа 10000.0 в коде для читаемости и единообразия
LEGEND_MIN_MAP_HEIGHT_M_FOR_RATIO = 10000.0
# Отношение ширины легенды к её высоте
LEGEND_WIDTH_TO_HEIGHT_RATIO = 0.133  # ширина = высота * 0.133
# Отступ легенды от краёв карты как доля от высоты легенды
LEGEND_MARGIN_RATIO = 0.067  # margin = высота_легенды * 0.067
# Количество меток высоты на легенде (мин, макс и промежуточные)
LEGEND_NUM_LABELS = 5
# Отступ текста от цветовой полосы легенды в метрах
LEGEND_TEXT_OFFSET_M = 5
# Толщина рамки вокруг цветовой полосы легенды в метрах
LEGEND_BORDER_WIDTH_M = 2
# Размер шрифта для подписей легенды как доля от высоты легенды (увеличено в 2 раза)
LEGEND_LABEL_FONT_RATIO = 0.26  # размер_шрифта = высота_легенды * 0.26
# Диапазон размера шрифта подписей легенды в пикселях (мин и макс, увеличено в 2 раза)
LEGEND_LABEL_FONT_MIN_PX = 24
LEGEND_LABEL_FONT_MAX_PX = 96
# Толщина обводки текста легенды в метрах
LEGEND_TEXT_OUTLINE_WIDTH_M = 2
# Дополнительный отступ вокруг легенды для разрыва линий сетки (метры)
LEGEND_GRID_GAP_PADDING_M = 10
# Цвет фона легенды (RGBA): белый полупрозрачный для визуального выделения
LEGEND_BACKGROUND_COLOR = (255, 255, 255, 230)
# Отступ фона легенды от краёв цветовой полосы (метры)
LEGEND_BACKGROUND_PADDING_M = 8
# Горизонтальная позиция легенды: доля от ширины последнего километрового квадрата
# (0.5 = середина)
LEGEND_HORIZONTAL_POSITION_RATIO = 0.5
# Вертикальный отступ нижней границы легенды от первой горизонтальной линии сетки
# (в долях от шага сетки)
LEGEND_VERTICAL_OFFSET_RATIO = 0.15
# Дополнительный отступ заголовка легенды вверх (доля от высоты легенды)
LEGEND_TITLE_OFFSET_RATIO = 0.10

# --- Кэш DEM тайлов
_DEM_CACHE_MAX_SIZE = 100

# --- Радиогоризонт (Radio Horizon)
# Использовать ретина-тайлы для радиогоризонта (False = 256px, экономит память в 4 раза)
RADIO_HORIZON_USE_RETINA = False
# Цветовая шкала для карты радиогоризонта: (t, (R, G, B))
# t — нормализованное значение от 0.0 (0 м) до 1.0 (RADIO_HORIZON_MAX_HEIGHT_M)
RADIO_HORIZON_COLOR_RAMP: list[tuple[float, tuple[int, int, int]]] = [
    (0.0, (0, 128, 0)),  # тёмно-зелёный: 0 м (прямая видимость без подъёма)
    (0.1, (50, 205, 50)),  # лайм: ~50 м
    (0.2, (144, 238, 144)),  # светло-зелёный: ~100 м
    (0.4, (255, 255, 0)),  # жёлтый: ~200 м
    (0.6, (255, 165, 0)),  # оранжевый: ~300 м
    (0.8, (255, 69, 0)),  # красно-оранжевый: ~400 м
    (1.0, (139, 0, 0)),  # тёмно-красный: 500+ м
]
# Максимальная высота шкалы радиогоризонта (метры)
RADIO_HORIZON_MAX_HEIGHT_M = 500.0
# Коэффициент атмосферной рефракции (k=4/3 для стандартной атмосферы)
RADIO_HORIZON_REFRACTION_K = 4.0 / 3.0
# Шаг сетки расчёта (каждый N-й пиксель) для оптимизации производительности
# Автоматически увеличивается для больших изображений
RADIO_HORIZON_GRID_STEP = 4
# Цвет для точек за пределами радиогоризонта (недостижимые)
RADIO_HORIZON_UNREACHABLE_COLOR = (64, 64, 64)  # тёмно-серый
# Максимальное количество пикселей DEM для радиогоризонта (16 млн = 4000×4000)
# При превышении DEM автоматически даунсэмплится
RADIO_HORIZON_MAX_DEM_PIXELS = 16_000_000
# Коэффициент прозрачности цветовой карты радиогоризонта при наложении на
# топографическую основу
# 0.0 = полностью прозрачный (только топо), 1.0 = полностью непрозрачный (только цвета)
RADIO_HORIZON_TOPO_OVERLAY_ALPHA = 0.7
# Эпсилон для ограничения координат интерполяции внутри границ DEM
RADIO_HORIZON_INTERPOLATION_EDGE_EPSILON = 1.001
# Минимальное расстояние (в квадрате пикселей) для трассировки LOS
RADIO_HORIZON_LOS_MIN_DISTANCE_PX_SQ = 1.0
# Параметры дискретизации трассировки LOS
RADIO_HORIZON_LOS_STEPS_MAX = 200
RADIO_HORIZON_LOS_STEPS_MIN = 2
RADIO_HORIZON_LOS_STEP_DIVISOR = 2
# Инициализация и порог отсутствия данных для угла затенения
RADIO_HORIZON_MAX_ELEVATION_ANGLE_INIT = -1e30
RADIO_HORIZON_MAX_ELEVATION_ANGLE_NO_DATA_THRESHOLD = -1e29
# Минимальный запас высоты в целевой точке (метры)
RADIO_HORIZON_TARGET_HEIGHT_CLEARANCE_M = 1.0
# Размер LUT для цветовой шкалы радиогоризонта
RADIO_HORIZON_LUT_SIZE = 1024
# Параметры пустого изображения для вырожденных случаев
RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX = (1, 1)
RADIO_HORIZON_EMPTY_IMAGE_COLOR = (128, 128, 128)
# Пороги адаптивного шага сетки по количеству пикселей
RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_LARGE = 64_000_000
RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_MEDIUM = 16_000_000
RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL = 4_000_000
# Значения шага сетки для крупных изображений
RADIO_HORIZON_GRID_STEP_LARGE = 32
RADIO_HORIZON_GRID_STEP_MEDIUM = 16
RADIO_HORIZON_GRID_STEP_SMALL = 8
# Единица измерения для легенды радиогоризонта
RADIO_HORIZON_LEGEND_UNIT_LABEL = 'м'
# Минимальная высота для легенды радиогоризонта
RADIO_HORIZON_MIN_HEIGHT_M = 0.0

# --- Зона обнаружения РЛС (Radar Coverage)
# Использовать ретина-тайлы для РЛС
RADAR_COVERAGE_USE_RETINA = False
# Цвет затемнения вне сектора обзора (RGBA)
RADAR_COVERAGE_SECTOR_SHADOW_COLOR = (0, 0, 0, 160)
# Цвет границ сектора (RGBA)
RADAR_COVERAGE_SECTOR_BORDER_COLOR = (255, 255, 0, 200)
# Толщина линий границ сектора (метры)
RADAR_COVERAGE_SECTOR_BORDER_WIDTH_M = 8.0
# Цвет дуг потолка (RGBA)
RADAR_COVERAGE_CEILING_ARC_COLOR = (255, 200, 0, 180)
# Толщина линий дуг потолка (метры)
RADAR_COVERAGE_CEILING_ARC_WIDTH_M = 4.0
# Высоты для дуг потолка (метры, типовые высоты БпЛА)
RADAR_COVERAGE_CEILING_HEIGHTS_M = (500, 1000, 3000, 5000)
# Размер маркера РЛС (метры)
RADAR_COVERAGE_MARKER_SIZE_M = 150.0
# Цвет маркера РЛС (RGB)
RADAR_COVERAGE_MARKER_COLOR = (0, 0, 255)
# Длина линии направления маркера (метры)
RADAR_COVERAGE_MARKER_DIR_LENGTH_M = 300.0
# Шаг вращения азимута Shift+колесо (градусы)
RADAR_COVERAGE_AZIMUTH_STEP_DEG = 5.0

# Минимальный радиус дуги потолка РЛС (пиксели)
RADAR_CEILING_ARC_MIN_RADIUS_PX = 5
# Максимальная ширина сектора для подписей на обоих лучах (градусы)
RADAR_CEILING_LABEL_MAX_SECTOR_DEG = 330.0
# Нижняя граница нормализованного угла для определения перевёрнутости текста (градусы)
TEXT_UPSIDE_DOWN_LOW_DEG = 90
# Верхняя граница нормализованного угла для определения перевёрнутости текста (градусы)
TEXT_UPSIDE_DOWN_HIGH_DEG = 270

# Атрибут DWM для тёмной темы заголовка окна (Windows API)
DWMWA_USE_IMMERSIVE_DARK_MODE = 20

# Битовая маска атрибута «скрытый файл» Windows API
WIN32_FILE_ATTRIBUTE_HIDDEN = 0x2

# --- Профиль радиолинии (Link Profile)
LINK_PROFILE_DEFAULT_FREQ_MHZ = 900.0
LINK_PROFILE_REFRACTION_K = 4.0 / 3.0
LINK_PROFILE_NUM_SAMPLES = 500
LINK_PROFILE_DEFAULT_ANTENNA_A_M = 10.0
LINK_PROFILE_DEFAULT_ANTENNA_B_M = 10.0
SPEED_OF_LIGHT_MPS = 299_792_458.0
# Врезка профиля
LINK_PROFILE_INSET_HEIGHT_RATIO = 0.25
LINK_PROFILE_INSET_BG_COLOR = (255, 255, 255, 255)
LINK_PROFILE_INSET_MARGIN_H = 0.03   # горизонтальные поля (лево = право), доля ширины
LINK_PROFILE_INSET_MARGIN_V = 0.18   # вертикальные поля (верх = низ), доля высоты
LINK_PROFILE_TERRAIN_FILL_COLOR = (139, 119, 101, 200)
LINK_PROFILE_LOS_COLOR = (255, 0, 0)
LINK_PROFILE_FRESNEL_FILL_COLOR = (255, 165, 0, 60)
LINK_PROFILE_FRESNEL_BORDER_COLOR = (255, 0, 0, 255)
LINK_PROFILE_LOS_LINE_COLOR = (0, 0, 0, 255)
LINK_PROFILE_LINE_WIDTH_PX = 10
LINK_PROFILE_POINT_A_COLOR = (0, 0, 255)
LINK_PROFILE_POINT_B_COLOR = (255, 0, 0)
# Использовать ретина-тайлы для Link Profile (False = 256px)
LINK_PROFILE_USE_RETINA = False

# --- Логирование ---
# Дублировать лог в файл с fsync (переживает OOM/crash)
LOG_FSYNC_TO_FILE = True

# --- OOM Prevention ---
# Доля доступной RAM, которую можно использовать
MEMORY_SAFETY_RATIO = 0.75
# Минимум свободной памяти, которую нужно оставить (МБ)
MEMORY_MIN_FREE_MB = 512
