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

# Максимальное число параллельных HTTP-запросов
ASYNC_MAX_CONCURRENCY = 20

# Ограничение на итоговое число пикселей результирующего кадра (без припуска)
MAX_OUTPUT_PIXELS = 300_000_000

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

# Длина линий крестиков при отключенной сетке (пиксели)
GRID_CROSS_LENGTH_PX = 50

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

# Адаптивный размер шрифта подписей сетки (в километрах)
GRID_LABEL_FONT_KM = 0.05  # 50 метров на карте
# Диапазон клампа в пикселях для подписей сетки
GRID_LABEL_FONT_MIN_PX = 12
GRID_LABEL_FONT_MAX_PX = 120

# Включить наложение белой маски поверх карты

WGS84_CODE = 4326
SK42_CODE = 4284


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
# Цвет обычных изогипс (RGB)
CONTOUR_COLOR = (30, 30, 30)
# Толщина обычных изогипс в пикселях (задаётся здесь)
CONTOUR_WIDTH = 1
# Каждая N‑я изогипса считается «индексной» (выделенной)
CONTOUR_INDEX_EVERY = 5
# Цвет индексных изогипс (RGB)
CONTOUR_INDEX_COLOR = (20, 20, 20)
# Толщина индексных изогипс в пикселях (задаётся здесь)
CONTOUR_INDEX_WIDTH = 2
# --- Подписи изогипс
CONTOUR_LABELS_ENABLED = True
CONTOUR_LABEL_INDEX_ONLY = False
CONTOUR_LABEL_SPACING_PX = 300
CONTOUR_LABEL_MIN_SEG_LEN_PX = 40
CONTOUR_LABEL_EDGE_MARGIN_PX = 8
CONTOUR_LABEL_TEXT_COLOR = (20, 20, 20)
CONTOUR_LABEL_OUTLINE_COLOR = (255, 255, 255)
CONTOUR_LABEL_OUTLINE_WIDTH = 0
# Подложка для подписи изолиний отключена по умолчанию; для включения задайте RGBA-цвет
CONTOUR_LABEL_BG_RGBA = None

CONTOUR_LABEL_BG_PADDING = 3

# --- Разрывы линий контуров для подписей
# Включить разрывы линий в местах размещения подписей
CONTOUR_LABEL_GAP_ENABLED = True
# Дополнительный отступ вокруг подписей для разрыва линии (пиксели)
CONTOUR_LABEL_GAP_PADDING = 5

# Старый фиксированный размер (для бэкапа/совместимости)
CONTOUR_LABEL_FONT_SIZE = 18
# Новый масштабируемый размер шрифта в долях километра
# Одинаковый размер для всех изолиний (обычных и индексных)
CONTOUR_LABEL_FONT_KM = 0.06
# Диапазон клампа в пикселях, чтобы подписи оставались читаемыми
CONTOUR_LABEL_FONT_MIN_PX = 16
CONTOUR_LABEL_FONT_MAX_PX = 128
CONTOUR_LABEL_FONT_PATH = None
CONTOUR_LABEL_FONT_BOLD = True
CONTOUR_LABEL_FORMAT = '{:.0f}'

# --- Оценка поворота
# Отсчёт вдоль восточной оси для оценки угла (метры)
EAST_VECTOR_SAMPLE_M = 200.0

# --- Вспомогательные константы для подписей сетки
GRID_LABEL_THOUSAND_DIV = 1000
GRID_LABEL_MOD = 100
# Доля шага сетки для смещения подписей от линий сетки (0.0 - на линии, 0.5 - в центре квадрата)
GRID_LABEL_OFFSET_FRACTION = 0.125  # 1/8 шага сетки

# --- Общие константы для валидаторов/обработчиков GUI
# Минимальное число аргументов команды скролла (tk Scrollbar callback)

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

# --- Центрированный крест на карте
# Цвет креста (RGB)
CENTER_CROSS_COLOR = (255, 255, 255)
# Толщина линий креста (пиксели)
CENTER_CROSS_LINE_WIDTH_PX = 1
# Полная длина линии креста (пиксели)
CENTER_CROSS_LENGTH_PX = 40

# --- Контрольная точка: крест, аналогичный центральному, но красного цвета
# Цвет креста контрольной точки (RGB)
CONTROL_POINT_CROSS_COLOR = (220, 0, 0)
# Толщина линий креста контрольной точки (пиксели)
CONTROL_POINT_CROSS_LINE_WIDTH_PX = 5
# Полная длина линии креста контрольной точки (пиксели)
CONTROL_POINT_CROSS_LENGTH_PX = 240

# --- Вспомогательные константы для генерации контуров в service.py
# Квантование координат при сцеплении отрезков (чем больше — тем грубее)
SEED_POLYLINE_QUANT_FACTOR = 8
# Вес для усреднения четырёх значений в ячейке (1/4) в алгоритме marching squares
MARCHING_SQUARES_CENTER_WEIGHT = 0.25

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

# Пиксельный припуск по краю временного блока при отрисовке контуров
CONTOUR_BLOCK_EDGE_PAD_PX = 1
# Размер очереди для второго прохода отрисовки контуров
CONTOUR_PASS2_QUEUE_MAXSIZE = 4
# Периодичность логирования использования памяти (каждые N тайлов)
CONTOUR_LOG_MEMORY_EVERY_TILES = 50

# --- Легенда высот на карте (elevation legend)
# Желаемая высота легенды как доля от высоты карты (10%)
LEGEND_HEIGHT_RATIO = 0.10
# Минимальная высота легенды в километровых квадратах (для карт ниже порога высоты)
LEGEND_MIN_HEIGHT_GRID_SQUARES = 1.0
# Порог высоты карты (в километрах), ниже которого легенда занимает минимум 1 км-квадрат
# Используется вместо «магического» числа 10.0 в коде для читаемости и единообразия
LEGEND_MIN_MAP_HEIGHT_KM_FOR_RATIO = 10.0
# Отношение ширины легенды к её высоте
LEGEND_WIDTH_TO_HEIGHT_RATIO = 0.133  # ширина = высота * 0.133
# Отступ легенды от краёв карты как доля от высоты легенды
LEGEND_MARGIN_RATIO = 0.067  # margin = высота_легенды * 0.067
# Количество меток высоты на легенде (мин, макс и промежуточные)
LEGEND_NUM_LABELS = 5
# Отступ текста от цветовой полосы легенды в пикселях
LEGEND_TEXT_OFFSET_PX = 5
# Толщина рамки вокруг цветовой полосы легенды в пикселях
LEGEND_BORDER_WIDTH_PX = 2
# Размер шрифта для подписей легенды как доля от высоты легенды (увеличено на 30%)
LEGEND_LABEL_FONT_RATIO = 0.13  # размер_шрифта = высота_легенды * 0.13
# Диапазон размера шрифта подписей легенды в пикселях (мин и макс)
LEGEND_LABEL_FONT_MIN_PX = 12
LEGEND_LABEL_FONT_MAX_PX = 48
# Толщина обводки текста легенды в пикселях
LEGEND_TEXT_OUTLINE_WIDTH_PX = 2
# Дополнительный отступ вокруг легенды для разрыва линий сетки (пиксели)
LEGEND_GRID_GAP_PADDING_PX = 10
# Цвет фона легенды (RGBA): белый полупрозрачный для визуального выделения
LEGEND_BACKGROUND_COLOR = (255, 255, 255, 230)
# Отступ фона легенды от краёв цветовой полосы (пиксели)
LEGEND_BACKGROUND_PADDING_PX = 8
# Горизонтальная позиция легенды: доля от ширины последнего километрового квадрата (0.5 = середина)
LEGEND_HORIZONTAL_POSITION_RATIO = 0.5
# Вертикальный отступ нижней границы легенды от первой горизонтальной линии сетки (в долях от шага сетки)
LEGEND_VERTICAL_OFFSET_RATIO = 0.15
