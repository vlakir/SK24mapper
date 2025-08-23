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
