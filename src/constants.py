GOOGLE_STATIC_MAPS_URL = 'https://maps.googleapis.com/maps/api/staticmap'  # Базовый URL Google Static Maps API
STATIC_SIZE_PX = (
    640  # Базовый размер тайла по одной стороне (px) для statics (максимум: обычно 640)
)
STATIC_SCALE = 2  # Масштаб 2 => фактический тайл ~1280x1280 пикселей
MAX_ZOOM = 18  # Максимальный уровень приближения (переменная, подбирается автоматически вниз при больших размерах)
EARTH_RADIUS_M = 6378137.0  # Радиус Земли для Web Mercator (метры)
TILE_SIZE = 256  # Базовый размер тайла Web Mercator (пикселей)

ASYNC_MAX_CONCURRENCY = 20  # Максимальное число параллельных HTTP-запросов
MAX_OUTPUT_PIXELS = 150_000_000  # Ограничение на итоговое число пикселей результирующего кадра (без припуска)
PIL_DISABLE_LIMIT = False  # Отключить защиту Pillow от «бомб декомпрессии» (используйте с осторожностью)

ROTATION_PAD_RATIO = (
    0.07  # Доля припуска на сторону под поворот (7% от меньшей стороны)
)
ROTATION_PAD_MIN_PX = 128  # Минимальный припуск на сторону (px)

# припуск по осям в дополнение к заданным квадратам
ADDITIVE_RATIO = 0.3

GRID_STEP_M = 1000  # Шаг километровой сетки (метры)
GRID_COLOR = (0, 0, 0)  # Цвет линий сетки (RGB)

GRID_FONT_PATH: (
    None  # Путь к TTF/OTF обычного шрифта (если None — использовать DejaVu из Pillow)
)
GRID_TEXT_COLOR = (0, 0, 0)  # Цвет текста подписи (RGB)
GRID_TEXT_OUTLINE_COLOR = (255, 255, 255)  # Цвет обводки текста (RGB)

GRID_LABEL_BG_COLOR = (255, 255, 0)  # Цвет жёлтой подложки под подписью (RGB)

GRID_FONT_BOLD = True  # Использовать жирный шрифт для подписей (если доступен)

GRID_FONT_PATH = None  # Путь к шрифту TTF/OTF (если None — DejaVuSans.ttf)
GRID_FONT_PATH_BOLD = (
    None  # Путь к жирному шрифту TTF/OTF (если None — DejaVuSans-Bold.ttf)
)

ENABLE_WHITE_MASK = True  # Включить наложение белой маски поверх карты
