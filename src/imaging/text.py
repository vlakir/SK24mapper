"""Text rendering utilities - fonts, labels, outlines."""

import logging
import math

from PIL import ImageDraw, ImageFont

from shared.constants import (
    GRID_FONT_PATH,
    GRID_FONT_PATH_BOLD,
    GRID_LABEL_BG_COLOR,
    GRID_TEXT_COLOR,
    GRID_TEXT_OUTLINE_COLOR,
    GRID_TEXT_OUTLINE_WIDTH,
)

logger = logging.getLogger(__name__)


def draw_text_with_outline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = GRID_TEXT_COLOR,
    outline: tuple[int, int, int] = GRID_TEXT_OUTLINE_COLOR,
    outline_width: int = GRID_TEXT_OUTLINE_WIDTH,
    anchor: str = 'lt',
) -> None:
    """Рисует текст с «обводкой» для лучшей читаемости."""
    x, y = xy
    if outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text(
                    (x + dx, y + dy),
                    text,
                    font=font,
                    fill=outline,
                    anchor=anchor,
                )
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def draw_label_with_bg(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    anchor: str,
    img_size: tuple[int, int],
    bg_color: tuple[int, int, int] = GRID_LABEL_BG_COLOR,
    padding: int = 6,
) -> None:
    """
    Рисует жёлтую подложку под подписью, затем сам текст (с обводкой).

    Подложка рисуется только если после обрезки по границам изображения прямоугольник
    не вырожден.
    """
    x, y = xy
    w, h = img_size
    bbox = draw.textbbox((x, y), text, font=font, anchor=anchor)
    left = max(0, math.floor(bbox[0] - padding))
    top = max(0, math.floor(bbox[1] - padding))
    right = min(w, math.ceil(bbox[2] + padding))
    bottom = min(h, math.ceil(bbox[3] + padding))
    if right > left and bottom > top:
        draw.rectangle([left, top, right, bottom], fill=bg_color)
    draw_text_with_outline(draw, xy, text, font=font, anchor=anchor)


def draw_label_with_subscript_bg(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    parts: list[tuple[str, bool]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    subscript_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    anchor: str,
    img_size: tuple[int, int],
    bg_color: tuple[int, int, int] = GRID_LABEL_BG_COLOR,
    padding: int = 6,
) -> None:
    """
    Рисует текст с подстрочными индексами и жёлтой подложкой.

    Args:
        draw: ImageDraw объект для рисования
        xy: Координаты начала текста (интерпретируются согласно anchor)
        parts: Список кортежей (текст, is_subscript). Если is_subscript=True,
               текст рисуется подстрочным шрифтом со смещением вниз.
        font: Основной шрифт
        subscript_font: Шрифт для подстрочных индексов (меньшего размера)
        anchor: Якорь позиционирования ('mt' = middle-top, 'lt' = left-top, и т.д.)
        img_size: Размер изображения (width, height)
        bg_color: Цвет фона подложки
        padding: Отступ подложки от текста

    """
    x, y = xy
    w, h = img_size

    # Вычисляем общую ширину и высоту текста для определения bbox
    total_width = 0.0
    max_height = 0.0
    max_ascent = 0.0

    # Получаем метрики основного шрифта для baseline
    if isinstance(font, ImageFont.FreeTypeFont):
        main_ascent, main_descent = (float(value) for value in font.getmetrics())
    else:
        main_ascent, _main_descent = 0.0, 0.0

    if isinstance(subscript_font, ImageFont.FreeTypeFont):
        sub_ascent, sub_descent = (
            float(value) for value in subscript_font.getmetrics()
        )
    else:
        _sub_ascent, _sub_descent = 0.0, 0.0

    # Вычисляем размеры каждой части
    part_metrics = []
    for text_part, is_sub in parts:
        f = subscript_font if is_sub else font
        bbox_part = draw.textbbox((0, 0), text_part, font=f, anchor='lt')
        part_w = float(bbox_part[2] - bbox_part[0])
        part_h = float(bbox_part[3] - bbox_part[1])
        part_metrics.append((text_part, is_sub, part_w, part_h, f))
        total_width += part_w
        if not is_sub:
            max_height = max(max_height, part_h)
            max_ascent = max(max_ascent, main_ascent)

    # Если нет основного текста, используем размеры subscript
    if max_height == 0:
        for _, _, _, part_h, _ in part_metrics:
            max_height = max(max_height, part_h)

    # Определяем начальную позицию X в зависимости от anchor
    if anchor.startswith('m'):  # middle
        start_x = x - total_width / 2
    elif anchor.startswith('r'):  # right
        start_x = x - total_width
    else:  # left (default)
        start_x = x

    # Определяем позицию Y в зависимости от anchor
    if anchor.endswith('t'):  # top
        base_y = y
    elif anchor.endswith('m'):  # middle
        base_y = y - max_height / 2
    elif anchor.endswith('b'):  # bottom
        base_y = y - max_height
    else:
        base_y = y

    # Вычисляем общий bounding box для подложки
    bbox_left = start_x
    bbox_top = base_y
    bbox_right = start_x + total_width
    bbox_bottom = base_y + max_height

    # Рисуем подложку
    bg_left = max(0, math.floor(bbox_left - padding))
    bg_top = max(0, math.floor(bbox_top - padding))
    bg_right = min(w, math.ceil(bbox_right + padding))
    bg_bottom = min(h, math.ceil(bbox_bottom + padding))
    if bg_right > bg_left and bg_bottom > bg_top:
        draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=bg_color)

    # Рисуем текст по частям
    current_x = start_x
    for text_part, is_sub, part_w, part_h, f in part_metrics:
        text_y = base_y + max_height - part_h if is_sub else base_y

        draw_text_with_outline(
            draw, (current_x, text_y), text_part, font=f, anchor='lt'
        )
        current_x += part_w


def load_grid_font(font_size: int = 86) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Подгружает масштабируемый шрифт для подписей (предпочтительно — жирный).

    Порядок:
      1) GRID_FONT_PATH_BOLD (если задан) — жирный шрифт.
      2) GRID_FONT_PATH (если задан) — обычный шрифт.
      3) Системные шрифты (Windows, Linux, macOS).
      4) Резерв: встроенный маленький шрифт PIL.
    """
    if GRID_FONT_PATH_BOLD:
        try:
            return ImageFont.truetype(GRID_FONT_PATH_BOLD, font_size)
        except Exception:
            logger.debug('Failed to load bold grid font from %s', GRID_FONT_PATH_BOLD)
    if GRID_FONT_PATH:
        try:
            return ImageFont.truetype(GRID_FONT_PATH, font_size)
        except Exception:
            logger.debug('Failed to load grid font from %s', GRID_FONT_PATH)

    _system_fonts = (
        # Windows
        'arialbd.ttf',
        'arial.ttf',
        'segoeuib.ttf',
        'segoeui.ttf',
        'tahomabd.ttf',
        'tahoma.ttf',
        # Linux (абсолютные пути — truetype() не ищет по системным каталогам)
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
        # macOS
        '/System/Library/Fonts/Helvetica.ttc',
        '/Library/Fonts/Arial Bold.ttf',
        '/Library/Fonts/Arial.ttf',
    )
    for name in _system_fonts:
        try:
            return ImageFont.truetype(name, font_size)
        except Exception:
            logger.debug('Шрифт %s не найден, пробуем следующий', name)
            continue
    logger.warning(
        'Не удалось загрузить ни один масштабируемый шрифт, '
        'подписи будут мелкими (запрошено %d px)',
        font_size,
    )
    return ImageFont.load_default()
