"""Legend drawing utilities - elevation legend for maps."""

import logging

from PIL import Image, ImageDraw, ImageFont

from geo.topography import meters_per_pixel
from imaging.text import draw_text_with_outline, load_grid_font
from shared.constants import (
    GRID_STEP_M,
    LEGEND_BACKGROUND_COLOR,
    LEGEND_BACKGROUND_PADDING_M,
    LEGEND_BORDER_WIDTH_M,
    LEGEND_GRID_GAP_PADDING_M,
    LEGEND_HEIGHT_MAX_RATIO,
    LEGEND_HEIGHT_MIN_RATIO,
    LEGEND_HEIGHT_RATIO,
    LEGEND_HORIZONTAL_POSITION_RATIO,
    LEGEND_LABEL_FONT_MAX_PX,
    LEGEND_LABEL_FONT_MIN_PX,
    LEGEND_LABEL_FONT_RATIO,
    LEGEND_MARGIN_RATIO,
    LEGEND_MIN_HEIGHT_GRID_SQUARES,
    LEGEND_MIN_MAP_HEIGHT_M_FOR_RATIO,
    LEGEND_NUM_LABELS,
    LEGEND_TEXT_OFFSET_M,
    LEGEND_TEXT_OUTLINE_WIDTH_M,
    LEGEND_TITLE_OFFSET_RATIO,
    LEGEND_VERTICAL_OFFSET_RATIO,
    LEGEND_WIDTH_TO_HEIGHT_RATIO,
    STATIC_SCALE,
)

logger = logging.getLogger(__name__)


def draw_elevation_legend(
    img: Image.Image,
    color_ramp: list[tuple[float, tuple[int, int, int]]],
    min_elevation_m: float,
    max_elevation_m: float,
    center_lat_wgs: float,
    zoom: int,
    scale: int = STATIC_SCALE,
    title: str | None = None,
    label_step_m: float | None = None,
) -> tuple[int, int, int, int]:
    """
    Рисует адаптивную легенду высот в правом нижнем углу карты.

    Высота легенды составляет ~10% от высоты карты, но не менее 1 километрового
    квадрата для карт высотой < 10 км. Все размеры масштабируются пропорционально.

    Args:
        img: Изображение для рисования
        color_ramp: Цветовая палитра [(t, (R, G, B)), ...] где t in [0, 1]
        min_elevation_m: Минимальная высота в метрах
        max_elevation_m: Максимальная высота в метрах
        center_lat_wgs: Широта центра карты в WGS84 (для расчёта пикселей на метр)
        zoom: Уровень масштаба карты
        scale: Масштабный коэффициент (обычно 1 или 2 для retina)
        title: Заголовок легенды (опционально)
        label_step_m: Шаг округления меток высоты (опционально)

    Returns:
        Кортеж (x1, y1, x2, y2) - границы легенды с отступом для разрыва сетки

    """

    def _wrap_legend_title(
        draw_obj: ImageDraw.ImageDraw,
        text: str,
        title_font: ImageFont.ImageFont,
        max_width_px: int,
    ) -> list[str]:
        def _text_width(candidate: str) -> int:
            bbox = draw_obj.textbbox((0, 0), candidate, font=title_font, anchor='lt')
            return int(bbox[2] - bbox[0])

        words = text.split()
        if not words:
            return [text]

        lines: list[str] = []
        current = ''
        for word in words:
            candidate = word if not current else f'{current} {word}'
            if _text_width(candidate) <= max_width_px:
                current = candidate
                continue

            if current:
                lines.append(current)

            current = word

        if current:
            lines.append(current)
        return lines

    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Рассчитываем метры на пиксель и пиксели на метр
    mpp = meters_per_pixel(center_lat_wgs, zoom, scale=scale)
    ppm = 1.0 / mpp if mpp > 0 else 0.0

    # Рассчитываем высоту карты в метрах
    map_height_m = h * mpp

    # Определяем высоту легенды: 10% от высоты карты, но не менее 1 км квадрата
    if map_height_m < LEGEND_MIN_MAP_HEIGHT_M_FOR_RATIO:
        # Для малых карт: минимум 1 километровый квадрат
        legend_height = int(LEGEND_MIN_HEIGHT_GRID_SQUARES * GRID_STEP_M * ppm)
    else:
        # Для больших карт: 10% от высоты
        legend_height = int(h * LEGEND_HEIGHT_RATIO)

    # Ограничиваем высоту легенды долей высоты карты
    min_legend_height = max(1, int(h * LEGEND_HEIGHT_MIN_RATIO))
    max_legend_height = max(min_legend_height, int(h * LEGEND_HEIGHT_MAX_RATIO))
    legend_height = max(min_legend_height, min(legend_height, max_legend_height))

    # Рассчитываем остальные размеры пропорционально высоте легенды
    legend_width = int(legend_height * LEGEND_WIDTH_TO_HEIGHT_RATIO)
    margin = int(legend_height * LEGEND_MARGIN_RATIO)

    # Рассчитываем адаптивный размер шрифта
    font_size = int(legend_height * LEGEND_LABEL_FONT_RATIO)
    font_size = max(LEGEND_LABEL_FONT_MIN_PX, min(font_size, LEGEND_LABEL_FONT_MAX_PX))

    # Загружаем шрифт для подписей и заголовка
    try:
        font = load_grid_font(font_size)
    except Exception:
        font = ImageFont.load_default()

    title_lines = None
    title_gap_px = 0
    title_line_height = 0
    title_line_gap_px = 0
    title_block_width = 0
    title_block_height = 0

    # Рассчитываем размер одного квадрата сетки в пикселях
    grid_square_px = GRID_STEP_M * ppm

    # Новая позиция легенды:
    # Горизонтально: в середине последнего полного километрового квадрата
    # Находим правую границу последнего полного квадрата (с учётом margin)
    last_square_right = w - margin
    # Центр последнего квадрата находится на расстоянии (0.5 * grid_square) от правого края
    legend_center_x = (
        last_square_right - grid_square_px * LEGEND_HORIZONTAL_POSITION_RATIO
    )
    # Позиция левого края легенды
    legend_x = int(legend_center_x - legend_width / 2.0)

    # Вертикально: нижняя граница легенды немного выше первой горизонтальной линии сетки
    # Первая горизонтальная линия снизу находится на высоте grid_square_px от нижнего края
    first_grid_line_y = h - grid_square_px
    # Поднимаем легенду на заданную долю от шага сетки
    legend_y = int(
        first_grid_line_y
        - legend_height
        - grid_square_px * LEGEND_VERTICAL_OFFSET_RATIO
    )

    # Рассчитываем оценку ширины текста для границ легенды
    text_width_estimate = font_size * 6  # примерно "1234 м"

    # Рисуем фон легенды (полупрозрачный белый прямоугольник)
    # Фон на 20% больше легенды в обоих направлениях, легенда по центру фона
    text_offset_px = max(1, round(LEGEND_TEXT_OFFSET_M * ppm))
    legend_total_width = legend_width + text_offset_px + text_width_estimate
    title_extra_offset_px = 0
    if title:
        title_gap_px = max(1, round(LEGEND_TEXT_OFFSET_M * ppm))
        max_title_width = legend_total_width
        title_lines = _wrap_legend_title(draw, title, font, max_title_width)
        title_sizes = [
            draw.textbbox((0, 0), line, font=font, anchor='lt') for line in title_lines
        ]
        title_line_height = int(max(1, *(bbox[3] - bbox[1] for bbox in title_sizes)))
        title_line_gap_px = max(0, round(title_line_height * 0.15))
        title_block_width = int(max(bbox[2] - bbox[0] for bbox in title_sizes))
        title_block_height = title_line_height * len(
            title_lines
        ) + title_line_gap_px * (len(title_lines) - 1)
        title_extra_offset_px = max(1, int(legend_height * LEGEND_TITLE_OFFSET_RATIO))
    legend_total_width = max(legend_total_width, title_block_width)
    legend_total_height = legend_height + (
        title_block_height + title_gap_px + title_extra_offset_px if title_lines else 0
    )

    # Увеличиваем фон на 20% (коэффициент 1.2), добавляя по 10% с каждой стороны
    bg_padding_px = max(1, round(LEGEND_BACKGROUND_PADDING_M * ppm))
    bg_padding_x = max(int(legend_total_width * 0.10), bg_padding_px)
    bg_padding_y = max(int(legend_total_height * 0.10), bg_padding_px)

    title_x = legend_x
    title_y = legend_y
    if title_lines:
        title_y = legend_y - title_gap_px - title_block_height - title_extra_offset_px

    bg_x1 = legend_x - bg_padding_x
    bg_y1 = title_y - bg_padding_y
    bg_x2 = legend_x + legend_total_width + bg_padding_x
    bg_y2 = legend_y + legend_height + bg_padding_y

    # Сдвигаем легенду внутрь изображения, если фон выходит за границы
    shift_x = 0
    if bg_x1 < 0:
        shift_x = -bg_x1
    if bg_x2 + shift_x > w:
        shift_x = w - bg_x2

    shift_y = 0
    if bg_y1 < 0:
        shift_y = -bg_y1
    if bg_y2 + shift_y > h:
        shift_y = h - bg_y2

    if shift_x or shift_y:
        legend_x += shift_x
        legend_y += shift_y
        title_x += shift_x
        title_y += shift_y
        bg_x1 += shift_x
        bg_x2 += shift_x
        bg_y1 += shift_y
        bg_y2 += shift_y

    # Рисуем фон через альфа-композитинг
    if img.mode != 'RGBA':
        # Создаём временное RGBA изображение для наложения фона
        temp_rgba = img.convert('RGBA')
        # Создаём слой с фоном
        bg_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg_overlay)
        bg_draw.rectangle(
            [bg_x1, bg_y1, bg_x2, bg_y2],
            fill=LEGEND_BACKGROUND_COLOR,
        )
        # Накладываем фон
        temp_rgba = Image.alpha_composite(temp_rgba, bg_overlay)
        # Конвертируем обратно в RGB и обновляем исходное изображение
        img_rgb = temp_rgba.convert('RGB')
        img.paste(img_rgb)
        # Обновляем draw object для дальнейшего рисования
        draw = ImageDraw.Draw(img)
    else:
        # Изображение уже в RGBA — рисуем фон напрямую через альфа-композитинг
        bg_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg_overlay)
        bg_draw.rectangle(
            [bg_x1, bg_y1, bg_x2, bg_y2],
            fill=LEGEND_BACKGROUND_COLOR,
        )
        # Накладываем фон на исходное RGBA изображение
        composited = Image.alpha_composite(img, bg_overlay)
        img.paste(composited)
        # Обновляем draw object для дальнейшего рисования
        draw = ImageDraw.Draw(img)

    # Рисуем цветовую полосу (снизу вверх: от низких высот к высоким)
    for i in range(legend_height):
        # t идёт от 1.0 (вверху) до 0.0 (внизу) - высокие высоты сверху
        t = 1.0 - (i / (legend_height - 1)) if legend_height > 1 else 0.0

        # Найти цвет в палитре для данного t
        color = None
        for j in range(1, len(color_ramp)):
            t0, c0 = color_ramp[j - 1]
            t1, c1 = color_ramp[j]
            if t <= t1 or j == len(color_ramp) - 1:
                # Линейная интерполяция
                local = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                r = int(c0[0] + (c1[0] - c0[0]) * local)
                g = int(c0[1] + (c1[1] - c0[1]) * local)
                b = int(c0[2] + (c1[2] - c0[2]) * local)
                color = (r, g, b)
                break

        if color:
            y_pos = legend_y + i
            draw.line(
                [(legend_x, y_pos), (legend_x + legend_width, y_pos)],
                fill=color,
                width=1,
            )

    # Рисуем рамку вокруг цветовой полосы
    border_width_px = max(1, round(LEGEND_BORDER_WIDTH_M * ppm))
    draw.rectangle(
        [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
        outline=(0, 0, 0),
        width=border_width_px,
    )

    # Добавляем заголовок легенды
    if title_lines:
        title_line_step = title_line_height + title_line_gap_px
        for index, line in enumerate(title_lines):
            line_y = title_y + index * title_line_step
            draw_text_with_outline(
                draw,
                (title_x, line_y),
                line,
                font=font,
                fill=(0, 0, 0),
                outline=(255, 255, 255),
                outline_width=max(1, round(LEGEND_TEXT_OUTLINE_WIDTH_M * ppm)),
                anchor='lt',
            )

    # Рисуем метки высоты
    for i in range(LEGEND_NUM_LABELS):
        t = i / (LEGEND_NUM_LABELS - 1) if LEGEND_NUM_LABELS > 1 else 0.0
        elevation = min_elevation_m + (max_elevation_m - min_elevation_m) * t
        if label_step_m:
            elevation = round(elevation / label_step_m) * label_step_m
        label_text = f'{int(elevation)} м'

        # Позиция метки (снизу вверх)
        label_y = legend_y + legend_height - int(t * legend_height)

        # Рисуем текст справа от цветовой полосы с обводкой для читаемости
        text_x = legend_x + legend_width + text_offset_px
        draw_text_with_outline(
            draw,
            (text_x, label_y),
            label_text,
            font=font,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            outline_width=max(1, round(LEGEND_TEXT_OUTLINE_WIDTH_M * ppm)),
            anchor='lm',
        )

    # Возвращаем границы легенды с отступом для разрыва линий сетки
    # Используем увеличенные размеры фона плюс дополнительный отступ
    gap_padding = max(1, round(LEGEND_GRID_GAP_PADDING_M * ppm))
    return (
        bg_x1 - gap_padding,
        bg_y1 - gap_padding,
        bg_x2 + gap_padding,
        bg_y2 + gap_padding,
    )
