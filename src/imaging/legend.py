"""Legend drawing utilities - elevation legend for maps."""

import logging

import numpy as np
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
    grid_font_size_m: float | None = None,
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
        grid_font_size_m: Размер шрифта сетки в метрах (опционально)

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

    # Рассчитываем размер шрифта
    if grid_font_size_m is not None and ppm > 0:
        font_size = max(10, round(grid_font_size_m * ppm))
    else:
        font_size = int(legend_height * LEGEND_LABEL_FONT_RATIO)
        font_size = max(
            LEGEND_LABEL_FONT_MIN_PX, min(font_size, LEGEND_LABEL_FONT_MAX_PX)
        )

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
    # Центр последнего квадрата находится на расстоянии (0.5 * grid_square)
    # от правого края
    legend_center_x = (
        last_square_right - grid_square_px * LEGEND_HORIZONTAL_POSITION_RATIO
    )
    # Позиция левого края легенды
    legend_x = int(legend_center_x - legend_width / 2.0)

    # Вертикально: нижняя граница легенды немного выше первой горизонтальной линии сетки
    # Первая горизонтальная линия снизу находится на высоте grid_square_px
    # от нижнего края
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

    import time as _legend_time
    _sub_t0 = _legend_time.monotonic()
    _subs: list[tuple[str, float]] = []

    def _mark(label: str) -> None:
        nonlocal _sub_t0
        now = _legend_time.monotonic()
        _subs.append((label, now - _sub_t0))
        _sub_t0 = now

    _mark('setup')

    # Рисуем полупрозрачный фон только в области легенды.
    #
    # Раньше создавался полноразмерный RGBA-оверлей и alpha_composite по
    # всему изображению (на 9580² карты это ~600ms на фон одного маленького
    # прямоугольника в углу!). Теперь:
    #   1) clamp region до границ изображения,
    #   2) crop кусок img (~800×400 px),
    #   3) накладываем полупрозрачную заливку на этот crop,
    #   4) paste обратно по координатам.
    # Для 9580² карты ускорение ~0.6s → ~0.01s — bg-rect занимает ~1% пикселей.
    bg_region = (
        max(0, bg_x1),
        max(0, bg_y1),
        min(w, bg_x2),
        min(h, bg_y2),
    )
    rx1, ry1, rx2, ry2 = bg_region
    if rx2 > rx1 and ry2 > ry1:
        bg_crop = img.crop(bg_region)
        if bg_crop.mode != 'RGBA':
            bg_crop_rgba = bg_crop.convert('RGBA')
            bg_crop.close()
            bg_crop = bg_crop_rgba
        overlay_small = Image.new(
            'RGBA', (rx2 - rx1, ry2 - ry1), LEGEND_BACKGROUND_COLOR
        )
        bg_blended = Image.alpha_composite(bg_crop, overlay_small)
        bg_crop.close()
        overlay_small.close()
        if img.mode == 'RGB':
            bg_paste = bg_blended.convert('RGB')
            bg_blended.close()
            img.paste(bg_paste, (rx1, ry1))
            bg_paste.close()
        else:
            img.paste(bg_blended, (rx1, ry1))
            bg_blended.close()
    # Обновляем draw object для дальнейшего рисования
    draw = ImageDraw.Draw(img)
    _mark('background')

    # Рисуем цветовую полосу одним vectorized paste вместо 600-900
    # PIL.line() вызовов в Python-loop'е (на адаптивной 10%-высоте
    # карты при mpp=0.78 м/px legend_height = 800+ строк = ~250 ms
    # только на overhead Python→PIL→C на каждой линии).
    #
    # np.interp делает кусочно-линейную интерполяцию по color_ramp
    # для всех y-позиций сразу, отдельно для каждого канала RGB.
    # Результат — `column` shape (H, 3); broadcast в (H, W, 3) даёт
    # gradient без копирования по width-оси. ascontiguousarray
    # материализует один раз для Image.fromarray.
    if legend_height > 0 and legend_width > 0:
        ts_desc = (
            1.0 - np.arange(legend_height, dtype=np.float64)
            / max(1, legend_height - 1)
        )
        ramp_ts = np.array([t for t, _ in color_ramp], dtype=np.float64)
        ramp_rgb = np.array([rgb for _, rgb in color_ramp], dtype=np.float64)
        channels = [
            np.interp(ts_desc, ramp_ts, ramp_rgb[:, c]) for c in range(3)
        ]
        column = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        gradient = np.broadcast_to(
            column[:, None, :], (legend_height, legend_width, 3),
        )
        gradient_img = Image.fromarray(np.ascontiguousarray(gradient), 'RGB')
        img.paste(gradient_img, (legend_x, legend_y))
        gradient_img.close()
    _mark('colorbar')

    # Рисуем рамку вокруг цветовой полосы
    border_width_px = max(1, round(LEGEND_BORDER_WIDTH_M * ppm))
    draw.rectangle(
        [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
        outline=(0, 0, 0),
        width=border_width_px,
    )
    _mark('border')

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
    _mark('title')

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
    _mark('labels')
    logger.info(
        'legend sub-timings: %s',
        ' '.join(f'{name}={t * 1000:.0f}ms' for name, t in _subs),
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
