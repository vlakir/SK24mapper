"""Потоковая отрисовка сетки и элементов на StreamingImage."""

from __future__ import annotations

import logging
import math
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shared.constants import STREAMING_STRIP_HEIGHT

if TYPE_CHECKING:
    from imaging.streaming import StreamingImage

logger = logging.getLogger(__name__)


def _get_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont:
    """Загружает шрифт или возвращает дефолтный."""
    try:
        if font_path:
            return ImageFont.truetype(font_path, font_size)
        # Пробуем стандартные шрифты
        for name in ['DejaVuSans.ttf', 'arial.ttf', 'Arial.ttf']:
            try:
                return ImageFont.truetype(name, font_size)
            except OSError:
                continue
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _line_intersects_strip(
    x1: float, y1: float, x2: float, y2: float, strip_y: int, strip_h: int
) -> bool:
    """Проверяет, пересекает ли линия полосу."""
    strip_y_end = strip_y + strip_h
    min_y = min(y1, y2)
    max_y = max(y1, y2)
    return max_y >= strip_y and min_y < strip_y_end


def _clip_line_to_strip(
    x1: float, y1: float, x2: float, y2: float, strip_y: int, strip_h: int
) -> tuple[float, float, float, float] | None:
    """Обрезает линию до границ полосы. Возвращает координаты относительно полосы."""
    strip_y_end = strip_y + strip_h

    # Если линия полностью вне полосы
    if max(y1, y2) < strip_y or min(y1, y2) >= strip_y_end:
        return None

    # Клиппинг по Y
    if y1 == y2:
        # Горизонтальная линия
        if strip_y <= y1 < strip_y_end:
            return (x1, y1 - strip_y, x2, y2 - strip_y)
        return None

    # Вычисляем пересечения с границами полосы
    t_top = (strip_y - y1) / (y2 - y1) if y2 != y1 else 0
    t_bottom = (strip_y_end - y1) / (y2 - y1) if y2 != y1 else 1

    if t_top > t_bottom:
        t_top, t_bottom = t_bottom, t_top

    t_start = max(0, t_top)
    t_end = min(1, t_bottom)

    if t_start >= t_end:
        return None

    # Вычисляем обрезанные координаты
    cx1 = x1 + t_start * (x2 - x1)
    cy1 = y1 + t_start * (y2 - y1)
    cx2 = x1 + t_end * (x2 - x1)
    cy2 = y1 + t_end * (y2 - y1)

    # Переводим в координаты полосы
    return (cx1, cy1 - strip_y, cx2, cy2 - strip_y)


def draw_grid_streaming(
    img: 'StreamingImage',
    grid_lines: list[tuple[tuple[float, float], tuple[float, float]]],
    grid_color: tuple[int, int, int],
    line_width: int,
) -> None:
    """
    Рисует линии сетки на StreamingImage (in-place).

    Args:
        img: Изображение (модифицируется in-place)
        grid_lines: Список линий [((x1, y1), (x2, y2)), ...]
        grid_color: Цвет линий RGB
        line_width: Толщина линий в пикселях

    """
    if not grid_lines:
        return

    logger.debug('Drawing %d grid lines', len(grid_lines))

    strip_h = STREAMING_STRIP_HEIGHT

    for strip_y in range(0, img.height, strip_h):
        strip_height = min(strip_h, img.height - strip_y)

        # Находим линии, пересекающие эту полосу
        lines_in_strip = []
        for (x1, y1), (x2, y2) in grid_lines:
            if _line_intersects_strip(x1, y1, x2, y2, strip_y, strip_height):
                clipped = _clip_line_to_strip(x1, y1, x2, y2, strip_y, strip_height)
                if clipped:
                    lines_in_strip.append(clipped)

        if not lines_in_strip:
            continue

        # Загружаем полосу
        strip_data = img.get_strip(strip_y, strip_height)
        pil_strip = Image.fromarray(strip_data)
        draw = ImageDraw.Draw(pil_strip)

        # Рисуем линии
        for cx1, cy1, cx2, cy2 in lines_in_strip:
            draw.line(
                [(cx1, cy1), (cx2, cy2)],
                fill=grid_color,
                width=line_width,
            )

        # Записываем обратно
        img.set_strip(strip_y, np.array(pil_strip))
        del draw
        pil_strip.close()
        del strip_data


def draw_labels_streaming(
    img: 'StreamingImage',
    labels: list[dict],
    font_path: str | None = None,
) -> None:
    """
    Рисует текстовые подписи на StreamingImage (in-place).

    Args:
        img: Изображение (модифицируется in-place)
        labels: Список подписей с параметрами:
            - text: str - текст подписи
            - x: int - X-координата
            - y: int - Y-координата
            - font_size: int - размер шрифта
            - color: tuple[int, int, int] - цвет текста
            - outline_color: tuple[int, int, int] | None - цвет обводки
            - outline_width: int - толщина обводки
            - bg_color: tuple[int, int, int, int] | None - цвет фона (RGBA)
            - anchor: str - якорь текста (по умолчанию 'mm')
        font_path: Путь к шрифту

    """
    if not labels:
        return

    logger.debug('Drawing %d labels', len(labels))

    strip_h = STREAMING_STRIP_HEIGHT

    # Группируем подписи по полосам
    # Сначала оцениваем размеры текста для каждой подписи
    label_bounds = []
    for label in labels:
        font_size = label.get('font_size', 12)
        font = _get_font(font_path, font_size)
        text = label.get('text', '')

        # Оцениваем размер текста
        try:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(text) * font_size // 2
            text_h = font_size

        x = label.get('x', 0)
        y = label.get('y', 0)
        anchor = label.get('anchor', 'mm')

        # Вычисляем границы с учётом якоря
        if 'l' in anchor:
            x_min = x
        elif 'r' in anchor:
            x_min = x - text_w
        else:  # 'm' или по умолчанию
            x_min = x - text_w // 2

        if 't' in anchor:
            y_min = y
        elif 'b' in anchor:
            y_min = y - text_h
        else:  # 'm' или по умолчанию
            y_min = y - text_h // 2

        outline_width = label.get('outline_width', 0)
        bg_padding = label.get('bg_padding', 2)
        padding = outline_width + bg_padding

        label_bounds.append({
            'label': label,
            'y_min': y_min - padding,
            'y_max': y_min + text_h + padding,
            'font': font,
        })

    for strip_y in range(0, img.height, strip_h):
        strip_height = min(strip_h, img.height - strip_y)
        strip_y_end = strip_y + strip_height

        # Находим подписи, пересекающие эту полосу
        labels_in_strip = [
            lb for lb in label_bounds
            if lb['y_max'] >= strip_y and lb['y_min'] < strip_y_end
        ]

        if not labels_in_strip:
            continue

        # Загружаем полосу
        strip_data = img.get_strip(strip_y, strip_height)
        pil_strip = Image.fromarray(strip_data)
        draw = ImageDraw.Draw(pil_strip)

        # Рисуем подписи
        for lb in labels_in_strip:
            label = lb['label']
            font = lb['font']
            text = label.get('text', '')
            x = label.get('x', 0)
            y = label.get('y', 0) - strip_y  # Переводим в координаты полосы
            color = label.get('color', (0, 0, 0))
            outline_color = label.get('outline_color')
            outline_width = label.get('outline_width', 0)
            bg_color = label.get('bg_color')
            anchor = label.get('anchor', 'mm')

            # Рисуем фон если задан
            if bg_color:
                try:
                    bbox = font.getbbox(text)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w = len(text) * font.size // 2
                    text_h = font.size

                # Вычисляем позицию фона
                if 'l' in anchor:
                    bg_x = x
                elif 'r' in anchor:
                    bg_x = x - text_w
                else:
                    bg_x = x - text_w // 2

                if 't' in anchor:
                    bg_y = y
                elif 'b' in anchor:
                    bg_y = y - text_h
                else:
                    bg_y = y - text_h // 2

                padding = label.get('bg_padding', 2)
                draw.rectangle(
                    [bg_x - padding, bg_y - padding,
                     bg_x + text_w + padding, bg_y + text_h + padding],
                    fill=bg_color[:3] if len(bg_color) == 4 else bg_color,
                )

            # Рисуем обводку
            if outline_color and outline_width > 0:
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text(
                                (x + dx, y + dy),
                                text,
                                font=font,
                                fill=outline_color,
                                anchor=anchor,
                            )

            # Рисуем текст
            draw.text((x, y), text, font=font, fill=color, anchor=anchor)

        # Записываем обратно
        img.set_strip(strip_y, np.array(pil_strip))
        del draw
        pil_strip.close()
        del strip_data


def draw_legend_streaming(
    img: 'StreamingImage',
    legend_rect: tuple[int, int, int, int],
    color_ramp: list[tuple[float, tuple[int, int, int]]],
    min_value: float,
    max_value: float,
    labels: list[tuple[float, str]],
    font_path: str | None = None,
    font_size: int = 14,
    border_color: tuple[int, int, int] = (0, 0, 0),
    border_width: int = 1,
    text_color: tuple[int, int, int] = (0, 0, 0),
    text_outline_color: tuple[int, int, int] | None = (255, 255, 255),
    text_outline_width: int = 1,
    bg_color: tuple[int, int, int, int] | None = None,
    bg_padding: int = 5,
) -> None:
    """
    Рисует легенду высот на StreamingImage (in-place).

    Args:
        img: Изображение (модифицируется in-place)
        legend_rect: (x, y, width, height) позиция и размер легенды
        color_ramp: Цветовая шкала [(t, (R, G, B)), ...]
        min_value: Минимальное значение
        max_value: Максимальное значение
        labels: Подписи значений [(value, text), ...]
        font_path: Путь к шрифту
        font_size: Размер шрифта
        border_color: Цвет рамки
        border_width: Толщина рамки
        text_color: Цвет текста
        text_outline_color: Цвет обводки текста
        text_outline_width: Толщина обводки текста
        bg_color: Цвет фона (RGBA)
        bg_padding: Отступ фона

    """
    x, y, w, h = legend_rect

    logger.debug('Drawing legend at (%d, %d) size %dx%d', x, y, w, h)

    strip_h = STREAMING_STRIP_HEIGHT

    # Определяем полосы, которые пересекает легенда
    legend_y_min = y - bg_padding if bg_color else y
    legend_y_max = y + h + bg_padding if bg_color else y + h

    # Создаём LUT для цветовой шкалы
    def interpolate_color(t: float) -> tuple[int, int, int]:
        """Интерполирует цвет по шкале."""
        t = max(0.0, min(1.0, t))
        for i in range(len(color_ramp) - 1):
            t0, c0 = color_ramp[i]
            t1, c1 = color_ramp[i + 1]
            if t0 <= t <= t1:
                if t1 == t0:
                    return c0
                ratio = (t - t0) / (t1 - t0)
                return (
                    int(c0[0] + ratio * (c1[0] - c0[0])),
                    int(c0[1] + ratio * (c1[1] - c0[1])),
                    int(c0[2] + ratio * (c1[2] - c0[2])),
                )
        return color_ramp[-1][1]

    font = _get_font(font_path, font_size)

    for strip_y in range(0, img.height, strip_h):
        strip_height = min(strip_h, img.height - strip_y)
        strip_y_end = strip_y + strip_height

        # Проверяем пересечение с легендой
        if legend_y_max < strip_y or legend_y_min >= strip_y_end:
            continue

        # Загружаем полосу
        strip_data = img.get_strip(strip_y, strip_height)
        pil_strip = Image.fromarray(strip_data)
        draw = ImageDraw.Draw(pil_strip)

        # Координаты в полосе
        local_y = y - strip_y
        local_legend_y_min = legend_y_min - strip_y
        local_legend_y_max = legend_y_max - strip_y

        # Рисуем фон если задан
        if bg_color:
            bg_y0 = max(0, local_legend_y_min)
            bg_y1 = min(strip_height, local_legend_y_max)
            if bg_y0 < bg_y1:
                draw.rectangle(
                    [x - bg_padding, bg_y0, x + w + bg_padding, bg_y1],
                    fill=bg_color[:3] if len(bg_color) == 4 else bg_color,
                )

        # Рисуем цветовую полосу
        for row in range(h):
            row_y = local_y + row
            if 0 <= row_y < strip_height:
                # Нормализованное значение (0 внизу, 1 вверху)
                t = 1.0 - row / h if h > 0 else 0.5
                color = interpolate_color(t)
                draw.line([(x, row_y), (x + w, row_y)], fill=color)

        # Рисуем рамку
        if border_width > 0:
            # Верхняя линия
            if 0 <= local_y < strip_height:
                draw.line([(x, local_y), (x + w, local_y)], fill=border_color, width=border_width)
            # Нижняя линия
            if 0 <= local_y + h < strip_height:
                draw.line([(x, local_y + h), (x + w, local_y + h)], fill=border_color, width=border_width)
            # Левая линия
            for row in range(max(0, local_y), min(strip_height, local_y + h + 1)):
                if 0 <= row < strip_height:
                    for bw in range(border_width):
                        if x + bw < img.width:
                            draw.point((x + bw, row), fill=border_color)
            # Правая линия
            for row in range(max(0, local_y), min(strip_height, local_y + h + 1)):
                if 0 <= row < strip_height:
                    for bw in range(border_width):
                        if x + w - bw >= 0:
                            draw.point((x + w - bw, row), fill=border_color)

        # Рисуем подписи
        for value, text in labels:
            if max_value != min_value:
                t = (value - min_value) / (max_value - min_value)
            else:
                t = 0.5
            label_y = local_y + int((1.0 - t) * h)

            if 0 <= label_y < strip_height:
                label_x = x + w + 5  # Отступ от полосы

                # Обводка
                if text_outline_color and text_outline_width > 0:
                    for dx in range(-text_outline_width, text_outline_width + 1):
                        for dy in range(-text_outline_width, text_outline_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text(
                                    (label_x + dx, label_y + dy),
                                    text,
                                    font=font,
                                    fill=text_outline_color,
                                    anchor='lm',
                                )

                draw.text((label_x, label_y), text, font=font, fill=text_color, anchor='lm')

        # Записываем обратно
        img.set_strip(strip_y, np.array(pil_strip))
        del draw
        pil_strip.close()
        del strip_data


def draw_cross_streaming(
    img: 'StreamingImage',
    center_x: int,
    center_y: int,
    length: int,
    width: int,
    color: tuple[int, int, int],
) -> None:
    """
    Рисует центральный крест на StreamingImage (in-place).

    Args:
        img: Изображение (модифицируется in-place)
        center_x: X-координата центра
        center_y: Y-координата центра
        length: Длина линий креста
        width: Толщина линий
        color: Цвет креста RGB

    """
    half_len = length // 2

    # Горизонтальная линия
    h_line = ((center_x - half_len, center_y), (center_x + half_len, center_y))
    # Вертикальная линия
    v_line = ((center_x, center_y - half_len), (center_x, center_y + half_len))

    draw_grid_streaming(img, [h_line, v_line], color, width)


def draw_control_point_streaming(
    img: 'StreamingImage',
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
    label: str | None = None,
    font_path: str | None = None,
    font_size: int = 14,
    label_color: tuple[int, int, int] = (0, 0, 0),
    label_outline_color: tuple[int, int, int] | None = (255, 255, 255),
    label_outline_width: int = 1,
) -> None:
    """
    Рисует контрольную точку (треугольник) на StreamingImage (in-place).

    Args:
        img: Изображение (модифицируется in-place)
        x: X-координата центра
        y: Y-координата центра
        size: Размер треугольника
        color: Цвет треугольника RGB
        label: Текст подписи
        font_path: Путь к шрифту
        font_size: Размер шрифта
        label_color: Цвет подписи
        label_outline_color: Цвет обводки подписи
        label_outline_width: Толщина обводки

    """
    # Вычисляем вершины треугольника (вершиной вверх)
    half_size = size // 2
    height = int(size * math.sqrt(3) / 2)

    # Вершины: верхняя, левая нижняя, правая нижняя
    top = (x, y - height // 2)
    left = (x - half_size, y + height // 2)
    right = (x + half_size, y + height // 2)

    strip_h = STREAMING_STRIP_HEIGHT

    # Определяем полосы, которые пересекает треугольник
    tri_y_min = top[1]
    tri_y_max = left[1]

    for strip_y in range(0, img.height, strip_h):
        strip_height = min(strip_h, img.height - strip_y)
        strip_y_end = strip_y + strip_height

        # Проверяем пересечение
        if tri_y_max < strip_y or tri_y_min >= strip_y_end:
            continue

        # Загружаем полосу
        strip_data = img.get_strip(strip_y, strip_height)
        pil_strip = Image.fromarray(strip_data)
        draw = ImageDraw.Draw(pil_strip)

        # Координаты в полосе
        local_top = (top[0], top[1] - strip_y)
        local_left = (left[0], left[1] - strip_y)
        local_right = (right[0], right[1] - strip_y)

        # Рисуем треугольник
        draw.polygon([local_top, local_left, local_right], fill=color, outline=color)

        # Записываем обратно
        img.set_strip(strip_y, np.array(pil_strip))
        del draw
        pil_strip.close()
        del strip_data

    # Рисуем подпись если задана
    if label:
        label_y = y + height // 2 + font_size // 2 + 5  # Под треугольником
        labels = [{
            'text': label,
            'x': x,
            'y': label_y,
            'font_size': font_size,
            'color': label_color,
            'outline_color': label_outline_color,
            'outline_width': label_outline_width,
            'anchor': 'mt',
        }]
        draw_labels_streaming(img, labels, font_path)


def draw_polylines_streaming(
    img: 'StreamingImage',
    polylines: list[list[tuple[float, float]]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    """
    Рисует полилинии на StreamingImage (in-place).

    Args:
        img: Изображение (модифицируется in-place)
        polylines: Список полилиний, каждая - список точек [(x, y), ...]
        color: Цвет линий RGB
        width: Толщина линий в пикселях

    """
    if not polylines:
        return

    # Преобразуем полилинии в отдельные сегменты
    lines = []
    for polyline in polylines:
        for i in range(len(polyline) - 1):
            x1, y1 = polyline[i]
            x2, y2 = polyline[i + 1]
            lines.append(((x1, y1), (x2, y2)))

    draw_grid_streaming(img, lines, color, width)


def draw_rotated_labels_streaming(
    img: 'StreamingImage',
    labels: list[dict],
    font_path: str | None = None,
) -> None:
    """
    Рисует повёрнутые текстовые подписи на StreamingImage (in-place).

    Используется для подписей изолиний, которые выровнены вдоль линий.

    Args:
        img: Изображение (модифицируется in-place)
        labels: Список подписей с параметрами:
            - text: str - текст подписи
            - x: float - X-координата центра
            - y: float - Y-координата центра
            - angle_rad: float - угол поворота в радианах
            - font_size: int - размер шрифта
            - color: tuple[int, int, int] - цвет текста
            - outline_color: tuple[int, int, int] | None - цвет обводки
            - outline_width: int - толщина обводки
            - bg_color: tuple[int, int, int, int] | None - цвет фона (RGBA)
        font_path: Путь к шрифту

    """
    if not labels:
        return

    logger.debug('Drawing %d rotated labels', len(labels))

    strip_h = STREAMING_STRIP_HEIGHT

    # Pre-compute label bounds (accounting for rotation)
    label_info: list[dict] = []
    for label in labels:
        font_size = label.get('font_size', 12)
        font = _get_font(font_path, font_size)
        text = label.get('text', '')
        angle_rad = label.get('angle_rad', 0.0)

        # Estimate text size
        try:
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(text) * font_size // 2
            text_h = font_size

        # Add padding for background
        bg_padding = label.get('bg_padding', 2)
        box_w = text_w + 2 * bg_padding
        box_h = text_h + 2 * bg_padding

        # Account for rotation - use rotated bounding box
        angle_deg = math.degrees(angle_rad)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        rotated_w = box_w * cos_a + box_h * sin_a
        rotated_h = box_w * sin_a + box_h * cos_a

        x = label.get('x', 0)
        y = label.get('y', 0)

        # Bounding box in image coordinates
        half_w = rotated_w / 2 + 2
        half_h = rotated_h / 2 + 2
        y_min = y - half_h
        y_max = y + half_h

        label_info.append({
            'label': label,
            'font': font,
            'text_w': text_w,
            'text_h': text_h,
            'box_w': box_w,
            'box_h': box_h,
            'angle_deg': angle_deg,
            'y_min': y_min,
            'y_max': y_max,
            'half_w': half_w,
            'half_h': half_h,
        })

    for strip_y in range(0, img.height, strip_h):
        strip_height = min(strip_h, img.height - strip_y)
        strip_y_end = strip_y + strip_height

        # Find labels intersecting this strip
        labels_in_strip = [
            info for info in label_info
            if info['y_max'] >= strip_y and info['y_min'] < strip_y_end
        ]

        if not labels_in_strip:
            continue

        # Load strip as RGBA for alpha compositing
        strip_data = img.get_strip(strip_y, strip_height)
        # Convert to RGBA
        if strip_data.shape[2] == 3:  # noqa: PLR2004
            strip_rgba = np.zeros(
                (strip_height, strip_data.shape[1], 4), dtype=np.uint8
            )
            strip_rgba[:, :, :3] = strip_data
            strip_rgba[:, :, 3] = 255
        else:
            strip_rgba = strip_data.copy()

        pil_strip = Image.fromarray(strip_rgba, mode='RGBA')

        # Draw each label
        for info in labels_in_strip:
            label = info['label']
            font = info['font']
            text = label.get('text', '')
            x = label.get('x', 0)
            y = label.get('y', 0)
            angle_deg = info['angle_deg']
            text_w = info['text_w']
            text_h = info['text_h']
            box_w = info['box_w']
            box_h = info['box_h']

            color = label.get('color', (0, 0, 0))
            outline_color = label.get('outline_color')
            outline_width = label.get('outline_width', 0)
            bg_color = label.get('bg_color')
            bg_padding = label.get('bg_padding', 2)

            # Create label box image
            box_img = Image.new('RGBA', (int(box_w), int(box_h)), (0, 0, 0, 0))
            box_draw = ImageDraw.Draw(box_img)

            # Draw background
            if bg_color:
                box_draw.rectangle(
                    [0, 0, box_w - 1, box_h - 1],
                    fill=bg_color,
                )

            # Draw outline
            text_x = bg_padding
            text_y = bg_padding
            if outline_color and outline_width > 0:
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            box_draw.text(
                                (text_x + dx, text_y + dy),
                                text,
                                font=font,
                                fill=outline_color,
                            )

            # Draw text
            box_draw.text((text_x, text_y), text, font=font, fill=color)

            # Rotate the label box
            if abs(angle_deg) > 0.1:  # noqa: PLR2004
                rotated = box_img.rotate(
                    angle_deg,
                    expand=True,
                    resample=Image.Resampling.BICUBIC,
                )
            else:
                rotated = box_img

            # Compute paste position (center the rotated image at x, y)
            rot_w, rot_h = rotated.size
            paste_x = int(x - rot_w / 2)
            paste_y = int(y - rot_h / 2) - strip_y  # Convert to strip coordinates

            # Clip to strip bounds
            src_x0 = max(0, -paste_x)
            src_y0 = max(0, -paste_y)
            src_x1 = min(rot_w, pil_strip.width - paste_x)
            src_y1 = min(rot_h, strip_height - paste_y)

            if src_x1 > src_x0 and src_y1 > src_y0:
                dst_x = max(0, paste_x)
                dst_y = max(0, paste_y)

                # Crop rotated image to visible part
                cropped = rotated.crop((src_x0, src_y0, src_x1, src_y1))

                # Alpha composite
                pil_strip.alpha_composite(cropped, dest=(dst_x, dst_y))

            box_img.close()
            if rotated is not box_img:
                rotated.close()

        # Convert back to RGB and write
        result_rgb = np.array(pil_strip.convert('RGB'))
        img.set_strip(strip_y, result_rgb)

        pil_strip.close()
        del strip_data, strip_rgba, result_rgb
