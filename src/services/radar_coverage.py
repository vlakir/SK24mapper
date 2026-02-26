"""Специфика визуализации зоны обнаружения РЛС: сектор, затемнение, дуги потолка."""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from shared.constants import (
    RADAR_CEILING_ARC_MIN_RADIUS_PX,
    RADAR_CEILING_LABEL_MAX_SECTOR_DEG,
    RADAR_COVERAGE_CEILING_ARC_COLOR,
    RADAR_COVERAGE_CEILING_ARC_WIDTH_M,
    RADAR_COVERAGE_CEILING_HEIGHTS_M,
    RADAR_COVERAGE_SECTOR_BORDER_COLOR,
    RADAR_COVERAGE_SECTOR_BORDER_WIDTH_M,
    TEXT_UPSIDE_DOWN_HIGH_DEG,
    TEXT_UPSIDE_DOWN_LOW_DEG,
)


def draw_sector_overlay(
    img: Image.Image,
    cx: float,
    cy: float,
    azimuth_deg: float,
    sector_width_deg: float,
    max_range_px: float,
    pixel_size_m: float,
    elevation_max_deg: float = 30.0,
    ceiling_heights: tuple[float, ...] = RADAR_COVERAGE_CEILING_HEIGHTS_M,
    font: ImageFont.FreeTypeFont | None = None,
    rotation_deg: float = 0.0,
) -> None:
    """
    Рисует затемнение вне сектора, границы сектора и дуги потолка.

    Работает in-place на RGBA-изображении.

    Args:
        img: RGBA-изображение для рисования.
        cx: Координата X РЛС в пикселях изображения.
        cy: Координата Y РЛС в пикселях изображения.
        azimuth_deg: Азимут направления РЛС (0=север, по часовой).
        sector_width_deg: Ширина сектора обзора (градусы).
        max_range_px: Максимальная дальность обнаружения в пикселях.
        pixel_size_m: Размер пикселя в метрах.
        elevation_max_deg: Максимальный угол места (для расчёта дуг потолка).
        ceiling_heights: Высоты для дуг потолка (метры).
        font: Шрифт для подписей дуг потолка (None = без подписей).
        rotation_deg: Угол поворота карты (компенсация для совмещения
            с пиксельной маской).

    """
    if img.mode != 'RGBA':
        return

    w, h = img.size
    half_sector = sector_width_deg / 2.0

    # Effective azimuth in image coordinates: compensate map rotation.
    # The pixel-level sector mask was applied before rotation, so it rotated
    # with the image.  Border lines are drawn after rotation and must match.
    eff_az = azimuth_deg - rotation_deg

    # PIL angle convention: 0°=east, CW on screen (Y down)
    # Geographic azimuth 0°=north → PIL = azimuth - 90°
    pil_center_angle = eff_az - 90.0
    pil_start = pil_center_angle - half_sector
    pil_end = pil_center_angle + half_sector

    # Bounding box for sector arc
    r = max_range_px + 10
    bbox = [cx - r, cy - r, cx + r, cy + r]

    # No shadow overlay — pixels outside the sector are already colored
    # as unreachable by the coverage kernel (same gray as blocked areas).

    # === Границы сектора (линии) ===
    border_w = max(1, round(RADAR_COVERAGE_SECTOR_BORDER_WIDTH_M / pixel_size_m))
    border_color = RADAR_COVERAGE_SECTOR_BORDER_COLOR
    draw = ImageDraw.Draw(img)

    # Левая граница сектора
    left_angle_rad = math.radians(eff_az - half_sector)
    lx = cx + max_range_px * math.sin(left_angle_rad)
    ly = cy - max_range_px * math.cos(left_angle_rad)
    draw.line([(cx, cy), (lx, ly)], fill=border_color, width=border_w)

    # Правая граница сектора
    right_angle_rad = math.radians(eff_az + half_sector)
    rx = cx + max_range_px * math.sin(right_angle_rad)
    ry = cy - max_range_px * math.cos(right_angle_rad)
    draw.line([(cx, cy), (rx, ry)], fill=border_color, width=border_w)

    # Внешняя дуга сектора
    draw.arc(bbox, start=pil_start, end=pil_end, fill=border_color, width=border_w)

    # === 3. Дуги потолка ===
    _draw_ceiling_arcs(
        img,
        draw,
        cx,
        cy,
        eff_az,
        sector_width_deg,
        elevation_max_deg,
        pixel_size_m,
        ceiling_heights,
        max_range_px,
        font,
    )
    del draw


def _draw_ceiling_arcs(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    azimuth_deg: float,
    sector_width_deg: float,
    elevation_max_deg: float,
    pixel_size_m: float,
    ceiling_heights: tuple[float, ...],
    max_range_px: float,
    font: ImageFont.FreeTypeFont | None = None,
) -> None:
    """
    Рисует дуги потолка внутри сектора с подписями высот.

    Дуга потолка — это расстояние, на котором цель на высоте H
    оказывается на максимальном угле места РЛС:
    range = H / tan(elevation_max_deg)
    """
    if elevation_max_deg <= 0:
        return

    half_sector = sector_width_deg / 2.0
    tan_max = math.tan(math.radians(elevation_max_deg))
    if tan_max <= 0:
        return

    arc_color = RADAR_COVERAGE_CEILING_ARC_COLOR
    arc_width = max(1, round(RADAR_COVERAGE_CEILING_ARC_WIDTH_M / pixel_size_m))

    pil_center_angle = azimuth_deg - 90.0
    pil_start = pil_center_angle - half_sector
    pil_end = pil_center_angle + half_sector

    for height_m in ceiling_heights:
        # Дальность, на которой потолок совпадает с max elevation
        ceiling_range_m = height_m / tan_max
        ceiling_range_px = ceiling_range_m / pixel_size_m

        if ceiling_range_px > max_range_px:
            continue  # Дуга за пределами дальности РЛС
        if ceiling_range_px < RADAR_CEILING_ARC_MIN_RADIUS_PX:
            continue  # Слишком маленькая

        r = ceiling_range_px
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.arc(bbox, start=pil_start, end=pil_end, fill=arc_color, width=arc_width)

        # Подписи высот вдоль лучей границ сектора.
        # Текст поворачивается вдоль луча для читаемости.
        if font is not None:
            label = f'{int(height_m)}м'
            perp_px = 40

            # Правый луч
            right_rad = math.radians(azimuth_deg + half_sector)
            _paste_rotated_label(
                img,
                label,
                font,
                arc_color,
                cx,
                cy,
                ceiling_range_px,
                right_rad,
                perp_px,
                side='right',
            )

            # Левый луч (только если сектор < 330°)
            if sector_width_deg < RADAR_CEILING_LABEL_MAX_SECTOR_DEG:
                left_rad = math.radians(azimuth_deg - half_sector)
                _paste_rotated_label(
                    img,
                    label,
                    font,
                    arc_color,
                    cx,
                    cy,
                    ceiling_range_px,
                    left_rad,
                    perp_px,
                    side='left',
                )


def _paste_rotated_label(
    img: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, ...],
    cx: float,
    cy: float,
    radius_px: float,
    ray_rad: float,
    perp_px: float,
    side: str = 'right',
) -> None:
    """
    Рисует подпись, повёрнутую вдоль луча, и вставляет в изображение.

    Args:
        img: Целевое RGBA-изображение.
        text: Текст подписи.
        font: Шрифт.
        color: Цвет текста (RGBA или RGB).
        cx: Координата X центра (позиция РЛС) в пикселях.
        cy: Координата Y центра (позиция РЛС) в пикселях.
        radius_px: Расстояние от центра до дуги (пиксели).
        ray_rad: Угол луча в радианах (географический азимут).
        perp_px: Перпендикулярный отступ от луча (пиксели).
        side: 'right' — подпись справа от луча (CW), 'left' — слева (CCW).

    """
    # 1. Рисуем текст горизонтально на прозрачном холсте
    tmp_measure = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(tmp_measure).textbbox((0, 0), text, font=font)
    tmp_measure.close()
    del tmp_measure
    tw = int(bbox[2] - bbox[0] + 4)
    th = int(bbox[3] - bbox[1] + 4)
    tmp = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((-bbox[0] + 2, -bbox[1] + 2), text, fill=color, font=font)

    # 2. Угол поворота текста: вдоль луча.
    #    Географ. азимут 0°=север → на экране луч идёт вверх → текст надо
    #    повернуть на -(azimuth - 90°) = (90° - azimuth) для PIL rotate.
    ray_deg = math.degrees(ray_rad)
    rotate_angle = 90.0 - ray_deg

    # Если текст окажется вверх ногами (> ±90° от горизонта), развернём на 180°.
    norm = rotate_angle % 360
    if TEXT_UPSIDE_DOWN_LOW_DEG < norm < TEXT_UPSIDE_DOWN_HIGH_DEG:
        rotate_angle += 180.0

    rotated = tmp.rotate(rotate_angle, resample=Image.Resampling.BICUBIC, expand=True)

    # 3. Точка привязки: на луче + перпендикулярный отступ наружу
    on_ray_x = cx + radius_px * math.sin(ray_rad)
    on_ray_y = cy - radius_px * math.cos(ray_rad)

    if side == 'right':
        label_x = on_ray_x + perp_px * math.cos(ray_rad)
        label_y = on_ray_y + perp_px * math.sin(ray_rad)
    else:
        label_x = on_ray_x - perp_px * math.cos(ray_rad)
        label_y = on_ray_y - perp_px * math.sin(ray_rad)

    # Центрируем повёрнутое изображение на точке привязки
    paste_x = int(label_x - rotated.width / 2)
    paste_y = int(label_y - rotated.height / 2)

    img.paste(rotated, (paste_x, paste_y), rotated)
    tmp.close()
    rotated.close()
    del tmp, rotated
