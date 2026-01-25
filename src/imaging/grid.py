"""Grid drawing utilities - kilometer grid for SK-42 coordinate system."""

import logging
import math

from PIL import Image, ImageDraw
from pyproj import CRS, Transformer

from geo.topography import crs_sk42_geog, latlng_to_pixel_xy, meters_per_pixel
from imaging.text import (
    calculate_adaptive_grid_font_size,
    draw_label_with_bg,
    load_grid_font,
)
from shared.constants import (
    GRID_COLOR,
    GRID_CROSS_LENGTH_M,
    GRID_LABEL_BG_COLOR,
    GRID_LABEL_MOD,
    GRID_LABEL_THOUSAND_DIV,
    GRID_STEP_M,
    MIN_POINTS_FOR_LINE,
    STATIC_SCALE,
)
from shared.progress import ConsoleProgress

logger = logging.getLogger(__name__)


def draw_axis_aligned_km_grid(
    img: Image.Image,
    center_lat_sk42: float,
    center_lng_sk42: float,
    center_lat_wgs: float,
    center_lng_wgs: float,
    zoom: int,
    crs_sk42_gk: CRS,
    t_sk42_to_wgs: Transformer,
    step_m: int = GRID_STEP_M,
    color: tuple[int, int, int] = GRID_COLOR,
    width_m: float = 5.0,
    scale: int = STATIC_SCALE,
    grid_font_size_m: float = 100.0,
    grid_text_margin_m: float = 50.0,
    grid_label_bg_padding_m: float = 10.0,
    legend_bounds: tuple[int, int, int, int] | None = None,
    *,
    display_grid: bool = True,
    rotation_deg: float = 0.0,
) -> None:
    """
    Рисует километровую сетку (СК‑42/Гаусса–Крюгера) строго по осям изображения.

    Линии сетки прерываются в местах пересечения с легендой высот (если она была
    отрисована и переданы её границы через ``legend_bounds``). При отключённой
    опции отображения сетки (``display_grid=False``) линии и подписи не рисуются —
    вместо них выводятся крестики толщиной 1 пиксель в точках пересечений бывших
    линий сетки.

    Args:
        img: Изображение (Pillow Image), на котором выполняется рисование.
        center_lat_sk42: Широта центра карты в СК‑42 (градусы), используется для
            вычисления положения сетки в проекции Гаусса–Крюгера.
        center_lng_sk42: Долгота центра карты в СК‑42 (градусы), используется для
            вычисления положения сетки в проекции Гаусса–Крюгера.
        center_lat_wgs: Широта центра карты в WGS‑84 (градусы), используется для
            расчёта метры-на‑пиксель (масштаба).
        center_lng_wgs: Долгота центра карты в WGS‑84 (градусы), используется для
            точного преобразования координат сетки в пиксели.
        zoom: Уровень масштабирования карты (число тайлов Web Mercator по оси).
        crs_sk42_gk: Объект CRS (pyproj.CRS) для СК‑42 в зоне Гаусса–Крюгера,
            в которой находится карта.
        t_sk42_to_wgs: Трансформер (pyproj.Transformer) для преобразования
            координат СК‑42 → WGS‑84 (учитывает параметры Хельмерта).
        step_m: Шаг сетки в метрах; по умолчанию 1000 (GRID_STEP_M).
        color: Цвет линий/крестиков сетки в формате RGB.
        width_m: Толщина линий сетки в метрах (конвертируется в пиксели по масштабу).
        scale: Масштабный коэффициент рендеринга тайлов (обычно 1 или 2 для retina).
        grid_font_size_m: Размер шрифта для подписей сетки в метрах.
        grid_text_margin_m: Отступ подписей от краёв изображения в метрах.
        grid_label_bg_padding_m: Внутренний отступ подложки под подписью в метрах.
        legend_bounds: Опциональные границы легенды высот (x1, y1, x2, y2) — в этой
            области линии сетки прерываются, крестики не рисуются.
        display_grid: Признак полного отображения сетки. Если True — рисуются линии
            и подписи; если False — рисуются только крестики на пересечениях.
        rotation_deg: Угол поворота изображения в градусах. Используется для
            корректного преобразования координат сетки при повороте карты.

    Returns:
        None: Функция изменяет переданное изображение на месте, ничего не возвращает.

    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    mpp = meters_per_pixel(center_lat_wgs, zoom, scale=scale)
    ppm = 1.0 / mpp  # pixels per meter

    # Конвертация параметров из метров в пиксели
    width_px = max(1, round(width_m * ppm))
    grid_font_size = max(10, round(grid_font_size_m * ppm))
    grid_text_margin = max(0, round(grid_text_margin_m * ppm))
    grid_label_bg_padding = max(0, round(grid_label_bg_padding_m * ppm))

    # Вычисляем адаптивный размер шрифта на основе масштаба карты
    adaptive_font_size = calculate_adaptive_grid_font_size(mpp)
    # Используем заданный размер шрифта, если он больше адаптивного
    final_font_size = max(adaptive_font_size, grid_font_size)
    font = load_grid_font(final_font_size)

    cx, cy = w / 2.0, h / 2.0

    # Предвычисляем sin/cos для поворота (изображение было повёрнуто на rotation_deg)
    # PIL rotate() работает в системе координат с Y вниз, поэтому
    # используем отрицательный угол
    rotation_rad = math.radians(-rotation_deg)
    cos_rot = math.cos(rotation_rad)
    sin_rot = math.sin(rotation_rad)

    # Important: PROJ/pyproj uses (X, Y) = (easting, northing) when always_xy=True.
    # Military notation in this app is X = northing (север), Y = easting (восток).
    # Therefore, x0_gk is easting (восток), y0_gk is northing (север).
    t_sk42gk_from_sk42 = Transformer.from_crs(
        crs_sk42_geog,
        crs_sk42_gk,
        always_xy=True,
    )
    t_sk42gk_to_sk42 = Transformer.from_crs(
        crs_sk42_gk,
        crs_sk42_geog,
        always_xy=True,
    )
    x0_gk, y0_gk = t_sk42gk_from_sk42.transform(center_lng_sk42, center_lat_sk42)

    # Вычисляем "мировые" пиксельные координаты центра карты для точного преобразования
    cx_world, cy_world = latlng_to_pixel_xy(center_lat_wgs, center_lng_wgs, zoom)

    def gk_to_pixel(x_gk: float, y_gk: float) -> tuple[float, float]:
        """
        Преобразует координаты СК-42 ГК (метры) в пиксели изображения.

        Выполняет полную цепочку преобразований:
        СК-42 ГК → СК-42 географические → WGS-84 → Web Mercator
        пиксели → пиксели изображения
        После этого применяется поворот на угол rotation_deg вокруг центра изображения.
        """
        # СК-42 ГК (метры) → СК-42 географические (градусы)
        lng_sk42, lat_sk42 = t_sk42gk_to_sk42.transform(x_gk, y_gk)
        # СК-42 географические → WGS-84
        lng_wgs, lat_wgs = t_sk42_to_wgs.transform(lng_sk42, lat_sk42)
        # WGS-84 → "мировые" пиксели Web Mercator
        x_world, y_world = latlng_to_pixel_xy(lat_wgs, lng_wgs, zoom)
        # "Мировые" пиксели → пиксели изображения (относительно центра, до поворота)
        x_pre = cx + (x_world - cx_world) * scale
        y_pre = cy + (y_world - cy_world) * scale
        # Применяем поворот вокруг центра изображения
        dx = x_pre - cx
        dy = y_pre - cy
        x_px = cx + dx * cos_rot - dy * sin_rot
        y_px = cy + dx * sin_rot + dy * cos_rot
        return x_px, y_px

    def floor_to_step(v: float, step: float) -> int:
        return int(math.floor(v / step) * step)

    def draw_cross_at_intersection(x_px: float, y_px: float) -> None:
        """Рисует крестик в точке пересечения линий сетки."""
        cross_len_px = max(1, round(GRID_CROSS_LENGTH_M * ppm))
        half = max(1, cross_len_px // 2)
        cross_width = 1  # Толщина крестика всегда 1 пиксель
        # Вертикальная линия крестика
        draw.line(
            [(x_px, y_px - half), (x_px, y_px + half)], fill=color, width=cross_width
        )
        # Горизонтальная линия крестика
        draw.line(
            [(x_px - half, y_px), (x_px + half, y_px)], fill=color, width=cross_width
        )

    def draw_line_with_gap(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        gap_rect: tuple[int, int, int, int] | None,
    ) -> None:
        """Рисует линию с разрывом в области gap_rect (если указана)."""
        if gap_rect is None:
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width_px)
            return

        gx1, gy1, gx2, gy2 = gap_rect

        # Вертикальная линия (x постоянна)
        if abs(x2 - x1) < 1:
            x = x1
            # Проверяем пересечение с gap_rect
            if gx1 <= x <= gx2:
                # Линия проходит через область легенды по X
                # Рисуем сегменты до и после легенды
                if y1 < gy1:
                    # Сегмент сверху до легенды
                    draw.line([(x, y1), (x, gy1)], fill=color, width=width_px)
                if y2 > gy2:
                    # Сегмент снизу от легенды
                    draw.line([(x, gy2), (x, y2)], fill=color, width=width_px)
            else:
                # Нет пересечения - рисуем полностью
                draw.line([(x1, y1), (x2, y2)], fill=color, width=width_px)

        # Горизонтальная линия (y постоянна)
        elif abs(y2 - y1) < 1:
            y = y1
            # Проверяем пересечение с gap_rect
            if gy1 <= y <= gy2:
                # Линия проходит через область легенды по Y
                # Рисуем сегменты до и после легенды
                if x1 < gx1:
                    # Сегмент слева до легенды
                    draw.line([(x1, y), (gx1, y)], fill=color, width=width_px)
                if x2 > gx2:
                    # Сегмент справа от легенды
                    draw.line([(gx2, y), (x2, y)], fill=color, width=width_px)
            else:
                # Нет пересечения - рисуем полностью
                draw.line([(x1, y1), (x2, y2)], fill=color, width=width_px)
        else:
            # Диагональная линия - рисуем полностью (не применимо для сетки)
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width_px)

    half_w_m = (w / 2.0) / ppm
    half_h_m = (h / 2.0) / ppm

    gx_left_m = floor_to_step(x0_gk - half_w_m, step_m)
    gx_right_m = floor_to_step(x0_gk + half_w_m, step_m) + step_m
    gy_down_m = floor_to_step(y0_gk - half_h_m, step_m)
    gy_up_m = floor_to_step(y0_gk + half_h_m, step_m) + step_m

    # Прогресс: линии + подписи (верх+низ на вертикалях, лево+право на горизонталях)
    n_vert = int((gx_right_m - gx_left_m) / step_m) + 1
    n_horz = int((gy_up_m - gy_down_m) / step_m) + 1

    if display_grid:
        total_units = n_vert + n_horz + n_vert * 2 + n_horz * 2
    else:
        total_units = n_vert * n_horz  # Только крестики на пересечениях

    grid_progress = ConsoleProgress(total=max(1, total_units), label='Сетка и подписи')

    # Количество сегментов для отрисовки линий сетки как ломаных
    # Это позволяет точно отобразить кривизну линий на больших картах
    grid_line_segments = max(10, int((gy_up_m - gy_down_m) / step_m) + 1)

    if display_grid:
        # Текущая логика отрисовки полной сетки (линии + подписи)
        # Вертикальные линии и подписи X
        x_m = gx_left_m
        while x_m <= gx_right_m:
            # Строим ломаную линию из множества точек для точного отображения кривизны
            line_points: list[tuple[float, float]] = []
            for i in range(grid_line_segments + 1):
                y_seg = gy_down_m + (gy_up_m - gy_down_m) * i / grid_line_segments
                px, py = gk_to_pixel(x_m, y_seg)
                line_points.append((px, py))

            # Рисуем ломаную линию
            if len(line_points) >= MIN_POINTS_FOR_LINE:
                draw.line(line_points, fill=color, width=width_px)
            grid_progress.step_sync(1)

            # Подписываем квадрат справа от вертикали: берём центр правого квадрата
            x_label_m = x_m + step_m / 2.0
            x_digits = math.floor(x_label_m / GRID_LABEL_THOUSAND_DIV) % GRID_LABEL_MOD
            x_label = f'{x_digits:02d}'

            # Вычисляем точную позицию центра квадрата через gk_to_pixel
            # Верх - центр квадрата на уровне верхней границы карты
            label_x_top, _ = gk_to_pixel(x_label_m, gy_up_m)
            draw_label_with_bg(
                draw,
                (label_x_top, grid_text_margin),
                x_label,
                font=font,
                anchor='ma',
                img_size=(w, h),
                bg_color=GRID_LABEL_BG_COLOR,
                padding=grid_label_bg_padding,
            )
            grid_progress.step_sync(1)
            # Низ - центр квадрата на уровне нижней границы карты
            label_x_bot, _ = gk_to_pixel(x_label_m, gy_down_m)
            draw_label_with_bg(
                draw,
                (label_x_bot, h - grid_text_margin),
                x_label,
                font=font,
                anchor='ms',
                img_size=(w, h),
                bg_color=GRID_LABEL_BG_COLOR,
                padding=grid_label_bg_padding,
            )
            grid_progress.step_sync(1)
            x_m += step_m

        # Горизонтальные линии и подписи Y
        # Количество сегментов для горизонтальных линий
        grid_line_segments_h = max(10, int((gx_right_m - gx_left_m) / step_m) + 1)

        y_m = gy_down_m
        while y_m <= gy_up_m:
            # Строим ломаную линию из множества точек для точного отображения кривизны
            line_points_h: list[tuple[float, float]] = []
            for i in range(grid_line_segments_h + 1):
                x_seg = gx_left_m + (gx_right_m - gx_left_m) * i / grid_line_segments_h
                px, py = gk_to_pixel(x_seg, y_m)
                line_points_h.append((px, py))

            # Рисуем ломаную линию
            if len(line_points_h) >= MIN_POINTS_FOR_LINE:
                draw.line(line_points_h, fill=color, width=width_px)
            grid_progress.step_sync(1)

            # Подписываем квадрат выше горизонтали: берём центр верхнего квадрата
            y_label_m = y_m + step_m / 2.0
            y_digits = math.floor(y_label_m / GRID_LABEL_THOUSAND_DIV) % GRID_LABEL_MOD
            y_label = f'{y_digits:02d}'

            # Вычисляем точную позицию центра квадрата через gk_to_pixel
            # Лево - центр квадрата на уровне левой границы карты
            _, label_y_left = gk_to_pixel(gx_left_m, y_label_m)
            draw_label_with_bg(
                draw,
                (grid_text_margin, label_y_left),
                y_label,
                font=font,
                anchor='lm',
                img_size=(w, h),
                bg_color=GRID_LABEL_BG_COLOR,
                padding=grid_label_bg_padding,
            )
            grid_progress.step_sync(1)
            # Право - центр квадрата на уровне правой границы карты
            _, label_y_right = gk_to_pixel(gx_right_m, y_label_m)
            draw_label_with_bg(
                draw,
                (w - grid_text_margin, label_y_right),
                y_label,
                font=font,
                anchor='rm',
                img_size=(w, h),
                bg_color=GRID_LABEL_BG_COLOR,
                padding=grid_label_bg_padding,
            )
            grid_progress.step_sync(1)
            y_m += step_m
    else:
        # Новая логика: рисуем только крестики в точках пересечения
        x_m = gx_left_m
        while x_m <= gx_right_m:
            y_m = gy_down_m
            while y_m <= gy_up_m:
                # Используем точное преобразование для каждой точки пересечения
                x_px, y_px = gk_to_pixel(x_m, y_m)

                # Рисуем крестик только если он не в области легенды
                if legend_bounds is None:
                    draw_cross_at_intersection(x_px, y_px)
                else:
                    gx1, gy1, gx2, gy2 = legend_bounds
                    if not (gx1 <= x_px <= gx2 and gy1 <= y_px <= gy2):
                        draw_cross_at_intersection(x_px, y_px)

                y_m += step_m
                grid_progress.step_sync(1)

            x_m += step_m

    grid_progress.close()
