import contextlib
import logging
import math

from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer

from constants import (
    GRID_COLOR,
    GRID_CROSS_LENGTH_PX,
    GRID_FONT_BOLD,
    GRID_FONT_PATH,
    GRID_FONT_PATH_BOLD,
    GRID_LABEL_BG_COLOR,
    GRID_LABEL_MOD,
    GRID_LABEL_OFFSET_FRACTION,
    GRID_LABEL_THOUSAND_DIV,
    GRID_STEP_M,
    GRID_TEXT_COLOR,
    GRID_TEXT_OUTLINE_COLOR,
    GRID_TEXT_OUTLINE_WIDTH,
    LEGEND_BACKGROUND_COLOR,
    LEGEND_BORDER_WIDTH_PX,
    LEGEND_GRID_GAP_PADDING_PX,
    LEGEND_HEIGHT_RATIO,
    LEGEND_HORIZONTAL_POSITION_RATIO,
    LEGEND_LABEL_FONT_MAX_PX,
    LEGEND_LABEL_FONT_MIN_PX,
    LEGEND_LABEL_FONT_RATIO,
    LEGEND_MARGIN_RATIO,
    LEGEND_MIN_HEIGHT_GRID_SQUARES,
    LEGEND_MIN_MAP_HEIGHT_KM_FOR_RATIO,
    LEGEND_NUM_LABELS,
    LEGEND_TEXT_OFFSET_PX,
    LEGEND_TEXT_OUTLINE_WIDTH_PX,
    LEGEND_VERTICAL_OFFSET_RATIO,
    LEGEND_WIDTH_TO_HEIGHT_RATIO,
    STATIC_SCALE,
)
from progress import ConsoleProgress, LiveSpinner
from topography import crs_sk42_geog, latlng_to_pixel_xy, meters_per_pixel

logger = logging.getLogger(__name__)


def assemble_and_crop(
    images: list[Image.Image],
    tiles_x: int,
    tiles_y: int,
    eff_tile_px: int,
    crop_rect: tuple[int, int, int, int],
) -> Image.Image:
    """Склеивает все тайлы в общий холст и обрезает область с припуском."""
    grid_w_px = tiles_x * eff_tile_px
    grid_h_px = tiles_y * eff_tile_px

    paste_progress = ConsoleProgress(total=tiles_x * tiles_y, label='Склейка тайлов')
    canvas = Image.new('RGB', (grid_w_px, grid_h_px))
    idx = 0
    for j in range(tiles_y):
        for i in range(tiles_x):
            img = images[idx]
            if img.size != (eff_tile_px, eff_tile_px):
                img = img.resize((eff_tile_px, eff_tile_px), Image.Resampling.LANCZOS)
            canvas.paste(img, (i * eff_tile_px, j * eff_tile_px))
            # Proactively release memory of individual tile after paste
            try:
                if hasattr(images[idx], 'close'):
                    images[idx].close()
            except Exception as e:
                logger.debug(f'Failed to close tile image: {e}')
            # Replace consumed image with a tiny placeholder to keep type invariant
            # (avoid assigning None which breaks type expectations and mypy contracts)
            if isinstance(images, list):
                images[idx] = Image.new('RGB', (1, 1))
            idx += 1
            paste_progress.step_sync(1)
    paste_progress.close()

    crop_x, crop_y, crop_w, crop_h = crop_rect
    crop_progress = ConsoleProgress(total=1, label='Обрезка (расширенная область)')
    out = canvas.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    crop_progress.step_sync(1)
    crop_progress.close()
    # Drop large temporary canvas and input list refs before returning
    with contextlib.suppress(Exception):
        del canvas
    try:
        # Remove None placeholders to free list memory sooner
        images.clear()
    except Exception as e:
        logger.debug(f'Failed to clear images list: {e}')
    return out


def rotate_keep_size(
    img: Image.Image,
    angle_deg: float,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Поворачивает изображение с expand=True и центр-кропает к исходному размеру,.

    чтобы избежать «срезанных» углов.
    """
    spinner = LiveSpinner('Поворот карты')
    spinner.start()
    try:
        w, h = img.size
        rotated = img.rotate(
            angle=angle_deg,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=fill,
        )
        rw, rh = rotated.size
        cx, cy = rw // 2, rh // 2
        left = int(cx - w / 2)
        top = int(cy - h / 2)
        return rotated.crop((left, top, left + w, top + h))
    finally:
        spinner.stop('Поворот карты: готово')


def center_crop(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Центрированный кроп изображения до размеров (out_w x out_h)."""
    spinner = LiveSpinner('Финальный центр-кроп')
    spinner.start()
    try:
        w, h = img.size
        left = max(0, (w - out_w) // 2)
        top = max(0, (h - out_h) // 2)
        return img.crop((left, top, left + out_w, top + out_h))
    finally:
        spinner.stop('Финальный центр-кроп: готово')


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


def load_grid_font(font_size: int = 86) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Подгружает масштабируемый шрифт для подписей (предпочтительно — жирный).

    Порядок:
      1) GRID_FONT_PATH_BOLD (если задан) — жирный шрифт.
      2) GRID_FONT_PATH (если задан) — обычный шрифт.
      3) DejaVuSans-Bold.ttf, если GRID_FONT_BOLD=True, иначе DejaVuSans.ttf.
      4) Резерв: встроенный маленький шрифт PIL.
    """
    if GRID_FONT_PATH_BOLD:
        try:
            return ImageFont.truetype(GRID_FONT_PATH_BOLD, font_size)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug('Failed to load bold grid font from %s', GRID_FONT_PATH_BOLD)
    if GRID_FONT_PATH:
        try:
            return ImageFont.truetype(GRID_FONT_PATH, font_size)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug('Failed to load grid font from %s', GRID_FONT_PATH)
    if GRID_FONT_BOLD:
        try:
            return ImageFont.truetype('DejaVuSans-Bold.ttf', font_size)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug('Failed to load DejaVuSans-Bold.ttf, will try regular')
    try:
        return ImageFont.truetype('DejaVuSans.ttf', font_size)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.debug('Failed to load DejaVuSans.ttf, using default font')
    return ImageFont.load_default()


def calculate_adaptive_grid_font_size(mpp: float) -> int:
    """
    Вычисляет адаптивный размер шрифта для подписей сетки.

    Args:
        mpp: meters per pixel (масштаб карты)

    Returns:
        Размер шрифта в пикселях, ограниченный диапазоном

    """
    from constants import (
        GRID_LABEL_FONT_KM,
        GRID_LABEL_FONT_MAX_PX,
        GRID_LABEL_FONT_MIN_PX,
    )

    try:
        # Целевой физический размер в метрах → размер в пикселях
        px = round((GRID_LABEL_FONT_KM * 1000.0) / max(1e-9, mpp))
    except Exception:
        px = 86  # Fallback на старое значение по умолчанию

    # Ограничиваем диапазон для читаемости
    px = max(GRID_LABEL_FONT_MIN_PX, min(px, GRID_LABEL_FONT_MAX_PX))

    logger.info(
        'Адаптивный размер шрифта сетки: %d px (mpp=%.6f, target=%.3f km, range=[%d,%d])',
        px,
        mpp,
        GRID_LABEL_FONT_KM,
        GRID_LABEL_FONT_MIN_PX,
        GRID_LABEL_FONT_MAX_PX,
    )

    return px


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
    width_px: int = 4,
    scale: int = STATIC_SCALE,
    grid_font_size: int = 86,
    grid_text_margin: int = 43,
    grid_label_bg_padding: int = 6,
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
        width_px: Толщина линий сетки в пикселях (игнорируется, если display_grid=False
            и рисуются крестики).
        scale: Масштабный коэффициент рендеринга тайлов (обычно 1 или 2 для retina).
        grid_font_size: Базовый размер шрифта для подписей сетки (px); фактически
            может быть переопределён адаптивным расчётом.
        grid_text_margin: Отступ подписей от краёв изображения (px).
        grid_label_bg_padding: Внутренний отступ подложки под подписью (px).
        legend_bounds: Опциональные границы легенды высот (x1, y1, x2, y2) — в этой
            области линии сетки прерываются, крестики не рисуются.
        display_grid: Признак полного отображения сетки. Если True — рисуются линии
            и подписи; если False — рисуются только крестики на пересечениях.

    Returns:
        None: Функция изменяет переданное изображение на месте, ничего не возвращает.

    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    mpp = meters_per_pixel(center_lat_wgs, zoom, scale=scale)
    # Вычисляем адаптивный размер шрифта на основе масштаба карты
    adaptive_font_size = calculate_adaptive_grid_font_size(mpp)
    font = load_grid_font(adaptive_font_size)
    ppm = 1.0 / mpp

    cx, cy = w / 2.0, h / 2.0

    # Предвычисляем sin/cos для поворота (изображение было повёрнуто на rotation_deg)
    # PIL rotate() работает в системе координат с Y вниз, поэтому используем отрицательный угол
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
        СК-42 ГК → СК-42 географические → WGS-84 → Web Mercator пиксели → пиксели изображения
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
        half = GRID_CROSS_LENGTH_PX // 2
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
            if len(line_points) >= 2:
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
            if len(line_points_h) >= 2:
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


def apply_white_mask(img: Image.Image, opacity: float) -> Image.Image:
    """Накладывает поверх изображения белую полупрозрачную маску (только для карты)."""
    opacity = max(0.0, min(1.0, opacity))
    if opacity == 0.0:
        return img
    spinner = LiveSpinner('Наложение маски')
    spinner.start()
    try:
        rgba = img.convert('RGBA')
        overlay = Image.new('RGBA', rgba.size, (255, 255, 255, round(255 * opacity)))
        composited = Image.alpha_composite(rgba, overlay)
        return composited.convert('RGB')
    finally:
        spinner.stop('Наложение маски: готово')


def draw_elevation_legend(
    img: Image.Image,
    color_ramp: list[tuple[float, tuple[int, int, int]]],
    min_elevation_m: float,
    max_elevation_m: float,
    center_lat_wgs: float,
    zoom: int,
    scale: int = STATIC_SCALE,
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

    Returns:
        Кортеж (x1, y1, x2, y2) - границы легенды с отступом для разрыва сетки

    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Рассчитываем метры на пиксель и пиксели на метр
    mpp = meters_per_pixel(center_lat_wgs, zoom, scale=scale)
    ppm = 1.0 / mpp if mpp > 0 else 0.0

    # Рассчитываем высоту карты в километрах
    map_height_km = (h * mpp) / 1000.0

    # Определяем высоту легенды: 10% от высоты карты, но не менее 1 км квадрата
    if map_height_km < LEGEND_MIN_MAP_HEIGHT_KM_FOR_RATIO:
        # Для малых карт: минимум 1 километровый квадрат
        legend_height = int(LEGEND_MIN_HEIGHT_GRID_SQUARES * GRID_STEP_M * ppm)
    else:
        # Для больших карт: 10% от высоты
        legend_height = int(h * LEGEND_HEIGHT_RATIO)

    # Обеспечиваем минимальную читаемость
    legend_height = max(legend_height, 100)

    # Рассчитываем остальные размеры пропорционально высоте легенды
    legend_width = int(legend_height * LEGEND_WIDTH_TO_HEIGHT_RATIO)
    margin = int(legend_height * LEGEND_MARGIN_RATIO)

    # Рассчитываем адаптивный размер шрифта
    font_size = int(legend_height * LEGEND_LABEL_FONT_RATIO)
    font_size = max(LEGEND_LABEL_FONT_MIN_PX, min(font_size, LEGEND_LABEL_FONT_MAX_PX))

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
    legend_total_width = legend_width + LEGEND_TEXT_OFFSET_PX + text_width_estimate
    legend_total_height = legend_height

    # Увеличиваем фон на 20% (коэффициент 1.2), добавляя по 10% с каждой стороны
    bg_padding_x = int(legend_total_width * 0.10)
    bg_padding_y = int(legend_total_height * 0.10)

    bg_x1 = legend_x - bg_padding_x
    bg_y1 = legend_y - bg_padding_y
    bg_x2 = legend_x + legend_total_width + bg_padding_x
    bg_y2 = legend_y + legend_height + bg_padding_y

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
    draw.rectangle(
        [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
        outline=(0, 0, 0),
        width=LEGEND_BORDER_WIDTH_PX,
    )

    # Добавляем подписи высот
    try:
        font = load_grid_font(font_size)
    except Exception:
        font = ImageFont.load_default()

    # Рисуем метки высоты
    for i in range(LEGEND_NUM_LABELS):
        t = i / (LEGEND_NUM_LABELS - 1) if LEGEND_NUM_LABELS > 1 else 0.0
        elevation = min_elevation_m + (max_elevation_m - min_elevation_m) * t
        label_text = f'{int(elevation)} м'

        # Позиция метки (снизу вверх)
        label_y = legend_y + legend_height - int(t * legend_height)

        # Рисуем текст справа от цветовой полосы с обводкой для читаемости
        text_x = legend_x + legend_width + LEGEND_TEXT_OFFSET_PX
        draw_text_with_outline(
            draw,
            (text_x, label_y),
            label_text,
            font=font,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            outline_width=LEGEND_TEXT_OUTLINE_WIDTH_PX,
            anchor='lm',
        )

    # Возвращаем границы легенды с отступом для разрыва линий сетки
    # Используем увеличенные размеры фона плюс дополнительный отступ
    gap_padding = LEGEND_GRID_GAP_PADDING_PX
    return (
        bg_x1 - gap_padding,
        bg_y1 - gap_padding,
        bg_x2 + gap_padding,
        bg_y2 + gap_padding,
    )
