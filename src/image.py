import logging
import math

from PIL import Image, ImageDraw, ImageFont
from pyproj import CRS, Transformer

from constants import (
    CURRENT_PROFILE,
    GRID_COLOR,
    GRID_FONT_BOLD,
    GRID_FONT_PATH,
    GRID_FONT_PATH_BOLD,
    GRID_LABEL_BG_COLOR,
    GRID_LABEL_MOD,
    GRID_LABEL_THOUSAND_DIV,
    GRID_STEP_M,
    GRID_TEXT_COLOR,
    GRID_TEXT_OUTLINE_COLOR,
    GRID_TEXT_OUTLINE_WIDTH,
    STATIC_SCALE,
)
from progress import ConsoleProgress, LiveSpinner
from topography import crs_sk42_geog, meters_per_pixel


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
            idx += 1
            paste_progress.step_sync(1)
    paste_progress.close()

    crop_x, crop_y, crop_w, crop_h = crop_rect
    crop_progress = ConsoleProgress(total=1, label='Обрезка (расширенная область)')
    out = canvas.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
    crop_progress.step_sync(1)
    crop_progress.close()
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


def draw_text_with_outline(  # noqa: PLR0913
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


def draw_label_with_bg(  # noqa: PLR0913
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


def draw_axis_aligned_km_grid(  # noqa: PLR0913, PLR0915
    img: Image.Image,
    center_lat_sk42: float,
    center_lng_sk42: float,
    center_lat_wgs: float,
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
) -> None:
    """
    Рисует сетку 1 км (СК‑42/Гаусса–Крюгера) строго по осям экрана и подписывает.

    только 4-й и 5-й младшие знаки (две последние цифры тысяч метров).
    """
    # Аргумент предусмотрен для возможных будущих преобразований
    _ = t_sk42_to_wgs
    draw = ImageDraw.Draw(img)
    font = load_grid_font(grid_font_size)
    w, h = img.size

    mpp = meters_per_pixel(center_lat_wgs, zoom, scale=scale)
    ppm = 1.0 / mpp

    cx, cy = w / 2.0, h / 2.0

    t_sk42gk_from_sk42 = Transformer.from_crs(
        crs_sk42_geog,
        crs_sk42_gk,
        always_xy=True,
    )
    x0_gk, y0_gk = t_sk42gk_from_sk42.transform(center_lng_sk42, center_lat_sk42)

    def floor_to_step(v: float, step: float) -> int:
        return int(math.floor(v / step) * step)

    half_w_m = (w / 2.0) / ppm
    half_h_m = (h / 2.0) / ppm

    gx_left_m = floor_to_step(x0_gk - half_w_m, step_m)
    gx_right_m = floor_to_step(x0_gk + half_w_m, step_m) + step_m
    gy_down_m = floor_to_step(y0_gk - half_h_m, step_m)
    gy_up_m = floor_to_step(y0_gk + half_h_m, step_m) + step_m

    # Прогресс: линии + подписи (верх+низ на вертикалях, лево+право на горизонталях)
    n_vert = int((gx_right_m - gx_left_m) / step_m) + 1
    n_horz = int((gy_up_m - gy_down_m) / step_m) + 1
    total_units = n_vert + n_horz + n_vert * 2 + n_horz * 2
    grid_progress = ConsoleProgress(total=max(1, total_units), label='Сетка и подписи')

    # Вертикальные линии и подписи X
    x_m = gx_left_m
    while x_m <= gx_right_m:
        dx_m = x_m - x0_gk
        x_px = cx + dx_m * ppm
        draw.line([(x_px, 0), (x_px, h)], fill=color, width=width_px)
        grid_progress.step_sync(1)

        # Подписываем квадрат справа от вертикали: берём центр правого квадрата
        x_label_m = x_m + step_m / 2.0
        x_digits = math.floor(x_label_m / GRID_LABEL_THOUSAND_DIV) % GRID_LABEL_MOD
        x_label = f'{x_digits:02d}'

        # Сдвиг подписей вправо на половину шага сетки (к центру квадрата справа)
        half_step_px = (step_m / 2) * ppm

        # Верх - сдвигаем вправо
        draw_label_with_bg(
            draw,
            (x_px + half_step_px, grid_text_margin),
            x_label,
            font=font,
            anchor='ma',
            img_size=(w, h),
            bg_color=GRID_LABEL_BG_COLOR,
            padding=grid_label_bg_padding,
        )
        grid_progress.step_sync(1)
        # Низ - сдвигаем вправо
        draw_label_with_bg(
            draw,
            (x_px + half_step_px, h - grid_text_margin),
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
    y_m = gy_down_m
    while y_m <= gy_up_m:
        dy_m = y_m - y0_gk
        y_px = cy - dy_m * ppm
        draw.line([(0, y_px), (w, y_px)], fill=color, width=width_px)
        grid_progress.step_sync(1)

        # Подписываем квадрат выше горизонтали: берём центр верхнего квадрата
        y_label_m = y_m + step_m / 2.0
        y_digits = math.floor(y_label_m / GRID_LABEL_THOUSAND_DIV) % GRID_LABEL_MOD
        y_label = f'{y_digits:02d}'

        # Сдвиг подписей вверх на половину шага сетки (к центру квадрата выше)
        half_step_px = (step_m / 2) * ppm

        # Лево - сдвигаем вверх
        draw_label_with_bg(
            draw,
            (grid_text_margin, y_px - half_step_px),
            y_label,
            font=font,
            anchor='lm',
            img_size=(w, h),
            bg_color=GRID_LABEL_BG_COLOR,
            padding=grid_label_bg_padding,
        )
        grid_progress.step_sync(1)
        # Право - сдвигаем вверх
        draw_label_with_bg(
            draw,
            (w - grid_text_margin, y_px - half_step_px),
            y_label,
            font=font,
            anchor='rm',
            img_size=(w, h),
            bg_color=GRID_LABEL_BG_COLOR,
            padding=grid_label_bg_padding,
        )
        grid_progress.step_sync(1)
        y_m += step_m

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
