from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from constants import (
    CONTOUR_INDEX_EVERY,
    CONTOUR_LABEL_BG_PADDING,
    CONTOUR_LABEL_BG_RGBA,
    CONTOUR_LABEL_EDGE_MARGIN_PX,
    CONTOUR_LABEL_FONT_BOLD,
    CONTOUR_LABEL_FONT_KM,
    CONTOUR_LABEL_FONT_MAX_PX,
    CONTOUR_LABEL_FONT_MIN_PX,
    CONTOUR_LABEL_FONT_PATH,
    CONTOUR_LABEL_FONT_SIZE,
    CONTOUR_LABEL_FORMAT,
    CONTOUR_LABEL_INDEX_ONLY,
    CONTOUR_LABEL_MIN_SEG_LEN_PX,
    CONTOUR_LABEL_OUTLINE_COLOR,
    CONTOUR_LABEL_OUTLINE_WIDTH,
    CONTOUR_LABEL_SPACING_M,
    CONTOUR_LABEL_TEXT_COLOR,
    CONTOUR_LABELS_ENABLED,
    GRID_FONT_PATH,
    GRID_FONT_PATH_BOLD,
)

"""Unified contour labeling utilities.

This module hosts label rendering logic extracted from service.py.
The function below is intentionally minimal and mirrors the previous
working behavior to avoid regressions while completing the migration.
Additionally, it now logs detailed diagnostics to help pinpoint why
labels might not appear (filters, collisions, edges, etc.).
"""

if TYPE_CHECKING:
    # PIL Image type hint (runtime available via PIL import above)
    from PIL.Image import Image as PILImage

logger = logging.getLogger(__name__)

MIN_POLYLINE_POINTS = 2


def draw_contour_labels(
    img: PILImage,
    seed_polylines: dict[int, list[list[tuple[float, float]]]],
    levels: list[float],
    crop_rect: tuple[int, int, int, int] | None,
    seed_ds: int,
    mpp: float,
    *,
    dry_run: bool = False,
) -> list[tuple[int, int, int, int]]:
    """
    Place contour labels on image and/or compute their bounding boxes.

    Parameters mirror the previous inlined implementation in service.py.
    When dry_run=True, only the list of label bounding boxes is returned
    (no drawing). When dry_run=False, labels are rendered onto the image
    and the list of actually placed boxes is returned.
    """
    if not CONTOUR_LABELS_ENABLED:
        logger.info('Подписи изолиний отключены (CONTOUR_LABELS_ENABLED=False)')
        return []

    w, h = img.size

    # font (chosen dynamically based on meters-per-pixel)
    fp = CONTOUR_LABEL_FONT_PATH or (
        GRID_FONT_PATH_BOLD if CONTOUR_LABEL_FONT_BOLD else GRID_FONT_PATH
    )
    name = 'DejaVuSans-Bold.ttf' if CONTOUR_LABEL_FONT_BOLD else 'DejaVuSans.ttf'
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    font_reported = False

    def get_font_px() -> int:
        try:
            px = round((CONTOUR_LABEL_FONT_KM * 1000.0) / max(1e-9, mpp))
        except Exception:
            px = CONTOUR_LABEL_FONT_SIZE
        # clamp to readable range
        if px < CONTOUR_LABEL_FONT_MIN_PX:
            px = int(CONTOUR_LABEL_FONT_MIN_PX)
        if px > CONTOUR_LABEL_FONT_MAX_PX:
            px = int(CONTOUR_LABEL_FONT_MAX_PX)
        return px

    def get_font(is_index: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        nonlocal font_reported
        size = get_font_px()
        if size in font_cache:
            return font_cache[size]
        try:
            f: ImageFont.FreeTypeFont | ImageFont.ImageFont = (
                ImageFont.truetype(fp, size) if fp else ImageFont.truetype(name, size)
            )
        except Exception as ex:
            logger.warning(
                'Не удалось загрузить шрифт "%s" size=%d: %s', fp or name, size, ex
            )
            f = ImageFont.load_default()
        font_cache[size] = f
        if not font_reported:
            font_reported = True
            logger.info(
                'Настройки шрифта подписей: size_px=%d (mpp=%.6f, km=%.3f, clamp=[%d,%d]), bold=%s, path=%s',
                size,
                mpp,
                CONTOUR_LABEL_FONT_KM,
                int(CONTOUR_LABEL_FONT_MIN_PX),
                int(CONTOUR_LABEL_FONT_MAX_PX),
                CONTOUR_LABEL_FONT_BOLD,
                fp or name,
            )
        return f

    placed: list[tuple[int, int, int, int]] = []

    def poly_to_crop(poly: list[tuple[float, float]]) -> list[tuple[float, float]]:
        # seed_polylines are in seed grid coordinates starting at (0,0) of the crop
        # Convert to image pixel coords by scaling only
        return [(x * seed_ds, y * seed_ds) for (x, y) in poly]

    def intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)

    spacing_px = max(1, round(CONTOUR_LABEL_SPACING_M / max(1e-9, mpp)))

    logger.info(
        'Подписи изолиний: старт (levels=%d, img=%dx%d, seed_ds=%d, dry_run=%s). '
        'Пороги: spacing_m=%.1f (spacing_px=%d), min_seg=%d, edge=%d, outline=%d, gap_bg=%s',
        len(levels),
        w,
        h,
        int(seed_ds),
        dry_run,
        float(CONTOUR_LABEL_SPACING_M),
        int(spacing_px),
        int(CONTOUR_LABEL_MIN_SEG_LEN_PX),
        int(CONTOUR_LABEL_EDGE_MARGIN_PX),
        int(CONTOUR_LABEL_OUTLINE_WIDTH),
        bool(CONTOUR_LABEL_BG_RGBA),
    )

    total_attempts = 0
    total_skipped_short = 0
    total_skipped_edge = 0
    total_skipped_bbox = 0
    total_skipped_collision = 0

    for li, level in enumerate(levels):
        if CONTOUR_LABEL_INDEX_ONLY and (li % max(1, int(CONTOUR_INDEX_EVERY)) != 0):
            continue
        is_index_line = li % max(1, int(CONTOUR_INDEX_EVERY)) == 0
        text = CONTOUR_LABEL_FORMAT.format(level)
        level_polys = seed_polylines.get(li, [])
        logger.info(
            'Уровень %d/%d (%.1fm): полилиний=%d, индексная=%s',
            li + 1,
            len(levels),
            level,
            len(level_polys),
            is_index_line,
        )
        level_placed = 0
        level_attempts = 0
        level_skipped_short = 0
        level_skipped_edge = 0
        level_skipped_bbox = 0
        level_skipped_collision = 0

        for poly in level_polys:
            pts = poly_to_crop(poly)
            if len(pts) < MIN_POLYLINE_POINTS:
                continue
            # segment lengths and total
            seg_l_list: list[float] = []
            total_len = 0.0
            for i in range(1, len(pts)):
                dx = pts[i][0] - pts[i - 1][0]
                dy = pts[i][1] - pts[i - 1][1]
                length = math.hypot(dx, dy)
                seg_l_list.append(length)
                total_len += length
            if total_len < max(
                CONTOUR_LABEL_MIN_SEG_LEN_PX, spacing_px * 0.8
            ):
                level_skipped_short += 1
                total_skipped_short += 1
                continue

            target = spacing_px
            while target < total_len:
                acc = 0.0
                idx = -1
                for i, seg_l in enumerate(seg_l_list):
                    if acc + seg_l >= target:
                        idx = i
                        break
                    acc += seg_l
                if idx < 0:
                    break

                t = (target - acc) / max(1e-9, seg_l_list[idx])
                x0p, y0p = pts[idx]
                x1p, y1p = pts[idx + 1]
                px = x0p + (x1p - x0p) * t
                py = y0p + (y1p - y0p) * t
                level_attempts += 1
                total_attempts += 1

                # Compute angle (radians) in image coordinate system (y downward)
                dx = x1p - x0p
                dy = y1p - y0p
                ang_rad = math.atan2(-dy, dx)
                # Normalize to keep text upright
                if ang_rad < -math.pi / 2:
                    ang_rad += math.pi
                elif ang_rad > math.pi / 2:
                    ang_rad -= math.pi

                if not (
                    CONTOUR_LABEL_EDGE_MARGIN_PX
                    <= px
                    <= w - CONTOUR_LABEL_EDGE_MARGIN_PX
                    and CONTOUR_LABEL_EDGE_MARGIN_PX
                    <= py
                    <= h - CONTOUR_LABEL_EDGE_MARGIN_PX
                ):
                    level_skipped_edge += 1
                    total_skipped_edge += 1
                    target += spacing_px
                    continue

                # Prepare text box
                tmp = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
                td = ImageDraw.Draw(tmp)
                fnt = get_font(is_index_line)
                left, top, right, bottom = td.textbbox((0, 0), text, font=fnt)
                tw, th = right - left, bottom - top
                pad = int(CONTOUR_LABEL_BG_PADDING)
                bw, bh = int(tw + 2 * pad), int(th + 2 * pad)

                box = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
                bd = ImageDraw.Draw(box)
                if CONTOUR_LABEL_BG_RGBA:
                    bd.rectangle((0, 0, bw - 1, bh - 1), fill=CONTOUR_LABEL_BG_RGBA)

                ow = max(0, int(CONTOUR_LABEL_OUTLINE_WIDTH))
                if ow > 0:
                    for ox in (-ow, 0, ow):
                        for oy in (-ow, 0, ow):
                            if ox == 0 and oy == 0:
                                continue
                            bd.text(
                                (pad - left + ox, pad - top + oy),
                                text,
                                font=fnt,
                                fill=CONTOUR_LABEL_OUTLINE_COLOR,
                            )
                bd.text(
                    (pad - left, pad - top),
                    text,
                    font=fnt,
                    fill=CONTOUR_LABEL_TEXT_COLOR,
                )

                ang_deg = math.degrees(ang_rad)
                rot = box.rotate(
                    ang_deg, expand=True, resample=Image.Resampling.BICUBIC
                )
                rw, rh = rot.size
                x0b = round(px - rw / 2)
                y0b = round(py - rh / 2)
                x1b = x0b + rw
                y1b = y0b + rh
                bbox = (x0b, y0b, x1b, y1b)
                if (
                    x0b < 0
                    or y0b < 0
                    or x1b > w
                    or y1b > h
                    or any(intersects(bbox, bb) for bb in placed)
                ):
                    # classify reason: bbox or collision
                    if x0b < 0 or y0b < 0 or x1b > w or y1b > h:
                        level_skipped_bbox += 1
                        total_skipped_bbox += 1
                    else:
                        level_skipped_collision += 1
                        total_skipped_collision += 1
                    target += spacing_px
                    continue

                if not dry_run:
                    if img.mode == 'RGBA':
                        img.alpha_composite(rot, dest=(x0b, y0b))
                    else:
                        img.paste(rot, (x0b, y0b), rot)

                placed.append(bbox)
                level_placed += 1
                target += spacing_px

        logger.info(
            'Уровень %.1fm: размещено=%d, попыток=%d, отклонено: коротких=%d, у_края=%d, bbox=%d, коллизии=%d',
            level,
            level_placed,
            level_attempts,
            level_skipped_short,
            level_skipped_edge,
            level_skipped_bbox,
            level_skipped_collision,
        )

    logger.info(
        'Подписи изолиний: итого размещено=%d, попыток=%d; отклонено: коротких=%d, у_края=%d, bbox=%d, коллизии=%d',
        len(placed),
        total_attempts,
        total_skipped_short,
        total_skipped_edge,
        total_skipped_bbox,
        total_skipped_collision,
    )

    return placed
