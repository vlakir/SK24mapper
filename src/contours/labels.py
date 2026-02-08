from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from imaging.text import load_grid_font
from shared.constants import (
    CONTOUR_FONT_SIZE_RATIO,
    CONTOUR_INDEX_EVERY,
    CONTOUR_LABEL_BG_PADDING,
    CONTOUR_LABEL_BG_RGBA,
    CONTOUR_LABEL_EDGE_MARGIN_M,
    CONTOUR_LABEL_FONT_MAX_PX,
    CONTOUR_LABEL_FONT_MIN_PX,
    CONTOUR_LABEL_FONT_SIZE,
    CONTOUR_LABEL_FORMAT,
    CONTOUR_LABEL_INDEX_ONLY,
    CONTOUR_LABEL_MIN_SEG_LEN_M,
    CONTOUR_LABEL_OUTLINE_COLOR,
    CONTOUR_LABEL_OUTLINE_WIDTH,
    CONTOUR_LABEL_SPACING_M,
    CONTOUR_LABEL_TEXT_COLOR,
    CONTOUR_LABELS_ENABLED,
    MIN_POLYLINE_POINTS,
)

"""Unified contour labeling utilities.

This module hosts label rendering logic extracted from service.py.
The function below is intentionally minimal and mirrors the previous
working behavior to avoid regressions while completing the migration.
Additionally, it now logs detailed diagnostics to help pinpoint why
labels might not appear (filters, collisions, edges, etc.).
"""

if TYPE_CHECKING:
    from collections.abc import Callable

    # PIL Image type hint (runtime available via PIL import above)
    from PIL.Image import Image as PILImage

SeedPolylines = dict[int, list[list[tuple[float, float]]]]
BBox = tuple[int, int, int, int]


logger = logging.getLogger(__name__)


@dataclass
class LabelStats:
    placed: int = 0
    attempts: int = 0
    skipped_short: int = 0
    skipped_edge: int = 0
    skipped_bbox: int = 0
    skipped_collision: int = 0


def intersects(a: BBox, b: BBox) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


@dataclass(frozen=True)
class LabelConfig:
    w: int
    h: int
    spacing_px: int
    min_seg_px: int
    edge_margin_px: int
    dry_run: bool


def resolve_label_settings(
    label_spacing_m: float | None,
    label_min_seg_len_m: float | None,
    label_edge_margin_m: float | None,
    label_font_m: float | None,
) -> tuple[float, float, float, float]:
    spacing = (
        label_spacing_m if label_spacing_m is not None else CONTOUR_LABEL_SPACING_M
    )
    min_seg = (
        label_min_seg_len_m
        if label_min_seg_len_m is not None
        else CONTOUR_LABEL_MIN_SEG_LEN_M
    )
    edge_margin = (
        label_edge_margin_m
        if label_edge_margin_m is not None
        else CONTOUR_LABEL_EDGE_MARGIN_M
    )
    font_m = (
        label_font_m if label_font_m is not None else 100.0 * CONTOUR_FONT_SIZE_RATIO
    )
    return spacing, min_seg, edge_margin, font_m


def build_font_getter(
    mpp: float,
    label_font_m: float,
) -> Callable[[], ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    font_reported = False

    def get_font_px() -> int:
        try:
            px = round(label_font_m / max(1e-9, mpp))
        except Exception:
            px = CONTOUR_LABEL_FONT_SIZE
        if px < CONTOUR_LABEL_FONT_MIN_PX:
            px = int(CONTOUR_LABEL_FONT_MIN_PX)
        if px > CONTOUR_LABEL_FONT_MAX_PX:
            px = int(CONTOUR_LABEL_FONT_MAX_PX)
        return px

    def get_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        nonlocal font_reported
        size = get_font_px()
        if size in font_cache:
            return font_cache[size]
        f = load_grid_font(size)
        font_cache[size] = f
        if not font_reported:
            font_reported = True
            logger.info(
                'Настройки шрифта подписей изолиний: size_px=%d '
                '(mpp=%.6f, target=%.1f м, clamp=[%d,%d])',
                size,
                mpp,
                label_font_m,
                int(CONTOUR_LABEL_FONT_MIN_PX),
                int(CONTOUR_LABEL_FONT_MAX_PX),
            )
        return f

    return get_font


def normalize_angle(ang_rad: float) -> float:
    if ang_rad < -math.pi / 2:
        return ang_rad + math.pi
    if ang_rad > math.pi / 2:
        return ang_rad - math.pi
    return ang_rad


def segment_lengths(
    pts: list[tuple[float, float]],
) -> tuple[list[float], float]:
    seg_l_list: list[float] = []
    total_len = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        length = math.hypot(dx, dy)
        seg_l_list.append(length)
        total_len += length
    return seg_l_list, total_len


def find_segment_for_target(
    seg_l_list: list[float],
    target: float,
) -> tuple[int, float] | None:
    acc = 0.0
    for i, seg_l in enumerate(seg_l_list):
        if acc + seg_l >= target:
            return i, acc
        acc += seg_l
    return None


def render_label_box(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    tmp = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    td = ImageDraw.Draw(tmp)
    left, top, right, bottom = td.textbbox((0, 0), text, font=font)
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
                    font=font,
                    fill=CONTOUR_LABEL_OUTLINE_COLOR,
                )
    bd.text(
        (pad - left, pad - top),
        text,
        font=font,
        fill=CONTOUR_LABEL_TEXT_COLOR,
    )
    return box


def compute_bbox(
    px: float,
    py: float,
    rot: Image.Image,
) -> BBox:
    rw, rh = rot.size
    x0b = round(px - rw / 2)
    y0b = round(py - rh / 2)
    x1b = x0b + rw
    y1b = y0b + rh
    return x0b, y0b, x1b, y1b


def classify_bbox(
    bbox: BBox,
    placed_boxes: list[BBox],
    config: LabelConfig,
) -> str | None:
    x0b, y0b, x1b, y1b = bbox
    if x0b < 0 or y0b < 0 or x1b > config.w or y1b > config.h:
        return 'bbox'
    if any(intersects(bbox, bb) for bb in placed_boxes):
        return 'collision'
    return None


def within_edge_margin(px: float, py: float, config: LabelConfig) -> bool:
    return (
        config.edge_margin_px <= px <= config.w - config.edge_margin_px
        and config.edge_margin_px <= py <= config.h - config.edge_margin_px
    )


def prepare_label(
    text: str,
    ang_rad: float,
    px: float,
    py: float,
    placed: list[BBox],
    config: LabelConfig,
    get_font: Callable[[], ImageFont.FreeTypeFont | ImageFont.ImageFont],
) -> tuple[BBox | None, Image.Image | None, str | None]:
    font = get_font()
    box = render_label_box(text, font)
    ang_deg = math.degrees(ang_rad)
    rot = box.rotate(ang_deg, expand=True, resample=Image.Resampling.BICUBIC)
    bbox = compute_bbox(px, py, rot)
    reason = classify_bbox(bbox, placed, config)
    if reason:
        return None, None, reason
    return bbox, rot, None


def process_polyline(
    img: PILImage,
    poly: list[tuple[float, float]],
    text: str,
    seed_ds: int,
    config: LabelConfig,
    placed: list[BBox],
    get_font: Callable[[], ImageFont.FreeTypeFont | ImageFont.ImageFont],
    stats: LabelStats,
    total_stats: LabelStats,
) -> None:
    pts = [(x * seed_ds, y * seed_ds) for (x, y) in poly]
    if len(pts) < MIN_POLYLINE_POINTS:
        return
    seg_l_list, total_len = segment_lengths(pts)
    if total_len < max(config.min_seg_px, config.spacing_px * 0.8):
        stats.skipped_short += 1
        total_stats.skipped_short += 1
        return

    target = config.spacing_px
    while target < total_len:
        segment = find_segment_for_target(seg_l_list, target)
        if segment is None:
            break
        idx, acc = segment
        t = (target - acc) / max(1e-9, seg_l_list[idx])
        x0p, y0p = pts[idx]
        x1p, y1p = pts[idx + 1]
        px = x0p + (x1p - x0p) * t
        py = y0p + (y1p - y0p) * t
        stats.attempts += 1
        total_stats.attempts += 1

        ang_rad = normalize_angle(math.atan2(-(y1p - y0p), x1p - x0p))
        if not within_edge_margin(px, py, config):
            stats.skipped_edge += 1
            total_stats.skipped_edge += 1
            target += config.spacing_px
            continue

        bbox, rot, reason = prepare_label(
            text,
            ang_rad,
            px,
            py,
            placed,
            config,
            get_font,
        )
        if reason:
            if reason == 'bbox':
                stats.skipped_bbox += 1
                total_stats.skipped_bbox += 1
            else:
                stats.skipped_collision += 1
                total_stats.skipped_collision += 1
            target += config.spacing_px
            continue

        if bbox is None or rot is None:
            logger.warning(
                'Параметры подписи не подготовлены: bbox=%s, rot=%s', bbox, rot
            )
            target += config.spacing_px
            continue

        x0b, y0b, _, _ = bbox
        if not config.dry_run:
            if img.mode == 'RGBA':
                img.alpha_composite(rot, dest=(x0b, y0b))
            else:
                img.paste(rot, (x0b, y0b), rot)

        placed.append(bbox)
        stats.placed += 1
        target += config.spacing_px


def place_labels_for_levels(
    img: PILImage,
    seed_polylines: SeedPolylines,
    levels: list[float],
    seed_ds: int,
    config: LabelConfig,
    get_font: Callable[[], ImageFont.FreeTypeFont | ImageFont.ImageFont],
) -> list[BBox]:
    placed: list[BBox] = []
    total_stats = LabelStats()
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
        level_stats = LabelStats()

        for poly in level_polys:
            process_polyline(
                img,
                poly,
                text,
                seed_ds,
                config,
                placed,
                get_font,
                level_stats,
                total_stats,
            )

        logger.info(
            'Уровень %.1fm: размещено=%d, попыток=%d, отклонено: '
            'коротких=%d, у_края=%d, bbox=%d, коллизии=%d',
            level,
            level_stats.placed,
            level_stats.attempts,
            level_stats.skipped_short,
            level_stats.skipped_edge,
            level_stats.skipped_bbox,
            level_stats.skipped_collision,
        )

    logger.info(
        'Подписи изолиний: итого размещено=%d, попыток=%d; отклонено: '
        'коротких=%d, у_края=%d, bbox=%d, коллизии=%d',
        len(placed),
        total_stats.attempts,
        total_stats.skipped_short,
        total_stats.skipped_edge,
        total_stats.skipped_bbox,
        total_stats.skipped_collision,
    )
    return placed


def draw_contour_labels(
    img: PILImage,
    seed_polylines: SeedPolylines,
    levels: list[float],
    crop_rect: tuple[int, int, int, int] | None,
    seed_ds: int,
    mpp: float,
    *,
    dry_run: bool = False,
    label_spacing_m: float | None = None,
    label_min_seg_len_m: float | None = None,
    label_edge_margin_m: float | None = None,
    label_font_m: float | None = None,
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
    if crop_rect is not None:
        logger.debug('Прямоугольник обрезки: %s', crop_rect)

    (
        label_spacing_m,
        label_min_seg_len_m,
        label_edge_margin_m,
        label_font_m,
    ) = resolve_label_settings(
        label_spacing_m,
        label_min_seg_len_m,
        label_edge_margin_m,
        label_font_m,
    )
    get_font = build_font_getter(mpp, label_font_m)

    spacing_px = max(1, round(label_spacing_m / max(1e-9, mpp)))
    min_seg_px = max(1, round(label_min_seg_len_m / max(1e-9, mpp)))
    edge_margin_px = max(1, round(label_edge_margin_m / max(1e-9, mpp)))
    config = LabelConfig(
        w=w,
        h=h,
        spacing_px=spacing_px,
        min_seg_px=min_seg_px,
        edge_margin_px=edge_margin_px,
        dry_run=dry_run,
    )

    logger.info(
        'Подписи изолиний: старт '
        '(levels=%d, img=%dx%d, seed_ds=%d, dry_run=%s). '
        'Пороги: spacing_m=%.1f (spacing_px=%d), min_seg_m=%.1f '
        '(min_seg_px=%d), edge_m=%.1f (edge_px=%d), outline=%d, '
        'gap_bg=%s',
        len(levels),
        w,
        h,
        int(seed_ds),
        dry_run,
        float(label_spacing_m),
        int(spacing_px),
        float(label_min_seg_len_m),
        int(min_seg_px),
        float(label_edge_margin_m),
        int(edge_margin_px),
        int(CONTOUR_LABEL_OUTLINE_WIDTH),
        bool(CONTOUR_LABEL_BG_RGBA),
    )

    return place_labels_for_levels(
        img,
        seed_polylines,
        levels,
        seed_ds,
        config,
        get_font,
    )
