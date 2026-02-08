from __future__ import annotations

from dataclasses import dataclass

from shared.constants import (
    CONTOUR_ADAPTIVE_ALPHA,
    CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M,
    CONTOUR_ADAPTIVE_MAX_SCALE,
    CONTOUR_ADAPTIVE_MIN_SCALE,
    CONTOUR_FONT_SIZE_RATIO,
    CONTOUR_INTERVAL_M,
    CONTOUR_LABEL_EDGE_MARGIN_M,
    CONTOUR_LABEL_FONT_SCALE_ALPHA,
    CONTOUR_LABEL_GAP_PADDING_M,
    CONTOUR_LABEL_MIN_SEG_LEN_M,
    CONTOUR_LABEL_SPACING_M,
)


@dataclass(frozen=True)
class ContourAdaptiveParams:
    scale: float
    interval_m: float
    label_spacing_m: float
    label_min_seg_len_m: float
    label_edge_margin_m: float
    label_font_m: float
    label_gap_padding_m: float


def _clamp(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def compute_contour_adaptive_params(
    map_size_m: float,
    grid_font_size_m: float = 100.0,
) -> ContourAdaptiveParams:
    if map_size_m <= 0 or CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M <= 0:
        scale = 1.0
    else:
        scale = (
            map_size_m / CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M
        ) ** CONTOUR_ADAPTIVE_ALPHA
    scale = _clamp(scale, CONTOUR_ADAPTIVE_MIN_SCALE, CONTOUR_ADAPTIVE_MAX_SCALE)
    font_scale = scale**CONTOUR_LABEL_FONT_SCALE_ALPHA
    contour_font_m = grid_font_size_m * CONTOUR_FONT_SIZE_RATIO
    return ContourAdaptiveParams(
        scale=scale,
        interval_m=CONTOUR_INTERVAL_M * scale,
        label_spacing_m=CONTOUR_LABEL_SPACING_M * scale,
        label_min_seg_len_m=CONTOUR_LABEL_MIN_SEG_LEN_M * scale,
        label_edge_margin_m=CONTOUR_LABEL_EDGE_MARGIN_M * scale,
        label_font_m=contour_font_m * font_scale,
        label_gap_padding_m=CONTOUR_LABEL_GAP_PADDING_M * scale,
    )
