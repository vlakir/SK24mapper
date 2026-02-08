"""Tests for contours.adaptive module."""

import pytest

from contours.adaptive import compute_contour_adaptive_params
from shared.constants import (
    CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M,
    CONTOUR_ADAPTIVE_MAX_SCALE,
    CONTOUR_ADAPTIVE_MIN_SCALE,
    CONTOUR_FONT_SIZE_RATIO,
    CONTOUR_LABEL_FONT_SCALE_ALPHA,
)


class TestContourAdaptiveParams:
    def test_scale_at_base_size(self):
        params = compute_contour_adaptive_params(CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M)
        assert params.scale == pytest.approx(1.0)

    def test_scale_clamped_low(self):
        params = compute_contour_adaptive_params(CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M * 0.01)
        assert params.scale == pytest.approx(CONTOUR_ADAPTIVE_MIN_SCALE)

    def test_scale_clamped_high(self):
        params = compute_contour_adaptive_params(CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M * 100.0)
        assert params.scale == pytest.approx(CONTOUR_ADAPTIVE_MAX_SCALE)

    def test_font_scale_uses_alpha(self):
        grid_font_size_m = 100.0  # default
        params = compute_contour_adaptive_params(
            CONTOUR_ADAPTIVE_BASE_MAP_SIZE_M * 4.0,
            grid_font_size_m,
        )
        expected_font_scale = params.scale ** CONTOUR_LABEL_FONT_SCALE_ALPHA
        contour_font_m = grid_font_size_m * CONTOUR_FONT_SIZE_RATIO
        assert params.label_font_m == pytest.approx(
            contour_font_m * expected_font_scale
        )