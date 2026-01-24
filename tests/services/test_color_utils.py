"""Tests for services.color_utils module."""
import pytest
from services.color_utils import lerp, build_color_lut, color_at_lut, ColorMapper


class TestLerp:
    def test_lerp_at_zero(self):
        assert lerp(10.0, 20.0, 0.0) == 10.0

    def test_lerp_at_one(self):
        assert lerp(10.0, 20.0, 1.0) == 20.0

    def test_lerp_at_half(self):
        assert lerp(10.0, 20.0, 0.5) == 15.0

    def test_lerp_negative(self):
        assert lerp(-10.0, 10.0, 0.5) == 0.0


class TestBuildColorLut:
    def test_simple_ramp(self):
        ramp = [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]
        lut = build_color_lut(ramp, lut_size=11)
        assert len(lut) == 11
        assert lut[0] == (0, 0, 0)
        assert lut[-1] == (255, 255, 255)
        # Middle should be gray
        assert lut[5] == (128, 128, 128)

    def test_multi_segment_ramp(self):
        ramp = [
            (0.0, (255, 0, 0)),
            (0.5, (0, 255, 0)),
            (1.0, (0, 0, 255)),
        ]
        lut = build_color_lut(ramp, lut_size=5)
        assert lut[0] == (255, 0, 0)
        assert lut[2] == (0, 255, 0)
        assert lut[4] == (0, 0, 255)


class TestColorAtLut:
    def test_at_zero(self):
        lut = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
        assert color_at_lut(lut, 0.0) == (0, 0, 0)

    def test_at_one(self):
        lut = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
        assert color_at_lut(lut, 1.0) == (255, 255, 255)

    def test_clamp_negative(self):
        lut = [(0, 0, 0), (255, 255, 255)]
        assert color_at_lut(lut, -0.5) == (0, 0, 0)

    def test_clamp_over_one(self):
        lut = [(0, 0, 0), (255, 255, 255)]
        assert color_at_lut(lut, 1.5) == (255, 255, 255)


class TestColorMapper:
    def test_color_at(self):
        ramp = [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]
        mapper = ColorMapper(ramp, lut_size=256)
        assert mapper.color_at(0.0) == (0, 0, 0)
        assert mapper.color_at(1.0) == (255, 255, 255)

    def test_lut_property(self):
        ramp = [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]
        mapper = ColorMapper(ramp, lut_size=100)
        assert len(mapper.lut) == 100
