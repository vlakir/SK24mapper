"""Tests for render.compose module."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from render.compose import compose_final_image


class TestComposeFinalImage:
    """Tests for compose_final_image function."""

    def test_no_rotation(self):
        """Should return same size image when no rotation."""
        img = Image.new('RGB', (100, 100), color='red')
        result = compose_final_image(img)
        assert result.size == (100, 100)

    def test_with_rotation(self):
        """Should rotate image and keep size."""
        img = Image.new('RGB', (100, 100), color='blue')
        result = compose_final_image(img, rotate_deg=45.0)
        assert result.size == (100, 100)

    def test_zero_rotation(self):
        """Zero rotation should return same size."""
        img = Image.new('RGB', (200, 150), color='green')
        result = compose_final_image(img, rotate_deg=0.0)
        assert result.size == (200, 150)

    def test_90_degree_rotation(self):
        """90 degree rotation should work."""
        img = Image.new('RGB', (100, 100), color='yellow')
        result = compose_final_image(img, rotate_deg=90.0)
        assert result.size == (100, 100)

    def test_negative_rotation(self):
        """Negative rotation should work."""
        img = Image.new('RGB', (100, 100), color='purple')
        result = compose_final_image(img, rotate_deg=-30.0)
        assert result.size == (100, 100)

    def test_returns_rgb_image(self):
        """Should return RGB image."""
        img = Image.new('RGB', (100, 100), color='red')
        result = compose_final_image(img)
        assert result.mode == 'RGB'

    def test_180_degree_rotation(self):
        """180 degree rotation should work."""
        img = Image.new('RGB', (100, 100), color='cyan')
        result = compose_final_image(img, rotate_deg=180.0)
        assert result.size == (100, 100)

    def test_small_rotation(self):
        """Small rotation angle should work."""
        img = Image.new('RGB', (100, 100), color='magenta')
        result = compose_final_image(img, rotate_deg=5.0)
        assert result.size == (100, 100)

    def test_large_image(self):
        """Should handle larger images."""
        img = Image.new('RGB', (1000, 800), color='orange')
        result = compose_final_image(img, rotate_deg=15.0)
        assert result.size == (1000, 800)


