"""Tests for image_io module."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from imaging.io import build_save_kwargs, save_jpeg


class TestBuildSaveKwargs:
    """Tests for build_save_kwargs function."""

    def test_default_quality(self):
        """Default quality should be 95."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert kwargs['quality'] == 95

    def test_custom_quality(self):
        """Custom quality should be used."""
        kwargs = build_save_kwargs(Path('test.jpg'), quality=80)
        assert kwargs['quality'] == 80

    def test_quality_clamped_to_min(self):
        """Quality below 10 should be clamped to 10."""
        kwargs = build_save_kwargs(Path('test.jpg'), quality=5)
        assert kwargs['quality'] == 10

    def test_quality_clamped_to_max(self):
        """Quality above 100 should be clamped to 100."""
        kwargs = build_save_kwargs(Path('test.jpg'), quality=150)
        assert kwargs['quality'] == 100

    def test_format_is_jpeg(self):
        """Format should always be JPEG."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert kwargs['format'] == 'JPEG'

    def test_optimize_not_set(self):
        """Optimize is intentionally not set: cost ~+800 ms per save
        for ~10% smaller files at q=95 on map imagery — bad trade-off."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert 'optimize' not in kwargs

    def test_progressive_not_set(self):
        """Progressive is intentionally not set: ~+100 ms per save and
        no perceptible benefit for files we never stream over a network."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert 'progressive' not in kwargs

    def test_subsampling_default(self):
        """Subsampling is intentionally left at libjpeg default (4:2:0),
        not forced to 0 (4:4:4) — chroma in satellite/elevation maps is
        already smooth, the extra precision isn't worth the encode cost."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert 'subsampling' not in kwargs

    def test_exif_empty(self):
        """EXIF should be empty bytes."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert kwargs['exif'] == b''


class TestSaveJpeg:
    """Tests for save_jpeg function."""

    def test_save_jpeg(self, tmp_path):
        """Should save JPEG without errors."""
        img = Image.new('RGB', (50, 50), color='red')
        path = tmp_path / "test.jpg"
        save_kwargs = build_save_kwargs(path)
        save_jpeg(img, path, save_kwargs)
        assert path.exists()


