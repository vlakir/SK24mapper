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

    def test_optimize_enabled(self):
        """Optimize should be enabled."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert kwargs['optimize'] is True

    def test_progressive_enabled(self):
        """Progressive should be enabled."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert kwargs['progressive'] is True

    def test_subsampling_zero(self):
        """Subsampling should be 0 (best quality)."""
        kwargs = build_save_kwargs(Path('test.jpg'))
        assert kwargs['subsampling'] == 0

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


