"""Tests for preview module."""

import pytest
from PIL import Image

from preview import publish_preview_image


class TestPublishPreviewImage:
    """Tests for publish_preview_image function."""

    def test_returns_bool(self):
        """Should return boolean."""
        img = Image.new('RGB', (100, 100), color='red')
        result = publish_preview_image(img)
        assert isinstance(result, bool)

    def test_no_callback_returns_false(self):
        """Without callback set, should return False."""
        img = Image.new('RGB', (100, 100), color='blue')
        result = publish_preview_image(img)
        assert result is False

    def test_accepts_image(self):
        """Should accept PIL Image."""
        img = Image.new('RGB', (200, 200), color='green')
        # Should not raise
        publish_preview_image(img)
