"""Tests for preview module."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from gui.preview import publish_preview_image


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

    @patch('gui.preview._publish_preview_image')
    def test_publish_preview_image_calls_impl(self, mock_pub):
        """Should call underlying publish function."""
        img = MagicMock()
        publish_preview_image(img)
        mock_pub.assert_called_once_with(img, None, None)
