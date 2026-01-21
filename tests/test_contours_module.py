"""Tests for contours module (shim)."""

import pytest

import contours


class TestContoursModule:
    """Tests for contours module."""

    def test_has_path(self):
        """Module should have __path__ attribute."""
        assert hasattr(contours, '__path__')

    def test_path_is_list(self):
        """__path__ should be a list."""
        assert isinstance(contours.__path__, list)

    def test_path_contains_contours(self):
        """__path__ should contain 'contours' directory."""
        assert len(contours.__path__) > 0
        assert 'contours' in contours.__path__[0]
