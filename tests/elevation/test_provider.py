"""Tests for elevation_provider module."""

import pytest

from elevation.provider import TileKey


class TestTileKey:
    """Tests for TileKey dataclass."""

    def test_create(self):
        """Should create TileKey with all fields."""
        key = TileKey(z=15, x=100, y=200, retina=True)
        assert key.z == 15
        assert key.x == 100
        assert key.y == 200
        assert key.retina is True

    def test_path_parts_no_retina(self):
        """path_parts should return correct parts without retina."""
        key = TileKey(z=12, x=1234, y=5678, retina=False)
        parts = key.path_parts()
        assert parts == ('12', '1234', '5678.pngraw')

    def test_path_parts_with_retina(self):
        """path_parts should return correct parts with retina."""
        key = TileKey(z=12, x=1234, y=5678, retina=True)
        parts = key.path_parts()
        assert parts == ('12', '1234', '5678@2x.pngraw')

    def test_hashable(self):
        """TileKey should be hashable for use as dict key."""
        key1 = TileKey(z=10, x=100, y=200, retina=False)
        key2 = TileKey(z=10, x=100, y=200, retina=False)
        d = {key1: 'value'}
        assert d[key2] == 'value'

    def test_equality(self):
        """Equal TileKeys should be equal."""
        key1 = TileKey(z=10, x=100, y=200, retina=True)
        key2 = TileKey(z=10, x=100, y=200, retina=True)
        assert key1 == key2

    def test_inequality_different_z(self):
        """TileKeys with different z should not be equal."""
        key1 = TileKey(z=10, x=100, y=200, retina=True)
        key2 = TileKey(z=11, x=100, y=200, retina=True)
        assert key1 != key2

    def test_inequality_different_retina(self):
        """TileKeys with different retina should not be equal."""
        key1 = TileKey(z=10, x=100, y=200, retina=True)
        key2 = TileKey(z=10, x=100, y=200, retina=False)
        assert key1 != key2

    def test_inequality_different_x(self):
        """TileKeys with different x should not be equal."""
        key1 = TileKey(z=10, x=100, y=200, retina=True)
        key2 = TileKey(z=10, x=101, y=200, retina=True)
        assert key1 != key2

    def test_inequality_different_y(self):
        """TileKeys with different y should not be equal."""
        key1 = TileKey(z=10, x=100, y=200, retina=True)
        key2 = TileKey(z=10, x=100, y=201, retina=True)
        assert key1 != key2

    def test_path_parts_zero_coords(self):
        """path_parts should handle zero coordinates."""
        key = TileKey(z=0, x=0, y=0, retina=False)
        parts = key.path_parts()
        assert parts == ('0', '0', '0.pngraw')

    def test_path_parts_large_coords(self):
        """path_parts should handle large coordinates."""
        key = TileKey(z=20, x=1048575, y=1048575, retina=True)
        parts = key.path_parts()
        assert parts == ('20', '1048575', '1048575@2x.pngraw')

    def test_use_in_set(self):
        """TileKey should work in sets."""
        key1 = TileKey(z=10, x=100, y=200, retina=False)
        key2 = TileKey(z=10, x=100, y=200, retina=False)
        key3 = TileKey(z=10, x=100, y=201, retina=False)
        s = {key1, key2, key3}
        assert len(s) == 2
