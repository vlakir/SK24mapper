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

    def test_use_in_set(self):
        """TileKey should work in sets."""
        key1 = TileKey(z=10, x=100, y=200, retina=False)
        key2 = TileKey(z=10, x=100, y=200, retina=False)
        key3 = TileKey(z=10, x=100, y=201, retina=False)
        s = {key1, key2, key3}
        assert len(s) == 2
