"""Color utilities for elevation and radio horizon colorization."""

from __future__ import annotations

import numpy as np


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


def build_color_lut(
    ramp: list[tuple[float, tuple[int, int, int]]],
    lut_size: int = 2048,
) -> list[tuple[int, int, int]]:
    """
    Build a lookup table (LUT) from a color ramp.

    Args:
        ramp: List of (t, (R, G, B)) tuples where t is in [0, 1]
        lut_size: Size of the resulting LUT

    Returns:
        List of RGB tuples

    """
    lut: list[tuple[int, int, int]] = []
    for i in range(lut_size):
        t = i / (lut_size - 1) if lut_size > 1 else 0.0
        # find segment
        for j in range(1, len(ramp)):
            t0, c0 = ramp[j - 1]
            t1, c1 = ramp[j]
            if t <= t1 or j == len(ramp) - 1:
                local = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                r = round(lerp(c0[0], c1[0], local))
                g = round(lerp(c0[1], c1[1], local))
                b = round(lerp(c0[2], c1[2], local))
                lut.append((r, g, b))
                break
    return lut


def color_at_lut(lut: list[tuple[int, int, int]], t: float) -> tuple[int, int, int]:
    """
    Get color from LUT at normalized position t.

    Args:
        lut: Color lookup table
        t: Normalized value in [0, 1]

    Returns:
        RGB tuple

    """
    if t <= 0.0:
        return lut[0]
    if t >= 1.0:
        return lut[-1]
    idx = int(t * (len(lut) - 1))
    return lut[idx]


class ColorMapper:
    """Helper class for mapping values to colors using a LUT."""

    def __init__(
        self,
        ramp: list[tuple[float, tuple[int, int, int]]],
        lut_size: int = 2048,
    ) -> None:
        self._lut_list = build_color_lut(ramp, lut_size)
        self._lut = np.array(self._lut_list, dtype=np.uint8)

    def color_at(self, t: float) -> tuple[int, int, int]:
        """Get color at normalized position t in [0, 1]."""
        return color_at_lut(self._lut_list, t)

    @property
    def lut(self) -> np.ndarray:
        """Access the underlying LUT as numpy array."""
        return self._lut
