from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

"""Generic seed polyline builder (marching squares) placeholder.

This module is introduced as part of service.py split to isolate contour
construction logic. Implementation will be migrated here iteratively to avoid
functional regressions.
"""

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class SeedGrid:
    w: int
    h: int
    step: int


@dataclass
class ContourLevels:
    base: float
    interval: float
    index_every: int


@dataclass
class Sampling:
    get_value: Callable[[int, int], float]


@dataclass
class CoordMap:
    to_px: Callable[[int, int], tuple[float, float]]


@dataclass
class BuildOpts:
    quant: float


def build_seed_polylines(
    grid: SeedGrid,
    levels: ContourLevels,
    sampling: Sampling,
    coord: CoordMap,
    opts: BuildOpts,
) -> list[list[tuple[float, float]]]:
    """
    Unified generator of seed polylines (to be implemented).

    The actual marching-squares implementation will be moved here in a later
    iteration. For now, this placeholder keeps a stable API so that service.py
    can gradually adopt it without breaking.
    """
    msg = (
        'build_seed_polylines is not yet wired; migration will follow in next iteration'
    )
    raise NotImplementedError(msg)
