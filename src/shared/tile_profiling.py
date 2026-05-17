"""
Tile-fetch profiling primitives.

Collects per-tile fetch/decode timings and cache-hit counts inside a
contextvar scope so any async path that calls record_tile_fetch() during
a map build contributes to the same aggregate. Kept in a separate module
to avoid the cycle that would arise if shared.diagnostics — which already
imports from geo.topography — also had to be imported by geo.topography.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class TileFetchStats:
    count: int = 0
    memory_hits: int = 0  # in-process LRU (no HTTP, no disk)
    disk_hits: int = 0  # local bytes file (no HTTP)
    cache_hits: int = 0  # aiohttp_client_cache SQLite (HTTP layer hit)
    cache_misses: int = 0  # network fetch
    fetch_seconds: float = 0.0
    decode_seconds: float = 0.0
    per_tile_fetch: list[float] = field(default_factory=list)
    per_tile_decode: list[float] = field(default_factory=list)


_tile_stats_var: ContextVar[TileFetchStats | None] = ContextVar(
    'tile_fetch_stats', default=None
)


@contextmanager
def collect_tile_stats() -> Iterator[TileFetchStats]:
    stats = TileFetchStats()
    token = _tile_stats_var.set(stats)
    try:
        yield stats
    finally:
        _tile_stats_var.reset(token)


def record_tile_fetch(
    *,
    fetch_seconds: float,
    decode_seconds: float,
    from_cache: bool,
    from_memory: bool = False,
    from_disk: bool = False,
) -> None:
    stats = _tile_stats_var.get()
    if stats is None:
        return
    stats.count += 1
    if from_memory:
        stats.memory_hits += 1
    elif from_disk:
        stats.disk_hits += 1
    elif from_cache:
        stats.cache_hits += 1
    else:
        stats.cache_misses += 1
    stats.fetch_seconds += fetch_seconds
    stats.decode_seconds += decode_seconds
    stats.per_tile_fetch.append(fetch_seconds)
    stats.per_tile_decode.append(decode_seconds)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def format_tile_stats(stats: TileFetchStats, *, label: str = 'tile-fetch') -> str:
    if stats.count == 0:
        return f'{label}: tiles=0'
    p50_f = _percentile(stats.per_tile_fetch, 0.5) * 1000.0
    p99_f = _percentile(stats.per_tile_fetch, 0.99) * 1000.0
    p50_d = _percentile(stats.per_tile_decode, 0.5) * 1000.0
    p99_d = _percentile(stats.per_tile_decode, 0.99) * 1000.0
    return (
        f'{label}: tiles={stats.count} '
        f'mem={stats.memory_hits}/disk={stats.disk_hits}/'
        f'sqlite={stats.cache_hits}/net={stats.cache_misses} '
        f'fetch={stats.fetch_seconds:.2f}s (p50={p50_f:.0f}ms, p99={p99_f:.0f}ms) '
        f'decode={stats.decode_seconds:.2f}s (p50={p50_d:.0f}ms, p99={p99_d:.0f}ms)'
    )
