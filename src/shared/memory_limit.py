"""
Apply RLIMIT_AS to the current process so MemoryError fires inside the
process before physical RAM exhaustion. Shared between main.py (GUI
parent), gui.workers._process_entry (build worker) and tools/perf_loop
(benchmark driver).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


def apply_rlimit_as(
    ratio: float,
    *,
    component: str = 'process',
) -> int | None:
    """
    Cap the current process's virtual-address space to `ratio` × physical RAM.

    Linux-only (other platforms: returns None without action). The cap
    is the SMALLER of `ratio × total_RAM` and the existing hard limit,
    so this is idempotent across nested process spawn.

    Args:
        ratio: Fraction of physical RAM to allow (e.g. 0.70).
        component: Label for the log message ('GUI', 'Worker', 'perf_loop').

    Returns:
        Effective limit in bytes, or None if it couldn't be set.

    """
    try:
        import resource

        import psutil
    except ImportError:
        return None

    mem = psutil.virtual_memory()
    limit_bytes = int(mem.total * ratio)
    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit_bytes = min(limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError) as e:
        logger.warning('%s: failed to set RLIMIT_AS: %s', component, e)
        return None

    logger.info(
        '%s: RLIMIT_AS set to %.0f MB (RAM=%.0f MB × ratio=%.2f, reserve '
        '~%.0f MB)',
        component,
        limit_bytes / _MB,
        mem.total / _MB,
        ratio,
        (mem.total - limit_bytes) / _MB,
    )
    return limit_bytes


def system_memory_available_mb() -> float | None:
    """
    System memory currently available for allocation (MB).

    Returns `psutil.virtual_memory().available` — это «available» в
    Linux-смысле: free + reclaimable cache. Лучше чем просто `free`.
    Returns None если psutil недоступен (тогда caller trust'ит
    desired operation — fallback).
    """
    try:
        import psutil

        return psutil.virtual_memory().available / _MB
    except Exception:
        return None


def compact_c_heap() -> None:
    """
    Ask the platform allocator to return free pages back to the OS.

    Linux (glibc): `malloc_trim(0)` — trims main + secondary arenas down
    to the high-water mark of live allocations. Critical when MALLOC
    _ARENA_MAX is small (we set it to 2 in main.py): без trim free pages
    остаются «зарезервированными» glibc'ом и считаются в RSS, что
    давит на system OOM-killer когда RAM tight.

    Windows: HeapCompact(GetProcessHeap(), 0) — coalesces free blocks
    and returns committed-but-unused pages.

    No-op for other platforms или при ошибке загрузки libc.

    Идемпотентна. Безопасна вызывать после любого heavy gc.collect().
    """
    import sys

    try:
        if sys.platform.startswith('linux'):
            import ctypes

            ctypes.CDLL('libc.so.6').malloc_trim(0)
        elif sys.platform == 'win32':
            import ctypes

            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.GetProcessHeap.restype = ctypes.c_void_p
            kernel32.HeapCompact.restype = ctypes.c_size_t
            kernel32.HeapCompact.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            heap = kernel32.GetProcessHeap()
            if heap:
                kernel32.HeapCompact(heap, 0)
    except Exception:
        logger.debug('compact_c_heap failed', exc_info=True)
