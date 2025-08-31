"""
Diagnostic utilities.

This module monitors system resources and detects potential hanging issues.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import psutil

_PSUTIL_AVAILABLE = True

logger = logging.getLogger(__name__)


def get_memory_info() -> dict[str, Any]:
    """Get comprehensive memory usage information."""
    if not _PSUTIL_AVAILABLE:
        return {'error': 'psutil not available'}

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        return {
            'process_rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'process_vms_mb': round(memory_info.vms / 1024 / 1024, 2),
            'system_total_mb': round(system_memory.total / 1024 / 1024, 2),
            'system_available_mb': round(system_memory.available / 1024 / 1024, 2),
            'system_used_percent': system_memory.percent,
            'process_memory_percent': round(process.memory_percent(), 2),
        }
    except Exception as e:
        return {'error': f'Failed to get memory info: {e}'}


def get_thread_info() -> dict[str, Any]:
    """Get information about active threads."""
    try:
        active_threads = threading.active_count()
        thread_names = [t.name for t in threading.enumerate()]

        info = {
            'active_count': active_threads,
            'thread_names': thread_names,
            'main_thread_alive': threading.main_thread().is_alive(),
        }

        if _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                info['system_threads'] = process.num_threads()
            except Exception as e:
                logger.debug(f'Failed to get system thread count: {e}')

        return info
    except Exception as e:
        return {'error': f'Failed to get thread info: {e}'}


def get_file_descriptor_info() -> dict[str, Any]:
    """Get information about open file descriptors."""
    if not _PSUTIL_AVAILABLE:
        return {'error': 'psutil not available'}

    try:
        process = psutil.Process()
        open_files = len(process.open_files())
        connections = len(process.connections())

        return {
            'open_files': open_files,
            'network_connections': connections,
            'pid': process.pid,
        }
    except Exception as e:
        return {'error': f'Failed to get file descriptor info: {e}'}


def get_sqlite_info() -> dict[str, Any]:
    """Get SQLite connection information from cache directories."""
    try:
        # Check cache directory for SQLite files
        repo_root = Path(__file__).resolve().parent.parent
        cache_dir = repo_root / '.cache'

        sqlite_files = []
        if cache_dir.exists():
            for sqlite_file in cache_dir.rglob('*.sqlite*'):
                try:
                    # Try to get file info
                    stat = sqlite_file.stat()
                    sqlite_files.append(
                        {
                            'file': str(sqlite_file),
                            'size_mb': round(stat.st_size / 1024 / 1024, 2),
                            'modified': time.ctime(stat.st_mtime),
                        }
                    )
                except Exception as e:
                    logger.debug(
                        f'Failed to get info for SQLite file {sqlite_file}: {e}'
                    )

        return {
            'cache_dir': str(cache_dir),
            'sqlite_files': sqlite_files,
            'total_files': len(sqlite_files),
        }
    except Exception as e:
        return {'error': f'Failed to get SQLite info: {e}'}


def get_system_load() -> dict[str, Any]:
    """Get system load and CPU information."""
    if not _PSUTIL_AVAILABLE:
        return {'error': 'psutil not available'}

    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else None

        info = {
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
        }

        if load_avg:
            info['load_avg_1min'] = load_avg[0]
            info['load_avg_5min'] = load_avg[1]
            info['load_avg_15min'] = load_avg[2]

        return info
    except Exception as e:
        return {'error': f'Failed to get system load: {e}'}


def log_comprehensive_diagnostics(
    operation: str = 'general', level: int = logging.INFO
) -> None:
    """Log comprehensive diagnostic information."""
    logger.log(level, f'=== DIAGNOSTIC INFO: {operation.upper()} ===')

    # Memory information
    memory_info = get_memory_info()
    logger.log(
        level,
        f'Memory - RSS: {memory_info.get("process_rss_mb", "N/A")}MB, '
        f'VMS: {memory_info.get("process_vms_mb", "N/A")}MB, '
        f'System Available: {memory_info.get("system_available_mb", "N/A")}MB '
        f'({memory_info.get("system_used_percent", "N/A")}% used)',
    )

    # Thread information
    thread_info = get_thread_info()
    logger.log(
        level,
        f'Threads - Active: {thread_info.get("active_count", "N/A")}, '
        f'System: {thread_info.get("system_threads", "N/A")}, '
        f'Main alive: {thread_info.get("main_thread_alive", "N/A")}',
    )

    # File descriptor information
    fd_info = get_file_descriptor_info()
    logger.log(
        level,
        f'Resources - Open files: {fd_info.get("open_files", "N/A")}, '
        f'Network connections: {fd_info.get("network_connections", "N/A")}',
    )

    # System load
    load_info = get_system_load()
    logger.log(
        level,
        f'System - CPU: {load_info.get("cpu_percent", "N/A")}%, '
        f'Load avg: {load_info.get("load_avg_1min", "N/A")}',
    )

    # SQLite cache info
    sqlite_info = get_sqlite_info()
    logger.log(level, f'SQLite - Cache files: {sqlite_info.get("total_files", "N/A")}')

    # Thread names for debugging
    thread_info = get_thread_info()
    if 'thread_names' in thread_info:
        logger.log(level, f'Active threads: {", ".join(thread_info["thread_names"])}')

    logger.log(level, f'=== END DIAGNOSTIC INFO: {operation.upper()} ===')


def log_memory_usage(context: str = '') -> None:
    """Quick memory usage logging."""
    memory_info = get_memory_info()
    logger.info(
        f'Memory usage{" (" + context + ")" if context else ""}: '
        f'RSS={memory_info.get("process_rss_mb", "N/A")}MB, '
        f'Available={memory_info.get("system_available_mb", "N/A")}MB'
    )


def log_thread_status(context: str = '') -> None:
    """Quick thread status logging."""
    thread_info = get_thread_info()
    logger.info(
        f'Thread status{" (" + context + ")" if context else ""}: '
        f'Active={thread_info.get("active_count", "N/A")}, '
        f'System={thread_info.get("system_threads", "N/A")}'
    )


def monitor_resource_changes(
    before_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monitor changes in resource usage."""
    current_info = {
        'memory': get_memory_info(),
        'threads': get_thread_info(),
        'files': get_file_descriptor_info(),
    }

    if before_info:
        changes: dict[str, dict[str, str]] = {}
        for category in current_info:
            if category in before_info:
                changes[category] = {}
                for key in current_info[category]:
                    if key in before_info[category]:
                        if isinstance(current_info[category][key], (int, float)):
                            diff = (
                                current_info[category][key] - before_info[category][key]
                            )
                            if diff != 0:
                                changes[category][key] = (
                                    f'{before_info[category][key]} -> {current_info[category][key]} ({diff:+})'
                                )

        if any(changes.values()):
            logger.info(f'Resource changes detected: {changes}')

    return current_info


class ResourceMonitor:
    """Context manager for monitoring resource usage during operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.start_resources = None

    def __enter__(self):
        self.start_time = time.time()
        self.start_resources = monitor_resource_changes()
        log_comprehensive_diagnostics(f'{self.operation_name} - START')
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        assert self.start_time is not None
        duration = time.time() - self.start_time
        logger.info(
            f"Operation '{self.operation_name}' completed in {duration:.2f} seconds"
        )

        # Log final resources and changes
        monitor_resource_changes(self.start_resources)
        log_comprehensive_diagnostics(f'{self.operation_name} - END')

        if exc_type:
            logger.error(
                f"Operation '{self.operation_name}' failed with "
                f'{exc_type.__name__}: {exc_val}'
            )


# Check if psutil is available and log warning if not
if not _PSUTIL_AVAILABLE:
    logger.warning(
        'psutil library not available - memory and system monitoring will be limited'
    )
