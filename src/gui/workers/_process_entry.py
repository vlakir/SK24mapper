"""
Точка входа дочернего процесса для создания карты.

Запускается через ``multiprocessing.Process``.
Проксирует события прогресса/превью в ``mp.Queue`` для опроса из GUI-потока.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import logging
import os
import sys
import time
import zlib
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as _MpEvent

    from domain.models import DownloadParams, MapMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IPC через Shared Memory
#
# Вместо прокачки сотен МБ через pipe (pickle → socket → unpickle),
# записываем данные в shared memory и передаём через Queue только
# имя блока + метаданные (~100 байт).
#
# PIL RGBA: tobytes() → zlib level=1 → SharedMemory
# PIL RGB/L: JPEG quality=95 → SharedMemory
# numpy: tobytes() → SharedMemory (без сжатия — быстрее)
# ---------------------------------------------------------------------------

_ZLIB_LEVEL = 1

# Sentinels для rh_cache dict values
_SHM_PIL_SENTINEL = '__shm_pil__'
_SHM_NPY_SENTINEL = '__shm_npy__'
_MIN_SENTINEL_TUPLE_LEN = 2


def _write_to_shm(data: bytes) -> str:
    """Записать bytes в новый SharedMemory блок, вернуть имя."""
    shm = SharedMemory(create=True, size=len(data))
    shm.buf[: len(data)] = data
    name = shm.name
    shm.close()  # close handle, but don't unlink — reader сделает unlink
    return name


def _read_from_shm(name: str, size: int) -> bytes:
    """Прочитать bytes из SharedMemory и освободить блок."""
    shm = SharedMemory(name=name, create=False)
    data = bytes(shm.buf[:size])
    shm.close()
    shm.unlink()
    return data


# ---------------------------------------------------------------------------
# PIL Image сериализация
# ---------------------------------------------------------------------------


def _serialize_pil(img: Image.Image) -> tuple[str, int, str, tuple[int, int]]:
    """
    Сжать PIL Image и записать в SharedMemory.

    Returns:
        (shm_name, data_length, mode, size)

    """
    t0 = time.monotonic()
    original_mode = img.mode
    if img.mode in ('RGBA', 'LA', 'PA'):
        raw = img.tobytes()
        data = zlib.compress(raw, _ZLIB_LEVEL)
        fmt = 'zlib'
    else:
        buf = io.BytesIO()
        save_img = img.convert('RGB') if img.mode not in ('RGB', 'L') else img
        save_img.save(buf, format='JPEG', quality=95)
        data = buf.getvalue()
        fmt = 'JPEG'

    shm_name = _write_to_shm(data)
    logger.info(
        'serialize_pil: %s %dx%d → %s %.1f MB → shm[%s] in %.3f sec',
        original_mode,
        *img.size,
        fmt,
        len(data) / 1e6,
        shm_name,
        time.monotonic() - t0,
    )
    return (shm_name, len(data), original_mode, img.size)


def deserialize_pil(
    shm_name: str,
    data_len: int,
    mode: str,
    _size: tuple[int, int],
) -> Image.Image:
    """Восстановить PIL Image из SharedMemory."""
    t0 = time.monotonic()
    Image.MAX_IMAGE_PIXELS = None
    data = _read_from_shm(shm_name, data_len)
    t_read = time.monotonic()

    # Attempt zlib decompression (RGBA path), fall back to JPEG/PNG
    raw_bytes: bytes | None = None
    with contextlib.suppress(zlib.error):
        raw_bytes = zlib.decompress(data)

    if raw_bytes is not None:
        del data
        img = Image.frombytes(mode, _size, raw_bytes)
        logger.info(
            'deserialize_pil: shm[%s] %.1f MB → %s %dx%d in %.3f sec (read=%.3f)',
            shm_name,
            data_len / 1e6,
            mode,
            *_size,
            time.monotonic() - t0,
            t_read - t0,
        )
        return img

    # JPEG/PNG fallback
    raw = Image.open(io.BytesIO(data))
    raw.load()
    del data
    t1 = time.monotonic()
    if raw.mode != mode:
        result = raw.convert(mode)
        logger.info(
            'deserialize_pil: shm[%s] img %.1f MB → %dx%d (read=%.3f, '
            'decode=%.3f, convert %s→%s=%.3f)',
            shm_name,
            data_len / 1e6,
            *_size,
            t_read - t0,
            t1 - t_read,
            raw.mode,
            mode,
            time.monotonic() - t1,
        )
        return result
    logger.info(
        'deserialize_pil: shm[%s] img %.1f MB → %s %dx%d in %.3f sec (read=%.3f)',
        shm_name,
        data_len / 1e6,
        mode,
        *_size,
        time.monotonic() - t0,
        t_read - t0,
    )
    return raw


# ---------------------------------------------------------------------------
# numpy array сериализация
# ---------------------------------------------------------------------------


def _serialize_numpy(arr: np.ndarray) -> tuple[str, int, str, tuple[int, ...]]:
    """
    Записать numpy array в SharedMemory (без сжатия).

    Returns:
        (shm_name, data_length, dtype_str, shape)

    """
    t0 = time.monotonic()
    data = arr.tobytes()
    shm_name = _write_to_shm(data)
    logger.info(
        'serialize_numpy: %s %s → shm[%s] %.1f MB in %.3f sec',
        arr.dtype,
        arr.shape,
        shm_name,
        len(data) / 1e6,
        time.monotonic() - t0,
    )
    return (shm_name, len(data), str(arr.dtype), arr.shape)


def _deserialize_numpy(
    shm_name: str,
    data_len: int,
    dtype_str: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Восстановить numpy array из SharedMemory."""
    t0 = time.monotonic()
    data = _read_from_shm(shm_name, data_len)
    arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape).copy()
    logger.info(
        'deserialize_numpy: shm[%s] %.1f MB → %s %s in %.3f sec',
        shm_name,
        data_len / 1e6,
        arr.dtype,
        arr.shape,
        time.monotonic() - t0,
    )
    return arr


# ---------------------------------------------------------------------------
# rh_cache сериализация (PIL + numpy + скаляры)
# ---------------------------------------------------------------------------


def _serialize_rh_cache(rh_cache: dict | None) -> dict | None:
    """Сериализовать rh_cache: PIL → shm, numpy → shm, остальное as-is."""
    if rh_cache is None:
        return None
    t0 = time.monotonic()
    result: dict = {}
    for key, value in rh_cache.items():
        if isinstance(value, Image.Image):
            result[key] = (_SHM_PIL_SENTINEL, *_serialize_pil(value))
        elif isinstance(value, np.ndarray):
            result[key] = (_SHM_NPY_SENTINEL, *_serialize_numpy(value))
        else:
            result[key] = value
    logger.info(
        'serialize_rh_cache: %d keys in %.3f sec',
        len(result),
        time.monotonic() - t0,
    )
    return result


def deserialize_rh_cache(rh_cache: dict | None) -> dict | None:
    """Десериализовать rh_cache: shm → PIL/numpy, остальное as-is."""
    if rh_cache is None:
        return None
    t0 = time.monotonic()
    result: dict = {}
    for key, value in rh_cache.items():
        if isinstance(value, tuple) and len(value) >= _MIN_SENTINEL_TUPLE_LEN:
            sentinel = value[0]
            if sentinel == _SHM_PIL_SENTINEL:
                _, shm_name, data_len, mode, size = value
                result[key] = deserialize_pil(shm_name, data_len, mode, size)
                continue
            if sentinel == _SHM_NPY_SENTINEL:
                _, shm_name, data_len, dtype_str, shape = value
                result[key] = _deserialize_numpy(shm_name, data_len, dtype_str, shape)
                continue
        result[key] = value
    logger.info(
        'deserialize_rh_cache: %d keys in %.3f sec',
        len(result),
        time.monotonic() - t0,
    )
    return result


# ---------------------------------------------------------------------------
# dem_grid сериализация
# ---------------------------------------------------------------------------


def serialize_dem_grid(dem_grid: object | None) -> object | None:
    """Сериализовать dem_grid (numpy) в SharedMemory."""
    if dem_grid is None or not isinstance(dem_grid, np.ndarray):
        return dem_grid
    return (_SHM_NPY_SENTINEL, *_serialize_numpy(dem_grid))


def deserialize_dem_grid(dem_grid: object | None) -> object | None:
    """Десериализовать dem_grid из SharedMemory."""
    if (
        isinstance(dem_grid, tuple)
        and len(dem_grid) >= _MIN_SENTINEL_TUPLE_LEN
        and dem_grid[0] == _SHM_NPY_SENTINEL
    ):
        _, shm_name, data_len, dtype_str, shape = dem_grid
        return _deserialize_numpy(shm_name, data_len, dtype_str, shape)
    return dem_grid


# ---------------------------------------------------------------------------
# QueueProgressSink — сериализует события в mp.Queue
# ---------------------------------------------------------------------------


class QueueProgressSink:
    """ProgressSink, пишущий события в mp.Queue для опроса из GUI-потока."""

    def __init__(self, queue: mp.Queue) -> None:
        self._q = queue

    def on_progress(self, done: int, total: int, label: str) -> None:
        self._q.put(('progress', done, total, label))

    def on_spinner(self, label: str) -> None:
        self._q.put(('spinner', label))

    def on_warning(self, text: str, field_updates: dict | None = None) -> None:
        self._q.put(('warning', text, field_updates))

    def on_preview(
        self,
        image: Image.Image,
        metadata: MapMetadata | None,
        dem_grid: object | None,
        rh_cache: dict | None,
    ) -> None:
        t0 = time.monotonic()
        img_data = _serialize_pil(image)
        dem_data = serialize_dem_grid(dem_grid)
        rh_data = _serialize_rh_cache(rh_cache)
        logger.info(
            'on_preview: total serialization %.3f sec',
            time.monotonic() - t0,
        )
        self._q.put(('preview', img_data, metadata, dem_data, rh_data))


# ---------------------------------------------------------------------------
# Главная функция дочернего процесса
# ---------------------------------------------------------------------------


def worker_process_main(
    params: DownloadParams,
    queue: mp.Queue,
    cancel_event: _MpEvent,
) -> None:
    """
    Запускается в дочернем процессе через ``mp.Process(target=...)``.

    Проксирует все события прогресса/превью в *queue*,
    а результат (success/error) — как финальное сообщение ``('finished', ...)``.
    """
    # Настроить sys.path — дочерний процесс не наследует PYTHONPATH автоматически
    src_dir = str(Path(__file__).resolve().parent.parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Настроить логирование — тот же формат что в main.py
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        _local = os.getenv('LOCALAPPDATA')
        if _local:
            # Windows
            _log_dir = Path(_local) / 'SK42' / 'log'
        else:
            # Linux/macOS — XDG_STATE_HOME
            _xdg_state = os.getenv('XDG_STATE_HOME', '')
            _log_dir = (
                Path(_xdg_state) / 'sk42mapper' / 'log'
                if _xdg_state
                else Path.home() / '.local' / 'state' / 'sk42mapper' / 'log'
            )
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_file = _log_dir / 'mil_mapper.log'
        handlers.append(logging.FileHandler(str(_log_file), encoding='utf-8'))
    except Exception:
        logger.debug('Failed to set up file logging', exc_info=True)
    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=handlers)

    _progress = importlib.import_module('shared.progress')
    _cancelled_error = _progress.CancelledError
    _cancel_token_cls = _progress.EventCancelToken

    sink = QueueProgressSink(queue)
    cancel = _cancel_token_cls(cancel_event)

    try:
        # Глубокая проверка API до старта тяжёлой работы
        sink.on_spinner('Проверка подключения…')
        _diag = importlib.import_module('shared.diagnostics')
        asyncio.run(
            _diag.run_deep_verification(
                api_key=params.api_key,
                settings=params.settings,
            )
        )

        _map_job = importlib.import_module('services.map_job')
        _map_job.run_map_job(params, sink, cancel)
        queue.put(('finished', True, ''))
    except _cancelled_error:
        queue.put(('finished', False, 'Операция отменена пользователем'))
    except Exception as e:
        logger.exception('Worker process failed')
        queue.put(('finished', False, str(e)))
