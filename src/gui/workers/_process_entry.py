"""
Точка входа дочернего процесса для создания карты.

Запускается через ``multiprocessing.Process``.
Проксирует события прогресса/превью в ``mp.Queue`` для опроса из GUI-потока.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import gc
import io
import logging
import os
import sys
import threading
import time
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
# PIL RGBA/LA/PA: tobytes() → raw bytes → SharedMemory (no compression)
# PIL RGB/L: JPEG quality=95 → SharedMemory
# numpy: np.copyto into shm view → SharedMemory (no compression)
# ---------------------------------------------------------------------------

# SharedMemory блоки, которые нужно держать открытыми до выхода процесса.
# На Windows mapping уничтожается при закрытии последнего хэндла,
# поэтому writer НЕ закрывает хэндл — reader откроет свой и сделает unlink.
_shm_keep_alive: list[SharedMemory] = []

# Sentinels для rh_cache dict values
_SHM_PIL_SENTINEL = '__shm_pil__'
_SHM_NPY_SENTINEL = '__shm_npy__'
_MIN_SENTINEL_TUPLE_LEN = 2


def _write_to_shm(data: bytes) -> str:
    """Записать bytes в новый SharedMemory блок, вернуть имя."""
    shm = SharedMemory(create=True, size=len(data))
    shm.buf[: len(data)] = data
    _shm_keep_alive.append(shm)  # НЕ закрываем — reader откроет и сделает unlink
    return shm.name


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
    Сериализовать PIL Image в SharedMemory.

    RGBA/LA/PA: raw bytes без сжатия. /dev/shm — это tmpfs в RAM, размер
    блока влияет только на доступную память во время transit (~1-2с),
    а zlib level=1 на 8404²RGBA стоил ~1с на компрессию в воркере и ~0.6с
    на декомпрессию в GUI.

    Реализация — zero-copy + numpy memcpy:
      np.asarray(img)   — zero-copy view PIL-буфера (PIL.Image имеет
                          __array_interface__)
      shm_view = np.ndarray(..., buffer=shm.buf)
      np.copyto(shm_view, arr)   — один memcpy ПРЯМО в shm, отпускает GIL
                                   во время копирования (важно: 4 worker'а
                                   в параллельном пуле _serialize_rh_cache
                                   реально работают параллельно).
    Старый путь img.tobytes() + shm.buf[:N]=data делал ДВА memcpy
    (PIL → bytes → shm) и держал GIL в tobytes — 4 worker'а
    сериализовались через GIL.

    RGB/L: JPEG quality=95 (для preview-изображения сжатие 200MB → 10MB
    действительно ценное и быстрое — libjpeg отпускает GIL и SIMD).

    Returns:
        (shm_name, data_length, mode, size). Reader использует `mode`
        для выбора десериализации (RGBA-семейство → raw, остальное → JPEG).

    """
    t0 = time.monotonic()
    original_mode = img.mode
    if img.mode in ('RGBA', 'LA', 'PA'):
        arr = np.asarray(img)
        nbytes = arr.nbytes
        shm = SharedMemory(create=True, size=nbytes)
        shm_view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np.copyto(shm_view, arr)
        _shm_keep_alive.append(shm)
        shm_name = shm.name
        data_len = nbytes
        fmt = 'raw'
    else:
        buf = io.BytesIO()
        save_img = img.convert('RGB') if img.mode not in ('RGB', 'L') else img
        save_img.save(buf, format='JPEG', quality=95)
        data = buf.getvalue()
        shm_name = _write_to_shm(data)
        data_len = len(data)
        fmt = 'JPEG'

    logger.info(
        'serialize_pil: %s %dx%d → %s %.1f MB → shm[%s] in %.3f sec',
        original_mode,
        *img.size,
        fmt,
        data_len / 1e6,
        shm_name,
        time.monotonic() - t0,
    )
    return (shm_name, data_len, original_mode, img.size)


def deserialize_pil(
    shm_name: str,
    data_len: int,
    mode: str,
    _size: tuple[int, int],
) -> Image.Image:
    """
    Восстановить PIL Image из SharedMemory.

    Диспетчеризация по mode: RGBA/LA/PA — raw bytes (соответствует
    raw-серилизации в _serialize_pil), всё остальное — JPEG/PNG decode.
    """
    t0 = time.monotonic()
    Image.MAX_IMAGE_PIXELS = None

    if mode in ('RGBA', 'LA', 'PA'):
        # Raw numpy path — match _serialize_pil's RGBA branch. Single memcpy:
        # shm → fresh numpy array (.copy() forces a deep copy so we can safely
        # unlink shm), then Image.fromarray is a zero-copy view over that
        # array. Old path was: bytes(shm.buf[:size]) → bytes → Image.frombytes,
        # which paid TWO memcpies (shm→bytes, bytes→PIL internal buffer).
        channels = {'RGBA': 4, 'LA': 2, 'PA': 2}[mode]
        w, h = _size
        shape = (h, w, channels)
        shm = SharedMemory(name=shm_name, create=False)
        try:
            shm_view = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
            arr = shm_view.copy()
        finally:
            shm.close()
            shm.unlink()
        t_read = time.monotonic()
        # PIL infers mode from array shape: (H,W,4)→RGBA, (H,W,2)→LA.
        # Omitting `mode=` keyword avoids the Pillow-13 deprecation warning.
        img = Image.fromarray(arr)
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

    # JPEG/PNG path (RGB/L from _serialize_pil's JPEG branch)
    data = _read_from_shm(shm_name, data_len)
    t_read = time.monotonic()
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
    Записать numpy array в SharedMemory одним memcpy через np.copyto в
    view на shm.buf. Раньше использовался arr.tobytes() — это аллокация
    отдельного bytes-объекта на 282MB + второй memcpy при записи в shm.

    Returns:
        (shm_name, data_length, dtype_str, shape)

    """
    t0 = time.monotonic()
    contiguous = np.ascontiguousarray(arr)
    nbytes = contiguous.nbytes
    shm = SharedMemory(create=True, size=nbytes)
    shm_view = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
    np.copyto(shm_view, contiguous)
    _shm_keep_alive.append(shm)
    logger.info(
        'serialize_numpy: %s %s → shm[%s] %.1f MB in %.3f sec',
        arr.dtype,
        arr.shape,
        shm.name,
        nbytes / 1e6,
        time.monotonic() - t0,
    )
    return (shm.name, nbytes, str(arr.dtype), arr.shape)


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


_SERIALIZE_RH_CACHE_MAX_WORKERS = 3


def _serialize_rh_cache(rh_cache: dict | None) -> dict | None:
    """
    Сериализовать rh_cache: PIL → shm, numpy → shm, остальное as-is.

    Для elev_color/RH/Radar/NSU тут лежат 2-4 RGBA-слоя для
    интерактивного alpha-слайдера в GUI. Heavy items
    (Image.Image / np.ndarray) идут параллельно в небольшом
    ThreadPoolExecutor; identity-dedupe гарантирует, что два ключа,
    смотрящие на один объект, не создают два shm-сегмента (overlay
    cache-HIT path: overlay_base = overlay_layer).

    Раньше делали последовательно из-за memory-bandwidth и kernel-
    contention при 4 потоках по 282 MB. После того как (1) MALLOC
    _ARENA_MAX=2 убрал arena fragmentation и (2) identity-dedupe убрал
    дубликат overlay, осталось 3 heavy items вместо 4. Re-bench под
    нынешние условия (workers=3):
      sequential : 413 ms
      parallel-3 : 356 ms   (~14 % быстрее)
      parallel-4 : 318 ms   (~23 % быстрее, но per-publish мы редко
                              имеем 4 разнотипных heavy items)
    На full-size 8k² overlay экономия ~80-150 ms за publish.
    """
    if rh_cache is None:
        return None
    t0 = time.monotonic()

    # First pass on the main thread: dedupe heavy values by id() and
    # collect work items. Scalars/strings/None go straight into the
    # output dict.
    aliased_count = 0
    pending: list[tuple[str, object]] = []
    pending_ids: dict[int, str] = {}  # id(value) → key that owns the alloc
    result: dict = {}
    for key, value in rh_cache.items():
        if isinstance(value, (Image.Image, np.ndarray)):
            owner_key = pending_ids.get(id(value))
            if owner_key is not None:
                # Defer aliasing to the second pass; mark the alias
                # slot with the owner so we can copy after dispatch.
                result[key] = ('__alias__', owner_key)
                aliased_count += 1
                continue
            pending_ids[id(value)] = key
            pending.append((key, value))
        else:
            result[key] = value

    # Heavy work in parallel. Each worker calls _serialize_pil /
    # _serialize_numpy which release the GIL inside np.copyto (memcpy
    # into shm), so thread parallelism actually buys wall-time despite
    # memory-bandwidth competition.
    def _serialize_one(value):
        if isinstance(value, Image.Image):
            return (_SHM_PIL_SENTINEL, *_serialize_pil(value))
        return (_SHM_NPY_SENTINEL, *_serialize_numpy(value))

    if pending:
        n_workers = min(_SERIALIZE_RH_CACHE_MAX_WORKERS, len(pending))
        if n_workers == 1:
            # No win from a single-worker pool — skip the executor
            # overhead and run inline.
            for key, value in pending:
                result[key] = _serialize_one(value)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers,
            ) as ex:
                future_to_key = {
                    ex.submit(_serialize_one, value): key
                    for key, value in pending
                }
                for fut in future_to_key:
                    result[future_to_key[fut]] = fut.result()

    # Second pass: resolve `__alias__` placeholders to the just-
    # serialized tuple of the owner key.
    for key, value in list(result.items()):
        if (
            isinstance(value, tuple)
            and len(value) == 2  # noqa: PLR2004
            and value[0] == '__alias__'
        ):
            result[key] = result[value[1]]

    logger.info(
        'serialize_rh_cache: %d keys (%d heavy parallel, %d aliased) '
        'in %.3f sec',
        len(result),
        len(pending),
        aliased_count,
        time.monotonic() - t0,
    )
    return result


def deserialize_rh_cache(rh_cache: dict | None) -> dict | None:
    """
    Десериализовать rh_cache: shm → PIL/numpy, остальное as-is.

    Identity-dedupe paired with `_serialize_rh_cache`: если два ключа
    хранят один и тот же serialized tuple (тот же shm_name — сериализатор
    делает это, когда исходные PIL/numpy объекты были одним объектом, как
    в overlay-cache HIT, где overlay_base = overlay_layer = один RGBA),
    мы должны десериализовать только ПЕРВЫЙ и отдать тот же объект
    остальным ключам. Иначе deserialize_pil unlink-нёт shm после первого
    чтения, и второй вызов получит пустой буфер → чёрный image без
    видимых ошибок.
    """
    if rh_cache is None:
        return None
    t0 = time.monotonic()
    result: dict = {}
    # shm_name → already-deserialized object. Keyed on shm_name so the
    # mapping survives whatever tuple-identity quirks queue-pickling may
    # introduce on the way over.
    by_shm: dict[str, object] = {}
    for key, value in rh_cache.items():
        if isinstance(value, tuple) and len(value) >= _MIN_SENTINEL_TUPLE_LEN:
            sentinel = value[0]
            if sentinel == _SHM_PIL_SENTINEL:
                _, shm_name, data_len, mode, size = value
                cached = by_shm.get(shm_name)
                if cached is not None:
                    result[key] = cached
                    continue
                obj = deserialize_pil(shm_name, data_len, mode, size)
                by_shm[shm_name] = obj
                result[key] = obj
                continue
            if sentinel == _SHM_NPY_SENTINEL:
                _, shm_name, data_len, dtype_str, shape = value
                cached = by_shm.get(shm_name)
                if cached is not None:
                    result[key] = cached
                    continue
                obj = _deserialize_numpy(shm_name, data_len, dtype_str, shape)
                by_shm[shm_name] = obj
                result[key] = obj
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


_DEM_GRID_IPC_MAX_DIM = 2048  # Cap dem_grid IPC size: 8404² float32 = 282MB → 4MB+.


def serialize_dem_grid(dem_grid: object | None) -> object | None:
    """
    Сериализовать dem_grid (numpy) в SharedMemory.

    Для IPC даунсэмплим точечной выборкой через stride-view — массив 8404×8404
    float32 (~282MB) превращается в ~2100×2100 (~18MB). Подсказка под
    курсором показывает целое число метров; на расстояниях 4 пикселя
    изображения значение меняется незначительно. GUI учитывает реальную
    форму массива относительно метаданных изображения.
    """
    if dem_grid is None or not isinstance(dem_grid, np.ndarray):
        return dem_grid
    max_dim = max(dem_grid.shape[0], dem_grid.shape[1])
    if max_dim > _DEM_GRID_IPC_MAX_DIM:
        ds = max(1, max_dim // _DEM_GRID_IPC_MAX_DIM)
        dem_grid = dem_grid[::ds, ::ds]
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
        # Free up VMS room BEFORE we start mmap'ing shm sergments. NSU /
        # RH / Radar builds at z=17 leave worker at ~4-5 GB RSS, and the
        # 8th shm mmap (282 MB RGBA overlay) periodically hit OSError 12
        # under RLIMIT_AS=0.70×RAM. gc.collect + malloc_trim drops
        # transient allocations + glibc arena fragmentation, typically
        # freeing 500-800 MB of address space.
        _release_vms_for_publish()

        # Track shm names created in THIS publish call so we can unlink
        # any orphans if a later serialize raises. Without this, a
        # partial failure (e.g. ENOMEM on the 8th mmap) left 7 shm
        # segments alive on /dev/shm + resource_tracker, and GUI got
        # nothing → black preview.
        keep_alive_before = len(_shm_keep_alive)
        try:
            # PIL JPEG-encode and numpy memcpy are independent and release
            # the GIL during their inner C/SIMD work. Running them
            # concurrently in a tiny threadpool turns sequential
            # 0.22s + 0.55s ≈ 0.78s into max(0.22, 0.28) wallclock.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                img_future = ex.submit(_serialize_pil, image)
                dem_future = ex.submit(serialize_dem_grid, dem_grid)
                img_data = img_future.result()
                dem_data = dem_future.result()
            rh_data = _serialize_rh_cache(rh_cache)
            logger.info(
                'on_preview: total serialization %.3f sec',
                time.monotonic() - t0,
            )
            self._q.put(('preview', img_data, metadata, dem_data, rh_data))
        except Exception:
            # Drop any shm we managed to create this call so /dev/shm
            # doesn't leak and resource_tracker isn't filled at shutdown.
            orphaned = _shm_keep_alive[keep_alive_before:]
            del _shm_keep_alive[keep_alive_before:]
            for shm in orphaned:
                with contextlib.suppress(Exception):
                    shm.close()
                with contextlib.suppress(Exception):
                    shm.unlink()
            logger.exception(
                'on_preview: serialization failed after %d shm (cleaned up)',
                len(orphaned),
            )
            raise


# ---------------------------------------------------------------------------
# Memory protection for child process
# ---------------------------------------------------------------------------


def _set_child_memory_limit() -> None:
    """
    Set RLIMIT_AS + bias OOM killer toward this child (Linux only).

    RLIMIT_AS budget = MEMORY_RLIMIT_RATIO × RAM (NOT RAM + swap). The
    old (total = RAM + swap) × 0.85 formula was unsafe on systems where
    total ≈ physical: on a 13 GB box with 4 GB swap that limit was
    14.8 GB, so the worker could touch ~all RAM before MemoryError fired,
    leaving nothing for GUI / IDE / OS → system-wide OOM where the
    kernel killer sometimes picks the IDE despite our oom_score_adj=1000
    bias on the worker (when worker RSS is small relative to IDE's, the
    score gradient flips).

    New formula: RAM × ratio, where ratio defaults to 0.65 — leaves
    ~35% of physical RAM (≈4.6 GB on a 13 GB box) for GUI + IDE + OS.
    The worker hits MemoryError inside its own process WELL before the
    system as a whole runs out, so oom_score_adj=1000 then just serves
    as an extra belt-and-braces for true global pressure.

    Additionally we set /proc/self/oom_score_adj = 1000 so the kernel's
    OOM killer picks this child first if the global pressure does
    somehow hit a wall (e.g. concurrent large allocation by another
    process). Worker death is recoverable; IDE death is not.
    """
    try:
        import resource

        import psutil
    except ImportError:
        return

    from shared.constants import MEMORY_RLIMIT_RATIO

    # Bias OOM killer toward this process — kernel kills us first if the
    # global memory pressure hits a wall. oom_score_adj range [-1000, 1000];
    # 1000 = "kill me first". Best-effort write; if /proc isn't writable
    # (containers etc.) we silently skip.
    with contextlib.suppress(OSError):
        with open('/proc/self/oom_score_adj', 'w') as f:
            f.write('1000\n')

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    # RAM-only cap (NOT RAM+swap): the worker must MemoryError before
    # physical RAM is exhausted, so GUI / IDE / OS keep working.
    limit_bytes = int(mem.total * MEMORY_RLIMIT_RATIO)

    try:
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit_bytes = min(limit_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        logger.info(
            'Child RLIMIT_AS set to %.0f MB (RAM=%.0f MB × ratio=%.2f, '
            'reserve ~%.0f MB for GUI+IDE+OS; swap=%.0f MB available '
            'as backing only — not counted in limit), oom_score_adj=1000',
            limit_bytes / (1024 * 1024),
            mem.total / (1024 * 1024),
            MEMORY_RLIMIT_RATIO,
            (mem.total - limit_bytes) / (1024 * 1024),
            swap.total / (1024 * 1024),
        )
    except (ValueError, OSError) as e:
        logger.warning('Failed to set child RLIMIT_AS: %s', e)


# ---------------------------------------------------------------------------
# Главная функция дочернего процесса
# ---------------------------------------------------------------------------


def _setup_worker_environment() -> None:
    """One-time setup: sys.path, logging, RLIMIT_AS. Called once per process."""
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

    # Set RLIMIT_AS in child process — spawn mode does NOT inherit it
    # from the parent. Without this, Linux overcommit lets malloc succeed
    # and OOM killer crashes the entire system instead of raising MemoryError.
    _set_child_memory_limit()


def _release_shm_after_build() -> None:
    """
    Release SharedMemory blocks created during a build.

    On Windows the mapping is destroyed when the last handle closes, so we
    spawn a background thread that sleeps 2s (giving the GUI's reader a
    chance to open its own handles) before dropping ours. The main loop
    returns immediately so the persistent worker is ready for the next
    build without freezing.

    On Linux POSIX shm survives writer close as long as the file exists,
    so we close immediately.
    """
    if not _shm_keep_alive:
        return

    if sys.platform == 'win32':
        # Detach the current keep-alive list and clean it up in the background.
        to_close = _shm_keep_alive.copy()
        _shm_keep_alive.clear()

        def _delayed_close() -> None:
            time.sleep(2)
            for shm in to_close:
                with contextlib.suppress(Exception):
                    shm.close()

        threading.Thread(target=_delayed_close, daemon=True).start()
    else:
        # Linux: drop writer handles immediately; readers can still open
        # by name as long as no one has unlink'd the segment.
        for shm in _shm_keep_alive:
            with contextlib.suppress(Exception):
                shm.close()
        _shm_keep_alive.clear()


def _run_one_build(
    params: DownloadParams,
    queue: mp.Queue,
    cancel_event: _MpEvent,
) -> None:
    """Process a single build request and surface the outcome via `queue`."""
    import gc

    from services import map_job
    from shared.progress import CancelledError, EventCancelToken

    sink = QueueProgressSink(queue)
    cancel = EventCancelToken(cancel_event)

    try:
        map_job.run_map_job(params, sink, cancel)
        queue.put(('finished', True, ''))
    except CancelledError:
        queue.put(('finished', False, 'Операция отменена пользователем'))
    except MemoryError:
        logger.exception('Worker process: MemoryError (RLIMIT_AS triggered)')
        queue.put(('finished', False, 'Недостаточно памяти для построения карты'))
    except Exception as e:
        logger.exception('Worker process failed')
        queue.put(('finished', False, str(e)))
    finally:
        _release_shm_after_build()
        # Active hint that next build starts from a clean reference graph.
        gc.collect()
        # Compact the C heap back to the OS. C extensions (numpy, PIL,
        # cv2, libjpeg) leave the allocator's heap inflated — on Linux
        # ~500 MB RSS after a SATELLITE retina build that tracemalloc
        # says is only ~55 MB Python-side. Returning that high-water
        # mark lets the next build start from near-baseline RSS instead
        # of compounding under RLIMIT_AS. No-op on unsupported runtimes.
        _compact_c_heap()


def _release_vms_for_publish() -> None:
    """
    Free VMS room before on_preview mmap's the publication shm segments.

    Under RLIMIT_AS = 0.70×RAM the worker's address space ceiling is
    tight (~9.3 GB on a 13 GB box). NSU / RH / RADAR builds leave RSS
    at ~4-5 GB by the time we reach the publish step, и 7-8 shm mmap
    суммарно ~600-800 MB добавляли к VMS. На 8-м mmap (большой RGBA
    overlay) периодически случался OSError 12 «Cannot allocate
    memory» — мапил в VA-space ровно на лимите.

    gc.collect() сначала роняет недостижимые объекты (огромные
    transient numpy в postprocess). malloc_trim возвращает свободные
    арены glibc обратно ядру — типично освобождает 500-800 MB
    address space, чего достаточно для 8th shm.
    """
    gc.collect()
    _compact_c_heap()


def _compact_c_heap() -> None:
    """
    Ask the platform allocator to return free pages back to the OS.

    Linux (glibc):
      `malloc_trim(0)` — trims the main arena and (since glibc 2.16) other
      arenas down to the high-water mark of live allocations. Pairs with
      MALLOC_ARENA_MAX=2 so the worker keeps a tight footprint between
      builds.

    Windows (MSVC CRT):
      `HeapCompact(GetProcessHeap(), 0)` — coalesces free blocks and
      returns committed-but-unused pages to the VMM. Windows has no
      glibc-style multi-arena reservation, so the pressure is lower in
      the first place, but compacting still helps when many short-lived
      transient buffers (decode scratch, label bitmaps) fragment the
      default heap.

    Other platforms (macOS, musl, etc): no-op.

    The call is synchronous but typically <50 ms; we run it after
    gc.collect() in the post-build finally so the cost is hidden behind
    the queue round-trip back to the GUI.
    """
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
        logger.debug('_compact_c_heap failed', exc_info=True)


def _rss_mb() -> float:
    """Return current process RSS in MB, or 0.0 on failure."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def persistent_worker_main(
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    cancel_event: _MpEvent,
) -> None:
    """
    Long-lived worker that processes build requests from `request_queue`.

    Replaces the previous one-build-per-process model. Eliminates per-build
    Python startup + pyproj/numpy/cv2 import cost (~1s end-to-end). The
    process exits when it receives a ('shutdown',) request or sentinel None.

    request_queue protocol:
      ('build', DownloadParams)  → perform a build
      ('shutdown',) | None       → exit the loop

    result_queue protocol: unchanged — progress/spinner/warning/preview
    messages from QueueProgressSink, then ('finished', success, error_msg).
    """
    _setup_worker_environment()
    pid = os.getpid()
    logger.info('Persistent worker process started (pid=%d)', pid)

    # Warm up Numba JIT in the background so the user's first
    # recompute / build doesn't pay the 2-5 s compile cost. Runs in a
    # daemon thread — if a build request arrives before warmup is
    # done, the build's njit calls will block on JIT compile as they
    # used to; otherwise the kernels are ready by the time the user
    # interacts.
    def _warmup() -> None:
        try:
            t0 = time.monotonic()
            from services.radio_horizon import warmup_numba_kernels
            warmup_numba_kernels()
            logger.info(
                'Numba JIT warmup done in %.2f s', time.monotonic() - t0,
            )
        except Exception:
            logger.warning('Numba warmup failed', exc_info=True)

    threading.Thread(
        target=_warmup, name='numba-warmup', daemon=True,
    ).start()

    builds_done = 0
    while True:
        try:
            msg = request_queue.get()
        except (EOFError, OSError):
            logger.info('Persistent worker: request queue closed, exiting')
            return
        if msg is None:
            return
        kind = msg[0]
        if kind == 'shutdown':
            logger.info(
                'Persistent worker: shutdown requested (builds_done=%d, RSS=%.0fMB)',
                builds_done, _rss_mb(),
            )
            return
        if kind != 'build':
            logger.warning('Persistent worker: unknown message kind=%r', kind)
            continue

        params: DownloadParams = msg[1]
        cancel_event.clear()
        rss_before = _rss_mb()
        logger.info(
            'Persistent worker: build #%d start (RSS=%.0fMB)',
            builds_done + 1, rss_before,
        )
        _run_one_build(params, result_queue, cancel_event)
        builds_done += 1
        rss_after = _rss_mb()
        logger.info(
            'Persistent worker: build #%d done (RSS=%.0fMB, Δ=%+.0fMB)',
            builds_done, rss_after, rss_after - rss_before,
        )


# Backwards-compatible alias for any external caller still importing the
# old name. Internally everything now goes through persistent_worker_main.
def worker_process_main(
    params: DownloadParams,
    queue: mp.Queue,
    cancel_event: _MpEvent,
) -> None:
    """One-shot wrapper kept for tests and any external callers."""
    _setup_worker_environment()
    _run_one_build(params, queue, cancel_event)
