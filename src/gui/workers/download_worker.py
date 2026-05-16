"""
Download worker — отправляет build-запросы в долгоживущий рабочий процесс.

Раньше каждое построение карты порождало новый ``multiprocessing.Process``
через ``mp.Process(target=worker_process_main, ...)``. Это давало ~1 секунду
накладных расходов на каждом ребилде: spawn интерпретатора + повторный
импорт pyproj/numpy/cv2/PIL.

Сейчас один воркер-процесс живёт от первого до последнего билда. ``start()``
просто кладёт ``('build', params)`` в request-очередь воркера и подписывает
reader-поток на чтение result-очереди. ``stop_and_join()`` не убивает воркер
— он остаётся ждать следующего запроса.

Воркер шарится между всеми DownloadWorker-инстансами через class-level
поля (``_shared_process``, ``_shared_req_queue``, ``_shared_out_queue``,
``_shared_cancel``). View.py гарантирует, что в любой момент активна не
больше одного DownloadWorker — старый ``stop_and_join`` зовётся до создания
нового.

Фоновый ``threading.Thread`` читает ``mp.Queue`` и десериализует тяжёлые
данные (preview, dem_grid, rh_cache), а лёгкие сигналы пересылаются
в GUI-поток через ``QTimer(50 ms)`` опрос ``queue.Queue``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue_mod
import threading
from typing import TYPE_CHECKING, ClassVar

from PIL import Image
from PySide6.QtCore import QObject, QTimer, Signal

from gui.workers._process_entry import (
    deserialize_dem_grid,
    deserialize_pil,
    deserialize_rh_cache,
    persistent_worker_main,
)

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as _MpEvent

    from domain.models import DownloadParams

logger = logging.getLogger(__name__)

# Интервал опроса внутренней очереди из GUI-потока (мс)
_POLL_INTERVAL_MS = 50


class DownloadWorker(QObject):
    """
    Handle одного build-запроса к долгоживущему воркеру.

    Сигналы полностью совместимы с предыдущим QThread/per-build Process
    вариантом, поэтому view.py подключается к ним без изменений.
    """

    finished = Signal(bool, str)  # success, error_message
    progress_update = Signal(int, int, str)  # done, total, label
    warning_received = Signal(str, object)  # warning text, field_updates dict|None
    # Fired from reader thread the moment a new preview arrives from the
    # worker, BEFORE its heavy PIL/numpy payload is deserialised. Gives the
    # GUI ~0.5–3s head-start to release the previous build's _rh_cache and
    # base image, instead of holding both builds simultaneously in RAM at
    # the deserialisation peak (which on z=17 elev_color caused OOM SIGKILL).
    preview_starting = Signal()
    preview_ready = Signal(
        Image.Image,
        object,
        object,
        object,
    )  # PIL Image, MapMetadata, dem_grid, rh_cache

    # ------------------------------------------------------------------
    # Class-level persistent worker (один процесс на всё приложение)
    # ------------------------------------------------------------------
    _shared_process: ClassVar[mp.Process | None] = None
    _shared_req_queue: ClassVar[mp.Queue | None] = None
    _shared_out_queue: ClassVar[mp.Queue | None] = None
    _shared_cancel: ClassVar[_MpEvent | None] = None
    _shared_lock: ClassVar[threading.Lock] = threading.Lock()
    # Watchdog: если RSS воркера после билда превышает порог — пометить
    # на респавн. На следующем _ensure_process воркер будет shutdown'нут
    # и спавнится свежий. Это защита от долгосрочных утечек.
    _shared_marked_for_replacement: ClassVar[bool] = False
    _max_rss_mb: ClassVar[int] = 4096  # 4 GB

    @classmethod
    def _replace_shared_process_locked(cls) -> None:
        """Shut down current worker (within already-held _shared_lock)."""
        if cls._shared_process is None:
            return
        if cls._shared_process.is_alive():
            try:
                if cls._shared_req_queue is not None:
                    cls._shared_req_queue.put(('shutdown',))
            except Exception:
                logger.debug('replace: shutdown put failed', exc_info=True)
            cls._shared_process.join(timeout=2)
            if cls._shared_process.is_alive():
                logger.warning(
                    'Worker did not exit on shutdown — killing for replacement'
                )
                cls._shared_process.kill()
                cls._shared_process.join(timeout=1)
        cls._shared_process = None
        cls._shared_req_queue = None
        cls._shared_out_queue = None
        cls._shared_cancel = None
        cls._shared_marked_for_replacement = False

    @classmethod
    def _ensure_process(cls) -> None:
        """Spawn the persistent worker on first build (or after death/RSS-replace)."""
        with cls._shared_lock:
            if cls._shared_marked_for_replacement:
                logger.info('Persistent worker: replacing due to RSS threshold')
                cls._replace_shared_process_locked()
            if cls._shared_process is not None and cls._shared_process.is_alive():
                return
            # Either first call, or previous worker died — start fresh.
            if cls._shared_process is not None:
                logger.warning(
                    'Persistent worker dead (exitcode=%s), respawning',
                    cls._shared_process.exitcode,
                )
            cls._shared_req_queue = mp.Queue()
            cls._shared_out_queue = mp.Queue()
            cls._shared_cancel = mp.Event()
            cls._shared_process = mp.Process(
                target=persistent_worker_main,
                args=(
                    cls._shared_req_queue,
                    cls._shared_out_queue,
                    cls._shared_cancel,
                ),
                daemon=True,
            )
            cls._shared_process.start()
            logger.info(
                'Persistent worker spawned (pid=%s)',
                cls._shared_process.pid,
            )

    @classmethod
    def _check_worker_rss_locked(cls) -> None:
        """Called from instance code; reads RSS, marks for replacement if too high."""
        if cls._shared_process is None or not cls._shared_process.is_alive():
            return
        try:
            import psutil
            proc = psutil.Process(cls._shared_process.pid)
            rss_mb = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            logger.debug('worker RSS check failed', exc_info=True)
            return
        if rss_mb > cls._max_rss_mb:
            logger.warning(
                'Worker RSS %.0fMB exceeds %dMB — will respawn next build',
                rss_mb, cls._max_rss_mb,
            )
            cls._shared_marked_for_replacement = True

    @classmethod
    def shutdown_shared_worker(cls, timeout_ms: int = 3000) -> None:
        """
        Tell the persistent worker to exit. Call on app shutdown.
        """
        with cls._shared_lock:
            if cls._shared_process is None or not cls._shared_process.is_alive():
                cls._shared_process = None
                cls._shared_req_queue = None
                cls._shared_out_queue = None
                cls._shared_cancel = None
                return
            try:
                if cls._shared_req_queue is not None:
                    cls._shared_req_queue.put(('shutdown',))
            except Exception:
                logger.debug('shutdown_shared_worker: put failed', exc_info=True)
            cls._shared_process.join(timeout=timeout_ms / 1000)
            if cls._shared_process.is_alive():
                logger.warning(
                    'Persistent worker did not exit on shutdown, killing'
                )
                cls._shared_process.kill()
                cls._shared_process.join(timeout=1)
            cls._shared_process = None
            cls._shared_req_queue = None
            cls._shared_out_queue = None
            cls._shared_cancel = None

    # ------------------------------------------------------------------
    # Instance API
    # ------------------------------------------------------------------

    def __init__(self, params: DownloadParams, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._params = params

        # Внутренняя лёгкая очередь: reader thread → GUI thread
        self._gui_queue: _queue_mod.Queue = _queue_mod.Queue()
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        # «Билд завершён» — reader выставляет, чтобы isRunning видела это
        # независимо от состояния воркер-процесса (он живёт всегда).
        self._build_finished = threading.Event()

        # QTimer для опроса _gui_queue (лёгкие сообщения, без десериализации)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_gui_queue)

    def start(self) -> None:
        """Send a build request to the persistent worker and start reader."""
        DownloadWorker._ensure_process()
        assert DownloadWorker._shared_req_queue is not None
        assert DownloadWorker._shared_out_queue is not None
        assert DownloadWorker._shared_cancel is not None

        DownloadWorker._shared_cancel.clear()
        self._reader_stop.clear()
        self._build_finished.clear()

        DownloadWorker._shared_req_queue.put(('build', self._params))

        # Reader thread consumes the shared out queue while THIS build is
        # active. stop_and_join sets _reader_stop so it exits cleanly without
        # tearing down the worker process.
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(DownloadWorker._shared_out_queue,),
            daemon=True,
            name='download-reader',
        )
        self._reader_thread.start()
        self._poll_timer.start()

    def request_cancel(self) -> None:
        """Сигналим воркеру отмену текущего билда."""
        if DownloadWorker._shared_cancel is not None:
            DownloadWorker._shared_cancel.set()
        logger.info('DownloadWorker: cancel requested')

    def isRunning(self) -> bool:  # noqa: N802 — Qt-style name kept for compat
        """True пока конкретный билд не получил 'finished'."""
        return (
            DownloadWorker._shared_process is not None
            and DownloadWorker._shared_process.is_alive()
            and not self._build_finished.is_set()
        )

    def stop_and_join(self, timeout_ms: int = 5000) -> None:
        """
        Stop watching the worker for this build.

        Не убивает persistent worker — он остаётся для следующего билда.
        Если билд в процессе и его нужно реально отменить — вызывающий
        должен сначала вызвать request_cancel().
        """
        self._poll_timer.stop()
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout_ms / 1000)
            self._reader_thread = None

    # ------------------------------------------------------------------
    # Reader thread: mp.Queue → десериализация → _gui_queue
    # ------------------------------------------------------------------

    def _reader_loop(self, out_queue: mp.Queue) -> None:
        """Фоновый поток: читает result-очередь, десериализует, кладёт в _gui_queue."""
        while not self._reader_stop.is_set():
            try:
                msg = out_queue.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            except Exception:
                logger.debug('Reader: queue.get failed', exc_info=True)
                break

            kind = msg[0]

            if kind in ('progress', 'spinner', 'warning'):
                self._gui_queue.put(msg)

            elif kind == 'preview':
                try:
                    # Notify GUI to drop previous build's cache NOW, before we
                    # spend 0.5–3s deserialising the new build's bitmaps. The
                    # slot in main thread runs via queued connection — old
                    # cache is freed concurrently with our deserialisation
                    # here, so peak RAM = max(old, new) instead of old+new.
                    self.preview_starting.emit()

                    _, img_data, metadata, dem_data, rh_data = msg
                    shm_name, data_len, mode, size = img_data
                    image = deserialize_pil(shm_name, data_len, mode, size)
                    dem_grid = deserialize_dem_grid(dem_data)
                    rh_cache = deserialize_rh_cache(rh_data)
                    logger.info(
                        'Reader: preview deserialized (%dx%d %s)',
                        size[0],
                        size[1],
                        mode,
                    )
                    self._gui_queue.put(
                        ('preview_ready', image, metadata, dem_grid, rh_cache)
                    )
                except Exception:
                    logger.exception('Reader: failed to deserialize preview')

            elif kind == 'finished':
                self._gui_queue.put(msg)
                self._build_finished.set()
                return  # reader для этого билда завершён

    # ------------------------------------------------------------------
    # GUI thread: опрос _gui_queue (только лёгкие объекты)
    # ------------------------------------------------------------------

    def _poll_gui_queue(self) -> None:
        """Забрать готовые сообщения из _gui_queue и проэмитить сигналы."""
        while True:
            try:
                msg = self._gui_queue.get_nowait()
            except _queue_mod.Empty:
                break

            kind = msg[0]

            if kind == 'progress':
                _, done, total, label = msg
                self.progress_update.emit(done, total, label)

            elif kind == 'spinner':
                _, label = msg
                self.progress_update.emit(0, 0, label)

            elif kind == 'warning':
                _, text, field_updates = msg
                self.warning_received.emit(text, field_updates)

            elif kind == 'preview_ready':
                _, image, metadata, dem_grid, rh_cache = msg
                self.preview_ready.emit(image, metadata, dem_grid, rh_cache)

            elif kind == 'finished':
                _, success, error_msg = msg
                self._poll_timer.stop()
                # Watchdog: после каждого билда проверяем RSS воркера.
                # Если выше порога — _ensure_process на следующем build()
                # сделает graceful respawn.
                DownloadWorker._check_worker_rss_locked()
                self.finished.emit(success, error_msg)
                return  # больше не опрашиваем

        # Если persistent worker внезапно умер посреди билда — это crash.
        proc = DownloadWorker._shared_process
        if (
            proc is not None
            and not proc.is_alive()
            and not self._build_finished.is_set()
        ):
            # Дать reader дочитать оставшееся (если есть)
            if self._reader_thread is not None and self._reader_thread.is_alive():
                return
            self._poll_timer.stop()
            exitcode = proc.exitcode
            logger.error(
                'Persistent worker crashed mid-build (exit=%s)', exitcode
            )
            self._build_finished.set()
            self.finished.emit(
                False,
                f'Рабочий процесс завершился аварийно (код {exitcode})',
            )
            # Не убираем _shared_process здесь — _ensure_process переспавнит
            # на следующий start().
