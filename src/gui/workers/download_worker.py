"""
Download worker — запускает создание карты в отдельном процессе.

Использует ``multiprocessing.Process`` вместо ``QThread``, чтобы обойти GIL
и не блокировать GUI-поток во время тяжёлых вычислений (поворот, наложение сетки и пр.).

Фоновый ``threading.Thread`` читает ``mp.Queue`` и десериализует тяжёлые
данные (preview, dem_grid, rh_cache), а лёгкие сигналы пересылаются
в GUI-поток через ``QTimer(50 ms)`` опрос ``queue.Queue``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue_mod
import threading
from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtCore import QObject, QTimer, Signal

from gui.workers._process_entry import (
    deserialize_dem_grid,
    deserialize_pil,
    deserialize_rh_cache,
    worker_process_main,
)

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as _MpEvent

    from domain.models import DownloadParams

logger = logging.getLogger(__name__)

# Интервал опроса внутренней очереди из GUI-потока (мс)
_POLL_INTERVAL_MS = 50


class DownloadWorker(QObject):
    """
    Обёртка над ``mp.Process`` для запуска создания карты.

    Сигналы полностью совместимы с предыдущим QThread-вариантом,
    поэтому view.py подключается к ним без изменений.
    """

    finished = Signal(bool, str)  # success, error_message
    progress_update = Signal(int, int, str)  # done, total, label
    warning_received = Signal(str, object)  # warning text, field_updates dict|None
    preview_ready = Signal(
        Image.Image,
        object,
        object,
        object,
    )  # PIL Image, MapMetadata, dem_grid, rh_cache

    def __init__(self, params: DownloadParams, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._params = params

        # IPC примитивы
        self._mp_queue: mp.Queue = mp.Queue()
        self._cancel_event: _MpEvent = mp.Event()
        self._process: mp.Process | None = None

        # Внутренняя лёгкая очередь: reader thread → GUI thread
        self._gui_queue: _queue_mod.Queue = _queue_mod.Queue()
        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()

        # QTimer для опроса _gui_queue (лёгкие сообщения, без десериализации)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_gui_queue)

    # ------------------------------------------------------------------
    # Public API (совместим со старым DownloadWorker)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Запуск дочернего процесса и reader-потока."""
        self._cancel_event.clear()
        self._reader_stop.clear()
        self._process = mp.Process(
            target=worker_process_main,
            args=(self._params, self._mp_queue, self._cancel_event),
            daemon=True,
        )
        self._process.start()

        # Фоновый поток: читает mp.Queue, десериализует тяжёлые данные,
        # кладёт готовые python-объекты в _gui_queue
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name='download-reader',
        )
        self._reader_thread.start()

        self._poll_timer.start()
        logger.info(
            'DownloadWorker: child process started (pid=%s)',
            self._process.pid,
        )

    def request_cancel(self) -> None:
        """Запрос отмены операции."""
        self._cancel_event.set()
        logger.info('DownloadWorker: cancel requested')

    def isRunning(self) -> bool:
        """True, если дочерний процесс ещё работает."""
        return self._process is not None and self._process.is_alive()

    def stop_and_join(self, timeout_ms: int = 5000) -> None:
        """
        Остановить процесс и дождаться завершения.

        Вызывается из ``_cleanup_download_worker`` и ``closeEvent``.
        """
        self._poll_timer.stop()
        self._reader_stop.set()

        if self._process is None:
            return
        if self._process.is_alive():
            # Сначала пробуем мягкую отмену
            self._cancel_event.set()
            self._process.join(timeout=timeout_ms / 1000)
            if self._process.is_alive():
                logger.warning(
                    'DownloadWorker: process did not terminate gracefully, killing'
                )
                self._process.kill()
                self._process.join(timeout=2)
        self._process = None

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
            self._reader_thread = None

    # ------------------------------------------------------------------
    # Reader thread: mp.Queue → десериализация → _gui_queue
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Фоновый поток: читает mp.Queue, десериализует, кладёт в _gui_queue."""
        while not self._reader_stop.is_set():
            try:
                # Блокирующее чтение с таймаутом, чтобы периодически
                # проверять _reader_stop
                msg = self._mp_queue.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            except Exception:
                logger.debug('Reader: queue.get failed', exc_info=True)
                break

            kind = msg[0]

            if kind in ('progress', 'spinner', 'warning'):
                # Лёгкие сообщения — пробрасываем как есть
                self._gui_queue.put(msg)

            elif kind == 'preview':
                # Тяжёлая десериализация — в этом потоке, НЕ в GUI
                try:
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
                return  # reader завершается

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
                self.finished.emit(success, error_msg)
                return  # больше не опрашиваем

        # Если процесс умер без отправки 'finished' — это crash
        if self._process is not None and not self._process.is_alive():
            # Дождаться, пока reader вычитает оставшиеся сообщения
            if self._reader_thread is not None and self._reader_thread.is_alive():
                return  # ещё читает
            self._poll_timer.stop()
            exitcode = self._process.exitcode
            if exitcode and exitcode != 0:
                logger.error(
                    'DownloadWorker: child process crashed (exit=%s)',
                    exitcode,
                )
                success = False
                self.finished.emit(
                    success,
                    f'Рабочий процесс завершился аварийно (код {exitcode})',
                )
