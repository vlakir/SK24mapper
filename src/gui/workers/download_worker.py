"""Download worker — запускает создание карты в отдельном процессе.

Использует ``multiprocessing.Process`` вместо ``QThread``, чтобы обойти GIL
и не блокировать GUI-поток во время тяжёлых вычислений (поворот, наложение сетки и пр.).

``QTimer(50 ms)`` опрашивает ``mp.Queue``, десериализует PIL Image
и эмитит Qt-сигналы точно так же, как раньше делал ``QThread``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtCore import QObject, QTimer, Signal

from gui.workers._process_entry import deserialize_pil, deserialize_rh_cache

if TYPE_CHECKING:
    from domain.models import DownloadParams

logger = logging.getLogger(__name__)

# Интервал опроса очереди из дочернего процесса (мс)
_POLL_INTERVAL_MS = 50


class DownloadWorker(QObject):
    """Обёртка над ``mp.Process`` для запуска создания карты.

    Сигналы полностью совместимы с предыдущим QThread-вариантом,
    поэтому view.py подключается к ним без изменений.
    """

    finished = Signal(bool, str)  # success, error_message
    progress_update = Signal(int, int, str)  # done, total, label
    preview_ready = Signal(
        Image.Image, object, object, object,
    )  # PIL Image, MapMetadata, dem_grid, rh_cache

    def __init__(self, params: DownloadParams, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._params = params

        # IPC примитивы
        self._queue: mp.Queue = mp.Queue()
        self._cancel_event: mp.Event = mp.Event()
        self._process: mp.Process | None = None

        # QTimer для опроса очереди
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(_POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_queue)

    # ------------------------------------------------------------------
    # Public API (совместим со старым DownloadWorker)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Запуск дочернего процесса."""
        from gui.workers._process_entry import worker_process_main

        self._cancel_event.clear()
        self._process = mp.Process(
            target=worker_process_main,
            args=(self._params, self._queue, self._cancel_event),
            daemon=True,
        )
        self._process.start()
        self._poll_timer.start()
        logger.info('DownloadWorker: child process started (pid=%s)', self._process.pid)

    def request_cancel(self) -> None:
        """Запрос отмены операции."""
        self._cancel_event.set()
        logger.info('DownloadWorker: cancel requested')

    def isRunning(self) -> bool:  # noqa: N802 — сохраняем Qt naming convention
        """True, если дочерний процесс ещё работает."""
        return self._process is not None and self._process.is_alive()

    def stop_and_join(self, timeout_ms: int = 5000) -> None:
        """Остановить процесс и дождаться завершения.

        Вызывается из ``_cleanup_download_worker`` и ``closeEvent``.
        """
        self._poll_timer.stop()
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

    # ------------------------------------------------------------------
    # Опрос очереди из GUI-потока
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        """Забрать все сообщения из очереди и проэмитить соответствующие сигналы."""
        import queue as _queue_mod

        while True:
            try:
                msg = self._queue.get_nowait()
            except _queue_mod.Empty:
                break

            kind = msg[0]

            if kind == 'progress':
                _, done, total, label = msg
                self.progress_update.emit(done, total, label)

            elif kind == 'spinner':
                _, label = msg
                self.progress_update.emit(0, 0, label)

            elif kind == 'preview':
                try:
                    _, img_data, metadata, dem_grid, rh_data = msg
                    pil_bytes, mode, size = img_data
                    image = deserialize_pil(pil_bytes, mode, size)
                    rh_cache = deserialize_rh_cache(rh_data)
                    logger.info(
                        'DownloadWorker: preview received (%dx%d %s)', size[0], size[1], mode
                    )
                    self.preview_ready.emit(image, metadata, dem_grid, rh_cache)
                except Exception:
                    logger.exception('DownloadWorker: failed to process preview message')

            elif kind == 'finished':
                _, success, error_msg = msg
                self._poll_timer.stop()
                self.finished.emit(success, error_msg)
                return  # больше не опрашиваем

        # Если процесс умер без отправки 'finished' — это crash
        if self._process is not None and not self._process.is_alive():
            self._poll_timer.stop()
            exitcode = self._process.exitcode
            if exitcode and exitcode != 0:
                logger.error('DownloadWorker: child process crashed (exit=%s)', exitcode)
                self.finished.emit(False, f'Рабочий процесс завершился аварийно (код {exitcode})')
