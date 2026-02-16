"""Download worker thread."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtCore import QThread, Signal

from shared.diagnostics import log_memory_usage, log_thread_status
from shared.progress import (
    CancelledError,
    clear_cancel_event,
    set_cancel_event,
    set_preview_image_callback,
    set_progress_callback,
    set_spinner_callbacks,
)

if TYPE_CHECKING:
    from gui.controller import MilMapperController

logger = logging.getLogger(__name__)


class DownloadWorker(QThread):
    """Worker thread for map download operations."""

    finished = Signal(bool, str)  # success, error_message
    progress_update = Signal(int, int, str)  # done, total, label
    preview_ready = Signal(
        Image.Image, object, object, object
    )  # PIL Image, MapMetadata, dem_grid, rh_cache

    def __init__(self, controller: MilMapperController) -> None:
        super().__init__()
        self._controller = controller
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        """Request cancellation of the running operation."""
        self._cancel_event.set()

    def run(self) -> None:
        """Execute download in background thread."""
        logger.info('DownloadWorker thread started')
        log_thread_status('worker thread start')
        log_memory_usage('worker thread start')

        # Регистрируем cancel_event глобально
        self._cancel_event.clear()
        set_cancel_event(self._cancel_event)

        try:
            # Setup thread-safe callbacks that emit signals instead of direct UI updates
            def preview_callback(
                img_obj: Image.Image,
                metadata: object | None,
                dem_grid: object | None,
                rh_cache: dict | None,
            ) -> bool:
                """Handle preview image from map generation."""
                try:
                    if isinstance(img_obj, Image.Image):
                        self.preview_ready.emit(img_obj, metadata, dem_grid, rh_cache)
                        return True
                except Exception:
                    logger.warning('Failed to process preview image')
                    return False
                else:
                    return False

            # Setup progress system with thread-safe callbacks

            set_spinner_callbacks(
                lambda label: self.progress_update.emit(0, 0, label),
                lambda _: None,
            )
            set_preview_image_callback(preview_callback)

            set_progress_callback(
                lambda done, total, label: self.progress_update.emit(done, total, label)
            )

            # Run the actual download
            if self._controller.download_map():
                logger.info('DownloadWorker completed successfully')
                succeeded = True
                self.finished.emit(succeeded, '')
            else:
                logger.warning('DownloadWorker: download returned False')
                succeeded = False
                self.finished.emit(succeeded, 'Загрузка не удалась')

        except CancelledError:
            logger.info('DownloadWorker cancelled by user')
            succeeded = False
            self.finished.emit(succeeded, 'Операция отменена пользователем')
        except Exception as e:
            logger.exception('DownloadWorker thread failed')
            log_memory_usage('worker thread error')
            succeeded = False
            self.finished.emit(succeeded, str(e))
        finally:
            clear_cancel_event()
            log_thread_status('worker thread end')
            log_memory_usage('worker thread end')
