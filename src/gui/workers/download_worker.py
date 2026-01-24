"""Download worker thread."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PIL import Image
from PySide6.QtCore import QThread, Signal

from shared.diagnostics import log_memory_usage, log_thread_status
from shared.progress import set_preview_image_callback, set_spinner_callbacks

if TYPE_CHECKING:
    from gui.controller import MilMapperController

logger = logging.getLogger(__name__)


class DownloadWorker(QThread):
    """Worker thread for map download operations."""

    finished = Signal(bool, str)  # success, error_message
    progress_update = Signal(int, int, str)  # done, total, label
    preview_ready = Signal(object)  # PIL Image object

    def __init__(self, controller: MilMapperController) -> None:
        super().__init__()
        self._controller = controller

    def run(self) -> None:
        """Execute download in background thread."""
        logger.info('DownloadWorker thread started')
        log_thread_status('worker thread start')
        log_memory_usage('worker thread start')

        try:
            # Setup thread-safe callbacks that emit signals instead of direct UI updates
            def preview_callback(img_obj: object) -> bool:
                """Handle preview image from map generation."""
                try:
                    if isinstance(img_obj, Image.Image):
                        self.preview_ready.emit(img_obj)
                        return True
                    return False
                except Exception as e:
                    logger.warning(f'Failed to process preview image: {e}')
                    return False

            # Setup progress system with thread-safe callbacks

            set_spinner_callbacks(
                lambda label: self.progress_update.emit(0, 0, label),
                lambda label: None,
            )
            set_preview_image_callback(preview_callback)

            # Import and set progress callback for ConsoleProgress updates
            from shared.progress import set_progress_callback

            set_progress_callback(
                lambda done, total, label: self.progress_update.emit(done, total, label)
            )

            # Run the actual download
            result = self._controller.download_map()

            log_memory_usage('worker thread after download')
            log_thread_status('worker thread after download')

            if result:
                logger.info('DownloadWorker completed successfully')
                self.finished.emit(True, '')
            else:
                logger.warning('DownloadWorker: download returned False')
                self.finished.emit(False, 'Загрузка не удалась')

        except Exception as e:
            logger.exception(f'DownloadWorker thread failed: {e}')
            log_memory_usage('worker thread error')
            self.finished.emit(False, str(e))
        finally:
            log_thread_status('worker thread end')
            log_memory_usage('worker thread end')
