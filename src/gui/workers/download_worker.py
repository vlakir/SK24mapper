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

from shared.progress import set_progress_callback

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

    def run(self) -> None:
        """Execute download in background thread."""
        logger.info('DownloadWorker thread started')
        log_thread_status('worker thread start')
        log_memory_usage('worker thread start')

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
                success = True
                self.finished.emit(success, '')
            else:
                logger.warning('DownloadWorker: download returned False')
                success = False
                self.finished.emit(success, 'Загрузка не удалась')

        except Exception as e:
            logger.exception('DownloadWorker thread failed')
            log_memory_usage('worker thread error')
            success = False
            self.finished.emit(success, str(e))
        finally:
            log_thread_status('worker thread end')
            log_memory_usage('worker thread end')
