"""
GUI-agnostic точка входа сервисного слоя для создания карт.

Эта функция одинаково работает из PySide6 (через multiprocessing),
из веб-фреймворка (через asyncio/WebSocket) и из тестов.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from service import download_satellite_rectangle
from shared.progress import (
    clear_cancel_event,
    set_cancel_event,
    set_preview_image_callback,
    set_progress_callback,
    set_spinner_callbacks,
    set_warning_callback,
)

if TYPE_CHECKING:
    from domain.models import DownloadParams
    from shared.progress import CancelToken, ProgressSink

logger = logging.getLogger(__name__)


def run_map_job(
    params: DownloadParams,
    sink: ProgressSink,
    cancel: CancelToken,
) -> None:
    """
    GUI-agnostic entry point. Вызывается из любого фронтенда.

    Прокидывает *sink* и *cancel* в глобальные callback-и _CbStore
    (обратная совместимость с MapDownloadService / LiveSpinner / ConsoleProgress),
    затем запускает ``download_satellite_rectangle`` через ``asyncio.run``.
    """
    # Прокидываем протоколы в глобальные callback-и для совместимости
    set_cancel_event(cancel)
    set_progress_callback(sink.on_progress)
    set_spinner_callbacks(sink.on_spinner, lambda _: None)
    set_preview_image_callback(sink.on_preview)
    set_warning_callback(getattr(sink, 'on_warning', None))

    try:
        asyncio.run(
            download_satellite_rectangle(
                center_x_sk42_gk=params.center_x,
                center_y_sk42_gk=params.center_y,
                width_m=params.width_m,
                height_m=params.height_m,
                api_key=params.api_key,
                output_path=params.output_path,
                settings=params.settings,
            )
        )
    finally:
        clear_cancel_event()
