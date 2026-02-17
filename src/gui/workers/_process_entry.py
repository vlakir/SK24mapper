"""Точка входа дочернего процесса для создания карты.

Запускается через ``multiprocessing.Process``.
Проксирует события прогресса/превью в ``mp.Queue`` для опроса из GUI-потока.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from domain.models import DownloadParams, MapMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Сериализация PIL Image для IPC
#
# Сырые tobytes() дают 500-700 MB для больших карт (13110×13110),
# что превышает лимит Windows named pipe (~2 GB с учётом pickle overhead).
# Решение: сжимаем JPEG (RGB) / PNG (RGBA) перед отправкой через Queue.
# ---------------------------------------------------------------------------

def _serialize_pil(img: Image.Image) -> tuple[bytes, str, tuple[int, int]]:
    """Сжать PIL Image для передачи через mp.Queue."""
    import io

    buf = io.BytesIO()
    original_mode = img.mode
    if img.mode in ('RGBA', 'LA', 'PA'):
        # PNG для изображений с альфа-каналом (lossless, быстрое сжатие)
        img.save(buf, format='PNG', compress_level=1)
    else:
        # JPEG для RGB/L — отличное сжатие, минимальные потери
        save_img = img.convert('RGB') if img.mode not in ('RGB', 'L') else img
        save_img.save(buf, format='JPEG', quality=95)
    return (buf.getvalue(), original_mode, img.size)


def deserialize_pil(data: bytes, mode: str, size: tuple[int, int]) -> Image.Image:
    """Восстановить PIL Image из сжатых данных IPC."""
    import io

    from PIL import Image as _Image  # noqa: N811

    # Снимаем лимит: данные сгенерированы нашим же кодом, не из внешних источников
    _Image.MAX_IMAGE_PIXELS = None
    img = _Image.open(io.BytesIO(data))
    img.load()  # принудительно читаем пиксели, пока BytesIO жив
    # Восстановить исходный mode (JPEG всегда загружается как RGB)
    if img.mode != mode:
        img = img.convert(mode)
    return img


def _serialize_rh_cache(rh_cache: dict | None) -> dict | None:
    """Сериализовать rh_cache, сжимая PIL Image."""
    if rh_cache is None:
        return None
    from PIL import Image as _Image  # noqa: N811

    result: dict = {}
    for key, value in rh_cache.items():
        if isinstance(value, _Image.Image):
            result[key] = ('__pil__', *_serialize_pil(value))
        else:
            result[key] = value
    return result


def deserialize_rh_cache(rh_cache: dict | None) -> dict | None:
    """Десериализовать rh_cache, восстанавливая PIL Image."""
    if rh_cache is None:
        return None
    result: dict = {}
    for key, value in rh_cache.items():
        if isinstance(value, tuple) and len(value) == 4 and value[0] == '__pil__':
            _, data, mode, size = value
            result[key] = deserialize_pil(data, mode, size)
        else:
            result[key] = value
    return result


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

    def on_preview(
        self,
        image: Image.Image,
        metadata: MapMetadata | None,
        dem_grid: object | None,
        rh_cache: dict | None,
    ) -> None:
        img_data = _serialize_pil(image)
        rh_data = _serialize_rh_cache(rh_cache)
        self._q.put(('preview', img_data, metadata, dem_grid, rh_data))


# ---------------------------------------------------------------------------
# Главная функция дочернего процесса
# ---------------------------------------------------------------------------

def worker_process_main(
    params: DownloadParams,
    queue: mp.Queue,
    cancel_event: mp.Event,
) -> None:
    """Запускается в дочернем процессе через ``mp.Process(target=...)``.

    Проксирует все события прогресса/превью в *queue*,
    а результат (success/error) — как финальное сообщение ``('finished', ...)``.
    """
    # Настроить sys.path — дочерний процесс не наследует PYTHONPATH автоматически
    src_dir = str(Path(__file__).resolve().parent.parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Настроить логирование в дочернем процессе — тот же формат и те же хендлеры что в main.py
    import os
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        _local = os.getenv('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
        _log_file = Path(_local) / 'SK42' / 'log' / 'mil_mapper.log'
        _log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(_log_file), encoding='utf-8'))
    except Exception:
        pass  # если не удалось — пишем только в stdout
    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=handlers)

    from shared.progress import CancelledError, EventCancelToken

    sink = QueueProgressSink(queue)
    cancel = EventCancelToken(cancel_event)

    try:
        from services.map_job import run_map_job

        run_map_job(params, sink, cancel)
        queue.put(('finished', True, ''))
    except CancelledError:
        queue.put(('finished', False, 'Операция отменена пользователем'))
    except Exception as e:
        logger.exception('Worker process failed')
        queue.put(('finished', False, str(e)))
