"""Потоковая обработка больших изображений через memory-mapped файлы."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

from shared.constants import (
    JPEG_MAX_DIMENSION,
    ROTATE_ANGLE_EPSILON,
    STREAMING_CLEANUP_TEMP,
    STREAMING_STRIP_HEIGHT,
    STREAMING_TEMP_DIR,
    TIFF_JPEG_QUALITY,
    TIFF_TILE_SIZE,
    TIFF_USE_BIGTIFF,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class StreamingImage:
    """
    Обёртка над memory-mapped numpy массивом для потоковой работы с большими изображениями.

    Attributes:
        width: Ширина изображения в пикселях
        height: Высота изображения в пикселях
        mmap_path: Путь к временному mmap-файлу
        array: Memory-mapped numpy массив shape=(height, width, 3), dtype=uint8

    """

    def __init__(
        self,
        width: int,
        height: int,
        temp_dir: str = STREAMING_TEMP_DIR,
        fill_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        """
        Создаёт новое изображение с выделением mmap-файла.

        Args:
            width: Ширина изображения в пикселях
            height: Высота изображения в пикселях
            temp_dir: Директория для временных файлов
            fill_color: Цвет заливки (RGB)

        """
        self.width = width
        self.height = height
        self._temp_dir = temp_dir
        self._closed = False

        # Создаём директорию если не существует
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

        # Создаём уникальный файл для mmap
        self.mmap_path = os.path.join(temp_dir, f'streaming_{uuid.uuid4().hex}.raw')

        # Размер файла в байтах (RGB, uint8)
        file_size = width * height * 3

        # Создаём файл нужного размера эффективно через truncate
        with open(self.mmap_path, 'wb') as f:
            f.truncate(file_size)

        # Открываем как memory-mapped массив
        self._mmap = np.memmap(
            self.mmap_path,
            dtype=np.uint8,
            mode='r+',
            shape=(height, width, 3),
        )

        # Заполняем цветом заливки эффективно через numpy
        if fill_color != (0, 0, 0):
            self._mmap[:] = fill_color
            self._mmap.flush()

        logger.debug(
            'StreamingImage created: %dx%d, file=%s, size=%.1f MB',
            width,
            height,
            self.mmap_path,
            file_size / (1024 * 1024),
        )

    @property
    def array(self) -> np.memmap:
        """Возвращает memory-mapped массив."""
        if self._closed:
            msg = 'StreamingImage is closed'
            raise ValueError(msg)
        return self._mmap

    @property
    def size(self) -> tuple[int, int]:
        """Возвращает размер (width, height) как у PIL.Image."""
        return (self.width, self.height)

    def paste_tile(
        self,
        tile_data: bytes | np.ndarray,
        x: int,
        y: int,
        tile_w: int | None = None,
        tile_h: int | None = None,
    ) -> None:
        """
        Вставляет тайл в указанную позицию.

        Args:
            tile_data: Данные тайла (JPEG bytes или numpy array)
            x: X-координата левого верхнего угла
            y: Y-координата левого верхнего угла
            tile_w: Ширина тайла (если None, определяется из данных)
            tile_h: Высота тайла (если None, определяется из данных)

        """
        if self._closed:
            msg = 'StreamingImage is closed'
            raise ValueError(msg)

        # Декодируем если это bytes
        if isinstance(tile_data, bytes):
            tile_arr = np.array(Image.open(BytesIO(tile_data)).convert('RGB'))
        else:
            tile_arr = tile_data
            if tile_arr.ndim == 2:
                # Grayscale -> RGB
                tile_arr = np.stack([tile_arr] * 3, axis=-1)

        th, tw = tile_arr.shape[:2]
        if tile_w is not None and tile_h is not None:
            if (tw, th) != (tile_w, tile_h):
                # Resize если размер не совпадает
                tile_arr = cv2.resize(
                    tile_arr, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR
                )
                th, tw = tile_h, tile_w

        # Вычисляем пересечение с границами изображения
        src_x0 = max(0, -x)
        src_y0 = max(0, -y)
        src_x1 = min(tw, self.width - x)
        src_y1 = min(th, self.height - y)

        dst_x0 = max(0, x)
        dst_y0 = max(0, y)
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        if src_x1 > src_x0 and src_y1 > src_y0:
            self._mmap[dst_y0:dst_y1, dst_x0:dst_x1] = tile_arr[
                src_y0:src_y1, src_x0:src_x1
            ]
            # NOTE: Removed flush() here - it was causing major performance issues
            # (disk sync after every tile). flush() is now called only once at the end
            # or when explicitly needed via explicit_flush() method.

    def flush(self) -> None:
        """
        Явно сбрасывает данные на диск.

        Вызывайте после завершения записи всех тайлов.
        """
        if self._closed:
            return
        self._mmap.flush()

    def get_strip(self, y: int, height: int) -> np.ndarray:
        """
        Возвращает горизонтальную полосу изображения (копию).

        Args:
            y: Y-координата начала полосы
            height: Высота полосы

        Returns:
            Копия данных полосы как numpy array

        """
        if self._closed:
            msg = 'StreamingImage is closed'
            raise ValueError(msg)

        y_end = min(y + height, self.height)
        return np.array(self._mmap[y:y_end])

    def set_strip(self, y: int, data: np.ndarray) -> None:
        """
        Записывает горизонтальную полосу в изображение.

        Args:
            y: Y-координата начала полосы
            data: Данные полосы (numpy array)

        """
        if self._closed:
            msg = 'StreamingImage is closed'
            raise ValueError(msg)

        h = data.shape[0]
        y_end = min(y + h, self.height)
        actual_h = y_end - y
        self._mmap[y:y_end] = data[:actual_h]
        # NOTE: Removed flush() here - caller should call flush() after all strips
        # are written to avoid excessive disk syncs.

    def close(self) -> None:
        """Закрывает mmap и удаляет временный файл."""
        if self._closed:
            return

        self._closed = True

        # Закрываем mmap
        if hasattr(self, '_mmap') and self._mmap is not None:
            del self._mmap
            self._mmap = None

        # Удаляем временный файл
        if STREAMING_CLEANUP_TEMP and hasattr(self, 'mmap_path'):
            try:
                if os.path.exists(self.mmap_path):
                    os.remove(self.mmap_path)
                    logger.debug('Removed temp file: %s', self.mmap_path)
            except OSError as e:
                logger.warning('Failed to remove temp file %s: %s', self.mmap_path, e)

    def __enter__(self) -> 'StreamingImage':
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor."""
        self.close()


def assemble_tiles_streaming(
    tile_data_list: Sequence[bytes | np.ndarray],
    tiles_x: int,
    tiles_y: int,
    eff_tile_px: int,
    crop_rect: tuple[int, int, int, int],
    temp_dir: str = STREAMING_TEMP_DIR,
) -> StreamingImage:
    """
    Собирает тайлы в StreamingImage.

    Args:
        tile_data_list: Список данных тайлов (JPEG bytes или numpy arrays)
        tiles_x: Количество тайлов по горизонтали
        tiles_y: Количество тайлов по вертикали
        eff_tile_px: Размер тайла в пикселях
        crop_rect: (x, y, width, height) области обрезки
        temp_dir: Директория для временных файлов

    Returns:
        StreamingImage с собранным изображением

    """
    crop_x, crop_y, crop_w, crop_h = crop_rect

    # Создаём результирующее изображение
    result = StreamingImage(crop_w, crop_h, temp_dir=temp_dir)

    logger.info(
        'Assembling %d tiles (%dx%d) into %dx%d image, eff_tile_px=%d',
        len(tile_data_list),
        tiles_x,
        tiles_y,
        crop_w,
        crop_h,
        eff_tile_px,
    )

    idx = 0
    for j in range(tiles_y):
        for i in range(tiles_x):
            if idx >= len(tile_data_list):
                break

            tile_data = tile_data_list[idx]

            # Координаты тайла на полном холсте
            tile_x0 = i * eff_tile_px
            tile_y0 = j * eff_tile_px

            # Пересечение с crop_rect
            inter_x0 = max(tile_x0, crop_x)
            inter_y0 = max(tile_y0, crop_y)
            inter_x1 = min(tile_x0 + eff_tile_px, crop_x + crop_w)
            inter_y1 = min(tile_y0 + eff_tile_px, crop_y + crop_h)

            if inter_x0 < inter_x1 and inter_y0 < inter_y1:
                # Декодируем тайл
                if isinstance(tile_data, bytes):
                    tile_arr = np.array(Image.open(BytesIO(tile_data)).convert('RGB'))
                else:
                    tile_arr = tile_data

                # Resize если нужно
                th, tw = tile_arr.shape[:2]
                if (tw, th) != (eff_tile_px, eff_tile_px):
                    # DEBUG: логируем resize операцию
                    if idx == 0:
                        logger.warning(
                            'RESIZE DETECTED: tile actual size %dx%d, expected %dx%d - THIS CAUSES QUALITY LOSS!',
                            tw, th, eff_tile_px, eff_tile_px
                        )
                    tile_arr = cv2.resize(
                        tile_arr,
                        (eff_tile_px, eff_tile_px),
                        interpolation=cv2.INTER_LINEAR,
                    )
                elif idx == 0:
                    logger.info(
                        'Tile size OK: %dx%d matches expected %dx%d',
                        tw, th, eff_tile_px, eff_tile_px
                    )

                # Вырезаем нужную часть тайла
                src_x0 = inter_x0 - tile_x0
                src_y0 = inter_y0 - tile_y0
                src_x1 = inter_x1 - tile_x0
                src_y1 = inter_y1 - tile_y0

                tile_crop = tile_arr[src_y0:src_y1, src_x0:src_x1]

                # Координаты вставки в результат
                dst_x = inter_x0 - crop_x
                dst_y = inter_y0 - crop_y

                # Вставляем
                result.paste_tile(tile_crop, dst_x, dst_y)

                # Освобождаем память
                del tile_arr
                del tile_crop

            idx += 1

    logger.info('Assembly complete: %dx%d', crop_w, crop_h)
    return result


def rotate_streaming(
    src: StreamingImage,
    angle_deg: float,
    fill: tuple[int, int, int] = (255, 255, 255),
    temp_dir: str = STREAMING_TEMP_DIR,
) -> StreamingImage:
    """
    Поворачивает изображение с сохранением размера.

    Оптимизации:
    - Пропуск поворота для углов < ROTATE_ANGLE_EPSILON
    - Параллельная обработка полос через ThreadPoolExecutor
    - Увеличенная высота полосы для снижения накладных расходов

    Args:
        src: Исходное изображение
        angle_deg: Угол поворота в градусах (против часовой стрелки)
        fill: Цвет заливки для углов
        temp_dir: Директория для временных файлов

    Returns:
        Новый StreamingImage с повёрнутым изображением (src закрывается)

    """
    from concurrent.futures import ThreadPoolExecutor
    from shared.constants import (
        ROTATE_NUM_WORKERS,
        ROTATE_STRIP_HEIGHT,
    )

    # Если угол близок к нулю, возвращаем исходный образ без копирования
    if abs(angle_deg) < ROTATE_ANGLE_EPSILON:
        logger.debug('Rotation angle %.4f° < %.4f°, skipping rotation',
                     abs(angle_deg), ROTATE_ANGLE_EPSILON)
        return src

    w, h = src.width, src.height
    strip_h = ROTATE_STRIP_HEIGHT
    num_strips = (h + strip_h - 1) // strip_h
    # Всегда используем параллельный режим для эффективности
    use_parallel = num_strips > 1

    logger.info(
        'Rotating image %dx%d by %.2f° (%s mode, %d strips of %d px)',
        w, h, angle_deg,
        'parallel' if use_parallel else 'sequential',
        num_strips, strip_h,
    )

    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    inv_matrix = cv2.invertAffineTransform(rotation_matrix)

    # Создаём результат
    dst = StreamingImage(w, h, temp_dir=temp_dir)

    # Предварительно вычисляем X-координаты (одинаковы для всех полос)
    dst_xs = np.arange(w, dtype=np.float32)

    def process_strip(dst_y: int) -> tuple[int, np.ndarray]:
        """Обрабатывает одну полосу и возвращает (y, данные)."""
        dst_y_end = min(dst_y + strip_h, h)
        strip_height = dst_y_end - dst_y

        # Координатная сетка для полосы
        dst_ys = np.arange(dst_y, dst_y_end, dtype=np.float32)
        dst_xx, dst_yy = np.meshgrid(dst_xs, dst_ys)

        # Обратное преобразование
        src_xx = inv_matrix[0, 0] * dst_xx + inv_matrix[0, 1] * dst_yy + inv_matrix[0, 2]
        src_yy = inv_matrix[1, 0] * dst_xx + inv_matrix[1, 1] * dst_yy + inv_matrix[1, 2]

        # Диапазон исходных строк
        src_y_min = max(0, int(np.floor(src_yy.min())) - 1)
        src_y_max = min(h, int(np.ceil(src_yy.max())) + 2)

        if src_y_max <= src_y_min:
            return dst_y, np.full((strip_height, w, 3), fill, dtype=np.uint8)

        # Загружаем исходные данные
        src_strip = src.get_strip(src_y_min, src_y_max - src_y_min)
        src_yy_local = src_yy - src_y_min

        # Интерполяция (cv2.remap требует float32)
        strip_data = cv2.remap(
            src_strip,
            src_xx.astype(np.float32),
            src_yy_local.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )

        return dst_y, strip_data

    # Список позиций полос
    strip_positions = list(range(0, h, strip_h))

    # Flush каждые N полос чтобы не накапливать dirty pages в RAM
    bytes_per_strip = w * strip_h * 3
    flush_interval = max(1, 500_000_000 // bytes_per_strip)

    if use_parallel and num_strips > 1:
        # Параллельная обработка с ограничением памяти
        # Обрабатываем пакетами чтобы не держать все результаты в RAM
        from concurrent.futures import as_completed

        with ThreadPoolExecutor(max_workers=ROTATE_NUM_WORKERS) as executor:
            # Отправляем задачи пакетами
            batch_size = flush_interval * ROTATE_NUM_WORKERS
            for batch_start in range(0, len(strip_positions), batch_size):
                batch_end = min(batch_start + batch_size, len(strip_positions))
                batch = strip_positions[batch_start:batch_end]

                # Обрабатываем пакет
                futures = {executor.submit(process_strip, pos): pos for pos in batch}
                results = []
                for future in as_completed(futures):
                    results.append(future.result())

                # Записываем результаты пакета
                for dst_y, strip_data in sorted(results, key=lambda x: x[0]):
                    dst.set_strip(dst_y, strip_data)
                    del strip_data

                # Flush после каждого пакета
                dst.flush()
                logger.debug('Rotation progress: %d/%d strips (flushed)',
                           batch_end, num_strips)
    else:
        # Последовательная обработка
        for idx, dst_y in enumerate(strip_positions):
            _, strip_data = process_strip(dst_y)
            dst.set_strip(dst_y, strip_data)
            del strip_data

            # Периодический flush
            if (idx + 1) % flush_interval == 0:
                dst.flush()

    dst.flush()
    src.close()
    logger.info('Rotation complete')
    return dst


def crop_streaming(
    src: StreamingImage,
    out_w: int,
    out_h: int,
    temp_dir: str = STREAMING_TEMP_DIR,
) -> StreamingImage:
    """
    Обрезает изображение по центру до заданного размера.

    Args:
        src: Исходное изображение
        out_w: Целевая ширина
        out_h: Целевая высота
        temp_dir: Директория для временных файлов

    Returns:
        StreamingImage с обрезанным изображением.
        Если размеры совпадают — возвращается src без копирования.
        Иначе создаётся новый образ, src закрывается.

    """
    # Если размеры совпадают, возвращаем исходный образ без копирования
    if src.width == out_w and src.height == out_h:
        logger.debug('Crop size matches source, skipping crop')
        return src

    # Вычисляем смещения для центрирования
    left = (src.width - out_w) // 2
    top = (src.height - out_h) // 2

    # Создаём результат
    dst = StreamingImage(out_w, out_h, temp_dir=temp_dir)

    # Копируем полосами
    strip_h = STREAMING_STRIP_HEIGHT
    total_strips = (out_h + strip_h - 1) // strip_h
    # Flush каждые N полос чтобы не накапливать dirty pages в RAM
    # ~500 MB на flush (достаточно часто чтобы не забить память)
    bytes_per_strip = out_w * strip_h * 3
    flush_interval = max(1, 500_000_000 // bytes_per_strip)

    logger.info(
        'Cropping image from %dx%d to %dx%d (%d strips, flush every %d)',
        src.width, src.height, out_w, out_h, total_strips, flush_interval,
    )

    for strip_idx, dst_y in enumerate(range(0, out_h, strip_h)):
        dst_y_end = min(dst_y + strip_h, out_h)
        strip_height = dst_y_end - dst_y

        src_y = top + dst_y

        # Загружаем полосу из src
        src_strip = src.get_strip(src_y, strip_height)

        # Вырезаем по горизонтали
        dst_strip = src_strip[:, left : left + out_w]

        dst.set_strip(dst_y, dst_strip)
        del src_strip
        del dst_strip

        # Периодический flush чтобы не накапливать dirty pages
        if (strip_idx + 1) % flush_interval == 0:
            dst.flush()
            logger.debug('Crop progress: %d/%d strips', strip_idx + 1, total_strips)

    dst.flush()
    src.close()
    logger.info('Crop complete: %dx%d', out_w, out_h)
    return dst


def save_streaming_tiff(
    img: StreamingImage,
    output_path: str,
    compression: str = 'jpeg',
    quality: int = TIFF_JPEG_QUALITY,
    tile_size: int = TIFF_TILE_SIZE,
    bigtiff: bool = TIFF_USE_BIGTIFF,
) -> None:
    """
    Сохраняет StreamingImage в TIFF файл.

    Args:
        img: Изображение для сохранения
        output_path: Путь к выходному файлу
        compression: Тип сжатия ('jpeg', 'lzw', 'zstd', None)
        quality: Качество JPEG (1-100)
        tile_size: Размер тайла в TIFF
        bigtiff: Использовать BigTIFF формат

    """
    import tifffile

    total_pixels = img.width * img.height
    logger.info(
        'Saving TIFF: %s (%dx%d = %.1fM pixels, compression=%s, quality=%d)',
        output_path,
        img.width,
        img.height,
        total_pixels / 1_000_000,
        compression,
        quality,
    )

    # Подготавливаем параметры сжатия
    compress_kwargs = {}
    if compression == 'jpeg':
        compress_kwargs = {'compression': 'jpeg', 'compressionargs': {'level': quality}}
    elif compression == 'lzw':
        compress_kwargs = {'compression': 'lzw'}
    elif compression == 'zstd':
        compress_kwargs = {'compression': 'zstd'}
    elif compression is None:
        compress_kwargs = {'compression': None}
    else:
        compress_kwargs = {'compression': compression}

    with tifffile.TiffWriter(output_path, bigtiff=bigtiff) as tif:
        # mmap array можно передать напрямую — tifffile читает его по частям
        # Важно: массив должен быть C-contiguous, что гарантируется при создании
        data = img.array
        if not data.flags['C_CONTIGUOUS']:
            logger.warning('Array is not C-contiguous, making copy')
            data = np.ascontiguousarray(data)

        tif.write(
            data,
            photometric='rgb',
            tile=(tile_size, tile_size),
            **compress_kwargs,
        )

    logger.info('TIFF saved: %s', output_path)


def save_streaming_jpeg(
    img: StreamingImage,
    output_path: str,
    quality: int = 90,
) -> None:
    """
    Сохраняет StreamingImage в JPEG файл.

    Для изображений больше JPEG_MAX_DIMENSION автоматически сохраняет как TIFF.

    Args:
        img: Изображение для сохранения
        output_path: Путь к выходному файлу
        quality: Качество JPEG (1-100)

    """
    # Проверяем ограничения JPEG
    if img.width > JPEG_MAX_DIMENSION or img.height > JPEG_MAX_DIMENSION:
        logger.warning(
            'Image size %dx%d exceeds JPEG limit %d, saving as TIFF instead',
            img.width,
            img.height,
            JPEG_MAX_DIMENSION,
        )
        # Меняем расширение на .tif
        tiff_path = str(Path(output_path).with_suffix('.tif'))
        save_streaming_tiff(img, tiff_path, compression='jpeg', quality=quality)
        return

    logger.info(
        'Saving JPEG: %s (%dx%d, quality=%d)',
        output_path,
        img.width,
        img.height,
        quality,
    )

    # Для JPEG нужно загрузить всё изображение в память
    # Это ограничение формата JPEG
    strip_h = STREAMING_STRIP_HEIGHT
    full_data = np.empty((img.height, img.width, 3), dtype=np.uint8)

    for y in range(0, img.height, strip_h):
        strip = img.get_strip(y, strip_h)
        y_end = min(y + strip_h, img.height)
        full_data[y:y_end] = strip
        del strip

    # Сохраняем через PIL
    pil_img = Image.fromarray(full_data)
    pil_img.save(output_path, 'JPEG', quality=quality)
    pil_img.close()

    del full_data
    logger.info('JPEG saved: %s', output_path)


def save_streaming_image(
    img: StreamingImage,
    output_path: str,
    quality: int = 90,
) -> str:
    """
    Сохраняет StreamingImage в файл, определяя формат по расширению.

    Args:
        img: Изображение для сохранения
        output_path: Путь к выходному файлу
        quality: Качество сжатия (1-100)

    Returns:
        Фактический путь к сохранённому файлу (может отличаться для больших JPEG)

    """
    ext = Path(output_path).suffix.lower()

    if ext in ('.tif', '.tiff'):
        save_streaming_tiff(img, output_path, compression='jpeg', quality=quality)
        return output_path
    elif ext in ('.jpg', '.jpeg'):
        # Проверяем размер
        if img.width > JPEG_MAX_DIMENSION or img.height > JPEG_MAX_DIMENSION:
            tiff_path = str(Path(output_path).with_suffix('.tif'))
            save_streaming_tiff(img, tiff_path, compression='jpeg', quality=quality)
            return tiff_path
        save_streaming_jpeg(img, output_path, quality=quality)
        return output_path
    else:
        # По умолчанию сохраняем как TIFF
        logger.warning('Unknown extension %s, saving as TIFF', ext)
        tiff_path = str(Path(output_path).with_suffix('.tif'))
        save_streaming_tiff(img, tiff_path, compression='jpeg', quality=quality)
        return tiff_path
