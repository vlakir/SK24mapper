"""Image composition utilities - tile assembly and cropping."""

import logging

from PIL import Image

from shared.progress import ConsoleProgress

logger = logging.getLogger(__name__)


def assemble_and_crop(
    images: list[Image.Image],
    tiles_x: int,
    tiles_y: int,
    eff_tile_px: int,
    crop_rect: tuple[int, int, int, int],
) -> Image.Image:
    """
    Склеивает тайлы напрямую в финальное изображение нужного размера.

    Оптимизировано: вместо создания полного холста и последующей обрезки,
    вставляем только пересекающиеся части тайлов напрямую в результат.
    Это экономит память и время для больших карт.
    """
    crop_x, crop_y, crop_w, crop_h = crop_rect

    paste_progress = ConsoleProgress(total=tiles_x * tiles_y, label='Склейка тайлов')
    # Создаём сразу финальное изображение нужного размера
    result = Image.new('RGB', (crop_w, crop_h))

    idx = 0
    for j in range(tiles_y):
        for i in range(tiles_x):
            img = images[idx]
            if img.size != (eff_tile_px, eff_tile_px):
                img = img.resize((eff_tile_px, eff_tile_px), Image.Resampling.LANCZOS)

            # Координаты тайла на полном холсте
            tile_x0 = i * eff_tile_px
            tile_y0 = j * eff_tile_px
            tile_x1 = tile_x0 + eff_tile_px
            tile_y1 = tile_y0 + eff_tile_px

            # Пересечение с crop_rect
            inter_x0 = max(tile_x0, crop_x)
            inter_y0 = max(tile_y0, crop_y)
            inter_x1 = min(tile_x1, crop_x + crop_w)
            inter_y1 = min(tile_y1, crop_y + crop_h)

            if inter_x0 < inter_x1 and inter_y0 < inter_y1:
                # Есть пересечение — вырезаем нужную часть тайла
                src_x0 = inter_x0 - tile_x0
                src_y0 = inter_y0 - tile_y0
                src_x1 = inter_x1 - tile_x0
                src_y1 = inter_y1 - tile_y0

                # Координаты вставки в результат
                dst_x = inter_x0 - crop_x
                dst_y = inter_y0 - crop_y

                # Вырезаем и вставляем
                tile_crop = img.crop((src_x0, src_y0, src_x1, src_y1))
                result.paste(tile_crop, (dst_x, dst_y))
                tile_crop.close()

            # Освобождаем память тайла
            try:
                if hasattr(images[idx], 'close'):
                    images[idx].close()
            except Exception as e:
                logger.debug(f'Failed to close tile image: {e}')
            if isinstance(images, list):
                images[idx] = None  # type: ignore[call-overload]

            paste_progress.step_sync(1)
            idx += 1

    paste_progress.close()
    return result
