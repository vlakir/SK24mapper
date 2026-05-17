"""Image transformation utilities - rotation, cropping, masking."""

import contextlib
import gc
import logging
import time

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ROTATE_ANGLE_EPS = 1e-6
_CV2_DIM_LIMIT = 32767  # SHRT_MAX — лимит cv2.warpAffine

_PIL_MODE_CHANNELS = {'RGB': 3, 'RGBA': 4, 'L': 1, 'P': 1}


def _pil_to_numpy_low_peak(img: Image.Image, *, close_src: bool) -> np.ndarray:
    """
    Превратить PIL.Image в numpy-массив с меньшим пиком памяти, чем np.array.

    `np.array(img)` идёт через `img.__array_interface__` → `img.tobytes()` →
    внутри `b"".join(output)`. На больших изображениях это даёт пик ~3-4×
    от размера: оригинал PIL + промежуточный list of chunks + final bytes
    + numpy. На 16972×16972 RGB это ~3.4 ГБ — реально вылетали в RLIMIT_AS
    при z=16 retina SATELLITE.

    Здесь мы:
      1) делаем `img.tobytes()` (пик ~2× размера ВНУТРИ tobytes — этого не
         избежать без C API);
      2) если `close_src=True`, СРАЗУ закрываем PIL (-1× размера из пика);
      3) `np.frombuffer(...).reshape(...)` без копирования: numpy
         "владеет" буфером bytes через `frombuffer`;
      4) `gc.collect()` чтобы освободить промежуточные chunks-list внутри
         tobytes до следующей крупной аллокации.

    Net peak при close_src=True: img + tobytes-peak (~2× транзитно) →
    после возврата только numpy (1×). Без close_src: + ещё 1×.
    """
    mode = img.mode
    w, h = img.size
    channels = _PIL_MODE_CHANNELS.get(mode)
    raw = img.tobytes()
    if close_src:
        with contextlib.suppress(Exception):
            img.close()
        # drop tobytes's internal chunks list (kept alive in some PIL paths
        # until the next major alloc evicts it).
        gc.collect()
    if channels is None or channels > 1:
        # frombuffer + reshape — это view, нет копии; .reshape((-1, ch)) для
        # многоканальных. Если канал unknown, оставляем плоским — caller
        # сам разберётся (но это fallback-кейс).
        shape = (h, w, channels) if channels else (-1,)
    else:
        shape = (h, w)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)

# Maximise OpenCV's internal threading once per process. In subprocess
# workers default may be 1 depending on build/env.
_cv2_threads_configured = False


def _ensure_cv2_threads() -> None:
    global _cv2_threads_configured
    if _cv2_threads_configured:
        return
    try:
        cores = cv2.getNumberOfCPUs()
        cv2.setNumThreads(cores)
        logger.info('cv2 threads: %d (cores=%d)', cv2.getNumThreads(), cores)
    except Exception as e:
        logger.debug('cv2.setNumThreads failed: %s', e)
    _cv2_threads_configured = True


def rotate_keep_size(
    img: Image.Image,
    angle_deg: float,
    fill: tuple[int, ...] = (255, 255, 255),
) -> Image.Image:
    """
    Поворачивает изображение на заданный угол (против часовой стрелки).

    Сохраняет исходный размер (обрезая углы).
    Использует OpenCV для ускорения операции.

    Args:
        img: Исходное изображение.
        angle_deg: Угол поворота в градусах (положительный — против часовой).
        fill: Цвет заливки для углов (по умолчанию белый).

    Returns:
        Повёрнутое изображение того же размера.

    """
    if abs(angle_deg) < ROTATE_ANGLE_EPS:
        return img.copy()

    _ensure_cv2_threads()

    w, h = img.size

    # PIL fallback для изображений, превышающих лимит cv2.warpAffine
    if w >= _CV2_DIM_LIMIT or h >= _CV2_DIM_LIMIT:
        return img.rotate(angle_deg, resample=Image.Resampling.BICUBIC, fillcolor=fill)

    t0 = time.monotonic()
    arr = np.array(img)
    t_to_np = time.monotonic()
    center = (w / 2, h / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # OpenCV использует BGR, но fill передаётся как RGB — для borderValue
    # порядок не важен, так как мы конвертируем обратно в тот же формат
    rotated = cv2.warpAffine(
        arr,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    t_warp = time.monotonic()
    del arr  # free input array before allocating output PIL image

    out = Image.fromarray(rotated)
    t_to_pil = time.monotonic()

    logger.info(
        'rotate_keep_size %dx%d %s: PIL→np=%.3fs warpAffine=%.3fs np→PIL=%.3fs',
        w, h, img.mode,
        t_to_np - t0,
        t_warp - t_to_np,
        t_to_pil - t_warp,
    )

    return out


def center_crop(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Обрезает изображение по центру до заданного размера."""
    w, h = img.size
    left = (w - out_w) // 2
    top = (h - out_h) // 2
    return img.crop((left, top, left + out_w, top + out_h))


def rotate_then_center_crop(
    img: Image.Image,
    angle_deg: float,
    out_w: int,
    out_h: int,
    fill: tuple[int, ...] = (255, 255, 255),
    *,
    close_input: bool = False,
    precrop_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """
    Rotate + center-crop in ONE cv2.warpAffine call.

    The trick: cv2.warpAffine takes (dst_w, dst_h) output size independently
    of input. By offsetting the rotation matrix translation by ((out_w -
    src_w)/2, (out_h - src_h)/2), the output is centred-cropped to the
    target size in the same pass. Skips:
      - a separate np→PIL→np cycle between rotate and crop (~150 ms on
        8488²RGB because PIL Image.fromarray for RGB has 3-byte → 4-byte
        repack cost on the way back),
      - the intermediate full pre-crop-sized buffer (8488² instead of 8404²
        — ~5% memory saving).

    For src ≥ dst with rotation needed this is the fast path; falls back
    to plain center_crop when angle is below the threshold.

    Если caller готов передать ownership `img` — пусть передаст
    `close_input=True`. В этом случае мы закрываем PIL.Image сразу после
    извлечения raw-байтов (внутри _pil_to_numpy_low_peak), что снимает
    1× размер из пиковой памяти. На 16972×16972 RGB это ~864 МБ, и без
    этого RLIMIT_AS может сработать (см. SATELLITE z=16 retina).

    Pre-crop fusion: if `precrop_box` (xs, ys, xe, ye) is given, the
    function does the crop in numpy (one contiguous-copy memcpy) before
    warpAffine, instead of relying on the caller to call PIL.crop first.
    On SATELLITE z=16 retina (19159² source → 16972² pre-crop), avoiding
    the separate PIL.crop + tobytes round-trip saves ~400 ms — PIL.crop
    forces a materialised copy, then _pil_to_numpy_low_peak does another
    one; here both fold into a single numpy slice + ascontiguousarray.
    """
    if abs(angle_deg) < ROTATE_ANGLE_EPS:
        # No rotation needed — plain crop. PIL.crop is fine here.
        if precrop_box is not None:
            # Apply precrop first, then center-crop (same final extent).
            img = img.crop(precrop_box)
        return center_crop(img, out_w, out_h)

    _ensure_cv2_threads()

    src_w, src_h = img.size
    src_mode = img.mode
    # PIL fallback for images that exceed cv2.warpAffine limits.
    if src_w >= _CV2_DIM_LIMIT or src_h >= _CV2_DIM_LIMIT:
        if precrop_box is not None:
            img = img.crop(precrop_box)
        rotated = img.rotate(
            angle_deg, resample=Image.Resampling.BICUBIC, fillcolor=fill
        )
        if close_input:
            with contextlib.suppress(Exception):
                img.close()
        return center_crop(rotated, out_w, out_h)

    t0 = time.monotonic()
    arr = _pil_to_numpy_low_peak(img, close_src=close_input)
    t_to_np = time.monotonic()

    if precrop_box is not None:
        # Slice → contiguous copy. Combined with the close_src=True
        # tobytes above, this is the only data-movement on the source.
        xs, ys, xe, ye = precrop_box
        arr = np.ascontiguousarray(arr[ys:ye, xs:xe])
        src_w = xe - xs
        src_h = ye - ys
        t_to_np = time.monotonic()  # restart measurement after precrop

    rotation_matrix = cv2.getRotationMatrix2D((src_w / 2, src_h / 2), angle_deg, 1.0)
    # Translate so the rotated image is centred on the (out_w, out_h) canvas.
    rotation_matrix[0, 2] += (out_w - src_w) / 2.0
    rotation_matrix[1, 2] += (out_h - src_h) / 2.0

    rotated = cv2.warpAffine(
        arr,
        rotation_matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    t_warp = time.monotonic()
    del arr  # освободить ~864 МБ raw-байтов до Image.fromarray

    out = Image.fromarray(rotated)
    t_to_pil = time.monotonic()

    logger.info(
        'rotate_then_center_crop %dx%d %s → %dx%d (close_input=%s): '
        'PIL→np=%.3fs warpAffine=%.3fs np→PIL=%.3fs',
        src_w, src_h, src_mode, out_w, out_h, close_input,
        t_to_np - t0,
        t_warp - t_to_np,
        t_to_pil - t_warp,
    )
    return out


def center_crop_np(arr: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """
    Center-crop a numpy image array (H, W[, C]) to (out_h, out_w).

    Returns a view, not a copy — cheap. For (H, W, 3) uint8 images this
    is identical in result to center_crop applied to a PIL.Image.fromarray
    of the same data, but skips the eager memcpy that PIL.Image.crop does
    (~100-160ms on 8488×8488 RGB).
    """
    h, w = arr.shape[:2]
    left = (w - out_w) // 2
    top = (h - out_h) // 2
    return arr[top:top + out_h, left:left + out_w]


def rotate_keep_size_np(
    arr: np.ndarray,
    angle_deg: float,
    fill: tuple[int, ...] = (255, 255, 255),
) -> np.ndarray:
    """
    Numpy-resident rotation: same semantics as rotate_keep_size but without
    the PIL↔np round-trip. Returns a fresh ndarray of the same shape.

    For (H, W, 3) RGB uint8 at 8488² this skips ~0.29s of np→PIL conversion
    that rotate_keep_size pays at the end. Use when the caller is already
    working in numpy and will keep doing so for at least one more pass
    (rotate → center_crop_np → ... → np→PIL once at the end).
    """
    if abs(angle_deg) < ROTATE_ANGLE_EPS:
        return arr.copy()

    _ensure_cv2_threads()
    h, w = arr.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        arr,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )


def apply_white_mask(img: Image.Image, opacity: float) -> Image.Image:
    """
    Накладывает полупрозрачную белую маску на изображение.

    Args:
        img: Исходное изображение (RGB).
        opacity: Непрозрачность маски (0.0 — без маски, 1.0 — полностью белое).

    Returns:
        Изображение с наложенной маской.

    """
    if opacity <= 0:
        return img
    white = Image.new('RGB', img.size, (255, 255, 255))
    return Image.blend(img, white, opacity)
