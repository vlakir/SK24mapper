"""DEM colorization utilities for elevation maps."""

import numpy as np
from PIL import Image

from services.color_utils import ColorMapper


def colorize_dem_tile_numpy(
    dem_rgb: np.ndarray,
    lo_rounded: float,
    hi_rounded: float,
    color_mapper: ColorMapper,
) -> np.ndarray:
    """
    Colorize a DEM tile using numpy for performance.

    Args:
        dem_rgb: RGB array from terrain-rgb tile (HxWx3, uint8)
        lo_rounded: Lower elevation bound (meters)
        hi_rounded: Upper elevation bound (meters)
        color_mapper: ColorMapper instance with precomputed LUT

    Returns:
        Colorized RGB array (HxWx3, uint8)

    """
    inv = 1.0 / (hi_rounded - lo_rounded) if hi_rounded > lo_rounded else 1.0

    # Precompute linear coefficients for t = ar*R + ag*G + ab*B + a0
    ar = 0.1 * 65536.0 * inv
    ag = 0.1 * 256.0 * inv
    ab = 0.1 * 1.0 * inv
    a0 = (-10000.0 - lo_rounded) * inv

    # Compute normalized elevation t
    t = (
        ar * dem_rgb[..., 0].astype(np.float32)
        + ag * dem_rgb[..., 1].astype(np.float32)
        + ab * dem_rgb[..., 2].astype(np.float32)
        + a0
    )

    # Clamp and map to LUT
    lut = color_mapper.lut
    lut_size = len(lut)
    indices = np.clip(
        (t * (lut_size - 1)).astype(np.int32),
        0,
        lut_size - 1,
    )

    # Build output RGB from LUT
    lut_arr = np.array(lut, dtype=np.uint8)
    return lut_arr[indices]


def colorize_dem_overlap(
    dem_img: Image.Image,
    overlap: tuple[int, int, int, int],
    tile_base_x: int,
    tile_base_y: int,
    lo_rounded: float,
    hi_rounded: float,
    color_mapper: ColorMapper,
) -> tuple[np.ndarray, int, int]:
    """
    Colorize a portion of DEM tile defined by overlap rectangle.

    Args:
        dem_img: PIL Image of DEM tile (RGB terrain-rgb format)
        overlap: (x0, y0, x1, y1) in global pixel coordinates
        tile_base_x: X origin of tile in global pixels
        tile_base_y: Y origin of tile in global pixels
        lo_rounded: Lower elevation bound (meters)
        hi_rounded: Upper elevation bound (meters)
        color_mapper: ColorMapper instance

    Returns:
        Tuple of (colorized_array, dest_x, dest_y) where dest coords are
        relative to crop origin

    """
    x0, y0, x1, y1 = overlap

    # Extract overlap region from tile
    arr = np.asarray(dem_img, dtype=np.uint8)
    sub = arr[
        y0 - tile_base_y : y1 - tile_base_y,
        x0 - tile_base_x : x1 - tile_base_x,
        :3,
    ]

    # Colorize
    colorized = colorize_dem_tile_numpy(sub, lo_rounded, hi_rounded, color_mapper)

    return colorized, x0, y0
