"""
Link Profile processor — terrain profile between two points
with LOS and Fresnel zones.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from geo.topography import latlng_to_pixel_xy
from imaging import load_grid_font
from imaging.text import draw_text_with_outline
from services.processors.radio_horizon import _load_topo
from shared.constants import (
    EARTH_RADIUS_M,
    LINK_PROFILE_ANTENNA_COLOR,
    LINK_PROFILE_ANTENNA_WIDTH_PX,
    LINK_PROFILE_FONT_MAX_PX,
    LINK_PROFILE_FONT_MIN_PX,
    LINK_PROFILE_GRID_BORDER_PX,
    LINK_PROFILE_GRID_COLOR,
    LINK_PROFILE_GRID_WIDTH_PX,
    LINK_PROFILE_INSET_BG_COLOR,
    LINK_PROFILE_INSET_HEIGHT_RATIO,
    LINK_PROFILE_INSET_MARGIN_H,
    LINK_PROFILE_INSET_MARGIN_V,
    LINK_PROFILE_LINE_WIDTH_PX,
    LINK_PROFILE_LOS_LINE_COLOR,
    LINK_PROFILE_NUM_SAMPLES,
    LINK_PROFILE_POINT_A_COLOR,
    LINK_PROFILE_POINT_B_COLOR,
    LINK_PROFILE_REFRACTION_K,
    LINK_PROFILE_TERRAIN_FILL_COLOR,
    SPEED_OF_LIGHT_MPS,
    MapType,
)

if TYPE_CHECKING:
    from services.map_context import MapDownloadContext

logger = logging.getLogger(__name__)

# Minimum points for meaningful analysis/drawing
_MIN_POINTS_FOR_LINE = 2
_MIN_PLOT_DIMENSION_PX = 20

# Fresnel zone thresholds (ITU-R recommendation)
_FRESNEL_GOOD_PCT = 60

# Grid line count range for auto-step selection
_MIN_GRID_LINES = 2
_MAX_GRID_LINES = 6


def extract_terrain_profile(
    dem: np.ndarray,
    point_a_px: tuple[float, float],
    point_b_px: tuple[float, float],
    pixel_size_m: float,
    num_samples: int = LINK_PROFILE_NUM_SAMPLES,
) -> dict:
    """
    Extract terrain profile along line A→B from DEM.

    Args:
        dem: 2D numpy array of elevations (meters).
        point_a_px: (col, row) of point A in DEM pixel coordinates.
        point_b_px: (col, row) of point B in DEM pixel coordinates.
        pixel_size_m: Meters per pixel.
        num_samples: Number of sample points along the profile.

    Returns:
        dict with keys: distances_m, elevations_m, total_distance_m

    """
    from scipy.ndimage import map_coordinates

    col_a, row_a = point_a_px
    col_b, row_b = point_b_px

    # Generate sample points along the line
    t = np.linspace(0, 1, num_samples)
    rows = row_a + t * (row_b - row_a)
    cols = col_a + t * (col_b - col_a)

    # Clamp to DEM bounds
    h, w = dem.shape
    rows = np.clip(rows, 0, h - 1)
    cols = np.clip(cols, 0, w - 1)

    # Bilinear interpolation
    elevations = map_coordinates(dem, [rows, cols], order=1, mode='nearest')

    # Compute distances
    dx = (col_b - col_a) * pixel_size_m
    dy = (row_b - row_a) * pixel_size_m
    total_distance_m = math.sqrt(dx * dx + dy * dy)
    distances_m = t * total_distance_m

    return {
        'distances_m': distances_m,
        'elevations_m': elevations,
        'total_distance_m': total_distance_m,
    }


def compute_link_analysis(
    profile: dict,
    antenna_a_m: float,
    antenna_b_m: float,
    freq_mhz: float,
    k: float = LINK_PROFILE_REFRACTION_K,
) -> dict:
    """
    Compute LOS, Earth curvature correction, and Fresnel zone analysis.

    Args:
        profile: Output of extract_terrain_profile.
        antenna_a_m: Antenna height at point A (meters above ground).
        antenna_b_m: Antenna height at point B (meters above ground).
        freq_mhz: Frequency in MHz.
        k: Atmospheric refraction coefficient (4/3 standard).

    Returns:
        dict with: los_heights_m, earth_correction_m, fresnel_radius_m,
                   clearance_m, has_obstruction, worst_clearance_m,
                   fresnel_clearance_pct, antenna_a_m, antenna_b_m

    """
    distances = profile['distances_m']
    elevations = profile['elevations_m']
    total_d = profile['total_distance_m']

    n = len(distances)
    elev_a = elevations[0]
    elev_b = elevations[-1]

    # Antenna tip heights (above sea level)
    h_a = elev_a + antenna_a_m
    h_b = elev_b + antenna_b_m

    # LOS line (straight line from A antenna tip to B antenna tip)
    if total_d > 0:
        los_heights = h_a + (h_b - h_a) * (distances / total_d)
    else:
        los_heights = np.full(n, h_a)

    # Earth curvature correction: d1*d2 / (2*K*R_earth)
    d1 = distances
    d2 = total_d - distances
    earth_correction = (d1 * d2) / (2.0 * k * EARTH_RADIUS_M)

    # Effective terrain height = ground elevation + earth curvature
    effective_terrain = elevations + earth_correction

    # Clearance = LOS height - effective terrain
    clearance = los_heights - effective_terrain

    # Fresnel zone radius: r = sqrt(wavelength * d1 * d2 / D)
    wavelength_m = SPEED_OF_LIGHT_MPS / (freq_mhz * 1e6)
    with np.errstate(invalid='ignore'):
        fresnel_radius = np.sqrt(wavelength_m * d1 * d2 / max(total_d, 1e-6))
    fresnel_radius = np.nan_to_num(fresnel_radius, nan=0.0)

    # Worst clearance (minimum clearance along the path, excluding endpoints)
    if n > _MIN_POINTS_FOR_LINE:
        inner_clearance = clearance[1:-1]
        worst_idx = np.argmin(inner_clearance)
        worst_clearance = float(inner_clearance[worst_idx])
        worst_fresnel = float(fresnel_radius[1:-1][worst_idx])
    else:
        worst_clearance = float(clearance[0]) if n > 0 else 0.0
        worst_fresnel = 0.0

    has_obstruction = worst_clearance < 0

    # Fresnel clearance percentage (worst clearance / Fresnel radius at that point)
    if worst_fresnel > 0:
        fresnel_clearance_pct = (worst_clearance / worst_fresnel) * 100.0
    else:
        fresnel_clearance_pct = 100.0

    return {
        'los_heights_m': los_heights,
        'earth_correction_m': earth_correction,
        'fresnel_radius_m': fresnel_radius,
        'clearance_m': clearance,
        'has_obstruction': has_obstruction,
        'worst_clearance_m': worst_clearance,
        'fresnel_clearance_pct': fresnel_clearance_pct,
        'antenna_a_m': antenna_a_m,
        'antenna_b_m': antenna_b_m,
    }


def render_profile_inset(
    link_data: dict,
    map_size: tuple[int, int],
) -> Image.Image:
    """
    Render profile inset diagram as RGBA image.

    Args:
        link_data: Combined dict with profile + analysis results.
        map_size: (width, height) of the final map image.

    Returns:
        RGBA PIL Image of the inset.

    """
    map_w, map_h = map_size
    inset_w = map_w
    inset_h = int(map_h * LINK_PROFILE_INSET_HEIGHT_RATIO)
    inset_h = max(inset_h, 120)  # minimum height

    inset = Image.new('RGBA', (inset_w, inset_h), LINK_PROFILE_INSET_BG_COLOR)
    draw = ImageDraw.Draw(inset)

    # Top border line (separation from map)
    draw.line([(0, 0), (inset_w - 1, 0)], fill=(100, 100, 100, 255), width=2)

    # Font — размер пропорционален высоте inset, ограничен константами
    font_size_px = int(inset_h * 0.07)
    font_size_px = max(
        LINK_PROFILE_FONT_MIN_PX, min(font_size_px, LINK_PROFILE_FONT_MAX_PX)
    )
    small_font_size_px = max(LINK_PROFILE_FONT_MIN_PX, font_size_px * 2 // 3)
    try:
        font = load_grid_font(font_size_px)
        small_font = load_grid_font(small_font_size_px)
    except Exception:
        from PIL import ImageFont

        font = ImageFont.load_default()
        small_font = font

    # Extract data
    distances = link_data['distances_m']
    elevations = link_data['elevations_m']
    los_heights = link_data['los_heights_m']
    fresnel_radius = link_data['fresnel_radius_m']
    earth_correction = link_data['earth_correction_m']
    total_d = link_data['total_distance_m']
    point_a_name = link_data.get('point_a_name', 'A')
    point_b_name = link_data.get('point_b_name', 'B')
    freq_mhz = link_data.get('freq_mhz', 900.0)
    antenna_a_m = link_data.get('antenna_a_m', 10.0)
    antenna_b_m = link_data.get('antenna_b_m', 10.0)
    worst_clearance = link_data.get('worst_clearance_m', 0.0)
    fresnel_pct = link_data.get('fresnel_clearance_pct', 100.0)

    # Gaps between axes and labels (proportional to font size)
    axis_label_gap = max(4, small_font_size_px // 2)
    tick_len = max(3, small_font_size_px // 3)

    # Elevation range (include LOS + Fresnel) — needed before margins
    effective_terrain = elevations + earth_correction
    all_heights = np.concatenate(
        [
            effective_terrain,
            los_heights + fresnel_radius,
            los_heights - fresnel_radius,
        ]
    )
    raw_min = float(np.nanmin(all_heights))
    raw_max = float(np.nanmax(all_heights))
    raw_range = max(raw_max - raw_min, 10.0)

    # Выбираем шаг оси, кратный 10 м, чтобы было 3–6 делений
    step = 10.0
    for candidate in (10, 20, 50, 100, 200, 500, 1000):
        n_lines = raw_range / candidate
        if _MIN_GRID_LINES <= n_lines <= _MAX_GRID_LINES:
            step = float(candidate)
            break

    # Округляем границы до шага
    e_min = math.floor((raw_min - raw_range * 0.05) / step) * step
    e_max = math.ceil((raw_max + raw_range * 0.1) / step) * step
    if e_max <= e_min:
        e_max = e_min + step

    # --- Adaptive margins: measure actual label sizes ---
    # Left margin: widest elevation label + gap
    max_elev_label_w = 0
    elev_val = e_min
    while elev_val <= e_max:
        bbox = small_font.getbbox(f'{int(elev_val)}')
        lw = bbox[2] - bbox[0] if bbox else 0
        max_elev_label_w = max(max_elev_label_w, lw)
        elev_val += step
    margin_left = max(
        int(inset_w * LINK_PROFILE_INSET_MARGIN_H),
        max_elev_label_w + axis_label_gap * 2,
    )

    # Right margin: unit label width + gap
    margin_right = max(
        int(inset_w * LINK_PROFILE_INSET_MARGIN_H),
        small_font_size_px * 2,
    )

    # Top margin: small padding (no title)
    margin_top = max(
        int(inset_h * 0.06),
        axis_label_gap * 2,
    )

    # Bottom margin: tick + gap + distance labels + gap + point names + gap + status
    bottom_content = (
        tick_len
        + axis_label_gap  # tick marks + gap
        + small_font_size_px  # distance labels
        + axis_label_gap  # gap
        + small_font_size_px  # point name labels
        + axis_label_gap  # gap
        + small_font_size_px  # status line
        + 6  # bottom padding
    )
    margin_bottom = max(
        int(inset_h * LINK_PROFILE_INSET_MARGIN_V),
        bottom_content,
    )

    plot_x0 = margin_left
    plot_x1 = inset_w - margin_right
    plot_y0 = margin_top
    plot_y1 = inset_h - margin_bottom
    plot_w = plot_x1 - plot_x0
    plot_h = plot_y1 - plot_y0

    if plot_w < _MIN_PLOT_DIMENSION_PX or plot_h < _MIN_PLOT_DIMENSION_PX:
        return inset

    def to_screen(d_m: float, elev_m: float) -> tuple[int, int]:
        """Convert (distance, elevation) to screen (x, y)."""
        x = plot_x0 + int(d_m / total_d * plot_w) if total_d > 0 else plot_x0
        y = plot_y1 - int(((elev_m - e_min) / (e_max - e_min)) * plot_h)
        return x, y

    # --- Plot area border ---
    draw.rectangle(
        [(plot_x0, plot_y0), (plot_x1, plot_y1)],
        outline=LINK_PROFILE_GRID_COLOR,
        width=LINK_PROFILE_GRID_BORDER_PX,
    )

    # --- Draw grid lines ---
    # Horizontal grid (elevation)
    elev_val = e_min
    while elev_val <= e_max:
        _, gy = to_screen(0, elev_val)
        draw.line(
            [(plot_x0, gy), (plot_x1, gy)],
            fill=LINK_PROFILE_GRID_COLOR,
            width=LINK_PROFILE_GRID_WIDTH_PX,
        )
        draw_text_with_outline(
            draw,
            (plot_x0 - axis_label_gap, gy),
            f'{int(elev_val)}',
            font=small_font,
            fill=(80, 80, 80),
            outline=(255, 255, 255),
            outline_width=1,
            anchor='rm',
        )
        elev_val += step

    # Vertical grid (distance) — same step as X axis labels
    dist_km = total_d / 1000.0
    if dist_km > 1:
        grid_step_m = max(1, int(dist_km / 6)) * 1000
    else:
        grid_step_m = max(100, int(total_d / 5 / 100) * 100)
    d_val_grid = grid_step_m
    while d_val_grid < total_d:
        gx, _ = to_screen(d_val_grid, e_min)
        draw.line(
            [(gx, plot_y0), (gx, plot_y1)],
            fill=LINK_PROFILE_GRID_COLOR,
            width=LINK_PROFILE_GRID_WIDTH_PX,
        )
        d_val_grid += grid_step_m

    # --- Draw terrain fill ---
    terrain_points = []
    for i in range(len(distances)):
        x, y = to_screen(distances[i], effective_terrain[i])
        terrain_points.append((x, y))

    # Close polygon at bottom
    if terrain_points:
        bottom_right = (terrain_points[-1][0], plot_y1)
        bottom_left = (terrain_points[0][0], plot_y1)
        terrain_polygon = [*terrain_points, bottom_right, bottom_left]
        draw.polygon(terrain_polygon, fill=LINK_PROFILE_TERRAIN_FILL_COLOR)

        # Terrain outline
        if len(terrain_points) >= _MIN_POINTS_FOR_LINE:
            draw.line(terrain_points, fill=(100, 80, 60, 255), width=2)

    # --- Fresnel zone color based on clearance percentage ---
    # >=60% — green (reliable link, ITU-R recommendation)
    # 0..60% — yellow (partial obstruction, degraded but possible)
    # <0% — red (LOS blocked by terrain)
    fresnel_pct_display = max(0.0, fresnel_pct)
    if fresnel_pct >= _FRESNEL_GOOD_PCT:
        fresnel_color = (0, 160, 0, 255)
    elif fresnel_pct >= 0:
        fresnel_color = (200, 180, 0, 255)
    else:
        fresnel_color = (200, 0, 0, 255)

    # --- Draw Fresnel zone ---
    fresnel_upper = []
    fresnel_lower = []
    for i in range(len(distances)):
        x_los, y_los = to_screen(distances[i], los_heights[i])
        r_px = int(((fresnel_radius[i]) / (e_max - e_min)) * plot_h)
        fresnel_upper.append((x_los, y_los - r_px))
        fresnel_lower.append((x_los, y_los + r_px))

    if fresnel_upper and fresnel_lower and len(fresnel_upper) >= _MIN_POINTS_FOR_LINE:
        draw.line(fresnel_upper, fill=fresnel_color, width=LINK_PROFILE_LINE_WIDTH_PX)
        draw.line(fresnel_lower, fill=fresnel_color, width=LINK_PROFILE_LINE_WIDTH_PX)

        # Frequency + Fresnel % label near the middle of the upper Fresnel arc
        mid_idx = len(fresnel_upper) // 2
        fu_x, fu_y = fresnel_upper[mid_idx]
        draw_text_with_outline(
            draw,
            (fu_x, fu_y - axis_label_gap),
            f'{freq_mhz:.0f} МГц',
            font=small_font,
            fill=fresnel_color[:3],
            outline=(255, 255, 255),
            outline_width=1,
            anchor='mb',
        )

    # --- Draw LOS line ---
    los_points = [
        to_screen(distances[i], los_heights[i]) for i in range(len(distances))
    ]
    if len(los_points) >= _MIN_POINTS_FOR_LINE:
        draw.line(
            los_points,
            fill=LINK_PROFILE_LOS_LINE_COLOR,
            width=LINK_PROFILE_LINE_WIDTH_PX,
        )

    # --- Draw antenna masts (black vertical lines at terrain→antenna top) ---
    antenna_color = LINK_PROFILE_ANTENNA_COLOR
    # Point A: from terrain surface to antenna top
    xa_base_x, xa_base_y = to_screen(0, effective_terrain[0])
    xa_top_x, xa_top_y = to_screen(0, effective_terrain[0] + antenna_a_m)
    draw.line(
        [(xa_base_x, xa_base_y), (xa_top_x, xa_top_y)],
        fill=antenna_color,
        width=LINK_PROFILE_ANTENNA_WIDTH_PX,
    )
    # Point B: from terrain surface to antenna top
    xb_base_x, xb_base_y = to_screen(total_d, effective_terrain[-1])
    xb_top_x, xb_top_y = to_screen(total_d, effective_terrain[-1] + antenna_b_m)
    draw.line(
        [(xb_base_x, xb_base_y), (xb_top_x, xb_top_y)],
        fill=antenna_color,
        width=LINK_PROFILE_ANTENNA_WIDTH_PX,
    )

    # --- Point name labels below the plot (под подписями оси расстояния) ---
    # Точка A — привязка к левому краю, не выходить за plot_x0
    # Точка B — привязка к правому краю, не выходить за plot_x1
    point_label_y = (
        plot_y1 + tick_len + axis_label_gap + small_font_size_px + axis_label_gap
    )
    draw_text_with_outline(
        draw,
        (max(xa_base_x, plot_x0), point_label_y),
        point_a_name,
        font=small_font,
        fill=LINK_PROFILE_POINT_A_COLOR,
        outline=(255, 255, 255),
        outline_width=1,
        anchor='lt',
    )
    draw_text_with_outline(
        draw,
        (min(xb_base_x, plot_x1), point_label_y),
        point_b_name,
        font=small_font,
        fill=LINK_PROFILE_POINT_B_COLOR,
        outline=(255, 255, 255),
        outline_width=1,
        anchor='rt',
    )

    # --- X axis labels (distance) ---
    if dist_km > 1:
        step_km = max(1, int(dist_km / 6))
        d_val = step_km
        while d_val <= dist_km:
            gx, _ = to_screen(d_val * 1000, e_min)
            draw.line(
                [(gx, plot_y1), (gx, plot_y1 + tick_len)],
                fill=(100, 100, 100, 200),
                width=1,
            )
            draw_text_with_outline(
                draw,
                (gx, plot_y1 + tick_len + axis_label_gap),
                f'{d_val}',
                font=small_font,
                fill=(80, 80, 80),
                outline=(255, 255, 255),
                outline_width=1,
                anchor='mt',
            )
            d_val += step_km
    else:
        # Sub-kilometer: use meters
        step_m = max(100, int(total_d / 5 / 100) * 100)
        d_val = step_m
        while d_val <= total_d:
            gx, _ = to_screen(d_val, e_min)
            draw.line(
                [(gx, plot_y1), (gx, plot_y1 + tick_len)],
                fill=(100, 100, 100, 200),
                width=1,
            )
            draw_text_with_outline(
                draw,
                (gx, plot_y1 + tick_len + axis_label_gap),
                f'{int(d_val)}м',
                font=small_font,
                fill=(80, 80, 80),
                outline=(255, 255, 255),
                outline_width=1,
                anchor='mt',
            )
            d_val += step_m

    # Distance unit label
    unit_label = 'км' if dist_km > 1 else 'м'
    draw_text_with_outline(
        draw,
        (plot_x1, plot_y1 + tick_len + axis_label_gap),
        unit_label,
        font=small_font,
        fill=(80, 80, 80),
        outline=(255, 255, 255),
        outline_width=1,
        anchor='lt',
    )

    # --- Status line ---
    status = (
        f'{total_d / 1000:.2f} км, просвет: {worst_clearance:.1f} м '
        f'(свободно {fresnel_pct_display:.0f}% зоны Френеля)'
    )
    status_color = fresnel_color[:3]

    # Bottom status line (centered, anchored to bottom of inset)
    draw_text_with_outline(
        draw,
        (inset_w // 2, inset_h - 6),
        status,
        font=small_font,
        fill=status_color,
        outline=(255, 255, 255),
        outline_width=1,
        anchor='mb',
    )

    return inset


def _wgs84_coords_from_gk(
    ctx: MapDownloadContext,
    x_gk: float,
    y_gk: float,
) -> tuple[float, float]:
    """Convert GK coordinates to WGS84 (lng, lat)."""
    lng_sk42, lat_sk42 = ctx.t_sk42_from_gk.transform(x_gk, y_gk)
    lng_wgs, lat_wgs = ctx.t_sk42_to_wgs.transform(lng_sk42, lat_sk42)
    return lng_wgs, lat_wgs


def _haversine_distance(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float,
) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def _extract_profile_from_tiles(
    ctx: MapDownloadContext,
    lat_a: float,
    lng_a: float,
    lat_b: float,
    lng_b: float,
    num_samples: int = LINK_PROFILE_NUM_SAMPLES,
) -> dict:
    """
    Extract terrain profile loading only DEM tiles along the A→B line.

    Instead of assembling the full-map DEM (~hundreds of MB), this loads
    only the tiles intersected by the A→B line (typically 5-20 tiles).
    Memory usage: a few MB instead of hundreds.
    """
    import asyncio

    from elevation.provider import ElevationTileProvider
    from infrastructure.http.client import resolve_cache_dir

    provider = ElevationTileProvider(
        client=ctx.client,
        api_key=ctx.api_key,
        use_retina=False,
        cache_root=resolve_cache_dir(),
    )

    zoom = ctx.zoom

    # Sample points along A→B in WGS84
    t = np.linspace(0, 1, num_samples)
    lats = lat_a + t * (lat_b - lat_a)
    lngs = lng_a + t * (lng_b - lng_a)

    # Determine which tile each sample falls in
    tile_groups: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    for i in range(num_samples):
        px_x, px_y = latlng_to_pixel_xy(lats[i], lngs[i], zoom)
        tx = int(px_x) // 256
        ty = int(px_y) // 256
        local_col = px_x - tx * 256
        local_row = px_y - ty * 256
        key = (tx, ty)
        if key not in tile_groups:
            tile_groups[key] = []
        tile_groups[key].append((local_col, local_row, i))

    logger.info(
        'Профиль радиолинии: загрузка %d DEM-тайлов (вместо %d для всей карты)',
        len(tile_groups),
        len(ctx.tiles),
    )

    # Load tiles and extract elevations via bilinear interpolation
    from scipy.ndimage import map_coordinates

    elevations = np.zeros(num_samples, dtype=np.float64)

    async def _process_tile(
        tile_key: tuple[int, int],
        samples: list[tuple[float, float, int]],
    ) -> list[tuple[int, float]]:
        tx, ty = tile_key
        async with ctx.semaphore:
            dem_tile = await provider.get_tile_dem(zoom, tx, ty)
        dem_arr = np.array(dem_tile, dtype=np.float32)
        h, w = dem_arr.shape
        rows = np.clip(np.array([s[1] for s in samples]), 0, h - 1)
        cols = np.clip(np.array([s[0] for s in samples]), 0, w - 1)
        tile_elevs = map_coordinates(dem_arr, [rows, cols], order=1, mode='nearest')
        return [(s[2], float(tile_elevs[j])) for j, s in enumerate(samples)]

    tasks = [_process_tile(k, v) for k, v in tile_groups.items()]
    results = await asyncio.gather(*tasks)
    for batch in results:
        for idx, elev in batch:
            elevations[idx] = elev

    # Distance via haversine (more accurate than pixel-based)
    total_distance_m = _haversine_distance(lat_a, lng_a, lat_b, lng_b)
    distances_m = t * total_distance_m

    return {
        'distances_m': distances_m,
        'elevations_m': elevations,
        'total_distance_m': total_distance_m,
    }


async def process_link_profile(ctx: MapDownloadContext) -> Image.Image:
    """
    Process link profile map — hybrid base + terrain profile
    with LOS/Fresnel analysis.
    """
    from shared.diagnostics import crash_log

    logger.info('Профиль радиолинии: старт')
    crash_log('process_link_profile: START')

    # 1. Load hybrid base
    crash_log('process_link_profile: _load_topo START (HYBRID)')
    base = await _load_topo(ctx, label='профиля радиолинии', style=MapType.HYBRID)
    crash_log(f'process_link_profile: _load_topo DONE, base={base.size}')

    # 2. Convert link point A and link point B to WGS84
    crash_log(
        f'process_link_profile: RAW settings: '
        f'A_x={ctx.settings.link_point_a_x}, A_y={ctx.settings.link_point_a_y}, '
        f'B_x={ctx.settings.link_point_b_x}, B_y={ctx.settings.link_point_b_y}'
    )
    crash_log(
        f'process_link_profile: GK coords: '
        f'A_x_gk={ctx.settings.link_point_a_x_sk42_gk:.0f}, '
        f'A_y_gk={ctx.settings.link_point_a_y_sk42_gk:.0f}, '
        f'B_x_gk={ctx.settings.link_point_b_x_sk42_gk:.0f}, '
        f'B_y_gk={ctx.settings.link_point_b_y_sk42_gk:.0f}'
    )
    lng_a_wgs, lat_a_wgs = _wgs84_coords_from_gk(
        ctx,
        ctx.settings.link_point_a_x_sk42_gk,
        ctx.settings.link_point_a_y_sk42_gk,
    )
    lng_b_wgs, lat_b_wgs = _wgs84_coords_from_gk(
        ctx,
        ctx.settings.link_point_b_x_sk42_gk,
        ctx.settings.link_point_b_y_sk42_gk,
    )
    crash_log(
        f'process_link_profile: A=({lat_a_wgs:.4f},{lng_a_wgs:.4f}) '
        f'B=({lat_b_wgs:.4f},{lng_b_wgs:.4f})'
    )

    # 3. Extract terrain profile — loads only DEM tiles along A→B (not full map!)
    crash_log('process_link_profile: _extract_profile START')
    profile = await _extract_profile_from_tiles(
        ctx,
        lat_a_wgs,
        lng_a_wgs,
        lat_b_wgs,
        lng_b_wgs,
    )
    crash_log(
        f'process_link_profile: _extract_profile DONE, '
        f'dist={profile["total_distance_m"]:.0f}m'
    )

    # 4. Compute LOS + Fresnel analysis
    analysis = compute_link_analysis(
        profile,
        antenna_a_m=ctx.settings.link_antenna_a_m,
        antenna_b_m=ctx.settings.link_antenna_b_m,
        freq_mhz=ctx.settings.link_freq_mhz,
        k=LINK_PROFILE_REFRACTION_K,
    )
    crash_log('process_link_profile: analysis DONE')

    # 5. Build link_data for postprocessing (inset rendering + markers)
    link_data = {**profile, **analysis}
    link_data['point_a_name'] = ctx.settings.link_point_a_name or 'A'
    link_data['point_b_name'] = ctx.settings.link_point_b_name or 'B'
    link_data['freq_mhz'] = ctx.settings.link_freq_mhz
    link_data['point_a_lat_wgs'] = lat_a_wgs
    link_data['point_a_lng_wgs'] = lng_a_wgs
    link_data['point_b_lat_wgs'] = lat_b_wgs
    link_data['point_b_lng_wgs'] = lng_b_wgs

    ctx.link_profile_data = link_data

    logger.info(
        'Профиль радиолинии: расстояние %.1f км, просвет %.1f м (%.0f%% Френеля)',
        profile['total_distance_m'] / 1000.0,
        analysis['worst_clearance_m'],
        analysis['fresnel_clearance_pct'],
    )

    return base
