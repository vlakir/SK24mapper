from __future__ import annotations

from pyproj import CRS

from shared.constants import (
    EPSG_SK42_GK_BASE,
    GK_FALSE_EASTING,
    GK_ZONE_CM_OFFSET_DEG,
    GK_ZONE_WIDTH_DEG,
    GK_ZONE_X_PREFIX_DIV,
    MAX_GK_ZONE,
    SK42_VALID_LAT_MAX,
    SK42_VALID_LAT_MIN,
    SK42_VALID_LON_MAX,
    SK42_VALID_LON_MIN,
)


def determine_zone(center_x_sk42_gk: float) -> int:
    """
    Determine GK zone number from X coordinate in SK-42 GK meters.

    Falls back to prefix calculation when X has prefixed zone digits.
    """
    zone = int(center_x_sk42_gk // GK_ZONE_X_PREFIX_DIV)
    if zone < 1 or zone > MAX_GK_ZONE:
        zone = max(
            1,
            min(
                MAX_GK_ZONE,
                int((center_x_sk42_gk - GK_FALSE_EASTING) // GK_ZONE_X_PREFIX_DIV) + 1,
            ),
        )
    return zone


def build_sk42_gk_crs(zone: int) -> CRS:
    """
    Build pyproj CRS for SK-42 Gauss-Krüger zone.

    Tries EPSG codes first; falls back to proj4 with Krassovsky ellipsoid.
    """
    try:
        return CRS.from_epsg(EPSG_SK42_GK_BASE + zone)
    except Exception:
        lon0 = zone * GK_ZONE_WIDTH_DEG - GK_ZONE_CM_OFFSET_DEG
        proj4 = (
            f'+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 '
            f'+x_0={GK_FALSE_EASTING} +y_0=0 +ellps=krass +units=m +no_defs +type=crs'
        )
        return CRS.from_proj4(proj4)


def validate_sk42_bounds(lng: float, lat: float) -> None:
    """Validate that SK-42 geographic coordinates are within expected bounds."""
    if not (
        SK42_VALID_LON_MIN <= lng <= SK42_VALID_LON_MAX
        and SK42_VALID_LAT_MIN <= lat <= SK42_VALID_LAT_MAX
    ):
        msg = (
            'Выбранная область вне зоны применимости СК-42. '
            'Карта не будет сформирована.'
        )
        raise SystemExit(msg)
