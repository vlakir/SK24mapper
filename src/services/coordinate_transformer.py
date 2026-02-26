"""Coordinate transformation utilities for SK-42 GK to WGS84 conversions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyproj import Transformer

from geo.coords_sk42 import build_sk42_gk_crs as _build_sk42_gk_crs
from geo.coords_sk42 import determine_zone as _determine_zone
from geo.coords_sk42 import validate_sk42_bounds as _validate_sk42_bounds
from geo.topography import (
    build_transformers_sk42,
    compute_rotation_deg_for_east_axis,
    crs_sk42_geog,
)

if TYPE_CHECKING:
    from pyproj import CRS

logger = logging.getLogger(__name__)


@dataclass
class CoordinateResult:
    """Result of coordinate transformation."""

    # SK-42 geographic coordinates
    center_lat_sk42: float
    center_lng_sk42: float

    # WGS84 coordinates
    center_lat_wgs: float
    center_lng_wgs: float

    # Rotation angle for grid alignment
    rotation_deg: float

    # CRS and transformers
    crs_sk42_gk: CRS
    t_sk42_from_gk: Transformer
    t_gk_from_sk42: Transformer
    t_sk42_to_wgs: Transformer
    t_wgs_to_sk42: Transformer

    # Zone info
    zone: int


class CoordinateTransformer:
    """Handles coordinate transformations between SK-42 GK and WGS84."""

    def __init__(
        self,
        center_x_gk: float,
        center_y_gk: float,
        helmert_params: tuple[float, ...] | None = None,
    ):
        """
        Initialize transformer with GK coordinates.

        Args:
            center_x_gk: Center X coordinate in SK-42 Gauss-Kruger (easting)
            center_y_gk: Center Y coordinate in SK-42 Gauss-Kruger (northing)
            helmert_params: Optional 7-parameter Helmert transformation tuple
                           (dx, dy, dz, rx, ry, rz, ds)

        """
        self.center_x_gk = center_x_gk
        self.center_y_gk = center_y_gk
        self.helmert_params = helmert_params

        # Determine zone and build CRS
        self.zone = _determine_zone(center_x_gk)
        self.crs_sk42_gk = _build_sk42_gk_crs(self.zone)

        # Build transformers
        self._build_transformers()

        # Compute coordinates
        self._compute_coordinates()

    def _build_transformers(self) -> None:
        """Build all required coordinate transformers."""
        # GK <-> SK-42 geographic
        self.t_sk42_from_gk = Transformer.from_crs(
            self.crs_sk42_gk, crs_sk42_geog, always_xy=True
        )
        self.t_gk_from_sk42 = Transformer.from_crs(
            crs_sk42_geog, self.crs_sk42_gk, always_xy=True
        )

        # SK-42 <-> WGS84 with optional Helmert
        if self.helmert_params:
            logger.info(
                'Используются пользовательские параметры Helmert: '
                f'dx={self.helmert_params[0]}, dy={self.helmert_params[1]}, '
                f'dz={self.helmert_params[2]}, rx={self.helmert_params[3]}", '
                f'ry={self.helmert_params[4]}", rz={self.helmert_params[5]}", '
                f'ds={self.helmert_params[6]}ppm'
            )
        else:
            logger.warning(
                'Дата трансформация СК-42↔WGS84 выполняется без явных региональных '
                'параметров; возможен систематический сдвиг 100–300 м. '
                'Укажите параметры Helmert в профиле.'
            )

        self.t_sk42_to_wgs, self.t_wgs_to_sk42 = build_transformers_sk42(
            custom_helmert=self.helmert_params,
        )

    def _compute_coordinates(self) -> None:
        """Compute SK-42 geographic and WGS84 coordinates from GK."""
        # GK -> SK-42 geographic
        self.center_lng_sk42, self.center_lat_sk42 = self.t_sk42_from_gk.transform(
            self.center_x_gk, self.center_y_gk
        )

        # Validate SK-42 bounds
        _validate_sk42_bounds(self.center_lng_sk42, self.center_lat_sk42)

        # SK-42 geographic -> WGS84
        self.center_lng_wgs, self.center_lat_wgs = self.t_sk42_to_wgs.transform(
            self.center_lng_sk42, self.center_lat_sk42
        )

    def get_wgs84_center(self) -> tuple[float, float]:
        """
        Get center coordinates in WGS84.

        Returns:
            Tuple of (latitude, longitude) in WGS84

        """
        return (self.center_lat_wgs, self.center_lng_wgs)

    def get_sk42_center(self) -> tuple[float, float]:
        """
        Get center coordinates in SK-42 geographic.

        Returns:
            Tuple of (latitude, longitude) in SK-42

        """
        return (self.center_lat_sk42, self.center_lng_sk42)

    def gk_to_wgs84(self, x_gk: float, y_gk: float) -> tuple[float, float]:
        """
        Convert GK coordinates to WGS84.

        Args:
            x_gk: X coordinate in GK (easting)
            y_gk: Y coordinate in GK (northing)

        Returns:
            Tuple of (latitude, longitude) in WGS84

        """
        lng_sk42, lat_sk42 = self.t_sk42_from_gk.transform(x_gk, y_gk)
        lng_wgs, lat_wgs = self.t_sk42_to_wgs.transform(lng_sk42, lat_sk42)
        return (lat_wgs, lng_wgs)

    def wgs84_to_gk(self, lat_wgs: float, lng_wgs: float) -> tuple[float, float]:
        """
        Convert WGS84 coordinates to GK.

        Args:
            lat_wgs: Latitude in WGS84
            lng_wgs: Longitude in WGS84

        Returns:
            Tuple of (x_gk, y_gk) in GK

        """
        lng_sk42, lat_sk42 = self.t_wgs_to_sk42.transform(lng_wgs, lat_wgs)
        x_gk, y_gk = self.t_gk_from_sk42.transform(lng_sk42, lat_sk42)
        return (x_gk, y_gk)

    def compute_rotation_deg(
        self, map_params: tuple[float, float, float, int, int, int, int]
    ) -> float:
        """
        Compute rotation angle for grid alignment.

        Args:
            map_params: Map parameters tuple for pixel coordinate conversion

        Returns:
            Rotation angle in degrees

        """
        return compute_rotation_deg_for_east_axis(
            self.center_lat_sk42,
            self.center_lng_sk42,
            map_params,
            self.crs_sk42_gk,
            self.t_sk42_to_wgs,
        )

    def get_result(
        self, map_params: tuple[float, float, float, int, int, int, int] | None = None
    ) -> CoordinateResult:
        """
        Get complete coordinate transformation result.

        Args:
            map_params: Map parameters tuple for rotation calculation.
            If None, rotation_deg will be 0.0.

        Returns:
            CoordinateResult with all computed values

        """
        rotation = self.compute_rotation_deg(map_params) if map_params else 0.0
        return CoordinateResult(
            center_lat_sk42=self.center_lat_sk42,
            center_lng_sk42=self.center_lng_sk42,
            center_lat_wgs=self.center_lat_wgs,
            center_lng_wgs=self.center_lng_wgs,
            rotation_deg=rotation,
            crs_sk42_gk=self.crs_sk42_gk,
            t_sk42_from_gk=self.t_sk42_from_gk,
            t_gk_from_sk42=self.t_gk_from_sk42,
            t_sk42_to_wgs=self.t_sk42_to_wgs,
            t_wgs_to_sk42=self.t_wgs_to_sk42,
            zone=self.zone,
        )


def is_point_within_bounds(
    point_x_gk: float,
    point_y_gk: float,
    center_x_gk: float,
    center_y_gk: float,
    width_m: float,
    height_m: float,
) -> bool:
    """
    Check whether a point lies within the map rectangle.

    Args:
        point_x_gk: Point X in GK (easting)
        point_y_gk: Point Y in GK (northing)
        center_x_gk: Map center X in GK (easting)
        center_y_gk: Map center Y in GK (northing)
        width_m: Map width in meters
        height_m: Map height in meters

    Returns:
        True if the point is inside (or on the boundary of) the map rectangle.

    """
    half_width = width_m / 2
    half_height = height_m / 2
    return (
        center_x_gk - half_width <= point_x_gk <= center_x_gk + half_width
        and center_y_gk - half_height <= point_y_gk <= center_y_gk + half_height
    )


def gk_to_sk42_raw(x_gk_easting: float, y_gk_northing: float) -> tuple[int, int]:
    """
    Convert GK easting/northing back to raw SK-42 integer format.

    The raw format uses 7-digit integers like ``5415000`` / ``7440000``
    where the first two digits encode the high part (zone prefix or latitude
    prefix) and the remaining five digits encode km offset * 1000.

    This is the inverse of the ``control_point_x_sk42_gk`` /
    ``control_point_y_sk42_gk`` properties in ``MapSettings``.

    Args:
        x_gk_easting: GK X coordinate (easting, e.g. 7_440_000)
        y_gk_northing: GK Y coordinate (northing, e.g. 6_015_000)

    Returns:
        (raw_x_north, raw_y_east) — integers in SK-42 raw format,
        e.g. ``(5415000, 7440000)``.

    """
    y_high = int(x_gk_easting // 1e5)
    y_low_km = (x_gk_easting % 1e5) / 1e3
    raw_y = y_high * 100_000 + round(y_low_km * 1000)

    x_high = int(y_gk_northing // 1e5)
    x_low_km = (y_gk_northing % 1e5) / 1e3
    raw_x = x_high * 100_000 + round(x_low_km * 1000)

    return (raw_x, raw_y)


def validate_control_point_bounds(
    control_x_gk: float,
    control_y_gk: float,
    center_x_gk: float,
    center_y_gk: float,
    width_m: float,
    height_m: float,
) -> None:
    """
    Validate that control point is within map bounds.

    .. deprecated::
        Use :func:`is_point_within_bounds` instead; this function raises
        ``ValueError`` on failure.

    Raises:
        ValueError: If control point is outside map bounds

    """
    if not is_point_within_bounds(
        control_x_gk, control_y_gk, center_x_gk, center_y_gk, width_m, height_m
    ):
        half_width = width_m / 2
        half_height = height_m / 2
        map_left = center_x_gk - half_width
        map_right = center_x_gk + half_width
        map_bottom = center_y_gk - half_height
        map_top = center_y_gk + half_height
        msg = (
            f'Контрольная точка X(север)={control_y_gk:.0f}, '
            f'Y(восток)={control_x_gk:.0f} '
            f'выходит за пределы карты. Границы карты: Y(восток)=[{map_left:.0f}, '
            f'{map_right:.0f}], '
            f'X(север)=[{map_bottom:.0f}, {map_top:.0f}]'
        )
        raise ValueError(msg)
