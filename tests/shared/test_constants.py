"""Tests for constants module."""

from shared.constants import (
    ADDITIVE_RATIO,
    DEM_CACHE_ENABLED,
    DESIRED_ZOOM,
    DOWNLOAD_CONCURRENCY,
    EARTH_RADIUS_M,
    EPSG_SK42_GK_BASE,
    GK_FALSE_EASTING,
    GK_ZONE_WIDTH_DEG,
    GRID_STEP_M,
    HTTP_TIMEOUT_DEFAULT,
    MAP_TYPE_LABELS_RU,
    MAPBOX_STYLE_BY_TYPE,
    MAX_GK_ZONE,
    MapType,
    RETINA_FACTOR,
    SK42_VALID_LAT_MAX,
    SK42_VALID_LAT_MIN,
    SK42_VALID_LON_MAX,
    SK42_VALID_LON_MIN,
    STATIC_SCALE,
    TILE_SIZE,
    TILE_SIZE_512,
    XYZ_TILE_SIZE,
    default_map_type,
    map_type_to_style_id,
)


class TestMapType:
    """Tests for MapType enum."""

    def test_all_map_types_exist(self):
        """All expected map types should exist."""
        assert MapType.SATELLITE
        assert MapType.HYBRID
        assert MapType.STREETS
        assert MapType.OUTDOORS
        assert MapType.ELEVATION_COLOR
        assert MapType.ELEVATION_CONTOURS
        assert MapType.ELEVATION_HILLSHADE

    def test_map_type_values(self):
        """Map type values should match names."""
        assert MapType.SATELLITE.value == 'SATELLITE'
        assert MapType.HYBRID.value == 'HYBRID'

    def test_all_map_types_have_labels(self):
        """All map types should have Russian labels."""
        for mt in MapType:
            assert mt in MAP_TYPE_LABELS_RU


class TestDefaultMapType:
    """Tests for default_map_type function."""

    def test_returns_satellite(self):
        """Default map type should be SATELLITE."""
        assert default_map_type() == MapType.SATELLITE

    def test_returns_map_type_instance(self):
        """Should return MapType instance."""
        assert isinstance(default_map_type(), MapType)


class TestMapTypeToStyleId:
    """Tests for map_type_to_style_id function."""

    def test_satellite_style(self):
        """SATELLITE should return satellite style."""
        result = map_type_to_style_id(MapType.SATELLITE)
        assert result == 'mapbox/satellite-v9'

    def test_hybrid_style(self):
        """HYBRID should return satellite-streets style."""
        result = map_type_to_style_id(MapType.HYBRID)
        assert result == 'mapbox/satellite-streets-v12'

    def test_streets_style(self):
        """STREETS should return streets style."""
        result = map_type_to_style_id(MapType.STREETS)
        assert result == 'mapbox/streets-v12'

    def test_outdoors_style(self):
        """OUTDOORS should return outdoors style."""
        result = map_type_to_style_id(MapType.OUTDOORS)
        assert result == 'mapbox/outdoors-v12'

    def test_elevation_returns_none(self):
        """Elevation map types should return None."""
        assert map_type_to_style_id(MapType.ELEVATION_COLOR) is None
        assert map_type_to_style_id(MapType.ELEVATION_CONTOURS) is None
        assert map_type_to_style_id(MapType.ELEVATION_HILLSHADE) is None

    def test_string_input(self):
        """Should accept string input."""
        result = map_type_to_style_id('SATELLITE')
        assert result == 'mapbox/satellite-v9'

    def test_invalid_string_fallback(self):
        """Invalid string should fallback to SATELLITE."""
        result = map_type_to_style_id('INVALID_TYPE')
        assert result == 'mapbox/satellite-v9'


class TestNumericConstants:
    """Tests for numeric constants."""

    def test_earth_radius(self):
        """Earth radius should be approximately 6371km."""
        assert 6_000_000 < EARTH_RADIUS_M < 7_000_000

    def test_tile_sizes(self):
        """Tile sizes should be standard values."""
        assert TILE_SIZE == 256
        assert TILE_SIZE_512 == 512
        assert XYZ_TILE_SIZE in (256, 512)

    def test_retina_factor(self):
        """Retina factor should be 2."""
        assert RETINA_FACTOR == 2

    def test_static_scale(self):
        """Static scale should be positive."""
        assert STATIC_SCALE > 0

    def test_desired_zoom(self):
        """Desired zoom should be reasonable."""
        assert 0 <= DESIRED_ZOOM <= 24

    def test_download_concurrency(self):
        """Download concurrency should be positive."""
        assert DOWNLOAD_CONCURRENCY > 0

    def test_grid_step(self):
        """Grid step should be 1000m (1km)."""
        assert GRID_STEP_M == 1000

    def test_additive_ratio(self):
        """Additive ratio should be small positive."""
        assert 0 < ADDITIVE_RATIO < 1

    def test_http_timeout(self):
        """HTTP timeout should be reasonable."""
        assert 5 <= HTTP_TIMEOUT_DEFAULT <= 60

    def test_dem_cache_enabled(self):
        """DEM cache enabled should be boolean."""
        assert isinstance(DEM_CACHE_ENABLED, bool)


class TestGKConstants:
    """Tests for Gauss-Kruger constants."""

    def test_gk_zone_width(self):
        """GK zone width should be 6 degrees."""
        assert GK_ZONE_WIDTH_DEG == 6

    def test_gk_false_easting(self):
        """GK false easting should be 500000m."""
        assert GK_FALSE_EASTING == 500_000

    def test_max_gk_zone(self):
        """Max GK zone should be 60."""
        assert MAX_GK_ZONE == 60

    def test_epsg_base(self):
        """EPSG base for SK-42 GK should be 28400."""
        assert EPSG_SK42_GK_BASE == 28400


class TestSK42Bounds:
    """Tests for SK-42 validity bounds."""

    def test_longitude_bounds(self):
        """Longitude bounds should cover former USSR."""
        assert SK42_VALID_LON_MIN < SK42_VALID_LON_MAX
        assert SK42_VALID_LON_MIN >= 0
        assert SK42_VALID_LON_MAX <= 360

    def test_latitude_bounds(self):
        """Latitude bounds should be reasonable."""
        assert SK42_VALID_LAT_MIN < SK42_VALID_LAT_MAX
        assert SK42_VALID_LAT_MIN >= 0
        assert SK42_VALID_LAT_MAX <= 90

    def test_moscow_in_bounds(self):
        """Moscow coordinates should be within bounds."""
        moscow_lon = 37.62
        moscow_lat = 55.75
        assert SK42_VALID_LON_MIN <= moscow_lon <= SK42_VALID_LON_MAX
        assert SK42_VALID_LAT_MIN <= moscow_lat <= SK42_VALID_LAT_MAX

