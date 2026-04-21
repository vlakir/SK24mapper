"""
Unit tests for NSU point bounds validation.

Tests _is_nsu_point_in_bounds logic and download-time filtering of OOB points.
"""

from __future__ import annotations

from services.coordinate_transformer import is_point_within_bounds, sk42_raw_to_gk

# --- Helpers ---


def _make_settings(
    from_x_high: int = 54,
    from_x_low: int = 14,
    from_y_high: int = 74,
    from_y_low: int = 43,
    to_x_high: int = 54,
    to_x_low: int = 23,
    to_y_high: int = 74,
    to_y_low: int = 49,
) -> object:
    """Create a MapSettings with given corner coords."""
    from domain.models import MapSettings

    return MapSettings(
        from_x_high=from_x_high,
        from_x_low=from_x_low,
        from_y_high=from_y_high,
        from_y_low=from_y_low,
        to_x_high=to_x_high,
        to_x_low=to_x_low,
        to_y_high=to_y_high,
        to_y_low=to_y_low,
        grid_width_m=5.0,
        grid_font_size_m=100.0,
        grid_text_margin_m=50.0,
        grid_label_bg_padding_m=10.0,
        mask_opacity=0.35,
    )


def _map_bounds(settings: object) -> tuple[float, float, float, float]:
    """Compute (center_x_gk, center_y_gk, width_m, height_m) from settings."""
    bl_x = settings.bottom_left_x_sk42_gk  # type: ignore[attr-defined]
    bl_y = settings.bottom_left_y_sk42_gk  # type: ignore[attr-defined]
    tr_x = settings.top_right_x_sk42_gk  # type: ignore[attr-defined]
    tr_y = settings.top_right_y_sk42_gk  # type: ignore[attr-defined]
    center_x = (bl_x + tr_x) / 2
    center_y = (bl_y + tr_y) / 2
    width_m = tr_x - bl_x
    height_m = tr_y - bl_y
    return center_x, center_y, width_m, height_m


def _check_point_in_bounds(
    settings: object, x_sk42: int, y_sk42: int,
) -> bool:
    """Mirror _is_nsu_point_in_bounds logic."""
    gk_e, gk_n = sk42_raw_to_gk(x_sk42, y_sk42)
    center_x, center_y, width_m, height_m = _map_bounds(settings)
    return is_point_within_bounds(gk_e, gk_n, center_x, center_y, width_m, height_m)


# ---------------------------------------------------------------------------
# Step 2: _is_nsu_point_in_bounds logic
# ---------------------------------------------------------------------------


class TestNsuPointInBounds:
    """Tests for NSU point bounds checking."""

    def test_point_inside_map_returns_true(self) -> None:
        """Point at map center is in bounds."""
        settings = _make_settings()
        center_x_raw = 5418500
        center_y_raw = 7446000
        assert _check_point_in_bounds(settings, center_x_raw, center_y_raw) is True

    def test_point_outside_map_returns_false(self) -> None:
        """Point far from map is out of bounds."""
        settings = _make_settings()
        far_x = 6000000  # Very far north
        far_y = 7440000
        assert _check_point_in_bounds(settings, far_x, far_y) is False

    def test_point_on_boundary_returns_true(self) -> None:
        """Point exactly on map edge is in bounds."""
        settings = _make_settings()
        center_x, center_y, width_m, height_m = _map_bounds(settings)
        from services.coordinate_transformer import gk_to_sk42_raw

        edge_x_gk = center_x - width_m / 2
        edge_y_gk = center_y
        raw_x, raw_y = gk_to_sk42_raw(edge_x_gk, edge_y_gk)
        assert _check_point_in_bounds(settings, raw_x, raw_y) is True

    def test_point_just_outside_boundary_returns_false(self) -> None:
        """Point just outside map edge is out of bounds."""
        settings = _make_settings()
        center_x, center_y, width_m, height_m = _map_bounds(settings)
        from services.coordinate_transformer import gk_to_sk42_raw

        outside_x_gk = center_x - width_m / 2 - 100
        outside_y_gk = center_y
        raw_x, raw_y = gk_to_sk42_raw(outside_x_gk, outside_y_gk)
        assert _check_point_in_bounds(settings, raw_x, raw_y) is False


# ---------------------------------------------------------------------------
# Step 3: Download-time filtering
# ---------------------------------------------------------------------------


class TestDownloadFilterNsu:
    """Tests for filtering OOB NSU points at download time."""

    def test_download_filters_oob_points(self) -> None:
        """Out-of-bounds NSU points are removed, in-bounds kept."""
        import json

        settings = _make_settings()
        center_x, center_y, width_m, height_m = _map_bounds(settings)

        inside1 = [5418500, 7446000]
        inside2 = [5419000, 7445000]
        outside1 = [6000000, 7440000]
        settings.nsu_target_points_json = json.dumps(  # type: ignore[attr-defined]
            [inside1, inside2, outside1],
        )

        valid, removed = [], []
        for i, (x_sk42, y_sk42) in enumerate(
            settings.nsu_target_points,  # type: ignore[attr-defined]
        ):
            gk_e, gk_n = sk42_raw_to_gk(x_sk42, y_sk42)
            if is_point_within_bounds(
                gk_e, gk_n, center_x, center_y, width_m, height_m,
            ):
                valid.append([x_sk42, y_sk42])
            else:
                removed.append(i + 1)

        assert len(valid) == 2
        assert len(removed) == 1
        assert removed[0] == 3

    def test_download_all_in_bounds_no_filtering(self) -> None:
        """No filtering when all points are within bounds."""
        import json

        settings = _make_settings()
        center_x, center_y, width_m, height_m = _map_bounds(settings)

        inside1 = [5418500, 7446000]
        inside2 = [5419000, 7445000]
        settings.nsu_target_points_json = json.dumps(  # type: ignore[attr-defined]
            [inside1, inside2],
        )

        valid, removed = [], []
        for i, (x_sk42, y_sk42) in enumerate(
            settings.nsu_target_points,  # type: ignore[attr-defined]
        ):
            gk_e, gk_n = sk42_raw_to_gk(x_sk42, y_sk42)
            if is_point_within_bounds(
                gk_e, gk_n, center_x, center_y, width_m, height_m,
            ):
                valid.append([x_sk42, y_sk42])
            else:
                removed.append(i + 1)

        assert len(valid) == 2
        assert len(removed) == 0
