"""Tests for profiles module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from domain.profiles import (
    CONTROL_POINT_PRECISION_TOLERANCE_M,
    _user_profiles_dir,
    delete_profile,
    ensure_profiles_dir,
    list_profiles,
    load_profile,
    profile_path,
    save_profile,
)
from domain.models import MapSettings


def create_test_settings(**overrides):
    """Create MapSettings with default values for testing."""
    defaults = {
        'from_x_high': 54,
        'from_y_high': 74,
        'to_x_high': 54,
        'to_y_high': 74,
        'from_x_low': 12,
        'from_y_low': 35,
        'to_x_low': 22,
        'to_y_low': 49,
        'output_path': 'test.jpg',
        'grid_width_m': 5.0,
        'grid_font_size_m': 100.0,
        'grid_text_margin_m': 50.0,
        'grid_label_bg_padding_m': 10.0,
        'mask_opacity': 0.35,
    }
    defaults.update(overrides)
    return MapSettings(**defaults)


class TestUserProfilesDir:
    """Tests for _user_profiles_dir function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = _user_profiles_dir()
        assert isinstance(result, Path)

    def test_local_profiles_preferred(self):
        """Local configs/profiles should be preferred if exists."""
        result = _user_profiles_dir()
        # Project has configs/profiles, so it should be used
        assert 'profiles' in str(result)


class TestEnsureProfilesDir:
    """Tests for ensure_profiles_dir function."""

    def test_returns_path(self):
        """Should return a Path object."""
        result = ensure_profiles_dir()
        assert isinstance(result, Path)

    def test_directory_exists(self):
        """Returned directory should exist."""
        result = ensure_profiles_dir()
        assert result.exists()
        assert result.is_dir()


class TestListProfiles:
    """Tests for list_profiles function."""

    def test_returns_list(self):
        """Should return a list."""
        result = list_profiles()
        assert isinstance(result, list)

    def test_profiles_sorted(self):
        """Profiles should be sorted."""
        result = list_profiles()
        assert result == sorted(result)

    def test_no_toml_extension(self):
        """Profile names should not have .toml extension."""
        result = list_profiles()
        for name in result:
            assert not name.endswith('.toml')


class TestProfilePath:
    """Tests for profile_path function."""

    def test_returns_path_with_toml(self):
        """Should return path with .toml extension."""
        result = profile_path('test_profile')
        assert result.suffix == '.toml'
        assert result.stem == 'test_profile'


class TestLoadProfile:
    """Tests for load_profile function."""

    def test_load_nonexistent_profile(self):
        """Loading nonexistent profile should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_profile('_nonexistent_profile_xyz_')

    def test_load_existing_profile(self):
        """Should load existing profile from configs/profiles."""
        profiles = list_profiles()
        if profiles:
            loaded = load_profile(profiles[0])
            assert loaded is not None
            assert hasattr(loaded, 'from_x_high')


class TestDeleteProfile:
    """Tests for delete_profile function."""

    def test_delete_nonexistent_profile(self):
        """Deleting nonexistent profile should not raise."""
        delete_profile('_nonexistent_profile_to_delete_')  # Should not raise


class TestControlPointPrecisionTolerance:
    """Tests for CONTROL_POINT_PRECISION_TOLERANCE_M constant."""

    def test_is_small_positive(self):
        """Tolerance should be small positive number."""
        assert CONTROL_POINT_PRECISION_TOLERANCE_M > 0
        assert CONTROL_POINT_PRECISION_TOLERANCE_M < 1
