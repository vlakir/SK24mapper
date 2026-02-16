
import pytest
from pathlib import Path
from domain.profiles import list_profiles, load_profile, save_profile, delete_profile, _user_profiles_dir
from domain.models import MapSettings

@pytest.fixture
def temp_profiles_dir(tmp_path, monkeypatch):
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    monkeypatch.setattr("domain.profiles._user_profiles_dir", lambda: profiles_dir)
    return profiles_dir

@pytest.fixture
def base_settings_dict():
    return {
        "from_x_high": 54, "from_y_high": 74, "to_x_high": 54, "to_y_high": 74,
        "from_x_low": 14, "from_y_low": 43, "to_x_low": 23, "to_y_low": 49,
        "output_path": "test.jpg",
        "grid_width_m": 5.0,
        "grid_font_size_m": 100.0,
        "grid_text_margin_m": 50.0,
        "grid_label_bg_padding_m": 10.0,
        "mask_opacity": 0.35
    }

class TestProfiles:
    def test_save_and_load_profile(self, temp_profiles_dir, base_settings_dict):
        settings = MapSettings(**base_settings_dict)
        save_profile("test_profile", settings)
        
        loaded = load_profile("test_profile")
        assert loaded.from_x_high == 54
        assert loaded.output_path == "test.jpg"

    def test_save_profile_by_absolute_path(self, temp_profiles_dir, base_settings_dict):
        settings = MapSettings(**base_settings_dict)
        path = temp_profiles_dir / "abs_profile.toml"
        save_profile(str(path.stem), settings)
        
        loaded = load_profile(str(path))
        assert loaded.from_x_high == 54

    def test_list_profiles(self, temp_profiles_dir, base_settings_dict):
        settings = MapSettings(**base_settings_dict)
        save_profile("a", settings)
        save_profile("b", settings)
        
        profiles = list_profiles()
        assert "a" in profiles
        assert "b" in profiles

    def test_delete_profile(self, temp_profiles_dir, base_settings_dict):
        settings = MapSettings(**base_settings_dict)
        save_profile("to_delete", settings)
        assert "to_delete" in list_profiles()
        
        delete_profile("to_delete")
        assert "to_delete" not in list_profiles()

    def test_load_nonexistent_profile(self, temp_profiles_dir):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent")

    def test_load_profile_with_control_point(self, temp_profiles_dir, base_settings_dict):
        # CX = 5414000, CY = 7443000
        d = base_settings_dict.copy()
        d.update({
            "control_point_enabled": True,
            "control_point_x": 5414000,
            "control_point_y": 7443000
        })
        settings = MapSettings(**d)
        save_profile("cp_profile", settings)
        loaded = load_profile("cp_profile")
        assert loaded.control_point_enabled is True
