"""Tests for TOML sectioned profile mapping layer."""

import tomlkit

from domain.models import MapSettings
from domain.toml_sections import (
    SECTION_MAP,
    flat_to_sectioned,
    sectioned_to_flat,
)


def _base_settings(**overrides):
    defaults = {
        'from_x_high': 54,
        'from_y_high': 74,
        'to_x_high': 54,
        'to_y_high': 74,
        'from_x_low': 14,
        'from_y_low': 43,
        'to_x_low': 23,
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


class TestFlatToSectioned:
    """Tests for flat_to_sectioned()."""

    def test_creates_expected_sections(self):
        flat = _base_settings().model_dump()
        result = flat_to_sectioned(flat)
        assert 'common' in result
        assert 'helmert' in result
        assert 'control_point' in result
        assert 'radio_horizon' in result
        assert 'radar_coverage' in result
        assert 'link_profile' in result

    def test_helmert_fields_in_section(self):
        flat = _base_settings().model_dump()
        result = flat_to_sectioned(flat)
        helmert = result['helmert']
        assert 'enabled' in helmert
        assert 'dx' in helmert
        assert 'dy' in helmert
        assert 'dz' in helmert
        assert 'rx_as' in helmert
        assert 'ds_ppm' in helmert
        # Flat name must NOT be in helmert section
        assert 'helmert_dx' not in helmert

    def test_link_profile_fields_in_section(self):
        flat = _base_settings(
            link_point_a_x=5420770,
            link_point_a_y=7448250,
            link_freq_mhz=1800.0,
        ).model_dump()
        result = flat_to_sectioned(flat)
        lp = result['link_profile']
        assert lp['point_a_x'] == 5420770
        assert lp['point_a_y'] == 7448250
        assert lp['freq_mhz'] == 1800.0
        # Flat names must NOT leak
        assert 'link_point_a_x' not in lp
        assert 'link_freq_mhz' not in lp

    def test_common_section_has_no_sectioned_fields(self):
        flat = _base_settings().model_dump()
        result = flat_to_sectioned(flat)
        common = result['common']
        # None of the mapped fields should appear in common
        for section_fields in SECTION_MAP.values():
            for flat_name in section_fields:
                assert flat_name not in common

    def test_common_section_has_core_fields(self):
        flat = _base_settings().model_dump()
        result = flat_to_sectioned(flat)
        common = result['common']
        assert 'from_x_high' in common
        assert 'output_path' in common
        assert 'map_type' in common
        assert 'mask_opacity' in common

    def test_values_preserved(self):
        s = _base_settings(helmert_dx=-99.5, radar_azimuth_deg=123.0)
        flat = s.model_dump()
        result = flat_to_sectioned(flat)
        assert result['helmert']['dx'] == -99.5
        assert result['radar_coverage']['azimuth_deg'] == 123.0


class TestSectionedToFlat:
    """Tests for sectioned_to_flat()."""

    def test_restores_flat_names(self):
        sectioned = {
            'common': {'from_x_high': 54, 'output_path': 'test.jpg'},
            'helmert': {'enabled': True, 'dx': -50.0},
            'link_profile': {'point_a_x': 5420770, 'freq_mhz': 900.0},
        }
        flat = sectioned_to_flat(sectioned)
        assert flat['helmert_enabled'] is True
        assert flat['helmert_dx'] == -50.0
        assert flat['link_point_a_x'] == 5420770
        assert flat['link_freq_mhz'] == 900.0
        assert flat['from_x_high'] == 54

    def test_backward_compat_flat_toml(self):
        """Old flat TOML (no sections) should pass through unchanged."""
        flat_data = {
            'from_x_high': 54,
            'helmert_dx': -50.0,
            'link_point_a_x': 5420770,
        }
        result = sectioned_to_flat(flat_data)
        assert result == flat_data

    def test_unknown_section_passes_through(self):
        """Unknown TOML sections should have their keys passed as-is."""
        data = {
            'common': {'from_x_high': 54},
            'future_feature': {'some_key': 42},
        }
        flat = sectioned_to_flat(data)
        assert flat['from_x_high'] == 54
        assert flat['some_key'] == 42


class TestRoundTrip:
    """Full round-trip: flat → sectioned → flat → MapSettings."""

    def test_round_trip_all_fields(self):
        original = _base_settings(
            link_point_a_x=5420770,
            link_point_a_y=7448250,
            link_point_a_name='РСТ 1',
            link_freq_mhz=1800.0,
            radar_azimuth_deg=217.0,
            control_point_enabled=True,
            control_point_x=5422328,
            control_point_y=7448442,
        )
        # flat → sectioned → TOML text → parse → flat → MapSettings
        flat1 = original.model_dump()
        sectioned = flat_to_sectioned(flat1)
        text = tomlkit.dumps(sectioned)
        parsed = dict(tomlkit.parse(text))
        flat2 = sectioned_to_flat(parsed)
        restored = MapSettings.model_validate(flat2)

        for field in MapSettings.model_fields:
            assert getattr(original, field) == getattr(restored, field), (
                f'Mismatch on field {field}: '
                f'{getattr(original, field)!r} != {getattr(restored, field)!r}'
            )

    def test_round_trip_preserves_link_profile_data(self):
        s = _base_settings(
            link_point_b_x=5417552,
            link_point_b_y=7444720,
            link_point_b_name='РСТ 2',
            link_antenna_a_m=15.0,
            link_antenna_b_m=25.0,
        )
        sectioned = flat_to_sectioned(s.model_dump())
        flat = sectioned_to_flat(sectioned)
        restored = MapSettings.model_validate(flat)
        assert restored.link_point_b_x == 5417552
        assert restored.link_point_b_y == 7444720
        assert restored.link_point_b_name == 'РСТ 2'
        assert restored.link_antenna_a_m == 15.0
        assert restored.link_antenna_b_m == 25.0


class TestSectionMapCompleteness:
    """Verify that SECTION_MAP covers all expected prefixed fields."""

    def test_all_helmert_fields_mapped(self):
        helmert_fields = [
            f for f in MapSettings.model_fields if f.startswith('helmert_')
        ]
        mapped = set(SECTION_MAP['helmert'].keys())
        for f in helmert_fields:
            assert f in mapped, f'Missing helmert mapping for {f}'

    def test_all_link_fields_mapped(self):
        link_fields = [
            f for f in MapSettings.model_fields if f.startswith('link_')
        ]
        mapped = set(SECTION_MAP['link_profile'].keys())
        for f in link_fields:
            assert f in mapped, f'Missing link_profile mapping for {f}'

    def test_all_radar_fields_mapped(self):
        radar_fields = [
            f for f in MapSettings.model_fields if f.startswith('radar_')
        ]
        mapped = set(SECTION_MAP['radar_coverage'].keys())
        for f in radar_fields:
            assert f in mapped, f'Missing radar_coverage mapping for {f}'

    def test_all_control_point_fields_mapped(self):
        cp_fields = [
            f for f in MapSettings.model_fields if f.startswith('control_point_')
        ]
        mapped = set(SECTION_MAP['control_point'].keys())
        for f in cp_fields:
            assert f in mapped, f'Missing control_point mapping for {f}'

    def test_no_duplicate_short_names_across_sections(self):
        """Short names can repeat across sections but flat names must be unique."""
        all_flat = []
        for fields in SECTION_MAP.values():
            all_flat.extend(fields.keys())
        assert len(all_flat) == len(set(all_flat)), 'Duplicate flat field names in SECTION_MAP'
