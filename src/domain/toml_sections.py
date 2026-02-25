"""Mapping layer between flat MapSettings fields and sectioned TOML format.

MapSettings remains a flat Pydantic model. This module provides two functions:
- flat_to_sectioned(): flat dict → sectioned dict (for TOML save)
- sectioned_to_flat(): sectioned dict → flat dict (for TOML load)
"""

from __future__ import annotations

# {section_name: {flat_field_name: short_name_in_toml}}
SECTION_MAP: dict[str, dict[str, str]] = {
    'helmert': {
        'helmert_enabled': 'enabled',
        'helmert_dx': 'dx',
        'helmert_dy': 'dy',
        'helmert_dz': 'dz',
        'helmert_rx_as': 'rx_as',
        'helmert_ry_as': 'ry_as',
        'helmert_rz_as': 'rz_as',
        'helmert_ds_ppm': 'ds_ppm',
    },
    'control_point': {
        'control_point_enabled': 'enabled',
        'control_point_x': 'x',
        'control_point_y': 'y',
        'control_point_name': 'name',
    },
    'radio_horizon': {
        'antenna_height_m': 'antenna_height_m',
        'max_flight_height_m': 'max_flight_height_m',
        'uav_height_reference': 'uav_height_reference',
        'radio_horizon_overlay_alpha': 'overlay_alpha',
    },
    'radar_coverage': {
        'radar_azimuth_deg': 'azimuth_deg',
        'radar_sector_width_deg': 'sector_width_deg',
        'radar_elevation_min_deg': 'elevation_min_deg',
        'radar_elevation_max_deg': 'elevation_max_deg',
        'radar_max_range_km': 'max_range_km',
        'radar_target_height_min_m': 'target_height_min_m',
        'radar_target_height_max_m': 'target_height_max_m',
    },
    'link_profile': {
        'link_point_a_x': 'point_a_x',
        'link_point_a_y': 'point_a_y',
        'link_point_a_name': 'point_a_name',
        'link_point_b_x': 'point_b_x',
        'link_point_b_y': 'point_b_y',
        'link_point_b_name': 'point_b_name',
        'link_freq_mhz': 'freq_mhz',
        'link_antenna_a_m': 'antenna_a_m',
        'link_antenna_b_m': 'antenna_b_m',
    },
}

# Reverse index: flat_field → (section, short_name)
_FLAT_TO_SECTION: dict[str, tuple[str, str]] = {}
for _section, _fields in SECTION_MAP.items():
    for _flat, _short in _fields.items():
        _FLAT_TO_SECTION[_flat] = (_section, _short)

# Reverse index: (section, short_name) → flat_field
_SECTION_TO_FLAT: dict[str, dict[str, str]] = {}
for _section, _fields in SECTION_MAP.items():
    _SECTION_TO_FLAT[_section] = {v: k for k, v in _fields.items()}


def flat_to_sectioned(flat: dict) -> dict:
    """Convert flat MapSettings dict to sectioned dict for TOML output."""
    result: dict = {'common': {}}
    for key, value in flat.items():
        if key in _FLAT_TO_SECTION:
            section, short_name = _FLAT_TO_SECTION[key]
            if section not in result:
                result[section] = {}
            result[section][short_name] = value
        else:
            result['common'][key] = value
    return result


def sectioned_to_flat(data: dict) -> dict:
    """Convert sectioned TOML dict to flat dict for MapSettings validation."""
    flat: dict = {}
    for key, value in data.items():
        if isinstance(value, dict) and key in _SECTION_TO_FLAT:
            # Known section — expand short names to flat names
            mapping = _SECTION_TO_FLAT[key]
            for short_name, field_value in value.items():
                flat_name = mapping.get(short_name, short_name)
                flat[flat_name] = field_value
        elif isinstance(value, dict) and key == 'common':
            # Common section — pass through as-is
            flat.update(value)
        elif isinstance(value, dict):
            # Unknown section — pass through keys as-is
            flat.update(value)
        else:
            # Top-level key (backward compat with flat TOML)
            flat[key] = value
    return flat
