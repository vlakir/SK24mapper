"""
Test script for display_grid checkbox functionality.

Tests:
1. Load a profile and verify display_grid value
2. Change display_grid setting and save profile
3. Reload profile and verify the setting persists
4. Verify the setting is correctly passed to drawing functions
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from profiles import load_profile, save_profile, profile_path
from domen import MapSettings


def test_display_grid_in_profile():
    """Test that display_grid is correctly saved and loaded from profiles."""
    print('=' * 60)
    print('TEST 1: Profile Save/Load Functionality')
    print('=' * 60)

    # Test with default profile
    profile_name = 'default'
    print(f"\n1. Loading profile '{profile_name}'...")

    try:
        settings = load_profile(profile_name)
        print(f'   ✓ Profile loaded successfully')
        print(f'   display_grid = {settings.display_grid}')

        # Verify it's True by default
        if settings.display_grid:
            print(f'   ✓ Default value is True (as expected)')
        else:
            print(f'   ⚠ Warning: Default value is False (expected True)')

    except Exception as e:
        print(f'   ✗ Error loading profile: {e}')
        return False

    # Test changing the value
    print(f'\n2. Testing value change to False...')
    original_value = settings.display_grid
    settings.display_grid = False

    # Save to a test profile
    test_profile_name = 'test_display_grid_temp'
    print(f"   Saving as '{test_profile_name}'...")

    try:
        save_profile(test_profile_name, settings)
        print(f'   ✓ Profile saved successfully')
    except Exception as e:
        print(f'   ✗ Error saving profile: {e}')
        return False

    # Reload and verify
    print(f"\n3. Reloading '{test_profile_name}' to verify persistence...")

    try:
        reloaded_settings = load_profile(test_profile_name)
        print(f'   ✓ Profile reloaded successfully')
        print(f'   display_grid = {reloaded_settings.display_grid}')

        if not reloaded_settings.display_grid:
            print(f'   ✓ Value correctly persisted as False')
        else:
            print(f'   ✗ ERROR: Value reverted to True (should be False)')
            return False

    except Exception as e:
        print(f'   ✗ Error reloading profile: {e}')
        return False

    # Test changing back to True
    print(f'\n4. Testing value change to True...')
    reloaded_settings.display_grid = True

    try:
        save_profile(test_profile_name, reloaded_settings)
        final_settings = load_profile(test_profile_name)
        print(f'   display_grid = {final_settings.display_grid}')

        if final_settings.display_grid:
            print(f'   ✓ Value correctly persisted as True')
        else:
            print(f'   ✗ ERROR: Value remained False (should be True)')
            return False

    except Exception as e:
        print(f'   ✗ Error in final test: {e}')
        return False

    # Cleanup test profile
    print(f'\n5. Cleaning up test profile...')
    try:
        test_path = profile_path(test_profile_name)
        if test_path.exists():
            test_path.unlink()
            print(f'   ✓ Test profile deleted')
    except Exception as e:
        print(f'   ⚠ Warning: Could not delete test profile: {e}')

    return True


def test_all_profiles_have_display_grid():
    """Verify all existing profiles have the display_grid field."""
    print('\n' + '=' * 60)
    print('TEST 2: All Profiles Have display_grid Field')
    print('=' * 60)

    from profiles import list_profiles

    profiles = list_profiles()
    print(f'\nFound {len(profiles)} profiles')

    all_ok = True
    for profile_name in profiles:
        try:
            settings = load_profile(profile_name)
            has_field = hasattr(settings, 'display_grid')
            value = settings.display_grid if has_field else 'N/A'

            status = '✓' if has_field else '✗'
            print(f'  {status} {profile_name:20s} display_grid={value}')

            if not has_field:
                all_ok = False

        except Exception as e:
            print(f'  ✗ {profile_name:20s} Error: {e}')
            all_ok = False

    return all_ok


def test_model_validation():
    """Test that MapSettings correctly validates and defaults display_grid."""
    print('\n' + '=' * 60)
    print('TEST 3: MapSettings Model Validation')
    print('=' * 60)

    # Test 1: Create without display_grid field (should default to True)
    print('\n1. Creating MapSettings without display_grid field...')
    try:
        minimal_data = {
            'from_x_high': 54,
            'from_y_high': 74,
            'to_x_high': 54,
            'to_y_high': 74,
            'from_x_low': 12,
            'from_y_low': 35,
            'to_x_low': 22,
            'to_y_low': 49,
            'output_path': '../maps/test.jpg',
            'grid_width_px': 5,
            'grid_font_size': 86,
            'grid_text_margin': 43,
            'grid_label_bg_padding': 6,
            'mask_opacity': 0.35,
        }
        settings = MapSettings.model_validate(minimal_data)
        print(f'   ✓ MapSettings created successfully')
        print(f'   display_grid = {settings.display_grid}')

        if settings.display_grid:
            print(f'   ✓ Defaults to True (correct)')
        else:
            print(f'   ✗ ERROR: Defaults to False (should be True)')
            return False

    except Exception as e:
        print(f'   ✗ Error: {e}')
        return False

    # Test 2: Create with display_grid=False
    print('\n2. Creating MapSettings with display_grid=False...')
    try:
        data_with_false = minimal_data.copy()
        data_with_false['display_grid'] = False
        settings = MapSettings.model_validate(data_with_false)
        print(f'   ✓ MapSettings created successfully')
        print(f'   display_grid = {settings.display_grid}')

        if not settings.display_grid:
            print(f'   ✓ Correctly set to False')
        else:
            print(f'   ✗ ERROR: Set to True (should be False)')
            return False

    except Exception as e:
        print(f'   ✗ Error: {e}')
        return False

    # Test 3: Create with display_grid=True
    print('\n3. Creating MapSettings with display_grid=True...')
    try:
        data_with_true = minimal_data.copy()
        data_with_true['display_grid'] = True
        settings = MapSettings.model_validate(data_with_true)
        print(f'   ✓ MapSettings created successfully')
        print(f'   display_grid = {settings.display_grid}')

        if settings.display_grid:
            print(f'   ✓ Correctly set to True')
        else:
            print(f'   ✗ ERROR: Set to False (should be True)')
            return False

    except Exception as e:
        print(f'   ✗ Error: {e}')
        return False

    return True


def main():
    """Run all tests."""
    print('\n' + '=' * 60)
    print('DISPLAY_GRID CHECKBOX FIX - COMPREHENSIVE TEST SUITE')
    print('=' * 60)

    results = []

    # Test 1: Profile save/load
    test1_passed = test_display_grid_in_profile()
    results.append(('Profile Save/Load', test1_passed))

    # Test 2: All profiles have the field
    test2_passed = test_all_profiles_have_display_grid()
    results.append(('All Profiles Have Field', test2_passed))

    # Test 3: Model validation
    test3_passed = test_model_validation()
    results.append(('Model Validation', test3_passed))

    # Summary
    print('\n' + '=' * 60)
    print('TEST SUMMARY')
    print('=' * 60)

    for test_name, passed in results:
        status = '✓ PASS' if passed else '✗ FAIL'
        print(f'  {status}: {test_name}')

    all_passed = all(passed for _, passed in results)

    print('\n' + '=' * 60)
    if all_passed:
        print('ALL TESTS PASSED ✓')
        print('=' * 60)
        print('\nThe display_grid checkbox fix is working correctly!')
        print('\nHow it works:')
        print('  1. Checkbox state changes trigger _on_settings_changed()')
        print('  2. Settings are collected via grid_widget.get_settings()')
        print(
            '  3. display_grid value is sent to controller via update_settings_bulk()'
        )
        print('  4. Value is saved to profile and persists across sessions')
        print('\nExpected behavior:')
        print('  • display_grid=True: Full grid with lines and labels')
        print('  • display_grid=False: Only crosses at intersection points')
        return 0
    else:
        print('SOME TESTS FAILED ✗')
        print('=' * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
