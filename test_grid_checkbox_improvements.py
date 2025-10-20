#!/usr/bin/env python3
"""
Test script to verify the grid checkbox improvements:
1. Checkbox is positioned first in the grid settings block
2. Other grid parameters are disabled when checkbox is unchecked
3. Checkbox responds to the first click correctly
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from gui.view import GridSettingsWidget


def test_grid_widget_layout():
    """Test that display_grid checkbox is first in layout."""
    print('TEST 1: Verifying checkbox position in layout...')

    widget = GridSettingsWidget()
    layout = widget.layout()

    # Get the widget at position (0, 0) - should be the checkbox
    first_item = layout.itemAtPosition(0, 0)
    if first_item is None:
        print('❌ FAILED: No widget at position (0, 0)')
        return False

    first_widget = first_item.widget()
    if first_widget is None:
        print('❌ FAILED: No widget found at position (0, 0)')
        return False

    # Check if it's the display_grid checkbox
    if first_widget == widget.display_grid_cb:
        print('✓ PASSED: display_grid checkbox is at position (0, 0)')

        # Verify other widgets are shifted down
        width_label_item = layout.itemAtPosition(1, 0)
        font_label_item = layout.itemAtPosition(2, 0)
        margin_label_item = layout.itemAtPosition(3, 0)
        padding_label_item = layout.itemAtPosition(4, 0)

        if all(
            [width_label_item, font_label_item, margin_label_item, padding_label_item]
        ):
            print('✓ PASSED: Other grid parameters shifted down correctly')
            return True
        else:
            print('❌ FAILED: Some grid parameters not found at expected positions')
            return False
    else:
        print(
            f'❌ FAILED: First widget is {type(first_widget).__name__}, not display_grid checkbox'
        )
        return False


def test_enable_disable_behavior():
    """Test that other controls are enabled/disabled based on checkbox state."""
    print('\nTEST 2: Verifying enable/disable behavior...')

    widget = GridSettingsWidget()

    # Initially checkbox should be checked and controls enabled
    if not widget.display_grid_cb.isChecked():
        print('❌ FAILED: Checkbox should be initially checked')
        return False

    if not all(
        [
            widget.width_spin.isEnabled(),
            widget.font_spin.isEnabled(),
            widget.margin_spin.isEnabled(),
            widget.padding_spin.isEnabled(),
            widget.width_label.isEnabled(),
            widget.font_label.isEnabled(),
            widget.margin_label.isEnabled(),
            widget.padding_label.isEnabled(),
        ]
    ):
        print('❌ FAILED: Controls should be enabled when checkbox is checked')
        return False

    print('✓ PASSED: Controls are enabled when checkbox is checked')

    # Uncheck the checkbox
    widget.display_grid_cb.setChecked(False)

    # Now all controls should be disabled
    if any(
        [
            widget.width_spin.isEnabled(),
            widget.font_spin.isEnabled(),
            widget.margin_spin.isEnabled(),
            widget.padding_spin.isEnabled(),
            widget.width_label.isEnabled(),
            widget.font_label.isEnabled(),
            widget.margin_label.isEnabled(),
            widget.padding_label.isEnabled(),
        ]
    ):
        print('❌ FAILED: Controls should be disabled when checkbox is unchecked')
        return False

    print('✓ PASSED: Controls are disabled when checkbox is unchecked')

    # Check the checkbox again
    widget.display_grid_cb.setChecked(True)

    # Controls should be enabled again
    if not all(
        [
            widget.width_spin.isEnabled(),
            widget.font_spin.isEnabled(),
            widget.margin_spin.isEnabled(),
            widget.padding_spin.isEnabled(),
        ]
    ):
        print('❌ FAILED: Controls should be re-enabled when checkbox is checked again')
        return False

    print('✓ PASSED: Controls are re-enabled when checkbox is checked again')
    return True


def test_set_settings_with_disabled_state():
    """Test that set_settings properly applies enable/disable state."""
    print('\nTEST 3: Verifying set_settings applies enable/disable state...')

    widget = GridSettingsWidget()

    # Load settings with display_grid = False
    settings = {
        'grid_width_px': 5,
        'grid_font_size': 120,
        'grid_text_margin': 43,
        'grid_label_bg_padding': 6,
        'display_grid': False,
    }

    widget.set_settings(settings)

    # Checkbox should be unchecked
    if widget.display_grid_cb.isChecked():
        print(
            '❌ FAILED: Checkbox should be unchecked after loading settings with display_grid=False'
        )
        return False

    print('✓ PASSED: Checkbox correctly unchecked after loading settings')

    # Controls should be disabled
    if any(
        [
            widget.width_spin.isEnabled(),
            widget.font_spin.isEnabled(),
            widget.margin_spin.isEnabled(),
            widget.padding_spin.isEnabled(),
        ]
    ):
        print(
            '❌ FAILED: Controls should be disabled after loading settings with display_grid=False'
        )
        return False

    print(
        '✓ PASSED: Controls are disabled after loading settings with display_grid=False'
    )

    # Load settings with display_grid = True
    settings['display_grid'] = True
    widget.set_settings(settings)

    # Checkbox should be checked
    if not widget.display_grid_cb.isChecked():
        print(
            '❌ FAILED: Checkbox should be checked after loading settings with display_grid=True'
        )
        return False

    print('✓ PASSED: Checkbox correctly checked after loading settings')

    # Controls should be enabled
    if not all(
        [
            widget.width_spin.isEnabled(),
            widget.font_spin.isEnabled(),
            widget.margin_spin.isEnabled(),
            widget.padding_spin.isEnabled(),
        ]
    ):
        print(
            '❌ FAILED: Controls should be enabled after loading settings with display_grid=True'
        )
        return False

    print(
        '✓ PASSED: Controls are enabled after loading settings with display_grid=True'
    )
    return True


def test_first_click_responsiveness():
    """Test that checkbox responds to the first click."""
    print('\nTEST 4: Verifying first click responsiveness...')

    widget = GridSettingsWidget()

    # Initial state: checked
    initial_state = widget.display_grid_cb.isChecked()
    initial_width_enabled = widget.width_spin.isEnabled()

    print(
        f'  Initial state: checkbox={initial_state}, width_spin enabled={initial_width_enabled}'
    )

    # Simulate first click (toggle)
    widget.display_grid_cb.setChecked(not initial_state)

    new_state = widget.display_grid_cb.isChecked()
    new_width_enabled = widget.width_spin.isEnabled()

    print(
        f'  After first click: checkbox={new_state}, width_spin enabled={new_width_enabled}'
    )

    # State should have changed
    if new_state == initial_state:
        print('❌ FAILED: Checkbox state did not change on first click')
        return False

    print('✓ PASSED: Checkbox state changed on first click')

    # Controls should have changed enabled state accordingly
    if new_state and not new_width_enabled:
        print('❌ FAILED: Controls not enabled when checkbox checked on first click')
        return False

    if not new_state and new_width_enabled:
        print('❌ FAILED: Controls not disabled when checkbox unchecked on first click')
        return False

    print('✓ PASSED: Controls responded correctly to first click')
    return True


def main():
    """Run all tests."""
    print('=' * 70)
    print('Grid Checkbox Improvements Test Suite')
    print('=' * 70)

    # Create QApplication (required for Qt widgets)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    results = []

    try:
        results.append(('Layout Position', test_grid_widget_layout()))
        results.append(('Enable/Disable Behavior', test_enable_disable_behavior()))
        results.append(
            ('set_settings Integration', test_set_settings_with_disabled_state())
        )
        results.append(
            ('First Click Responsiveness', test_first_click_responsiveness())
        )

        print('\n' + '=' * 70)
        print('TEST SUMMARY')
        print('=' * 70)

        for test_name, result in results:
            status = '✓ PASSED' if result else '❌ FAILED'
            print(f'{test_name}: {status}')

        all_passed = all(result for _, result in results)

        print('=' * 70)
        if all_passed:
            print('✓ ALL TESTS PASSED')
            print('\nAll three requirements are met:')
            print('1. ✓ Checkbox is positioned first in the grid settings block')
            print(
                '2. ✓ Other grid parameters become inactive when checkbox is unchecked'
            )
            print('3. ✓ Checkbox responds correctly to the first click')
        else:
            print('❌ SOME TESTS FAILED')
            failed_count = sum(1 for _, result in results if not result)
            print(f'\n{failed_count} test(s) failed. Please review the output above.')
        print('=' * 70)

        return 0 if all_passed else 1

    except Exception as e:
        print(f'\n❌ ERROR: Test suite crashed: {e}')
        import traceback

        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
