#!/usr/bin/env python3
"""
Test script to verify that slider values are preserved when preview is cleared.
This script simulates the issue scenario to confirm the fix works.
"""

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from PySide6.QtWidgets import QApplication
from gui.view import MainWindow
from gui.model import MilMapperModel
from gui.controller import MilMapperController


def test_slider_preservation():
    """Test that slider values are preserved when preview is cleared."""
    app = QApplication(sys.argv)

    # Create the MVC components
    model = MilMapperModel()
    controller = MilMapperController(model)
    view = MainWindow(model, controller)

    print('Testing slider value preservation...')

    # Set some custom slider values
    test_brightness = 120
    test_contrast = 80
    test_saturation = 150

    print(
        f'Setting test values - Brightness: {test_brightness}%, Contrast: {test_contrast}%, Saturation: {test_saturation}%'
    )

    view.brightness_slider.setValue(test_brightness)
    view.contrast_slider.setValue(test_contrast)
    view.saturation_slider.setValue(test_saturation)

    # Verify values are set
    assert view.brightness_slider.value() == test_brightness
    assert view.contrast_slider.value() == test_contrast
    assert view.saturation_slider.value() == test_saturation
    print('✓ Test values set successfully')

    # Clear preview UI (this is where the bug was)
    print('Clearing preview UI...')
    view._clear_preview_ui()

    # Check that slider values are preserved
    brightness_after = view.brightness_slider.value()
    contrast_after = view.contrast_slider.value()
    saturation_after = view.saturation_slider.value()

    print(
        f'After clearing - Brightness: {brightness_after}%, Contrast: {contrast_after}%, Saturation: {saturation_after}%'
    )

    # Verify values are preserved
    success = True
    if brightness_after != test_brightness:
        print(
            f'✗ Brightness value not preserved! Expected {test_brightness}, got {brightness_after}'
        )
        success = False
    else:
        print('✓ Brightness value preserved')

    if contrast_after != test_contrast:
        print(
            f'✗ Contrast value not preserved! Expected {test_contrast}, got {contrast_after}'
        )
        success = False
    else:
        print('✓ Contrast value preserved')

    if saturation_after != test_saturation:
        print(
            f'✗ Saturation value not preserved! Expected {test_saturation}, got {saturation_after}'
        )
        success = False
    else:
        print('✓ Saturation value preserved')

    # Check that internal _adj dict was updated correctly
    expected_brightness_adj = test_brightness / 100.0
    expected_contrast_adj = test_contrast / 100.0
    expected_saturation_adj = test_saturation / 100.0

    actual_brightness_adj = view._adj['brightness']
    actual_contrast_adj = view._adj['contrast']
    actual_saturation_adj = view._adj['saturation']

    print(
        f'Internal _adj values - Brightness: {actual_brightness_adj:.2f}, Contrast: {actual_contrast_adj:.2f}, Saturation: {actual_saturation_adj:.2f}'
    )

    if abs(actual_brightness_adj - expected_brightness_adj) > 0.01:
        print(
            f'✗ Internal brightness adjustment not preserved! Expected {expected_brightness_adj:.2f}, got {actual_brightness_adj:.2f}'
        )
        success = False
    else:
        print('✓ Internal brightness adjustment preserved')

    if abs(actual_contrast_adj - expected_contrast_adj) > 0.01:
        print(
            f'✗ Internal contrast adjustment not preserved! Expected {expected_contrast_adj:.2f}, got {actual_contrast_adj:.2f}'
        )
        success = False
    else:
        print('✓ Internal contrast adjustment preserved')

    if abs(actual_saturation_adj - expected_saturation_adj) > 0.01:
        print(
            f'✗ Internal saturation adjustment not preserved! Expected {expected_saturation_adj:.2f}, got {actual_saturation_adj:.2f}'
        )
        success = False
    else:
        print('✓ Internal saturation adjustment preserved')

    # Verify sliders are disabled after clearing
    if not view.brightness_slider.isEnabled():
        print('✓ Sliders correctly disabled after clearing')
    else:
        print('✗ Sliders should be disabled after clearing')
        success = False

    if success:
        print('\n🎉 ALL TESTS PASSED! The fix works correctly.')
        print('Slider values are preserved when preview is cleared.')
        return True
    else:
        print('\n❌ SOME TESTS FAILED! The fix needs review.')
        return False


if __name__ == '__main__':
    try:
        success = test_slider_preservation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'Test failed with exception: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
