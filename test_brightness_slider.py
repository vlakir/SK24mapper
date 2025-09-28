#!/usr/bin/env python3
"""
Test script to verify that the brightness slider can reach 400%.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PySide6.QtWidgets import QApplication
from gui.view import OutputSettingsWidget


def test_brightness_slider_range():
    """Test that brightness slider can handle 400% range."""
    print('Testing brightness slider range extension to 400%...')

    # Create the widget
    widget = OutputSettingsWidget()

    # Test the slider range
    slider = widget.brightness_slider
    min_val = slider.minimum()
    max_val = slider.maximum()

    print(f'Brightness slider range: {min_val} - {max_val}')

    # Test that we can set maximum value (400%)
    slider.setValue(400)
    current_val = slider.value()
    print(f'Set slider to 400, current value: {current_val}')

    # Test get_settings with 400% brightness
    settings = widget.get_settings()
    brightness_setting = settings.get('brightness', 0)
    print(f'Brightness setting at 400%: {brightness_setting}')

    # Test set_settings with 4.0 (400%) brightness
    test_settings = {
        'brightness': 4.0,
        'contrast': 1.0,
        'saturation': 1.0,
        'jpeg_quality': 95,
    }

    widget.set_settings(test_settings)
    new_slider_val = slider.value()
    print(f'After setting brightness to 4.0, slider value: {new_slider_val}')

    # Verify the results
    success = True
    if max_val != 400:
        print(f'ERROR: Expected max value 400, got {max_val}')
        success = False

    if current_val != 400:
        print(f'ERROR: Expected slider value 400, got {current_val}')
        success = False

    if abs(brightness_setting - 4.0) > 0.01:
        print(f'ERROR: Expected brightness setting 4.0, got {brightness_setting}')
        success = False

    if new_slider_val != 400:
        print(
            f'ERROR: Expected slider value 400 after set_settings, got {new_slider_val}'
        )
        success = False

    if success:
        print('✓ All tests passed! Brightness slider successfully extended to 400%')
    else:
        print('✗ Some tests failed!')

    return success


if __name__ == '__main__':
    app = QApplication(sys.argv)
    test_brightness_slider_range()
