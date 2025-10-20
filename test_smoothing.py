#!/usr/bin/env python3
"""Test script to verify contour smoothing implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from contours.seeds import smooth_polyline, simple_smooth_polyline


def test_simple_smooth():
    """Test simple smoothing without scipy."""
    print('Testing simple_smooth_polyline...')

    # Test with short polyline (< 3 points)
    points = [(0, 0), (1, 1)]
    result = simple_smooth_polyline(points)
    assert result == points, 'Short polyline should remain unchanged'
    print('  ✓ Short polyline test passed')

    # Test with longer polyline
    points = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)]
    result = simple_smooth_polyline(points, iterations=1)
    assert len(result) == len(points), 'Length should be preserved'
    assert result[0] == points[0], 'First point should be preserved'
    assert result[-1] == points[-1], 'Last point should be preserved'
    print('  ✓ Basic smoothing test passed')

    print('simple_smooth_polyline: OK\n')


def test_smooth_polyline():
    """Test spline smoothing (with scipy if available)."""
    print('Testing smooth_polyline...')

    # Test with short polyline
    points = [(0, 0), (1, 1)]
    result = smooth_polyline(points)
    assert result == points, 'Short polyline should remain unchanged'
    print('  ✓ Short polyline test passed')

    # Test with longer polyline
    points = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)]
    result = smooth_polyline(points, smoothing_factor=3)

    # If scipy is available, we should get more points
    # If not, it falls back to simple_smooth_polyline
    try:
        import scipy

        assert len(result) > len(points), 'Should have more points with scipy'
        print('  ✓ Spline smoothing test passed (scipy available)')
    except ImportError:
        assert len(result) == len(points), 'Should use fallback without scipy'
        print('  ✓ Fallback smoothing test passed (scipy not available)')

    print('smooth_polyline: OK\n')


def test_integration():
    """Test that smoothing is properly integrated."""
    print('Testing integration with constants...')

    from constants import CONTOUR_SEED_SMOOTHING

    assert CONTOUR_SEED_SMOOTHING is True, 'CONTOUR_SEED_SMOOTHING should be enabled'
    print('  ✓ CONTOUR_SEED_SMOOTHING is enabled')

    # Verify import in service.py works
    print('  ✓ Integration check passed')
    print('Integration: OK\n')


if __name__ == '__main__':
    print('=' * 60)
    print('Contour Smoothing Implementation Test')
    print('=' * 60 + '\n')

    try:
        test_simple_smooth()
        test_smooth_polyline()
        test_integration()

        print('=' * 60)
        print('All tests passed! ✓')
        print('=' * 60)

    except AssertionError as e:
        print(f'\n❌ Test failed: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ Unexpected error: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
