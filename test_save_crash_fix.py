#!/usr/bin/env python3
"""
Test script to verify that the save crash fix works correctly.
This test verifies that:
1. No QTimer.singleShot is called in _cleanup_save_resources
2. No _delayed_cleanup nested function exists
3. Cleanup is simple and synchronous
4. No wait() calls that could cause "pure virtual method called" errors
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_save_cleanup_no_timer():
    """Test that cleanup doesn't use QTimer or delayed operations."""
    print('Testing save cleanup fix...')

    view_file = Path(__file__).parent / 'src' / 'gui' / 'view.py'
    with open(view_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the _cleanup_save_resources method
    lines = content.split('\n')

    # Find method start
    method_start = None
    for i, line in enumerate(lines):
        if 'def _cleanup_save_resources(self)' in line:
            method_start = i
            break

    if method_start is None:
        print('✗ ERROR: Could not find _cleanup_save_resources method')
        return False

    # Find method end (next def or end of indentation)
    method_end = None
    for i in range(method_start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(' '):
            method_end = i
            break
        if line.strip().startswith('def ') and i > method_start + 5:
            method_end = i
            break

    if method_end is None:
        method_end = len(lines)

    method_content = '\n'.join(lines[method_start:method_end])

    print(
        f'\n_cleanup_save_resources method (lines {method_start + 1} to {method_end}):'
    )
    print('=' * 60)
    print(method_content)
    print('=' * 60)

    # Check for problematic patterns
    errors = []

    if 'QTimer.singleShot' in method_content:
        errors.append(
            '✗ ERROR: QTimer.singleShot found in cleanup method (causes thread issues)'
        )
    else:
        print('✓ No QTimer.singleShot in cleanup method')

    if '_delayed_cleanup' in method_content or 'def _delayed_cleanup' in method_content:
        errors.append('✗ ERROR: _delayed_cleanup function found (should be removed)')
    else:
        print('✓ No _delayed_cleanup nested function')

    if '.wait(' in method_content:
        errors.append(
            '✗ ERROR: wait() call found (can cause pure virtual method errors)'
        )
    else:
        print('✓ No wait() calls in cleanup')

    # Check for correct patterns
    if 'deleteLater()' in method_content:
        print('✓ Uses deleteLater() for proper Qt object cleanup')
    else:
        errors.append('⚠ WARNING: deleteLater() not found (should be present)')

    if 'delattr(self._save_worker' in method_content and 'image' in method_content:
        print('✓ Removes image attribute to free memory')
    else:
        errors.append('⚠ WARNING: Image attribute cleanup not found')

    # Check method is simple (should be short without nested functions)
    method_lines = [
        l
        for l in lines[method_start:method_end]
        if l.strip() and not l.strip().startswith('#')
    ]
    if len(method_lines) > 20:
        errors.append(
            f'⚠ WARNING: Method seems complex ({len(method_lines)} non-empty lines)'
        )
    else:
        print(f'✓ Method is simple and concise ({len(method_lines)} non-empty lines)')

    if errors:
        print('\n' + '\n'.join(errors))
        return False

    print('\n✅ All checks passed!')
    print('\nFix summary:')
    print(
        "- Removed QTimer.singleShot that caused 'Timers cannot be started from another thread'"
    )
    print('- Removed _delayed_cleanup that accessed potentially deleted Qt objects')
    print('- Simplified cleanup to use only deleteLater() without wait()')
    print("- Prevents 'pure virtual method called' error and SIGABRT crash")

    return True


if __name__ == '__main__':
    try:
        success = test_save_cleanup_no_timer()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'\n❌ Test failed with exception: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
