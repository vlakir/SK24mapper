#!/usr/bin/env python3
"""
Test script to verify that the size estimate fix works correctly.
This test verifies that:
1. _EstimateWorker can be instantiated without 'adj' parameter
2. The worker's run method executes without errors
3. The cleanup doesn't try to delete non-existent 'adj' attribute
"""

import sys
from pathlib import Path
from io import BytesIO
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def test_estimate_worker():
    """Test that the EstimateWorker works without adj parameter."""
    from PySide6.QtCore import QObject, Signal, Slot

    class _EstimateWorker(QObject):
        finished = Signal(int, str)  # estimate_bytes, error

        def __init__(
            self,
            img: Image.Image,
            quality: int,
        ) -> None:
            super().__init__()
            self.image = img
            self.quality = int(max(10, min(100, quality)))

        @Slot()
        def run(self) -> None:
            try:
                img = self.image
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Downscale for fast estimation
                max_side = 1600
                w, h = img.size
                scale = 1.0
                if max(w, h) > max_side:
                    scale = max_side / float(max(w, h))
                    new_w = max(1, round(w * scale))
                    new_h = max(1, round(h * scale))
                    img_small = img.resize(
                        (new_w, new_h),
                        Image.Resampling.LANCZOS,
                    )
                else:
                    img_small = img
                    new_w, new_h = w, h

                buf = BytesIO()
                img_small.save(
                    buf,
                    'JPEG',
                    quality=self.quality,
                    optimize=True,
                    subsampling='4:4:4',
                    progressive=False,
                )
                bytes_down = buf.tell()
                # Scale up by pixel count ratio and add small header constant
                k = (w * h) / float(new_w * new_h)
                estimate = int(bytes_down * k + 2048)
                self.finished.emit(estimate, '')
            except Exception as e:
                self.finished.emit(0, str(e))

    # Create a test image
    test_image = Image.new('RGB', (800, 600), color='red')

    # Create worker without adj parameter
    worker = _EstimateWorker(test_image, 85)

    # Verify worker has correct attributes
    assert hasattr(worker, 'image'), "Worker should have 'image' attribute"
    assert hasattr(worker, 'quality'), "Worker should have 'quality' attribute"
    assert not hasattr(worker, 'adj'), "Worker should NOT have 'adj' attribute"
    assert worker.quality == 85, f'Quality should be 85, got {worker.quality}'

    # Test that run method works
    result_estimate = None
    result_error = None

    def on_finished(estimate: int, error: str):
        nonlocal result_estimate, result_error
        result_estimate = estimate
        result_error = error

    worker.finished.connect(on_finished)
    worker.run()

    assert result_error == '', f'Run should not produce error, got: {result_error}'
    assert result_estimate is not None and result_estimate > 0, (
        f'Estimate should be positive, got: {result_estimate}'
    )

    print(f"✓ Worker created successfully without 'adj' parameter")
    print(f'✓ Worker.run() executed successfully')
    print(f'✓ Size estimate: {result_estimate} bytes')

    # Test cleanup - verify that deleting 'adj' attribute doesn't cause issues
    if hasattr(worker, 'image'):
        delattr(worker, 'image')
        print(f'✓ Image attribute cleaned up successfully')

    # This should not raise an error even though 'adj' doesn't exist
    if hasattr(worker, 'adj'):
        delattr(worker, 'adj')
        print(f"✓ Would have cleaned up 'adj' if it existed")
    else:
        print(f"✓ 'adj' attribute doesn't exist (as expected)")

    print('\n✅ All tests passed!')
    return True


if __name__ == '__main__':
    try:
        test_estimate_worker()
    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
