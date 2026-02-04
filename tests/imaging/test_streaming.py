"""Tests for streaming image processing module."""

import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from imaging.streaming import (
    StreamingImage,
    assemble_tiles_streaming,
    crop_streaming,
    rotate_streaming,
    save_streaming_image,
    save_streaming_jpeg,
    save_streaming_tiff,
)


class TestStreamingImage:
    """Tests for StreamingImage class."""

    def test_create_and_close(self):
        """Test creating and closing a StreamingImage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            img = StreamingImage(100, 100, temp_dir=temp_dir)
            assert img.width == 100
            assert img.height == 100
            assert img.size == (100, 100)
            assert os.path.exists(img.mmap_path)
            
            img.close()
            # After close, temp file should be removed
            assert not os.path.exists(img.mmap_path)

    def test_context_manager(self):
        """Test StreamingImage as context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with StreamingImage(50, 50, temp_dir=temp_dir) as img:
                assert img.width == 50
                assert img.height == 50
                mmap_path = img.mmap_path
                assert os.path.exists(mmap_path)
            
            # After exiting context, file should be removed
            assert not os.path.exists(mmap_path)

    def test_paste_tile_numpy(self):
        """Test pasting a numpy array tile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with StreamingImage(100, 100, temp_dir=temp_dir) as img:
                # Create a red tile
                tile = np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)
                img.paste_tile(tile, 10, 10)
                
                # Check that the tile was pasted
                strip = img.get_strip(10, 50)
                assert strip[0, 10, 0] == 255  # Red channel
                assert strip[0, 10, 1] == 0    # Green channel
                assert strip[0, 10, 2] == 0    # Blue channel

    def test_paste_tile_bytes(self):
        """Test pasting a JPEG bytes tile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with StreamingImage(100, 100, temp_dir=temp_dir) as img:
                # Create a PIL image and convert to bytes
                pil_img = Image.new('RGB', (50, 50), color=(0, 255, 0))
                import io
                buffer = io.BytesIO()
                pil_img.save(buffer, format='JPEG')
                tile_bytes = buffer.getvalue()
                
                img.paste_tile(tile_bytes, 25, 25)
                
                # Check that the tile was pasted (JPEG compression may alter values slightly)
                strip = img.get_strip(25, 50)
                assert strip[0, 25, 1] > 200  # Green channel should be high

    def test_get_strip(self):
        """Test getting a strip from the image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(128, 128, 128)) as img:
                strip = img.get_strip(0, 10)
                assert strip.shape == (10, 100, 3)
                assert np.all(strip == 128)

    def test_set_strip(self):
        """Test setting a strip in the image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with StreamingImage(100, 100, temp_dir=temp_dir) as img:
                # Create a strip with specific color
                strip = np.full((20, 100, 3), [100, 150, 200], dtype=np.uint8)
                img.set_strip(30, strip)
                
                # Read it back
                read_strip = img.get_strip(30, 20)
                assert np.array_equal(read_strip, strip)

    def test_fill_color(self):
        """Test that fill_color is applied correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with StreamingImage(50, 50, temp_dir=temp_dir, fill_color=(255, 128, 64)) as img:
                strip = img.get_strip(0, 50)
                assert strip[0, 0, 0] == 255
                assert strip[0, 0, 1] == 128
                assert strip[0, 0, 2] == 64


class TestAssembleTilesStreaming:
    """Tests for assemble_tiles_streaming function."""

    def test_single_tile(self):
        """Test assembling a single tile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a single tile
            tile = np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8)
            
            result = assemble_tiles_streaming(
                tile_data_list=[tile],
                tiles_x=1,
                tiles_y=1,
                eff_tile_px=256,
                crop_rect=(0, 0, 256, 256),
                temp_dir=temp_dir,
            )
            
            assert result.size == (256, 256)
            strip = result.get_strip(0, 256)
            assert strip[0, 0, 0] == 255
            result.close()

    def test_multiple_tiles(self):
        """Test assembling multiple tiles."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create 2x2 tiles with different colors
            tiles = [
                np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8),  # Red
                np.full((256, 256, 3), [0, 255, 0], dtype=np.uint8),  # Green
                np.full((256, 256, 3), [0, 0, 255], dtype=np.uint8),  # Blue
                np.full((256, 256, 3), [255, 255, 0], dtype=np.uint8),  # Yellow
            ]
            
            result = assemble_tiles_streaming(
                tile_data_list=tiles,
                tiles_x=2,
                tiles_y=2,
                eff_tile_px=256,
                crop_rect=(0, 0, 512, 512),
                temp_dir=temp_dir,
            )
            
            assert result.size == (512, 512)
            
            # Check corners
            strip_top = result.get_strip(0, 1)
            assert strip_top[0, 0, 0] == 255  # Red top-left
            assert strip_top[0, 256, 1] == 255  # Green top-right
            
            strip_bottom = result.get_strip(256, 1)
            assert strip_bottom[0, 0, 2] == 255  # Blue bottom-left
            assert strip_bottom[0, 256, 0] == 255  # Yellow bottom-right (red channel)
            
            result.close()

    def test_with_crop(self):
        """Test assembling with crop rect."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tile = np.full((256, 256, 3), [100, 100, 100], dtype=np.uint8)
            
            result = assemble_tiles_streaming(
                tile_data_list=[tile],
                tiles_x=1,
                tiles_y=1,
                eff_tile_px=256,
                crop_rect=(50, 50, 100, 100),  # Crop to center 100x100
                temp_dir=temp_dir,
            )
            
            assert result.size == (100, 100)
            result.close()


class TestRotateStreaming:
    """Tests for rotate_streaming function."""

    def test_zero_angle(self):
        """Test rotation with zero angle returns copy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(255, 0, 0))
            
            result = rotate_streaming(src, 0.0, temp_dir=temp_dir)
            
            assert result.size == (100, 100)
            strip = result.get_strip(0, 100)
            assert strip[0, 0, 0] == 255
            result.close()

    def test_90_degrees(self):
        """Test rotation by 90 degrees."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(0, 0, 0))
            # Draw a red line at top
            red_strip = np.full((10, 100, 3), [255, 0, 0], dtype=np.uint8)
            src.set_strip(0, red_strip)
            
            result = rotate_streaming(src, 90.0, fill=(255, 255, 255), temp_dir=temp_dir)
            
            assert result.size == (100, 100)
            # After 90 degree rotation, the red line should be on the left side
            result.close()

    def test_arbitrary_angle(self):
        """Test rotation by arbitrary angle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(128, 128, 128))
            
            result = rotate_streaming(src, 45.0, fill=(255, 255, 255), temp_dir=temp_dir)
            
            assert result.size == (100, 100)
            result.close()


class TestCropStreaming:
    """Tests for crop_streaming function."""

    def test_center_crop(self):
        """Test center cropping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = StreamingImage(200, 200, temp_dir=temp_dir, fill_color=(100, 100, 100))
            
            result = crop_streaming(src, 100, 100, temp_dir=temp_dir)
            
            assert result.size == (100, 100)
            strip = result.get_strip(0, 100)
            assert strip[0, 0, 0] == 100
            result.close()

    def test_same_size_crop(self):
        """Test cropping to same size returns copy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(50, 50, 50))
            
            result = crop_streaming(src, 100, 100, temp_dir=temp_dir)
            
            assert result.size == (100, 100)
            result.close()


class TestSaveStreaming:
    """Tests for save functions."""

    def test_save_tiff(self):
        """Test saving as TIFF."""
        with tempfile.TemporaryDirectory() as temp_dir:
            img = StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(255, 128, 64))
            output_path = os.path.join(temp_dir, 'test.tif')
            
            save_streaming_tiff(img, output_path)
            
            assert os.path.exists(output_path)
            # Verify the file can be read
            loaded = Image.open(output_path)
            assert loaded.size == (100, 100)
            loaded.close()
            img.close()

    def test_save_jpeg(self):
        """Test saving as JPEG."""
        with tempfile.TemporaryDirectory() as temp_dir:
            img = StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(255, 128, 64))
            output_path = os.path.join(temp_dir, 'test.jpg')
            
            save_streaming_jpeg(img, output_path)
            
            assert os.path.exists(output_path)
            loaded = Image.open(output_path)
            assert loaded.size == (100, 100)
            loaded.close()
            img.close()

    def test_save_streaming_image_auto_format(self):
        """Test save_streaming_image with automatic format detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            img = StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(200, 100, 50))
            
            # Test JPEG
            jpg_path = os.path.join(temp_dir, 'test.jpg')
            result_path = save_streaming_image(img, jpg_path)
            assert result_path == jpg_path
            assert os.path.exists(jpg_path)
            
            # Test TIFF
            img2 = StreamingImage(100, 100, temp_dir=temp_dir, fill_color=(200, 100, 50))
            tif_path = os.path.join(temp_dir, 'test.tif')
            result_path = save_streaming_image(img2, tif_path)
            assert result_path == tif_path
            assert os.path.exists(tif_path)
            
            img.close()
            img2.close()
