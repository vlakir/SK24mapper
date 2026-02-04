"""Tests for imaging.pyramid module."""

import numpy as np
import pytest
from PIL import Image

from imaging.pyramid import (
    ImagePyramid,
    build_pyramid_from_pil,
    build_pyramid_from_streaming,
    should_use_pyramid,
)
from imaging.streaming import StreamingImage


class TestImagePyramid:
    """Tests for ImagePyramid class."""

    def test_init(self):
        """Test pyramid initialization."""
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (50, 50), color='red')
        
        pyramid = ImagePyramid(
            levels=[img1, img2],
            original_size=(200, 200),
            scale_to_original=0.5,
        )
        
        assert pyramid.num_levels == 2
        assert pyramid.original_size == (200, 200)
        assert pyramid.scale_to_original == 0.5
        assert pyramid.base_level is img1
        assert pyramid.overview_level is img2
        
        pyramid.close()

    def test_empty_pyramid(self):
        """Test pyramid with no levels."""
        pyramid = ImagePyramid(
            levels=[],
            original_size=(100, 100),
            scale_to_original=1.0,
        )
        
        assert pyramid.num_levels == 0
        assert pyramid.base_level is None
        assert pyramid.overview_level is None

    def test_get_level_for_scale(self):
        """Test getting appropriate level for view scale."""
        img1 = Image.new('RGB', (1000, 1000), color='red')
        img2 = Image.new('RGB', (500, 500), color='green')
        img3 = Image.new('RGB', (250, 250), color='blue')
        
        pyramid = ImagePyramid(
            levels=[img1, img2, img3],
            original_size=(2000, 2000),
            scale_to_original=0.5,
        )
        
        # At high zoom, should return most detailed level
        level, scale = pyramid.get_level_for_scale(2.0)
        assert level is img1
        
        # At low zoom, should return less detailed level
        level, scale = pyramid.get_level_for_scale(0.1)
        assert level is img3
        
        pyramid.close()

    def test_close(self):
        """Test closing pyramid releases resources."""
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (50, 50), color='red')
        
        pyramid = ImagePyramid(
            levels=[img1, img2],
            original_size=(200, 200),
            scale_to_original=0.5,
        )
        
        pyramid.close()
        assert len(pyramid.levels) == 0


class TestBuildPyramidFromPil:
    """Tests for build_pyramid_from_pil function."""

    def test_small_image_no_scaling(self):
        """Test that small images are not scaled down."""
        img = Image.new('RGB', (500, 500), color='red')
        
        pyramid = build_pyramid_from_pil(img, max_base_size=2048)
        
        assert pyramid.base_level.size == (500, 500)
        assert pyramid.scale_to_original == 1.0
        assert pyramid.original_size == (500, 500)
        
        pyramid.close()

    def test_large_image_scaling(self):
        """Test that large images are scaled down."""
        img = Image.new('RGB', (4000, 3000), color='blue')
        
        pyramid = build_pyramid_from_pil(img, max_base_size=2048)
        
        # Should be scaled to fit within 2048
        assert max(pyramid.base_level.size) <= 2048
        assert pyramid.scale_to_original < 1.0
        assert pyramid.original_size == (4000, 3000)
        
        pyramid.close()

    def test_multiple_levels_created(self):
        """Test that multiple pyramid levels are created."""
        img = Image.new('RGB', (2000, 2000), color='green')
        
        pyramid = build_pyramid_from_pil(img, max_base_size=2048, min_level_size=256)
        
        # Should have multiple levels
        assert pyramid.num_levels >= 2
        
        # Each level should be smaller than the previous
        for i in range(1, pyramid.num_levels):
            prev_size = pyramid.levels[i - 1].size
            curr_size = pyramid.levels[i].size
            assert curr_size[0] < prev_size[0]
            assert curr_size[1] < prev_size[1]
        
        pyramid.close()

    def test_preserves_aspect_ratio(self):
        """Test that aspect ratio is preserved."""
        img = Image.new('RGB', (4000, 2000), color='yellow')
        
        pyramid = build_pyramid_from_pil(img, max_base_size=1000)
        
        base_w, base_h = pyramid.base_level.size
        original_ratio = 4000 / 2000
        base_ratio = base_w / base_h
        
        assert abs(original_ratio - base_ratio) < 0.01
        
        pyramid.close()


class TestBuildPyramidFromStreaming:
    """Tests for build_pyramid_from_streaming function."""

    def test_small_streaming_image(self):
        """Test pyramid from small StreamingImage."""
        streaming = StreamingImage(500, 400)
        # Fill with some data
        data = np.full((400, 500, 3), [255, 0, 0], dtype=np.uint8)
        streaming.set_strip(0, data)
        
        pyramid = build_pyramid_from_streaming(streaming, max_base_size=2048)
        
        assert pyramid.base_level.size == (500, 400)
        assert pyramid.scale_to_original == 1.0
        
        pyramid.close()
        streaming.close()

    def test_large_streaming_image(self):
        """Test pyramid from large StreamingImage."""
        streaming = StreamingImage(4000, 3000)
        
        pyramid = build_pyramid_from_streaming(streaming, max_base_size=2048)
        
        # Should be scaled down
        assert max(pyramid.base_level.size) <= 2048
        assert pyramid.scale_to_original < 1.0
        assert pyramid.original_size == (4000, 3000)
        
        pyramid.close()
        streaming.close()

    def test_streaming_data_preserved(self):
        """Test that image data is correctly transferred to pyramid."""
        streaming = StreamingImage(100, 100)
        # Fill with red
        red_data = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        streaming.set_strip(0, red_data)
        
        pyramid = build_pyramid_from_streaming(streaming, max_base_size=2048)
        
        # Check that base level has red pixels
        base_arr = np.array(pyramid.base_level)
        assert base_arr[50, 50, 0] == 255  # Red channel
        assert base_arr[50, 50, 1] == 0    # Green channel
        assert base_arr[50, 50, 2] == 0    # Blue channel
        
        pyramid.close()
        streaming.close()


class TestShouldUsePyramid:
    """Tests for should_use_pyramid function."""

    def test_small_image_no_pyramid(self):
        """Test that small images don't need pyramid."""
        assert should_use_pyramid(1000, 1000) is False
        assert should_use_pyramid(500, 500) is False

    def test_large_image_needs_pyramid(self):
        """Test that large images need pyramid."""
        assert should_use_pyramid(5000, 5000) is True
        assert should_use_pyramid(10000, 10000) is True

    def test_threshold_boundary(self):
        """Test behavior at threshold boundary."""
        # Just under threshold (4M pixels)
        assert should_use_pyramid(1999, 1999) is False
        
        # Just over threshold
        assert should_use_pyramid(2001, 2001) is True
