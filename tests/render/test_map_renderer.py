"""Tests for render.map_renderer module."""

from PIL import Image

from render.map_renderer import MapRenderer, blend_images, composite_with_mask


class TestBlendImages:
    """Tests for blend_images."""

    def test_blend_resizes_and_converts(self):
        """Overlay should be resized and modes converted to RGBA."""
        base = Image.new("RGB", (20, 20), color=(255, 0, 0))
        overlay = Image.new("L", (10, 10), color=128)

        result = blend_images(base, overlay, alpha=0.25)

        assert result.size == (20, 20)
        assert result.mode == "RGBA"


class TestCompositeWithMask:
    """Tests for composite_with_mask."""

    def test_composite_resizes_overlay_and_mask(self):
        """Overlay and mask should be resized to base size."""
        base = Image.new("RGB", (30, 20), color=(10, 10, 10))
        overlay = Image.new("RGB", (10, 10), color=(200, 200, 0))
        mask = Image.new("L", (10, 10), color=128)

        result = composite_with_mask(base, overlay, mask)

        assert result.size == (30, 20)
        assert result.mode == "RGB"


class TestMapRenderer:
    """Tests for MapRenderer class."""

    def test_render_empty_layers_returns_white(self):
        """Empty renderer should return white background image."""
        renderer = MapRenderer(12, 8)

        result = renderer.render()

        assert result.size == (12, 8)
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_add_layer_resizes(self):
        """add_layer should resize images to renderer size."""
        renderer = MapRenderer(16, 10)
        layer = Image.new("RGB", (8, 5), color=(100, 0, 0))

        renderer.add_layer(layer)

        assert renderer.layers[0].size == (16, 10)

    def test_render_composites_layers(self):
        """Renderer should composite layers and return RGB image."""
        renderer = MapRenderer(10, 10)
        base = Image.new("RGBA", (10, 10), color=(255, 0, 0, 255))
        overlay = Image.new("RGBA", (10, 10), color=(0, 255, 0, 128))

        renderer.add_layer(base)
        renderer.add_layer(overlay)

        result = renderer.render()

        assert result.mode == "RGB"
        assert result.size == (10, 10)
        assert result.getpixel((0, 0)) != (255, 0, 0)

    def test_clear_layers(self):
        """clear should reset stored layers."""
        renderer = MapRenderer(5, 5)
        renderer.add_layer(Image.new("RGB", (5, 5)))

        renderer.clear()

        assert renderer.layers == []