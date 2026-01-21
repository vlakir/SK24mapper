
import pytest
from pathlib import Path
from PIL import Image
from geometry import tile_overlap_rect_common
from image_io import build_save_kwargs, save_jpeg
from domen import MapSettings

def test_tile_overlap_rect_common():
    # Tile at (0,0), size 256. Crop rect (10, 10, 100, 100) -> overlap should be (10, 10, 110, 110)
    # Note: crop_rect is (x, y, w, h) in tile_overlap_rect_common
    res = tile_overlap_rect_common(0, 0, (10, 10, 100, 100), 256)
    assert res == (10, 10, 110, 110)
    
    # No overlap
    res = tile_overlap_rect_common(1, 1, (0, 0, 100, 100), 256)
    assert res is None

def test_build_save_kwargs():
    path = Path("test.jpg")
    kwargs = build_save_kwargs(path, quality=80)
    assert kwargs['quality'] == 80
    assert kwargs['format'] == 'JPEG'
    
    # Limits
    kwargs = build_save_kwargs(path, quality=200)
    assert kwargs['quality'] == 100
    kwargs = build_save_kwargs(path, quality=0)
    assert kwargs['quality'] == 10

def test_save_jpeg(tmp_path):
    img = Image.new('RGB', (10, 10), color='red')
    out_path = tmp_path / "test.jpg"
    kwargs = build_save_kwargs(out_path)
    save_jpeg(img, out_path, kwargs)
    assert out_path.exists()
    
    # Test with non-RGB
    img_l = Image.new('L', (10, 10), color=128)
    save_jpeg(img_l, out_path, kwargs)
    assert out_path.exists()

def test_map_settings_pydantic():
    # Valid
    s = MapSettings(
        from_x_high=54, from_y_high=74, to_x_high=54, to_y_high=74,
        from_x_low=14, from_y_low=43, to_x_low=14, to_y_low=43,
        output_path="test.jpg",
        grid_width_m=5.0,
        grid_font_size_m=100.0,
        grid_text_margin_m=50.0,
        grid_label_bg_padding_m=10.0,
        mask_opacity=0.35
    )
    assert s.from_x_high == 54
    
    # Invalid
    with pytest.raises(Exception):
        MapSettings(from_x_high="invalid")
