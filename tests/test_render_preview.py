
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
from render.compose import compose_final_image, save_image
from preview import publish_preview_image

def test_compose_final_image():
    img = Image.new('RGB', (100, 100), color='white')
    # No rotation
    res = compose_final_image(img)
    assert res is img
    
    # With rotation
    res_rot = compose_final_image(img, rotate_deg=90)
    assert res_rot is not img
    assert res_rot.size == (100, 100)

@patch('render.compose._save_jpeg')
def test_save_image(mock_save, tmp_path):
    img = Image.new('RGB', (10, 10))
    path = tmp_path / "test.jpg"
    
    # With explicit kwargs
    save_image(img, path, save_kwargs={'quality': 90})
    mock_save.assert_called_once()
    
    # Without kwargs (triggers fallback)
    save_image(img, path)
    assert mock_save.call_count == 2

@patch('preview._publish_preview_image')
def test_publish_preview_image(mock_pub):
    img = MagicMock()
    publish_preview_image(img)
    mock_pub.assert_called_once_with(img)
