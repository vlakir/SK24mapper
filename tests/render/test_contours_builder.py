import numpy as np
import pytest
from render.contours_builder import _refine_contour_subpixel, _build_contours_for_level, build_seed_polylines

def test_refine_contour_subpixel_basic():
    dem = np.array([
        [0, 0, 0],
        [0, 10, 0],
        [0, 0, 0]
    ], dtype=np.float32)
    
    # Контур вокруг центральной точки (1, 1) на уровне 5
    contour = np.array([[1, 0], [2, 1], [1, 2], [0, 1]], dtype=np.float32)
    level = 5.0
    
    refined = _refine_contour_subpixel(contour, dem, level)
    assert len(refined) == len(contour)
    # Проверяем, что точки сместились к центру
    for pt in refined:
        assert 0 <= pt[0] <= 2
        assert 0 <= pt[1] <= 2

def test_refine_contour_subpixel_empty():
    dem = np.zeros((3, 3), dtype=np.float32)
    contour = np.array([[1, 1]], dtype=np.float32) # Less than 2 points
    refined = _refine_contour_subpixel(contour, dem, 5.0)
    assert refined == []

def test_refine_contour_subpixel_out_of_bounds():
    dem = np.zeros((3, 3), dtype=np.float32)
    contour = np.array([[2, 2], [3, 3]], dtype=np.float32)
    refined = _refine_contour_subpixel(contour, dem, 5.0)
    assert len(refined) == 2
    assert refined[0] == (2.0, 2.0) # Should be appended as is due to boundary check

def test_build_contours_for_level_basic():
    dem = np.zeros((10, 10), dtype=np.float32)
    dem[2:7, 2:7] = 10.0
    level = 5.0
    
    polylines = _build_contours_for_level(dem, level, use_subpixel=False)
    assert len(polylines) > 0
    for poly in polylines:
        assert len(poly) >= 3

def test_build_contours_for_level_subpixel():
    dem = np.zeros((10, 10), dtype=np.float32)
    dem[2:7, 2:7] = 10.0
    level = 5.0
    
    polylines = _build_contours_for_level(dem, level, use_subpixel=True)
    assert len(polylines) > 0
    for poly in polylines:
        assert len(poly) >= 3

def test_build_seed_polylines_sequential():
    dem = np.zeros((10, 10), dtype=np.float32)
    dem[3:6, 3:6] = 10.0
    levels = [5.0]
    
    result = build_seed_polylines(dem, levels, 10, 10)
    assert 0 in result
    assert len(result[0]) > 0

def test_build_seed_polylines_parallel():
    dem = np.zeros((20, 20), dtype=np.float32)
    dem[5:15, 5:15] = 10.0
    levels = [2.0, 5.0, 8.0]
    
    # We need to make sure parallel branch is taken. 
    # num_workers = min(CONTOUR_PARALLEL_WORKERS, max(1, os.cpu_count() or 1), len(levels))
    # Assuming CPU count > 1 and CONTOUR_PARALLEL_WORKERS > 1
    result = build_seed_polylines(dem, levels, 20, 20)
    assert len(result) == 3
    for i in range(3):
        assert len(result[i]) > 0

def test_build_seed_polylines_list_input():
    dem_list = [[0.0] * 5 for _ in range(5)]
    dem_list[2][2] = 10.0
    levels = [5.0]
    result = build_seed_polylines(dem_list, levels, 5, 5)
    assert 0 in result
