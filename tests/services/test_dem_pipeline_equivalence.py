"""
Тесты пайплайн-эквивалентности DEM — главная защита от регрессий.

Группа 3: first-build vs recompute_coverage_fast,
PIL vs cv2 rotation dimensions, crop_size invariant.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from imaging.transforms import center_crop, rotate_keep_size
from services.radio_horizon import (
    compute_and_colorize_coverage,
    recompute_coverage_fast,
)

# ── Хелперы ──────────────────────────────────────────────────────────────


def make_test_dem(h: int, w: int) -> np.ndarray:
    """DEM с холмом в центре (не плоский, чтобы coverage отличался)."""
    dem = np.full((h, w), 150.0, dtype=np.float32)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    for row in range(h):
        for col in range(w):
            dist = np.sqrt((row - cy) ** 2 + (col - cx) ** 2)
            if dist < r:
                dem[row, col] = 150.0 + 200.0 * (1 - dist / r)
    return dem


def make_topo_base(w: int, h: int) -> Image.Image:
    """RGBA серое изображение — имитация топоосновы."""
    return Image.new('RGBA', (w, h), (128, 128, 128, 255))


def simulate_first_build(
    dem: np.ndarray,
    angle: float,
    target_w: int,
    target_h: int,
    antenna_row: int | None = None,
    antenna_col: int | None = None,
    antenna_height_m: float = 10.0,
    pixel_size_m: float = 10.0,
    grid_step: int = 8,
) -> Image.Image:
    """Имитация first-build пайплайна: coverage → resize(crop_size) → rotate → crop.

    Crop_size = DEM shape (w, h) — размер DEM до даунсэмплинга.
    """
    h, w = dem.shape
    if antenna_row is None:
        antenna_row = h // 2
    if antenna_col is None:
        antenna_col = w // 2

    crop_size = (w, h)  # (width, height)

    # 1. Compute coverage (на DEM)
    coverage = compute_and_colorize_coverage(
        dem=dem,
        antenna_row=antenna_row,
        antenna_col=antenna_col,
        antenna_height_m=antenna_height_m,
        pixel_size_m=pixel_size_m,
        grid_step=grid_step,
        max_height_m=500.0,
    )

    # 2. Resize to crop_size (для non-square DEM может отличаться)
    if coverage.size != crop_size:
        coverage = coverage.resize(crop_size, Image.Resampling.BILINEAR)

    # 3. Rotate
    if abs(angle) > 1e-6:
        coverage = rotate_keep_size(coverage, angle, fill=(0, 0, 0))

    # 4. Center crop
    coverage = center_crop(coverage, target_w, target_h)

    return coverage


# ── Тесты ────────────────────────────────────────────────────────────────


class TestFirstBuildRecomputeSizeMatch:
    """Главный регрессионный тест: first-build vs recompute_coverage_fast."""

    def _run_size_match(self, dem_h: int, dem_w: int, angle: float) -> None:
        dem = make_test_dem(dem_h, dem_w)
        target_w, target_h = dem_w - 20, dem_h - 20
        antenna_row, antenna_col = dem_h // 2, dem_w // 2
        grid_step = 8

        # First build
        first = simulate_first_build(
            dem, angle, target_w, target_h,
            antenna_row=antenna_row, antenna_col=antenna_col,
            grid_step=grid_step,
        )

        # Recompute
        topo = make_topo_base(dem_w, dem_h)
        crop_size = (dem_w, dem_h)
        result = recompute_coverage_fast(
            dem=dem,
            new_antenna_row=antenna_row,
            new_antenna_col=antenna_col,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo,
            overlay_alpha=0.5,
            max_height_m=500.0,
            grid_step=grid_step,
            final_size=(target_w, target_h),
            crop_size=crop_size,
            rotation_deg=angle,
        )

        # recompute_coverage_fast returns (blended, coverage) tuple
        if isinstance(result, tuple):
            blended, coverage = result
            recompute_img = coverage
        else:
            recompute_img = result

        assert first.size == recompute_img.size, (
            f'first={first.size}, recompute={recompute_img.size}'
        )

    def test_sizes_match_with_rotation(self) -> None:
        self._run_size_match(64, 64, 10.0)

    def test_sizes_match_without_rotation(self) -> None:
        self._run_size_match(64, 64, 0.0)

    def test_sizes_match_rectangular(self) -> None:
        self._run_size_match(96, 64, 10.0)

    @pytest.mark.parametrize('angle', [0, 5, 15, 30, 45])
    def test_sizes_match_various_angles(self, angle: float) -> None:
        self._run_size_match(64, 64, angle)


class TestCropSizeInvariant:
    """crop_size = DEM shape до даунсэмплинга."""

    def test_crop_size_equals_original_dem_shape(self) -> None:
        """resize(crop_size) → size == crop_size."""
        dem = make_test_dem(128, 96)
        crop_size = (96, 128)  # (w, h)
        img = Image.new('RGB', (dem.shape[1], dem.shape[0]), (100, 100, 100))
        resized = img.resize(crop_size, Image.Resampling.BILINEAR)
        assert resized.size == crop_size

    def test_crop_size_stored_before_downsample(self) -> None:
        """shape (128, 96) → crop_size = (96, 128) = (w, h)."""
        dem = make_test_dem(128, 96)
        h, w = dem.shape
        crop_size = (w, h)
        assert crop_size == (96, 128)


class TestPILvsCv2RotationDimensions:
    """PIL rotate vs cv2 rotate — одинаковые размеры."""

    def _check_dimensions(self, w: int, h: int, angle: float) -> None:
        # PIL path (rotate_keep_size)
        pil_img = Image.new('RGB', (w, h), (128, 128, 128))
        pil_rot = rotate_keep_size(pil_img, angle)
        pil_w, pil_h = pil_rot.size

        # cv2 path (как в _load_dem_for_cursor)
        arr = np.zeros((h, w), dtype=np.float32)
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cv2_rot = cv2.warpAffine(
            arr, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        cv2_h, cv2_w = cv2_rot.shape

        assert (pil_w, pil_h) == (cv2_w, cv2_h), (
            f'PIL=({pil_w},{pil_h}), cv2=({cv2_w},{cv2_h})'
        )

    def test_same_dimensions_square(self) -> None:
        self._check_dimensions(64, 64, 30.0)

    def test_same_dimensions_rectangular(self) -> None:
        self._check_dimensions(100, 60, 30.0)

    @pytest.mark.parametrize('angle', [0, 10, 30, 45, 90, -15])
    def test_same_dimensions_various_angles(self, angle: float) -> None:
        self._check_dimensions(80, 60, angle)


class TestDemGridShapeAfterRotateCrop:
    """DEM → rotate → crop → shape == (target_h, target_w)."""

    def _check_shape(self, src_size: int, target_w: int, target_h: int, angle: float) -> None:
        dem = np.zeros((src_size, src_size), dtype=np.float32)
        h, w = dem.shape

        # cv2 rotate (как _load_dem_for_cursor)
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            dem, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # numpy center crop
        rh, rw = rotated.shape
        left = (rw - target_w) // 2
        top = (rh - target_h) // 2
        left = max(0, left)
        top = max(0, top)
        cropped = rotated[top : top + target_h, left : left + target_w]

        assert cropped.shape == (target_h, target_w), (
            f'Cropped shape={cropped.shape}, expected=({target_h}, {target_w})'
        )

    def test_shape_matches_target(self) -> None:
        self._check_shape(128, 100, 100, 15.0)

    def test_shape_rectangular_target(self) -> None:
        self._check_shape(128, 80, 120, 15.0)

    def test_shape_large_rotation(self) -> None:
        self._check_shape(128, 100, 100, 45.0)


# ── Покрытие ветвей recompute_coverage_fast ──────────────────────────────


class TestRecomputeRotationResizeBranches:
    """Покрывает строки 832, 834: resize когда DEM < crop_size в rotation path."""

    def test_downsampled_dem_triggers_resize(self) -> None:
        """DEM 32×32 + crop_size=(64,64) → result/topo_base resize в rotation path."""
        from services.radio_horizon import downsample_dem

        original = make_test_dem(64, 64)
        dem = downsample_dem(original, 2)  # 32×32
        assert dem.shape == (32, 32)

        crop_size = (64, 64)
        final_size = (44, 44)
        topo = make_topo_base(32, 32)  # Тоже 32×32 — отличается от crop_size

        result = recompute_coverage_fast(
            dem=dem,
            new_antenna_row=16,
            new_antenna_col=16,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo,
            overlay_alpha=0.5,
            max_height_m=500.0,
            grid_step=8,
            final_size=final_size,
            crop_size=crop_size,
            rotation_deg=10.0,
        )

        blended, coverage = result
        assert blended.size == final_size
        assert coverage.size == final_size


class TestRecomputeCropPathResizeBranches:
    """Покрывает строки 876, 881-883: resize в crop path (без rotation)."""

    def test_crop_path_with_size_mismatch(self) -> None:
        """DEM 32×32 + crop_size=(64,64) → crop + resize в non-rotation path.

        Покрывает line 876 (result resize) и line 881 (topo_base resize).
        """
        from services.radio_horizon import downsample_dem

        original = make_test_dem(64, 64)
        dem = downsample_dem(original, 2)  # 32×32

        crop_size = (64, 64)
        final_size = (44, 44)
        topo = make_topo_base(32, 32)  # Same as DEM → enters if branch

        result = recompute_coverage_fast(
            dem=dem,
            new_antenna_row=16,
            new_antenna_col=16,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo,
            overlay_alpha=0.5,
            max_height_m=500.0,
            grid_step=8,
            final_size=final_size,
            crop_size=crop_size,
            rotation_deg=0.0,  # Без поворота → crop path
        )

        blended, coverage = result
        assert blended.size == final_size

    def test_crop_path_topo_different_size(self) -> None:
        """topo_base размером != DEM → elif ветка (line 882-883)."""
        from services.radio_horizon import downsample_dem

        original = make_test_dem(64, 64)
        dem = downsample_dem(original, 2)  # 32×32

        crop_size = (64, 64)
        final_size = (44, 44)
        # topo_base отличается от DEM size (32×32) → elif branch
        topo = make_topo_base(50, 50)

        result = recompute_coverage_fast(
            dem=dem,
            new_antenna_row=16,
            new_antenna_col=16,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo,
            overlay_alpha=0.5,
            max_height_m=500.0,
            grid_step=8,
            final_size=final_size,
            crop_size=crop_size,
            rotation_deg=0.0,
        )

        blended, coverage = result
        assert blended.size == final_size


class TestSectorWithTargetHeightMin:
    """Покрывает строки 711-718: sector_enabled + target_height_min_m > 0."""

    def test_sector_with_height_min_produces_image(self) -> None:
        """compute_and_colorize_coverage с sector + target_height_min_m > 0."""
        dem = make_test_dem(64, 64)
        result = compute_and_colorize_coverage(
            dem=dem,
            antenna_row=32,
            antenna_col=32,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=8,
            max_height_m=500.0,
            sector_enabled=True,
            radar_azimuth_deg=0.0,
            radar_sector_width_deg=90.0,
            elevation_min_deg=0.0,
            elevation_max_deg=45.0,
            max_range_m=500.0,
            target_height_min_m=50.0,
        )
        assert isinstance(result, Image.Image)
        assert result.size == (64, 64)

    def test_sector_height_min_affects_colorization(self) -> None:
        """Шкала [h_min, h_max] отличается от шкалы [0, h_max]."""
        dem = make_test_dem(64, 64)
        kwargs = dict(
            dem=dem,
            antenna_row=32,
            antenna_col=32,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=8,
            max_height_m=500.0,
            sector_enabled=True,
            radar_azimuth_deg=0.0,
            radar_sector_width_deg=360.0,
            max_range_m=1000.0,
        )
        without_min = compute_and_colorize_coverage(**kwargs, target_height_min_m=0.0)
        with_min = compute_and_colorize_coverage(**kwargs, target_height_min_m=100.0)

        arr_without = np.array(without_min)
        arr_with = np.array(with_min)
        # Разные шкалы → разные цвета (хотя бы немного)
        assert not np.array_equal(arr_without, arr_with)


