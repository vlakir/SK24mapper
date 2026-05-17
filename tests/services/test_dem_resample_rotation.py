"""
Тесты ресэмплинга и поворота DEM — чистые функции, round-trip, пиксельное выравнивание.

Группа 1: Инварианты чистых функций (downsample, rotate, crop)
Группа 2: Round-trip (downsample→upsample, rotate→inverse)
Группа 4: Пиксельное выравнивание (peak tracking, PIL vs cv2 path)
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from imaging.transforms import center_crop, rotate_keep_size
from services.radio_horizon import compute_downsample_factor, downsample_dem


# ── Хелперы ──────────────────────────────────────────────────────────────


def make_flat_dem(h: int, w: int, elevation: float = 200.0) -> np.ndarray:
    """Плоский DEM с постоянной высотой."""
    return np.full((h, w), elevation, dtype=np.float32)


def make_peak_dem(
    h: int,
    w: int,
    background: float = 100.0,
    peak: float = 999.0,
    peak_row: int | None = None,
    peak_col: int | None = None,
) -> np.ndarray:
    """DEM с единственным пиком."""
    dem = np.full((h, w), background, dtype=np.float32)
    if peak_row is None:
        peak_row = h // 2
    if peak_col is None:
        peak_col = w // 2
    dem[peak_row, peak_col] = peak
    return dem


def make_gradient_dem(h: int, w: int, lo: float = 0.0, hi: float = 1000.0) -> np.ndarray:
    """Линейный градиент (по строкам)."""
    row_vals = np.linspace(lo, hi, h, dtype=np.float32)
    return np.broadcast_to(row_vals[:, np.newaxis], (h, w)).copy()


# ── Группа 1: Чистые функции ────────────────────────────────────────────


class TestDownsampleDemInvariants:
    """Инварианты downsample_dem."""

    def test_preserves_mean_elevation_factor_2(self) -> None:
        dem = make_gradient_dem(64, 64, 100, 500)
        result = downsample_dem(dem, 2)
        assert abs(result.mean() - dem.mean()) < 1.0

    def test_preserves_mean_elevation_factor_4(self) -> None:
        dem = make_gradient_dem(64, 64, 100, 500)
        result = downsample_dem(dem, 4)
        assert abs(result.mean() - dem.mean()) < 1.0

    def test_output_within_original_range(self) -> None:
        dem = make_gradient_dem(64, 64, 50, 800)
        result = downsample_dem(dem, 2)
        assert result.min() >= dem.min()
        assert result.max() <= dem.max()

    def test_output_within_original_range_factor_4(self) -> None:
        dem = make_gradient_dem(64, 64, 50, 800)
        result = downsample_dem(dem, 4)
        assert result.min() >= dem.min()
        assert result.max() <= dem.max()

    def test_non_multiple_crop_preserves_range(self) -> None:
        """DEM 65×67, factor=4 → обрезается до кратного, значения в диапазоне."""
        dem = make_gradient_dem(65, 67, 10, 900)
        result = downsample_dem(dem, 4)
        assert result.shape == (16, 16)
        assert result.min() >= dem.min()
        assert result.max() <= dem.max()

    def test_uniform_dem_stays_uniform(self) -> None:
        dem = make_flat_dem(64, 64, 200.0)
        result = downsample_dem(dem, 2)
        np.testing.assert_allclose(result, 200.0)


class TestComputeDownsampleFactor:
    """Инварианты compute_downsample_factor."""

    @pytest.mark.parametrize(
        ('h', 'w'),
        [(100, 100), (500, 500), (1000, 2000), (4096, 4096), (10000, 10000)],
    )
    def test_always_returns_power_of_two(self, h: int, w: int) -> None:
        factor = compute_downsample_factor(h, w, max_pixels=1_000_000)
        assert factor >= 1
        # factor — степень двойки
        assert factor & (factor - 1) == 0, f'factor={factor} не степень двойки'

    def test_result_fits_under_limit(self) -> None:
        factor = compute_downsample_factor(10000, 10000, max_pixels=4_000_000)
        reduced = (10000 // factor) * (10000 // factor)
        assert reduced <= 4_000_000

    def test_factor_1_when_under_limit(self) -> None:
        factor = compute_downsample_factor(100, 100, max_pixels=1_000_000)
        assert factor == 1


class TestRotateKeepSizeInvariants:
    """Инварианты rotate_keep_size."""

    @pytest.mark.parametrize('angle', [5, 30, 45, 90, 180, -45])
    def test_preserves_center_pixel_various_angles(self, angle: int) -> None:
        """Центральный пиксель должен сохраниться после поворота ±5."""
        size = 64
        img = Image.new('L', (size, size), 50)
        # Ярко-белый крест 3×3 в центре
        arr = np.array(img)
        cx, cy = size // 2, size // 2
        arr[cy - 1 : cy + 2, cx - 1 : cx + 2] = 255
        img = Image.fromarray(arr)

        rotated = rotate_keep_size(img, angle, fill=(0,))
        rot_arr = np.array(rotated)
        center_val = rot_arr[cy, cx]
        assert center_val >= 200, f'Центральный пиксель = {center_val}, ожидается >= 200'

    @pytest.mark.parametrize(
        ('w', 'h'), [(64, 64), (100, 80), (37, 53)],
    )
    def test_preserves_dimensions(self, w: int, h: int) -> None:
        img = Image.new('RGB', (w, h), (128, 128, 128))
        rotated = rotate_keep_size(img, 15.0)
        assert rotated.size == (w, h)

    def test_fill_value_applied_to_corners(self) -> None:
        """При повороте 45° углы заполняются fill-цветом."""
        size = 64
        img = Image.new('RGB', (size, size), (100, 100, 100))
        fill = (255, 0, 0)
        rotated = rotate_keep_size(img, 45.0, fill=fill)
        arr = np.array(rotated)
        # Угол (0,0) должен быть близок к fill
        corner = arr[0, 0]
        assert corner[0] > 200, f'Угол R={corner[0]}, ожидается close to 255'

    def test_zero_fill_distinguishable(self) -> None:
        """DEM 100-500, fill=0 → углы ≈ 0, центр > 50."""
        size = 64
        dem_arr = make_gradient_dem(size, size, 100, 500)
        # Конвертируем в uint8 для PIL (нормализуем)
        norm = ((dem_arr - 100) / 400 * 200 + 50).astype(np.uint8)
        img = Image.fromarray(norm)
        rotated = rotate_keep_size(img, 45.0, fill=(0,))
        rot_arr = np.array(rotated)
        assert rot_arr[0, 0] < 10, 'Угол должен быть ≈ 0 (fill)'
        assert rot_arr[size // 2, size // 2] > 50, 'Центр должен быть > 50'


class TestCenterCropInvariants:
    """Инварианты center_crop."""

    def test_extracts_center_pixel(self) -> None:
        """Красная точка (50,50) в 100×100 → crop 10×10 → центр красный."""
        img = Image.new('RGB', (100, 100), (0, 0, 0))
        arr = np.array(img)
        arr[50, 50] = [255, 0, 0]
        img = Image.fromarray(arr)
        cropped = center_crop(img, 10, 10)
        crop_arr = np.array(cropped)
        # Центр кропа = (5, 5)
        assert crop_arr[5, 5, 0] == 255

    def test_extracts_center_pixel_odd_sizes(self) -> None:
        """101×101 → crop 11×11 → центр совпадает."""
        img = Image.new('L', (101, 101), 0)
        arr = np.array(img)
        arr[50, 50] = 255
        img = Image.fromarray(arr)
        cropped = center_crop(img, 11, 11)
        crop_arr = np.array(cropped)
        assert crop_arr[5, 5] == 255

    def test_asymmetric_crop(self) -> None:
        """200×100 → crop 50×30."""
        img = Image.new('RGB', (200, 100), (128, 128, 128))
        cropped = center_crop(img, 50, 30)
        assert cropped.size == (50, 30)


# ── Группа 2: Round-trip ────────────────────────────────────────────────


class TestDownsampleUpsampleRoundTrip:
    """Downsample → upsample round-trip."""

    def test_round_trip_correct_dimensions(self) -> None:
        dem = make_gradient_dem(64, 64, 0, 1000)
        down = downsample_dem(dem, 2)
        # Upsample обратно через PIL resize
        up_img = Image.fromarray(down).resize((64, 64), Image.Resampling.BILINEAR)
        up = np.array(up_img)
        assert up.shape == (64, 64)

    def test_round_trip_values_in_range(self) -> None:
        dem = make_gradient_dem(64, 64, 50, 800)
        down = downsample_dem(dem, 2)
        # cv2 resize для upsample
        up = cv2.resize(down, (64, 64), interpolation=cv2.INTER_LINEAR)
        assert up.min() >= dem.min() - 1.0
        assert up.max() <= dem.max() + 1.0

    def test_round_trip_no_nan_or_inf(self) -> None:
        dem = make_gradient_dem(64, 64, 0, 500)
        down = downsample_dem(dem, 2)
        up = cv2.resize(down, (64, 64), interpolation=cv2.INTER_LINEAR)
        assert np.all(np.isfinite(up))

    def test_round_trip_preserves_flat(self) -> None:
        dem = make_flat_dem(64, 64, 200.0)
        down = downsample_dem(dem, 2)
        up = cv2.resize(down, (64, 64), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_allclose(up, 200.0, atol=0.1)


class TestRotateRoundTrip:
    """rotate +θ → -θ → восстановление центра."""

    def test_rotate_inverse_restores_center(self) -> None:
        size = 64
        img = Image.new('L', (size, size), 50)
        arr = np.array(img)
        cx, cy = size // 2, size // 2
        arr[cy - 2 : cy + 3, cx - 2 : cx + 3] = 200
        img = Image.fromarray(arr)

        fwd = rotate_keep_size(img, 15.0, fill=(0,))
        back = rotate_keep_size(fwd, -15.0, fill=(0,))
        back_arr = np.array(back)
        center_val = back_arr[cy, cx]
        assert abs(int(center_val) - 200) < 10

    @pytest.mark.parametrize('angle', [5, 10, 30, 45])
    def test_rotate_inverse_parametrized(self, angle: int) -> None:
        size = 64
        img = Image.new('L', (size, size), 80)
        arr = np.array(img)
        cx, cy = size // 2, size // 2
        arr[cy - 2 : cy + 3, cx - 2 : cx + 3] = 220
        img = Image.fromarray(arr)

        fwd = rotate_keep_size(img, float(angle), fill=(0,))
        back = rotate_keep_size(fwd, float(-angle), fill=(0,))
        back_arr = np.array(back)
        center_val = back_arr[cy, cx]
        assert abs(int(center_val) - 220) < 15


class TestFullCycleInvariants:
    """Полный цикл: downsample → resize → rotate → crop → all finite."""

    def test_full_cycle_all_finite(self) -> None:
        dem = make_gradient_dem(128, 128, 0, 500)
        # 1. Downsample
        down = downsample_dem(dem, 2)
        # 2. Upsample (PIL resize back to original)
        up_img = Image.fromarray(down).resize((128, 128), Image.Resampling.BILINEAR)
        # 3. Rotate
        rotated = rotate_keep_size(up_img, 15.0, fill=(0,))
        # 4. Center crop
        cropped = center_crop(rotated, 100, 100)
        result = np.array(cropped)
        assert np.all(np.isfinite(result))
        assert result.shape == (100, 100)


# ── Группа 4: Пиксельное выравнивание ───────────────────────────────────


class TestPeakTracking:
    """Отслеживание пика через трансформации."""

    def test_peak_survives_rotation(self) -> None:
        """Пик 999 в центре, после rotate > 900."""
        size = 64
        dem = make_peak_dem(size, size, background=100.0, peak=999.0)
        # Конвертируем DEM → grayscale image (нормализовано)
        norm = ((dem - 100) / 899 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(norm)
        rotated = rotate_keep_size(img, 30.0, fill=(0,))
        rot_arr = np.array(rotated)
        cx, cy = size // 2, size // 2
        center_val = rot_arr[cy, cx]
        # В нормализованном пространстве пик = 255 → после поворота > 228 (≈ 900/999*255)
        assert center_val > 228, f'center = {center_val}'

    def test_peak_survives_rotate_and_crop(self) -> None:
        """Rotate + crop → пик в центре кропа."""
        size = 64
        dem = make_peak_dem(size, size, background=100.0, peak=999.0)
        norm = ((dem - 100) / 899 * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(norm)
        rotated = rotate_keep_size(img, 15.0, fill=(0,))
        cropped = center_crop(rotated, 40, 40)
        crop_arr = np.array(cropped)
        cx, cy = 20, 20
        center_val = crop_arr[cy, cx]
        assert center_val > 228, f'center after crop = {center_val}'

    def test_downsampled_peak_position(self) -> None:
        """Пик (32,32), factor=2 → argmax at (16,16)."""
        dem = make_peak_dem(64, 64, background=0.0, peak=999.0, peak_row=32, peak_col=32)
        down = downsample_dem(dem, 2)
        peak_pos = np.unravel_index(down.argmax(), down.shape)
        assert peak_pos == (16, 16)

    def test_downsampled_peak_position_factor_4(self) -> None:
        """Пик (64,64), factor=4 → argmax at (16,16)."""
        dem = make_peak_dem(128, 128, background=0.0, peak=999.0, peak_row=64, peak_col=64)
        down = downsample_dem(dem, 4)
        peak_pos = np.unravel_index(down.argmax(), down.shape)
        assert peak_pos == (16, 16)

    def test_dem_cursor_path_center_alignment(self) -> None:
        """cv2 rotation + numpy crop → центральный пик сохраняется."""
        size = 64
        dem = make_peak_dem(size, size, background=100.0, peak=999.0)
        angle = 15.0

        # cv2 path (как в _load_dem_for_cursor)
        h, w = dem.shape
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            dem, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        # numpy center crop
        tw, th = 40, 40
        left = (w - tw) // 2
        top = (h - th) // 2
        cropped = rotated[top : top + th, left : left + tw]

        center_val = cropped[th // 2, tw // 2]
        assert center_val > 900, f'cv2 path center = {center_val}'


class TestDemCursorVsImagePath:
    """Сравнение PIL path и cv2 path для DEM."""

    def test_center_pixel_same_value_both_paths(self) -> None:
        """PIL rotate vs cv2 rotate → центральный пиксель ≈ одинаковый."""
        size = 64
        dem = make_peak_dem(size, size, background=100.0, peak=999.0)
        angle = 15.0

        # PIL path
        dem_norm = ((dem - 100) / 899 * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(dem_norm)
        pil_rotated = rotate_keep_size(pil_img, angle, fill=(0,))
        pil_arr = np.array(pil_rotated).astype(np.float32)
        pil_center = pil_arr[size // 2, size // 2]

        # cv2 path (float32 DEM directly)
        h, w = dem.shape
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cv2_rotated = cv2.warpAffine(
            dem, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        cv2_center = cv2_rotated[size // 2, size // 2]
        # Нормализуем cv2 центр для сравнения
        cv2_center_norm = (cv2_center - 100) / 899 * 255

        assert abs(pil_center - cv2_center_norm) < 2.0, (
            f'PIL={pil_center:.1f}, cv2_norm={cv2_center_norm:.1f}'
        )

    def test_both_paths_same_dimensions(self) -> None:
        """PIL size == cv2 shape после rotation."""
        size = 64
        dem = make_flat_dem(size, size, 200.0)
        angle = 30.0

        # PIL path
        img = Image.fromarray(dem)
        pil_rot = rotate_keep_size(img, angle, fill=(0,))
        pil_size = pil_rot.size  # (w, h)

        # cv2 path
        h, w = dem.shape
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cv2_rot = cv2.warpAffine(
            dem, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        cv2_shape = cv2_rot.shape  # (h, w)

        assert pil_size == (cv2_shape[1], cv2_shape[0])
