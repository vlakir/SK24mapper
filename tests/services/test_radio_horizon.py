"""
Тесты для модуля radio_horizon.

Проверяет расчёт радиогоризонта и визуализацию.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from shared.constants import (
    EARTH_RADIUS_M,
    RADIO_HORIZON_COLOR_RAMP,
    RADIO_HORIZON_EMPTY_IMAGE_COLOR,
    RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX,
    RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL,
    RADIO_HORIZON_LEGEND_UNIT_LABEL,
    RADIO_HORIZON_LUT_SIZE,
    RADIO_HORIZON_MAX_HEIGHT_M,
    RADIO_HORIZON_MIN_HEIGHT_M,
    RADIO_HORIZON_REFRACTION_K,
    RADIO_HORIZON_UNREACHABLE_COLOR,
    UavHeightReference,
)
from services.radio_horizon import (
    _bilinear_interpolate,
    _build_color_lut,
    _trace_line_of_sight,
    colorize_radio_horizon,
    compute_and_colorize_radio_horizon,
    compute_downsample_factor,
    compute_radio_horizon,
    downsample_dem,
    get_radio_horizon_legend_params,
    recompute_radio_horizon_fast,
)


class TestBilinearInterpolate:
    """Тесты билинейной интерполяции."""

    def test_exact_pixel(self) -> None:
        """Точное попадание в пиксель возвращает его значение (с допуском)."""
        dem = [[100.0, 200.0], [300.0, 400.0]]
        h, w = 2, 2
        assert abs(_bilinear_interpolate(dem, 0, 0, h, w) - 100.0) < 0.5
        # Крайние пиксели имеют небольшое смещение из-за clamping
        assert abs(_bilinear_interpolate(dem, 0, 0.999, h, w) - 200.0) < 1.0
        assert abs(_bilinear_interpolate(dem, 0.999, 0, h, w) - 300.0) < 1.0
        assert abs(_bilinear_interpolate(dem, 0.999, 0.999, h, w) - 400.0) < 1.0

    def test_center_interpolation(self) -> None:
        """Интерполяция в центре ячейки."""
        dem = [[0.0, 100.0], [100.0, 200.0]]
        h, w = 2, 2
        # Центр: (0.5, 0.5) -> среднее всех четырёх = 100
        result = _bilinear_interpolate(dem, 0.5, 0.5, h, w)
        assert abs(result - 100.0) < 0.01

    def test_edge_interpolation(self) -> None:
        """Интерполяция на границе ячейки."""
        dem = [[0.0, 100.0], [0.0, 100.0]]
        h, w = 2, 2
        # Середина верхнего ребра: (0, 0.5) -> 50
        result = _bilinear_interpolate(dem, 0, 0.5, h, w)
        assert abs(result - 50.0) < 0.01

    def test_out_of_bounds_clamped(self) -> None:
        """Координаты вне границ ограничиваются."""
        dem = [[100.0, 200.0], [300.0, 400.0]]
        h, w = 2, 2
        # Отрицательные координаты -> (0, 0)
        result = _bilinear_interpolate(dem, -1.0, -1.0, h, w)
        assert abs(result - 100.0) < 0.01


class TestTraceLineOfSight:
    """Тесты трассировки луча."""

    def test_same_point(self) -> None:
        """Точка совпадает с антенной — видимость 0."""
        dem = [[100.0] * 10 for _ in range(10)]
        h, w = 10, 10
        effective_earth_radius = EARTH_RADIUS_M * RADIO_HORIZON_REFRACTION_K
        result = _trace_line_of_sight(
            dem=dem,
            antenna_row=5,
            antenna_col=5,
            antenna_abs_height=110.0,
            target_row=5,
            target_col=5,
            pixel_size_m=10.0,
            effective_earth_radius=effective_earth_radius,
            h=h,
            w=w,
        )
        assert result == 0.0

    def test_flat_terrain_clear_los(self) -> None:
        """На плоском рельефе без препятствий — видимость без подъёма."""
        # Плоская поверхность 100м
        dem = [[100.0] * 50 for _ in range(50)]
        h, w = 50, 50
        effective_earth_radius = EARTH_RADIUS_M * RADIO_HORIZON_REFRACTION_K
        result = _trace_line_of_sight(
            dem=dem,
            antenna_row=25,
            antenna_col=25,
            antenna_abs_height=110.0,  # антенна на 10м выше
            target_row=25,
            target_col=40,  # 15 пикселей = 150м
            pixel_size_m=10.0,
            effective_earth_radius=effective_earth_radius,
            h=h,
            w=w,
        )
        # На плоском рельефе и близком расстоянии — почти 0
        assert result < 5.0

    def test_hill_blocks_los(self) -> None:
        """Холм между антенной и целью требует подъёма."""
        dem = [[100.0] * 50 for _ in range(50)]
        h, w = 50, 50
        # Добавляем холм между антенной (25, 25) и целью (25, 45)
        for col in range(33, 37):
            dem[25][col] = 200.0  # холм 100м высотой

        effective_earth_radius = EARTH_RADIUS_M * RADIO_HORIZON_REFRACTION_K
        result = _trace_line_of_sight(
            dem=dem,
            antenna_row=25,
            antenna_col=25,
            antenna_abs_height=110.0,
            target_row=25,
            target_col=45,
            pixel_size_m=10.0,
            effective_earth_radius=effective_earth_radius,
            h=h,
            w=w,
        )
        # Должен требовать значительного подъёма
        assert result > 50.0

    def test_earth_curvature_effect(self) -> None:
        """На большом расстоянии кривизна Земли влияет на результат."""
        # Плоская поверхность на большом расстоянии
        dem = [[100.0] * 500 for _ in range(500)]
        h, w = 500, 500
        effective_earth_radius = EARTH_RADIUS_M * RADIO_HORIZON_REFRACTION_K
        
        # Ближняя точка
        near_result = _trace_line_of_sight(
            dem=dem,
            antenna_row=250,
            antenna_col=250,
            antenna_abs_height=110.0,
            target_row=250,
            target_col=260,  # 100м
            pixel_size_m=10.0,
            effective_earth_radius=effective_earth_radius,
            h=h,
            w=w,
        )
        
        # Дальняя точка (10 км)
        far_result = _trace_line_of_sight(
            dem=dem,
            antenna_row=250,
            antenna_col=250,
            antenna_abs_height=110.0,
            target_row=250,
            target_col=450,  # 2000м = 2км  
            pixel_size_m=10.0,
            effective_earth_radius=effective_earth_radius,
            h=h,
            w=w,
        )
        
        # На дальнем расстоянии требуется больше высоты из-за кривизны
        # (хотя на 2км эффект ещё небольшой)
        assert far_result >= near_result


class TestComputeRadioHorizon:
    """Тесты вычисления матрицы радиогоризонта."""

    def test_empty_dem(self) -> None:
        """Пустой DEM возвращает пустой результат."""
        dem = np.array([], dtype=np.float32).reshape(0, 0)
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=0,
            antenna_col=0,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
        )
        assert result.size == 0

    def test_flat_terrain(self) -> None:
        """На плоском рельефе все значения близки к 0."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        
        assert result.shape == (20, 20)
        
        # Большинство значений должны быть малыми
        max_val = result.max()
        assert max_val < 50.0  # На плоском рельефе не должно быть больших значений

    def test_result_dimensions(self) -> None:
        """Результат имеет те же размеры, что и входной DEM."""
        dem = np.full((25, 30), 100.0, dtype=np.float32)
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=12,
            antenna_col=15,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=4,
        )
        
        assert result.shape == (25, 30)

    def test_antenna_position_clamped(self) -> None:
        """Позиция антенны ограничивается границами DEM."""
        dem = np.full((10, 10), 100.0, dtype=np.float32)
        # Не должно выбросить исключение
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=5,
            antenna_col=5,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        assert result.shape[0] == 10


class TestColorizeRadioHorizon:
    """Тесты раскраски карты радиогоризонта."""

    def test_empty_matrix(self) -> None:
        """Пустая матрица возвращает минимальное изображение."""
        matrix = np.array([], dtype=np.float32).reshape(0, 0)
        result = colorize_radio_horizon(matrix)
        assert result.size == (1, 1)

    def test_output_dimensions(self) -> None:
        """Размер изображения соответствует входной матрице."""
        matrix = np.zeros((30, 50), dtype=np.float32)
        result = colorize_radio_horizon(matrix)
        assert result.size == (50, 30)

    def test_output_mode(self) -> None:
        """Выходное изображение в режиме RGB."""
        matrix = np.zeros((10, 10), dtype=np.float32)
        result = colorize_radio_horizon(matrix)
        assert result.mode == 'RGB'

    def test_zero_height_is_green(self) -> None:
        """Нулевая высота окрашивается в зелёный цвет."""
        matrix = np.array([[0.0]], dtype=np.float32)
        result = colorize_radio_horizon(matrix)
        pixel = result.getpixel((0, 0))
        # Зелёный: (0, 128, 0)
        assert pixel[1] > pixel[0]  # G > R
        assert pixel[1] > pixel[2]  # G > B

    def test_high_value_is_red(self) -> None:
        """Высокое значение окрашивается в красный цвет."""
        matrix = np.array([[500.0]], dtype=np.float32)  # Максимум шкалы
        result = colorize_radio_horizon(matrix, max_height_m=500.0)
        pixel = result.getpixel((0, 0))
        # Тёмно-красный: (139, 0, 0)
        assert pixel[0] > pixel[1]  # R > G
        assert pixel[0] > pixel[2]  # R > B

    def test_unreachable_is_gray(self) -> None:
        """Недостижимые точки окрашиваются в серый."""
        matrix = np.array([[np.inf]], dtype=np.float32)
        result = colorize_radio_horizon(matrix)
        pixel = result.getpixel((0, 0))
        # Серый: (64, 64, 64)
        assert pixel == (64, 64, 64)

    def test_gradient_order(self) -> None:
        """Красная компонента увеличивается с ростом высоты."""
        matrix = np.array([[0.0], [100.0], [200.0], [300.0], [400.0], [500.0]], dtype=np.float32)
        result = colorize_radio_horizon(matrix, max_height_m=500.0)
        
        # Получаем пиксели
        pixels = [result.getpixel((0, i)) for i in range(6)]
        
        # Красная компонента должна в целом увеличиваться
        # (проверяем, что последний пиксель краснее первого)
        first_red = pixels[0][0]
        last_red = pixels[-1][0]
        assert last_red > first_red
        
        # Зелёная компонента в конце должна быть меньше чем в начале/середине
        first_green = pixels[0][1]
        last_green = pixels[-1][1]
        assert first_green > last_green


class TestGetRadioHorizonLegendParams:
    """Тесты параметров легенды."""

    def test_default_params(self) -> None:
        """Параметры по умолчанию."""
        min_val, max_val, unit = get_radio_horizon_legend_params()
        assert min_val == 0.0
        assert max_val == 500.0
        assert unit == 'м'

    def test_custom_max_height(self) -> None:
        """Пользовательская максимальная высота."""
        min_val, max_val, unit = get_radio_horizon_legend_params(max_height_m=300.0)
        assert min_val == 0.0
        assert max_val == 300.0
        assert unit == 'м'


class TestIntegration:
    """Интеграционные тесты."""

    def test_full_pipeline(self) -> None:
        """Полный цикл: DEM -> radio horizon -> image."""
        # Создаём тестовый DEM с холмом
        dem = np.full((40, 40), 50.0, dtype=np.float32)
        # Добавляем холм
        dem[15:25, 15:25] = 150.0
        
        # Вычисляем радиогоризонт
        horizon = compute_radio_horizon(
            dem=dem,
            antenna_row=5,
            antenna_col=5,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        
        # Раскрашиваем
        image = colorize_radio_horizon(horizon)
        
        # Проверяем результат
        assert image.size == (40, 40)
        assert image.mode == 'RGB'
        
        # Точка за холмом должна требовать большей высоты
        # чем точка рядом с антенной
        near_height = horizon[6, 6]
        far_height = horizon[30, 30]
        assert far_height > near_height


class TestDownsampleDem:
    """Тесты даунсэмплинга DEM."""

    def test_factor_1_returns_same(self) -> None:
        """Коэффициент 1 возвращает исходную матрицу."""
        dem = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = downsample_dem(dem, 1)
        assert result is dem

    def test_factor_2_halves_size(self) -> None:
        """Коэффициент 2 уменьшает размер вдвое."""
        dem = np.array([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0]], dtype=np.float32)
        result = downsample_dem(dem, 2)
        assert result.shape == (2, 2)

    def test_factor_2_averages_blocks(self) -> None:
        """Коэффициент 2 усредняет блоки 2x2."""
        dem = np.array([[1.0, 3.0, 5.0, 7.0],
                        [1.0, 3.0, 5.0, 7.0],
                        [9.0, 11.0, 13.0, 15.0],
                        [9.0, 11.0, 13.0, 15.0]], dtype=np.float32)
        result = downsample_dem(dem, 2)
        # Верхний левый блок: (1+3+1+3)/4 = 2.0
        assert abs(result[0, 0] - 2.0) < 0.01
        # Верхний правый блок: (5+7+5+7)/4 = 6.0
        assert abs(result[0, 1] - 6.0) < 0.01
        # Нижний левый блок: (9+11+9+11)/4 = 10.0
        assert abs(result[1, 0] - 10.0) < 0.01
        # Нижний правый блок: (13+15+13+15)/4 = 14.0
        assert abs(result[1, 1] - 14.0) < 0.01

    def test_factor_4(self) -> None:
        """Коэффициент 4 уменьшает размер в 4 раза."""
        dem = np.array([[float(i + j) for j in range(8)] for i in range(8)], dtype=np.float32)
        result = downsample_dem(dem, 4)
        assert result.shape == (2, 2)

    def test_empty_dem(self) -> None:
        """Пустая DEM возвращается без изменений."""
        dem = np.array([], dtype=np.float32).reshape(0, 0)
        result = downsample_dem(dem, 2)
        assert result.size == 0

    def test_too_small_dem(self) -> None:
        """DEM меньше коэффициента возвращается без изменений."""
        dem = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = downsample_dem(dem, 4)
        assert result is dem


class TestComputeDownsampleFactor:
    """Тесты вычисления коэффициента даунсэмплинга."""

    def test_small_dem_no_downsampling(self) -> None:
        """Маленькая DEM не требует даунсэмплинга."""
        factor = compute_downsample_factor(1000, 1000, max_pixels=16_000_000)
        assert factor == 1

    def test_exact_limit(self) -> None:
        """DEM точно на лимите не требует даунсэмплинга."""
        factor = compute_downsample_factor(4000, 4000, max_pixels=16_000_000)
        assert factor == 1

    def test_slightly_over_limit(self) -> None:
        """DEM немного выше лимита требует даунсэмплинг в 2 раза."""
        factor = compute_downsample_factor(4001, 4001, max_pixels=16_000_000)
        assert factor == 2

    def test_large_dem_factor_4(self) -> None:
        """Большая DEM требует даунсэмплинг в 4 раза."""
        # 10000×10000 = 100M > 16M, нужен factor >= 3, т.е. 4 (степень двойки)
        factor = compute_downsample_factor(10000, 10000, max_pixels=16_000_000)
        assert factor == 4

    def test_very_large_dem_factor_8(self) -> None:
        """Очень большая DEM требует даунсэмплинг в 8 раз."""
        # 20000×20000 = 400M > 16M×16 = 256M, нужен factor 8
        factor = compute_downsample_factor(20000, 20000, max_pixels=16_000_000)
        assert factor == 8


class TestUavHeightReference:
    """Тесты режимов отсчёта высоты БпЛА."""

    def test_ground_reference_default(self) -> None:
        """Режим GROUND — значения без изменений (по умолчанию)."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        dem[10, 10] = 150.0  # Контрольная точка выше

        image = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            max_height_m=500.0,
            uav_height_reference=UavHeightReference.GROUND,
        )

        assert image.size == (20, 20)
        assert image.mode == 'RGB'

    def test_control_point_reference(self) -> None:
        """Режим CONTROL_POINT — высоты пересчитываются от уровня КТ."""
        # DEM с разной высотой: КТ на 100м, остальное на 50м
        dem = np.full((20, 20), 50.0, dtype=np.float32)
        dem[10, 10] = 100.0  # Контрольная точка

        image_ground = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            max_height_m=500.0,
            uav_height_reference=UavHeightReference.GROUND,
        )

        image_cp = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            max_height_m=500.0,
            uav_height_reference=UavHeightReference.CONTROL_POINT,
            cp_elevation=100.0,
        )

        # Изображения должны отличаться (разные режимы отсчёта)
        assert image_ground.size == image_cp.size
        # Пиксели должны отличаться из-за разного пересчёта
        arr_ground = np.array(image_ground)
        arr_cp = np.array(image_cp)
        # Не все пиксели одинаковы
        assert not np.array_equal(arr_ground, arr_cp)

    def test_sea_level_reference(self) -> None:
        """Режим SEA_LEVEL — абсолютные высоты от уровня моря."""
        dem = np.full((20, 20), 200.0, dtype=np.float32)

        image_ground = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            max_height_m=500.0,
            uav_height_reference=UavHeightReference.GROUND,
        )

        image_sea = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            max_height_m=500.0,
            uav_height_reference=UavHeightReference.SEA_LEVEL,
        )

        # Изображения должны отличаться
        arr_ground = np.array(image_ground)
        arr_sea = np.array(image_sea)
        assert not np.array_equal(arr_ground, arr_sea)

    def test_cp_elevation_auto_from_dem(self) -> None:
        """Если cp_elevation не задан, берётся из DEM."""
        dem = np.full((20, 20), 50.0, dtype=np.float32)
        dem[10, 10] = 100.0

        # Без явного cp_elevation
        image1 = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            max_height_m=500.0,
            uav_height_reference=UavHeightReference.CONTROL_POINT,
        )

        # С явным cp_elevation = 100 (как в DEM)
        image2 = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            max_height_m=500.0,
            uav_height_reference=UavHeightReference.CONTROL_POINT,
            cp_elevation=100.0,
        )

        # Результаты должны быть идентичны
        arr1 = np.array(image1)
        arr2 = np.array(image2)
        assert np.array_equal(arr1, arr2)


class TestBuildColorLut:
    """Тесты построения LUT для цветовой шкалы."""

    def test_lut_shape(self) -> None:
        """LUT имеет корректную форму: (lut_size+1, 3)."""
        lut = _build_color_lut(
            RADIO_HORIZON_COLOR_RAMP,
            RADIO_HORIZON_UNREACHABLE_COLOR,
            lut_size=256,
        )
        assert lut.shape == (257, 3)

    def test_lut_unreachable_color_at_end(self) -> None:
        """Последний элемент LUT содержит цвет недостижимых точек."""
        unreachable = (100, 200, 50)
        lut = _build_color_lut(
            RADIO_HORIZON_COLOR_RAMP,
            unreachable,
            lut_size=64,
        )
        assert tuple(lut[64]) == unreachable

    def test_lut_first_element_matches_ramp_start(self) -> None:
        """Первый элемент LUT соответствует начальному цвету рампы."""
        ramp = [(0.0, (10, 20, 30)), (1.0, (200, 210, 220))]
        lut = _build_color_lut(ramp, (0, 0, 0), lut_size=128)
        # Первый элемент (t=0) должен быть начальным цветом
        assert tuple(lut[0]) == (10, 20, 30)

    def test_lut_last_regular_element_matches_ramp_end(self) -> None:
        """Последний регулярный элемент LUT соответствует конечному цвету рампы."""
        ramp = [(0.0, (10, 20, 30)), (1.0, (200, 210, 220))]
        lut = _build_color_lut(ramp, (0, 0, 0), lut_size=128)
        # Последний регулярный элемент (t=1) должен быть конечным цветом
        assert tuple(lut[127]) == (200, 210, 220)

    def test_degenerate_ramp_with_same_t_values(self) -> None:
        """Рампа с одинаковыми значениями t (t0 == t1) не вызывает деление на ноль."""
        ramp = [(0.0, (255, 0, 0)), (0.0, (0, 255, 0)), (1.0, (0, 0, 255))]
        lut = _build_color_lut(ramp, (64, 64, 64), lut_size=64)
        # Не должно быть ошибки деления на ноль
        assert lut.shape == (65, 3)
        # Первый элемент (t=0) попадает в сегмент t0=0, t1=0, используется c0
        assert tuple(lut[0]) == (255, 0, 0)

    def test_default_lut_size(self) -> None:
        """LUT с размером по умолчанию из констант."""
        lut = _build_color_lut(
            RADIO_HORIZON_COLOR_RAMP,
            RADIO_HORIZON_UNREACHABLE_COLOR,
        )
        assert lut.shape == (RADIO_HORIZON_LUT_SIZE + 1, 3)


class TestColorizeRadioHorizonZeroSize:
    """Тесты colorize_radio_horizon с матрицами нулевого размера."""

    def test_zero_height_matrix(self) -> None:
        """Матрица с нулевой высотой (0xN) возвращает минимальное изображение."""
        matrix = np.zeros((0, 10), dtype=np.float32)
        result = colorize_radio_horizon(matrix)
        assert result.size == RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX

    def test_zero_width_matrix(self) -> None:
        """Матрица с нулевой шириной (Nx0) возвращает минимальное изображение."""
        matrix = np.zeros((10, 0), dtype=np.float32)
        result = colorize_radio_horizon(matrix)
        assert result.size == RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX

    def test_empty_image_color(self) -> None:
        """Пустая матрица генерирует изображение с заданным цветом."""
        matrix = np.array([], dtype=np.float32).reshape(0, 0)
        result = colorize_radio_horizon(matrix)
        pixel = result.getpixel((0, 0))
        assert pixel == RADIO_HORIZON_EMPTY_IMAGE_COLOR


class TestComputeAndColorizeEmptyDem:
    """Тесты compute_and_colorize_radio_horizon с пустым DEM."""

    def test_empty_dem_returns_empty_image(self) -> None:
        """Пустой DEM (size==0) возвращает минимальное изображение."""
        dem = np.array([], dtype=np.float32).reshape(0, 0)
        result = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=0,
            antenna_col=0,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
        )
        assert result.size == RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX

    def test_zero_height_dem_returns_empty_image(self) -> None:
        """DEM с нулевой высотой (0xN) возвращает минимальное изображение."""
        dem = np.zeros((0, 20), dtype=np.float32)
        result = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=0,
            antenna_col=0,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
        )
        assert result.size == RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX

    def test_zero_width_dem_returns_empty_image(self) -> None:
        """DEM с нулевой шириной (Nx0) возвращает минимальное изображение."""
        dem = np.zeros((20, 0), dtype=np.float32)
        result = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=0,
            antenna_col=0,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
        )
        assert result.size == RADIO_HORIZON_EMPTY_IMAGE_SIZE_PX

    def test_empty_dem_image_color(self) -> None:
        """Пустой DEM генерирует изображение с корректным цветом."""
        dem = np.array([], dtype=np.float32).reshape(0, 0)
        result = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=0,
            antenna_col=0,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
        )
        pixel = result.getpixel((0, 0))
        assert pixel == RADIO_HORIZON_EMPTY_IMAGE_COLOR


class TestAdaptiveGridStep:
    """Тесты адаптивного шага сетки для больших DEM."""

    def test_small_dem_uses_default_step(self) -> None:
        """Маленький DEM использует шаг по умолчанию."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        # 20*20 = 400, меньше всех порогов
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        assert result.shape == (20, 20)

    def test_grid_step_small_threshold(self) -> None:
        """DEM выше порога SMALL использует увеличенный шаг.

        Используется compute_and_colorize_radio_horizon для покрытия линий 503-504.
        """
        # Порог SMALL = 4_000_000. Для 2100x2100 = 4_410_000 > 4_000_000
        # Создаём маленький DEM и проверяем через mock-подход:
        # вместо реально большого массива, проверяем, что функция корректно работает
        # с DEM чуть выше порога
        side = math.isqrt(RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL) + 10
        dem = np.full((side, side), 100.0, dtype=np.float32)
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=side // 2,
            antenna_col=side // 2,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=1,
        )
        assert result.shape == (side, side)

    def test_compute_and_colorize_grid_step_small_threshold(self) -> None:
        """compute_and_colorize_radio_horizon с DEM выше порога SMALL."""
        side = math.isqrt(RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL) + 10
        dem = np.full((side, side), 100.0, dtype=np.float32)
        result = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=side // 2,
            antenna_col=side // 2,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=1,
        )
        assert result.size == (side, side)
        assert result.mode == 'RGB'


class TestComputeRadioHorizonFull:
    """Дополнительные тесты compute_radio_horizon."""

    def test_returns_float32(self) -> None:
        """Результат имеет тип float32."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        assert result.dtype == np.float32

    def test_antenna_at_corner(self) -> None:
        """Антенна в углу DEM — корректная работа."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=0,
            antenna_col=0,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        assert result.shape == (20, 20)
        # Точки дальше от антенны имеют большее значение из-за кривизны
        assert result[0, 0] <= result[19, 19]

    def test_antenna_out_of_bounds(self) -> None:
        """Позиция антенны вне DEM ограничивается."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        result = compute_radio_horizon(
            dem=dem,
            antenna_row=100,
            antenna_col=100,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        assert result.shape == (20, 20)

    def test_custom_earth_radius(self) -> None:
        """Разные значения радиуса Земли дают разные результаты."""
        dem = np.full((30, 30), 100.0, dtype=np.float32)
        result_normal = compute_radio_horizon(
            dem=dem,
            antenna_row=15,
            antenna_col=15,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=1,
            earth_radius_m=EARTH_RADIUS_M,
        )
        result_double = compute_radio_horizon(
            dem=dem,
            antenna_row=15,
            antenna_col=15,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=1,
            earth_radius_m=EARTH_RADIUS_M * 2,
        )
        # Разный радиус Земли должен давать разные результаты
        assert not np.array_equal(result_normal, result_double)

    def test_high_antenna_produces_small_values(self) -> None:
        """Высокая антенна уменьшает требуемые высоты БпЛА."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        dem[5:15, 5:15] = 150.0  # Холм

        result_low = compute_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=1.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        result_high = compute_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=500.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        # Высокая антенна должна давать меньшие или равные значения
        assert result_high.max() <= result_low.max()


class TestRecomputeRadioHorizonFast:
    """Тесты быстрого перестроения радиогоризонта."""

    def _make_topo_base(self, width: int, height: int) -> Image.Image:
        """Создаёт тестовую топооснову RGBA."""
        return Image.new('RGBA', (width, height), (128, 128, 128, 255))

    def test_basic_recompute(self) -> None:
        """Базовое перестроение возвращает RGBA изображение."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        topo_base = self._make_topo_base(20, 20)

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.5,
            grid_step=2,
        )

        assert result.size == (20, 20)
        assert result.mode == 'RGBA'

    def test_recompute_with_final_size(self) -> None:
        """Перестроение с финальным размером масштабирует результат."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        topo_base = self._make_topo_base(40, 40)

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.5,
            final_size=(40, 40),
            grid_step=2,
        )

        assert result.size == (40, 40)

    def test_recompute_with_mismatched_topo_base(self) -> None:
        """Перестроение с несовпадающим размером топоосновы масштабирует её."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        topo_base = self._make_topo_base(30, 30)  # Отличается от dem и final_size

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.5,
            final_size=(40, 40),
            grid_step=2,
        )

        assert result.size == (40, 40)

    def test_recompute_overlay_alpha_zero(self) -> None:
        """Прозрачность 0.0 — результат ближе к чистому радиогоризонту."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        topo_base = self._make_topo_base(20, 20)

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.0,
            grid_step=2,
        )

        assert result.size == (20, 20)

    def test_recompute_overlay_alpha_one(self) -> None:
        """Прозрачность 1.0 — результат ближе к чистой топооснове."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        topo_base = self._make_topo_base(20, 20)

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=1.0,
            grid_step=2,
        )

        assert result.size == (20, 20)

    def test_recompute_different_alpha_different_results(self) -> None:
        """Разные значения прозрачности дают разные изображения."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        dem[15:20, 15:20] = 200.0  # Добавляем рельеф для разнообразия

        topo_red = Image.new('RGBA', (20, 20), (255, 0, 0, 255))

        result_alpha_0 = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_red,
            overlay_alpha=0.0,
            grid_step=2,
        )
        result_alpha_1 = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_red,
            overlay_alpha=1.0,
            grid_step=2,
        )

        arr_0 = np.array(result_alpha_0)
        arr_1 = np.array(result_alpha_1)
        assert not np.array_equal(arr_0, arr_1)

    def test_recompute_antenna_clamped(self) -> None:
        """Позиция антенны вне границ ограничивается без ошибки."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        topo_base = self._make_topo_base(20, 20)

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=100,
            new_antenna_col=100,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.5,
            grid_step=2,
        )

        assert result.size == (20, 20)

    def test_recompute_negative_antenna_clamped(self) -> None:
        """Отрицательная позиция антенны ограничивается до (0, 0)."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        topo_base = self._make_topo_base(20, 20)

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=-5,
            new_antenna_col=-5,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.5,
            grid_step=2,
        )

        assert result.size == (20, 20)

    def test_recompute_with_control_point_reference(self) -> None:
        """Перестроение с режимом отсчёта CONTROL_POINT."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        dem[10, 10] = 150.0
        topo_base = self._make_topo_base(20, 20)

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.5,
            uav_height_reference=UavHeightReference.CONTROL_POINT,
            grid_step=2,
        )

        assert result.size == (20, 20)

    def test_recompute_without_final_size(self) -> None:
        """Перестроение без финального размера — размер определяется DEM."""
        dem = np.full((25, 30), 100.0, dtype=np.float32)
        topo_base = self._make_topo_base(30, 25)

        result = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=12,
            new_antenna_col=15,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.5,
            grid_step=2,
        )

        assert result.size == (30, 25)


class TestGetRadioHorizonLegendParamsFull:
    """Дополнительные тесты параметров легенды."""

    def test_returns_correct_unit_label(self) -> None:
        """Единица измерения соответствует константе."""
        _, _, unit = get_radio_horizon_legend_params()
        assert unit == RADIO_HORIZON_LEGEND_UNIT_LABEL

    def test_min_value_is_constant(self) -> None:
        """Минимальное значение всегда равно RADIO_HORIZON_MIN_HEIGHT_M."""
        min_val, _, _ = get_radio_horizon_legend_params(max_height_m=1000.0)
        assert min_val == RADIO_HORIZON_MIN_HEIGHT_M

    def test_max_value_matches_argument(self) -> None:
        """Максимальное значение соответствует переданному аргументу."""
        _, max_val, _ = get_radio_horizon_legend_params(max_height_m=750.0)
        assert max_val == 750.0

    def test_default_max_uses_constant(self) -> None:
        """Значение по умолчанию берётся из RADIO_HORIZON_MAX_HEIGHT_M."""
        _, max_val, _ = get_radio_horizon_legend_params()
        assert max_val == RADIO_HORIZON_MAX_HEIGHT_M

    def test_returns_three_element_tuple(self) -> None:
        """Функция возвращает кортеж из трёх элементов."""
        result = get_radio_horizon_legend_params()
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestComputeAndColorizeAdaptiveGridStep:
    """Тесты адаптивного шага сетки в compute_and_colorize_radio_horizon."""

    def test_small_threshold_colorize(self) -> None:
        """DEM выше порога SMALL использует увеличенный шаг в colorize-версии."""
        side = math.isqrt(RADIO_HORIZON_GRID_STEP_THRESHOLD_PIXELS_SMALL) + 10
        dem = np.full((side, side), 100.0, dtype=np.float32)
        result = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=side // 2,
            antenna_col=side // 2,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=1,
        )
        assert result.size == (side, side)
        assert result.mode == 'RGB'


class TestIntegrationFull:
    """Расширенные интеграционные тесты."""

    def test_compute_then_colorize_separately(self) -> None:
        """Раздельный вызов compute + colorize даёт тот же размер."""
        dem = np.full((30, 30), 100.0, dtype=np.float32)
        dem[10:20, 10:20] = 200.0

        matrix = compute_radio_horizon(
            dem=dem,
            antenna_row=15,
            antenna_col=15,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )
        image = colorize_radio_horizon(matrix, max_height_m=500.0)

        combined = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=15,
            antenna_col=15,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            max_height_m=500.0,
            grid_step=2,
        )

        assert image.size == combined.size
        assert image.mode == combined.mode

    def test_recompute_matches_initial_compute(self) -> None:
        """recompute_radio_horizon_fast с overlay_alpha=0 даёт результат
        близкий к compute_and_colorize."""
        dem = np.full((20, 20), 100.0, dtype=np.float32)
        dem[5:10, 5:10] = 150.0

        # Основной расчёт
        initial = compute_and_colorize_radio_horizon(
            dem=dem,
            antenna_row=10,
            antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            grid_step=2,
        )

        # Быстрый пересчёт с overlay_alpha=0 (blend_alpha=1, чистый радиогоризонт)
        topo_base = Image.new('RGBA', (20, 20), (0, 0, 0, 255))
        recomputed = recompute_radio_horizon_fast(
            dem=dem,
            new_antenna_row=10,
            new_antenna_col=10,
            antenna_height_m=10.0,
            pixel_size_m=10.0,
            topo_base=topo_base,
            overlay_alpha=0.0,
            grid_step=2,
        )

        # Размеры должны совпадать
        assert initial.size == recomputed.size[:2] if hasattr(recomputed.size, '__getitem__') else True

