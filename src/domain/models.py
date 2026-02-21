from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, field_validator

from shared.constants import (
    ADDITIVE_RATIO,
    MapType,
    UavHeightReference,
    default_map_type,
)


class MapMetadata(BaseModel):
    """Метаданные сгенерированной карты для информера координат."""

    center_x_gk: float
    center_y_gk: float
    center_lat_wgs: float
    center_lng_wgs: float
    meters_per_pixel: float
    rotation_deg: float
    width_px: int
    height_px: int
    zoom: int
    scale: int
    crop_x: int
    crop_y: int
    control_point_enabled: bool = False
    original_cp_x_gk: float | None = None
    original_cp_y_gk: float | None = None
    helmert_params: tuple[float, float, float, float, float, float, float] | None = None
    map_type: MapType | None = None


class MapSettings(BaseModel):
    """
    Установки, которые раньше были разбросаны по модулю,
    собраны в один датакласс.
    """

    model_config = {
        'extra': 'ignore',  # игнорировать лишние поля из профилей
        # (напр., устаревшие PNG настройки)
    }

    # Координаты области по углам (СК-42 Гаусса-Крюгера, метры)
    # Старшие разряды
    from_x_high: int
    from_y_high: int
    to_x_high: int
    to_y_high: int

    # Младшие разряды
    from_x_low: int
    from_y_low: int
    to_x_low: int
    to_y_low: int

    # Путь к итоговому файлу
    output_path: str

    # Тип карты (этап 1 — стилевые карты Mapbox)
    map_type: MapType = default_map_type()

    # Наложение изолиний поверх выбранного типа карты
    overlay_contours: bool = False

    # Толщина линий сетки (м)
    grid_width_m: float
    # Размер шрифта подписей (м)
    grid_font_size_m: float
    # Отступ подписи от края изображения (м)
    grid_text_margin_m: float
    # Внутренний отступ подложки вокруг текста (м)
    grid_label_bg_padding_m: float
    # Отображать полную сетку (True) или только крестики в точках пересечения (False)
    display_grid: bool = True
    # Прозрачность белой маски (0.0 — прозрачная, 1.0 — непрозрачная)
    mask_opacity: float

    # Коррекция изображения
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0

    # Опциональные 7 параметров Хельмерта (единицы: м, угловые секунды, ppm)
    helmert_dx: float | None = None
    helmert_dy: float | None = None
    helmert_dz: float | None = None
    helmert_rx_as: float | None = None
    helmert_ry_as: float | None = None
    helmert_rz_as: float | None = None
    helmert_ds_ppm: float | None = None

    # Контрольная точка
    control_point_enabled: bool = False
    control_point_x: int = 5415000  # Default: 54*100000 + 15*1000
    control_point_y: int = 7440000  # Default: 74*100000 + 40*1000
    control_point_name: str = (
        'НСУ'  # Название контрольной точки для отображения на карте
    )

    # Высота антенны над поверхностью земли (для карты радиогоризонта)
    antenna_height_m: float = 10.0
    # Максимальная высота полёта (для карты радиогоризонта, значения выше
    # отображаются серым)
    max_flight_height_m: float = 500.0
    # Режим отсчёта высоты БпЛА (для карты радиогоризонта)
    uav_height_reference: UavHeightReference = UavHeightReference.CONTROL_POINT
    # Прозрачность слоя радиогоризонта (насколько видна топооснова)
    # 0.0 = топооснова не видна, 1.0 = чистая топооснова
    radio_horizon_overlay_alpha: float = 0.3

    # Параметры РЛС (для карты зоны обнаружения)
    radar_azimuth_deg: float = 0.0  # Азимут направления РЛС (градусы, 0=север)
    radar_sector_width_deg: float = 90.0  # Ширина сектора обзора (градусы)
    radar_elevation_min_deg: float = 0.5  # Минимальный угол места (градусы)
    radar_elevation_max_deg: float = 30.0  # Максимальный угол места (градусы)
    radar_max_range_km: float = 15.0  # Максимальная дальность обнаружения (км)
    radar_target_height_min_m: float = 30.0  # Минимальная высота цели (м)
    radar_target_height_max_m: float = 5000.0  # Максимальная высота цели (м)

    # Валидации через Pydantic validators
    @field_validator('mask_opacity', 'radio_horizon_overlay_alpha')
    @classmethod
    def validate_opacity_fields(cls, v: float | str) -> float:
        v = float(v)
        if not (0.0 <= v <= 1.0):
            msg = 'Значение должно быть в диапазоне [0.0, 1.0]'
            raise ValueError(msg)
        return v

    @field_validator('contrast', 'saturation')
    @classmethod
    def validate_adjustments(cls, v: float | str) -> float:
        fv = float(v)
        # Допускаем диапазон 0.0–2.0 (0%–200%)
        fv = max(fv, 0.0)
        return min(fv, 2.0)

    @field_validator('brightness')
    @classmethod
    def validate_brightness(cls, v: float | str) -> float:
        fv = float(v)
        # Допускаем диапазон 0.0–2.0 (0%–200%)
        fv = max(fv, 0.0)
        return min(fv, 4.0)

    # Вычисляемые свойства из исходных параметров + ADDITIVE_RATIO
    @property
    def bottom_left_x_sk42_gk(self) -> float:
        # GK X (easting, горизонталь) собираем из "y" полей профиля
        return 1e3 * (self.from_y_low - ADDITIVE_RATIO) + 1e5 * self.from_y_high

    @property
    def bottom_left_y_sk42_gk(self) -> float:
        # GK Y (northing, вертикаль) собираем из "x" полей профиля
        return 1e3 * (self.from_x_low - ADDITIVE_RATIO) + 1e5 * self.from_x_high

    @property
    def top_right_x_sk42_gk(self) -> float:
        # GK X из "y"
        return 1e3 * (self.to_y_low + ADDITIVE_RATIO) + 1e5 * self.to_y_high

    @property
    def top_right_y_sk42_gk(self) -> float:
        # GK Y из "x"
        return 1e3 * (self.to_x_low + ADDITIVE_RATIO) + 1e5 * self.to_x_high

    @property
    def control_point_x_sk42_gk(self) -> float:
        # GK X (easting, горизонталь) из Y координаты контрольной точки — без припуска
        y_high = self.control_point_y // 100000
        y_low_km = (self.control_point_y % 100000) / 1000.0
        return 1e3 * y_low_km + 1e5 * y_high

    @property
    def control_point_y_sk42_gk(self) -> float:
        # GK Y (northing, вертикаль) из X координаты контрольной точки — без припуска
        x_high = self.control_point_x // 100000
        x_low_km = (self.control_point_x % 100000) / 1000.0
        return 1e3 * x_low_km + 1e5 * x_high

    @property
    def custom_helmert(
        self,
    ) -> tuple[float, float, float, float, float, float, float] | None:
        vals = (
            self.helmert_dx,
            self.helmert_dy,
            self.helmert_dz,
            self.helmert_rx_as,
            self.helmert_ry_as,
            self.helmert_rz_as,
            self.helmert_ds_ppm,
        )
        if any(v is None for v in vals):
            return None
        return vals  # type: ignore[return-value]


@dataclass
class DownloadParams:
    """
    Pickle-serializable параметры для запуска создания карты.

    Не зависит от Qt — может передаваться через multiprocessing.
    """

    center_x: float
    center_y: float
    width_m: float
    height_m: float
    api_key: str
    output_path: str
    settings: MapSettings
