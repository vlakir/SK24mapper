from pydantic import BaseModel, field_validator

from constants import ADDITIVE_RATIO, MapType, default_map_type


class MapSettings(BaseModel):
    """Установки, которые раньше были разбросаны по модулю, собраны в один датакласс."""

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

    # Толщина линий сетки (px)
    grid_width_px: int
    # Размер шрифта подписей (px)
    grid_font_size: int
    # Отступ подписи от края изображения (px)
    grid_text_margin: int
    # Внутренний отступ подложки вокруг текста (px)
    grid_label_bg_padding: int
    # Прозрачность белой маски (0.0 — прозрачная, 1.0 — непрозрачная)
    mask_opacity: float
    # Качество JPEG (10–100). По умолчанию — 95.
    jpeg_quality: int = 95

    # Коррекция изображения
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0

    # Валидации через Pydantic validators
    @field_validator('mask_opacity')
    @classmethod
    def validate_mask_opacity(cls, v: float | str) -> float:
        v = float(v)
        if not (0.0 <= v <= 1.0):
            msg = 'mask_opacity должен быть в диапазоне [0.0, 1.0]'
            raise ValueError(msg)
        return v

    @field_validator('jpeg_quality')
    @classmethod
    def validate_jpeg_quality(cls, v: int | str) -> int:
        iv = int(v)
        iv = max(iv, 10)
        return min(iv, 100)

    @field_validator('brightness', 'contrast', 'saturation')
    @classmethod
    def validate_adjustments(cls, v: float | str) -> float:
        fv = float(v)
        # Допускаем диапазон 0.0–2.0 (0%–200%)
        fv = max(fv, 0.0)
        return min(fv, 2.0)

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
