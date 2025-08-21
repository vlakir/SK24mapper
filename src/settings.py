from pydantic import BaseModel, field_validator

from constants import ADDITIVE_RATIO, MAX_ZOOM


class MapSettings(BaseModel):
    """Установки, которые раньше были разбросаны по модулю, собраны в один датакласс."""

    # Координаты области по углам (СК-42 Гаусса-Крюгера, метры)
    # Старшие разряды
    from_x_high: int = 54
    from_y_high: int = 74
    to_x_high: int = 54
    to_y_high: int = 74

    # Младшие разряды
    from_x_low: int = 14
    from_y_low: int = 43
    to_x_low: int = 18
    to_y_low: int = 49

    # Путь к итоговому файлу
    output_path: str = '../map.png'
    # Желательный зум (будет снижен автоматически при превышении лимитов)
    zoom: int = MAX_ZOOM
    # Толщина линий сетки (px)
    grid_width_px: int = 20
    # Размер шрифта подписей (px)
    grid_font_size: int = 86
    # Толщина обводки текста (px)
    grid_text_outline_width: int = 2
    # Отступ подписи от края изображения (px)
    grid_text_margin: int = 43
    # Внутренний отступ подложки вокруг текста (px)
    grid_label_bg_padding: int = 6
    # Прозрачность белой маски (0.0 — прозрачная, 1.0 — непрозрачная)
    mask_opacity: float = 0.35

    # Валидации через Pydantic validators
    @field_validator('mask_opacity')
    @classmethod
    def validate_mask_opacity(cls, v):
        v = float(v)
        if not (0.0 <= v <= 1.0):
            msg = 'mask_opacity должен быть в диапазоне [0.0, 1.0]'
            raise ValueError(msg)
        return v

    @field_validator('zoom')
    @classmethod
    def validate_zoom(cls, v):
        v = int(v)
        if v < 0:
            msg = 'zoom не может быть отрицательным'
            raise ValueError(msg)
        return v

    # Вычисляемые свойства из исходных параметров + ADDITIVE_RATIO
    @property
    def bottom_left_x_sk42_gk(self) -> float:
        return 1e3 * (self.from_x_low - ADDITIVE_RATIO) + 1e5 * self.from_x_high

    @property
    def bottom_left_y_sk42_gk(self) -> float:
        return 1e3 * (self.from_y_low - ADDITIVE_RATIO) + 1e5 * self.from_y_high

    @property
    def top_right_x_sk42_gk(self) -> float:
        return 1e3 * (self.to_x_low + ADDITIVE_RATIO) + 1e5 * self.to_x_high

    @property
    def top_right_y_sk42_gk(self) -> float:
        return 1e3 * (self.to_y_low + ADDITIVE_RATIO) + 1e5 * self.to_y_high
