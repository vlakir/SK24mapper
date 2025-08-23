from pydantic import BaseModel, field_validator

from constants import ADDITIVE_RATIO, MAX_ZOOM


class MapSettings(BaseModel):
    """Установки, которые раньше были разбросаны по модулю, собраны в один датакласс."""

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
    # Желательный зум (будет снижен автоматически при превышении лимитов)
    zoom: int = MAX_ZOOM

    # Источник тайлов Mapbox (Styles/XYZ)
    # Идентификатор стиля, например:
    #   "mapbox/satellite-v9" или "mapbox/satellite-streets-v12"
    style_id: str = 'mapbox/satellite-v9'
    # Размер тайла стиля (256 или 512) — влияет на плотность пикселей
    tile_size: int = 512
    # HiDPI: если true — добавлять "@2x" к URL (в 2 раза больше физических пикселей)
    use_retina: bool = True
    # Параллелизм загрузки
    concurrency: int = 20

    # Параметры старого статического источника (сохраняем для обратной совместимости)
    # Размер кадра (px) — ширина/высота (лимит 1280)
    static_tile_width_px: int = 1024
    static_tile_height_px: int = 1024
    # Формат изображения: png или jpg
    image_format: str = 'png'

    # Толщина линий сетки (px)
    grid_width_px: int
    # Размер шрифта подписей (px)
    grid_font_size: int
    # Толщина обводки текста (px)
    grid_text_outline_width: int
    # Отступ подписи от края изображения (px)
    grid_text_margin: int
    # Внутренний отступ подложки вокруг текста (px)
    grid_label_bg_padding: int
    # Прозрачность белой маски (0.0 — прозрачная, 1.0 — непрозрачная)
    mask_opacity: float

    # Валидации через Pydantic validators
    @field_validator('mask_opacity')
    @classmethod
    def validate_mask_opacity(cls, v: float | str) -> float:
        v = float(v)
        if not (0.0 <= v <= 1.0):
            msg = 'mask_opacity должен быть в диапазоне [0.0, 1.0]'
            raise ValueError(msg)
        return v

    @field_validator('zoom')
    @classmethod
    def validate_zoom(cls, v: int | str) -> int:
        v = int(v)
        if v < 0:
            msg = 'zoom не может быть отрицательным'
            raise ValueError(msg)
        return v

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
