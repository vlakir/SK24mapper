# SK24mapper — Инструкции для Claude Code

## О проекте
**SK24mapper (Mil Mapper 2.0)** — приложение для создания топографических карт со спутниковыми снимками и километровой сеткой в системе координат СК-42 / Гаусса-Крюгера.

## Запуск
```bash
cd /c/programming/SK24mapper && python src/main.py      # запуск GUI
cd /c/programming/SK24mapper && python -m pytest tests/ -x -q  # тесты
```
- Логи: `%LOCALAPPDATA%\SK42mapper\log\mil_mapper.log`
- Карты: `%APPDATA%\SK42mapper\maps\`
- Кэш тайлов: `%LOCALAPPDATA%\SK42mapper\.cache\`

## Архитектура

### Слои
```
src/
├── main.py                          # Точка входа
├── gui/
│   ├── view.py                      # PySide6 GUI (~1400 строк)
│   ├── controller.py                # Логика управления
│   └── model.py                     # Модель данных
├── services/
│   ├── map_download_service.py      # Оркестратор создания карт
│   ├── map_context.py               # MapDownloadContext — общий контекст
│   ├── processors/                  # Процессоры по типам карт
│   │   ├── radio_horizon.py         # Радиогоризонт
│   │   ├── elevation_color.py       # Цветная карта высот
│   │   ├── elevation_hillshade.py   # Теневая отмывка рельефа (IN PROGRESS)
│   │   ├── elevation_contours.py    # Изолинии (overlay поверх других режимов)
│   │   ├── radar_coverage.py        # Зона обнаружения РЛС
│   │   └── xyz_tiles.py             # Кастомные XYZ тайлы
│   ├── tile_coverage.py             # Расчёт покрытия тайлами
│   └── tile_fetcher.py              # Загрузка тайлов
├── geo/
│   └── topography.py                # DEM, hillshade, координатные преобразования
├── elevation/
│   └── provider.py                  # ElevationTileProvider — кэширование Terrain-RGB
├── shared/
│   └── constants.py                 # Все константы, enum MapType
└── contours/                        # Построение и подписи изолиний
```

### Ключевые паттерны
- **Процессоры** подключаются через `importlib.import_module()` в `map_download_service.py:_run_processor()`
- **Интерактивный alpha-слайдер**: процессоры сохраняют `ctx.rh_cache_topo_base` и `ctx.rh_cache_coverage` для blend
- **DEM для курсора/изолиний**: `ctx.raw_dem_for_cursor` — ОБЯЗАТЕЛЬНО должен совпадать по размеру с `ctx.crop_rect`
- **MapType enum** в `shared/constants.py` определяет все типы карт

## Зависимости
Python 3.13, PySide6, aiohttp, Pillow, pyproj, scipy, opencv-python (cv2), numpy

API: Mapbox (токен в `.env` или `.secrets.env`)

---

## ПРИОРИТЕТНАЯ ЗАДАЧА: Hillshade — проблема грубой сетки высот

### Статус: IN PROGRESS — требуется исследование

### Что сделано
1. Реализован процессор `elevation_hillshade.py` — теневая отмывка рельефа
2. Функция `compute_hillshade()` в `geo/topography.py` — GDAL-совместимая формула
3. Подключение в `map_download_service.py` — роутинг по `MapType.ELEVATION_HILLSHADE`
4. GUI: пункт в выпадающем меню, alpha-слайдер работает
5. Загрузка DEM на нативном zoom=14 (вместо display zoom) — меньше тайлов, чище данные
6. Изолинии восстановлены (resize DEM до display resolution для `ctx.raw_dem_for_cursor`)

### Нерешённая проблема
**Грубая пикселизация на hillshade карте** — видны крупные квадратные пиксели, особенно в верхней части карты. Рельеф выглядит как ступенчатая лестница, а не плавные склоны.

### Что уже исследовано
- **Mapbox Terrain-RGB нативное разрешение = zoom 14** (~5 м/пиксель). z15/z16 — интерполяция.
- **Качество данных неравномерно**: в тестовой области (широта ~57.8°, Кострома) верхние тайлы имеют всего 16 уникальных значений высоты на 256px (шаг ~122м постоянной высоты!), нижние — 43 значения.
- **Downsampling убран** — DEM загружается на z14 напрямую через `_load_dem_native()`.
- **Gaussian smoothing** (sigma=1.5) применяется к поверхности высот ДО вычисления градиентов — это правильный подход, но недостаточный.
- **z_factor=8.0** (вертикальное преувеличение) усиливает видимость, но также усиливает артефакты ступенчатости.

### Направления для исследования
1. **Альтернативные провайдеры DEM** — возможно, есть источники с лучшим разрешением для данного региона (SRTM 30m, Copernicus DEM 30m, ALOS)
2. **Адаптивное сглаживание** — разное sigma для участков с разным качеством данных
3. **Интерполяция высот** — бикубическая или сплайн-интерполяция DEM перед compute_hillshade
4. **Multidirectional hillshade** — освещение с нескольких направлений для снижения артефактов
5. **Проверка кэша** — убедиться что тайлы не закэшированы в деградированном качестве

### Ключевые файлы для этой задачи
- `src/services/processors/elevation_hillshade.py` — процессор (весь файл)
- `src/geo/topography.py` — `compute_hillshade()`, `assemble_dem()`, `meters_per_pixel()`
- `src/elevation/provider.py` — `ElevationTileProvider`, кэширование тайлов
- `src/shared/constants.py` — `HILLSHADE_*` константы
- `src/services/tile_coverage.py` — `compute_tile_coverage()`

### Текущие константы hillshade (в `shared/constants.py`)
```python
HILLSHADE_AZIMUTH_DEG = 315.0    # Северо-запад
HILLSHADE_ALTITUDE_DEG = 45.0    # Угол солнца
HILLSHADE_USE_RETINA = False     # 256px тайлы
HILLSHADE_Z_FACTOR = 8.0         # Вертикальное преувеличение
HILLSHADE_DEM_ZOOM = 14          # Нативное разрешение Mapbox
HILLSHADE_SMOOTH_SIGMA = 1.5     # Гауссово сглаживание DEM
```
