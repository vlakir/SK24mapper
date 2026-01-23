# Техническое задание: Оптимизация производительности SK24mapper

## Цель

Ускорить построение карт в режимах высот и радиогоризонта в 10-50 раз за счёт оптимизации вычислительно-интенсивных участков кода.

---

## Этап 1: Оптимизация модуля `radio_horizon.py` (приоритет: критический)

### 1.1. Внедрение Numba JIT для трассировки лучей

**Файл:** `src/radio_horizon.py`

**Задачи:**
1. Добавить зависимость `numba` в `pyproject.toml`
2. Переписать `_trace_line_of_sight_np()` с декоратором `@njit(cache=True)`
3. Переписать `_bilinear_interpolate_np()` с декоратором `@njit(cache=True)`
4. Убедиться, что все типы данных совместимы с Numba (использовать `np.float32`, `np.int32`)

**Пример реализации:**
```python
from numba import njit

@njit(cache=True)
def _trace_line_of_sight_numba(
    dem: np.ndarray,
    antenna_row: int,
    antenna_col: int,
    antenna_abs_height: float,
    target_row: int,
    target_col: int,
    pixel_size_m: float,
    effective_earth_radius: float,
) -> float:
    # Существующая логика без изменений
    # Numba скомпилирует в машинный код
    ...
```

**Ожидаемый результат:** ускорение в 10-30× для одиночного вызова

---

### 1.2. Параллелизация вычислений на грубой сетке

**Файл:** `src/radio_horizon.py`, функция `compute_radio_horizon()`

**Задачи:**
1. Вынести цикл вычисления `grid_values` в отдельную Numba-функцию
2. Использовать `@njit(parallel=True)` и `prange` для параллельного выполнения
3. Обеспечить thread-safe доступ к массиву результатов

**Пример реализации:**
```python
from numba import njit, prange

@njit(parallel=True, cache=True)
def _compute_grid_values_parallel(
    dem: np.ndarray,
    antenna_row: int,
    antenna_col: int,
    antenna_abs_height: float,
    pixel_size_m: float,
    effective_earth_radius: float,
    grid_step: int,
) -> np.ndarray:
    h, w = dem.shape
    grid_h = (h + grid_step - 1) // grid_step
    grid_w = (w + grid_step - 1) // grid_step
    grid_values = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    for gr in prange(grid_h):  # параллельный цикл по строкам
        row = min(gr * grid_step, h - 1)
        for gc in range(grid_w):
            col = min(gc * grid_step, w - 1)
            grid_values[gr, gc] = _trace_line_of_sight_numba(
                dem, antenna_row, antenna_col, antenna_abs_height,
                row, col, pixel_size_m, effective_earth_radius
            )
    return grid_values
```

**Ожидаемый результат:** дополнительное ускорение в 2-8× (зависит от числа ядер CPU)

---

### 1.3. Оптимизация интерполяции через scipy

**Файл:** `src/radio_horizon.py`, функция `compute_radio_horizon()`

**Задачи:**
1. Заменить ручной цикл интерполяции (строки 353-371) на `scipy.ndimage.zoom`
2. Добавить fallback на Numba-версию если scipy недоступен

**Пример реализации:**
```python
from scipy.ndimage import zoom

def compute_radio_horizon(...) -> np.ndarray:
    ...
    # Вычисляем на грубой сетке (уже оптимизировано в 1.2)
    grid_values = _compute_grid_values_parallel(...)
    
    # Интерполяция до полного размера
    # zoom_factors для достижения размера (h, w)
    zoom_h = h / grid_values.shape[0]
    zoom_w = w / grid_values.shape[1]
    result = zoom(grid_values, (zoom_h, zoom_w), order=1, mode='nearest')
    
    # Обрезаем до точного размера (zoom может дать ±1 пиксель)
    return result[:h, :w].astype(np.float32)
```

**Ожидаемый результат:** ускорение интерполяции в 5-10×

---

## Этап 2: Оптимизация модуля `topography.py` (приоритет: средний)

### 2.1. Кэширование DEM-тайлов

**Файл:** `src/topography.py`

**Задачи:**
1. Создать LRU-кэш для загруженных DEM-тайлов
2. Использовать `functools.lru_cache` или собственный кэш с ограничением по памяти
3. Добавить возможность сохранения кэша на диск для повторных запусков

**Пример реализации:**
```python
from functools import lru_cache

# Кэш на 100 тайлов (~400 MB при 512x512 float32)
@lru_cache(maxsize=100)
def get_dem_tile_cached(x: int, y: int, zoom: int) -> np.ndarray:
    return fetch_dem_tile(x, y, zoom)
```

**Ожидаемый результат:** ускорение при повторных запросах к тем же областям

---

### 2.2. Векторизация цветовой раскраски высот

**Файл:** `src/topography.py`

**Задачи:**
1. Проверить, что `colorize_elevation()` использует numpy vectorized операции
2. Если есть циклы — заменить на numpy broadcasting
3. Использовать предвычисленные LUT (lookup tables) для цветов

---

## Этап 3: Рефакторинг `service.py` (приоритет: низкий, но важен для maintainability)

### 3.1. Разбиение монолитной функции

**Файл:** `src/service.py`

**Задачи:**
1. Выделить загрузку тайлов в отдельный модуль `src/tiles/loader.py`
2. Выделить построение DEM в `src/dem/builder.py`
3. Выделить рендеринг карты в `src/render/map_renderer.py`
4. Устранить дублирование `build_seed_polylines()` (встречается дважды)

**Структура после рефакторинга:**
```
src/
├── dem/
│   ├── __init__.py
│   ├── builder.py      # Сборка DEM из тайлов
│   └── cache.py        # Кэширование DEM
├── tiles/
│   ├── loader.py       # Загрузка тайлов
│   └── ...
├── render/
│   ├── map_renderer.py # Рендеринг карты
│   └── ...
└── service.py          # Оркестрация (тонкий слой)
```

---

## Этап 4: Тестирование и бенчмаркинг

### 4.1. Создание бенчмарков

**Файл:** `tests/benchmarks/test_performance.py`

**Задачи:**
1. Создать тестовые DEM-матрицы разных размеров (1000×1000, 4000×4000, 8000×8000)
2. Измерить время выполнения до и после оптимизации
3. Добавить CI-проверку на регрессию производительности

**Пример:**
```python
import pytest
import time
import numpy as np
from radio_horizon import compute_radio_horizon

@pytest.mark.benchmark
def test_radio_horizon_performance():
    dem = np.random.rand(4000, 4000).astype(np.float32) * 500
    
    start = time.perf_counter()
    result = compute_radio_horizon(
        dem, antenna_row=2000, antenna_col=2000,
        antenna_height_m=10, pixel_size_m=10
    )
    elapsed = time.perf_counter() - start
    
    assert elapsed < 5.0, f"Radio horizon took {elapsed:.2f}s, expected < 5s"
```

---

### 4.2. Регрессионные тесты

**Задачи:**
1. Сохранить эталонные результаты для нескольких тестовых случаев
2. Проверить, что оптимизированный код даёт идентичные результаты (с допуском на float погрешность)

---

## План реализации по шагам

| Шаг | Задача | Файлы | Время | Риск |
|-----|--------|-------|-------|------|
| 1 | Добавить `numba` в зависимости | `pyproject.toml` | 5 мин | Низкий |
| 2 | Переписать `_trace_line_of_sight_np` на Numba | `radio_horizon.py` | 1 час | Средний |
| 3 | Переписать `_bilinear_interpolate_np` на Numba | `radio_horizon.py` | 30 мин | Низкий |
| 4 | Добавить параллельный цикл с `prange` | `radio_horizon.py` | 1 час | Средний |
| 5 | Заменить интерполяцию на `scipy.ndimage.zoom` | `radio_horizon.py` | 30 мин | Низкий |
| 6 | Написать бенчмарки | `tests/benchmarks/` | 1 час | Низкий |
| 7 | Тестирование на реальных данных | — | 2 часа | Низкий |
| 8 | Добавить кэширование DEM | `topography.py` | 2 часа | Средний |
| 9 | Рефакторинг `service.py` | Несколько файлов | 4-8 часов | Высокий |

---

## Зависимости для добавления

```toml
# pyproject.toml
[tool.poetry.dependencies]
numba = "^0.59.0"
scipy = "^1.12.0"  # если ещё не добавлен
```

---

## Критерии успеха

1. **Режим радиогоризонта**: время построения карты 4000×4000 < 5 секунд (было ~60-120 сек)
2. **Режим высот**: время построения < 3 секунд для той же области
3. **Регрессия**: результаты идентичны оригинальным (RMSE < 0.01)
4. **Память**: пиковое потребление не увеличилось более чем на 20%

---

## Риски и митигация

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| Numba не поддерживает какую-то конструкцию | Средняя | Переписать проблемный участок в Numba-совместимом стиле |
| Результаты отличаются из-за порядка операций | Низкая | Использовать `np.float64` для промежуточных вычислений |
| Увеличение времени первого запуска (JIT-компиляция) | Высокая | Использовать `cache=True` для сохранения скомпилированного кода |
| Проблемы с многопоточностью | Низкая | Numba `prange` безопасен для независимых итераций |
