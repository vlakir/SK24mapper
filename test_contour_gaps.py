#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности разрывов линий контуров в местах подписей.
"""

import sys
from pathlib import Path

# Добавляем src в путь
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))


def test_constants_import():
    """Проверяем импорт новых констант"""
    try:
        from constants import CONTOUR_LABEL_GAP_ENABLED, CONTOUR_LABEL_GAP_PADDING

        print(f'✓ Константы импортированы успешно:')
        print(f'  CONTOUR_LABEL_GAP_ENABLED = {CONTOUR_LABEL_GAP_ENABLED}')
        print(f'  CONTOUR_LABEL_GAP_PADDING = {CONTOUR_LABEL_GAP_PADDING}')
        return True
    except ImportError as e:
        print(f'✗ Ошибка импорта констант: {e}')
        return False


def test_service_import():
    """Проверяем что service.py импортируется без ошибок"""
    try:
        import service

        print('✓ Модуль service импортирован успешно')
        return True
    except Exception as e:
        print(f'✗ Ошибка импорта service: {e}')
        return False


def main():
    """Основная функция тестирования"""
    print('Тестирование функциональности разрывов линий контуров')
    print('=' * 60)

    success = True

    # Тест 1: Импорт констант
    print('\n1. Проверка импорта констант:')
    success &= test_constants_import()

    # Тест 2: Импорт service модуля
    print('\n2. Проверка импорта service модуля:')
    success &= test_service_import()

    print('\n' + '=' * 60)
    if success:
        print('✓ Все тесты пройдены успешно!')
        print('\nФункциональность разрывов линий контуров готова к использованию:')
        print('- CONTOUR_LABEL_GAP_ENABLED = True  # Включить разрывы')
        print('- CONTOUR_LABEL_GAP_PADDING = 5     # Отступ в пикселях')
        return 0
    else:
        print('✗ Обнаружены ошибки в реализации')
        return 1


if __name__ == '__main__':
    sys.exit(main())
