"""Утилиты для работы в portable режиме."""

import sys
from pathlib import Path


def is_portable_mode() -> bool:
    """
    Определяет, запущено ли приложение в portable режиме.

    Portable режим активируется, если имя исполняемого файла содержит '_portable'.
    Например: SK42_portable.exe

    Returns:
        bool: True если приложение в portable режиме, иначе False

    """
    exe_name = Path(sys.argv[0]).name.lower()
    return '_portable' in exe_name


def get_app_dir() -> Path:
    """
    Возвращает директорию приложения (где находится exe файл).

    Returns:
        Path: Путь к директории приложения

    """
    return Path(sys.argv[0]).resolve().parent


def get_portable_path(subdir: str) -> Path:
    """
    Возвращает путь к поддиректории для portable режима.

    В portable режиме все данные хранятся относительно exe файла:
    - cache/ - кэш тайлов
    - configs/ - конфигурационные файлы
    - logs/ - лог-файлы
    - maps/ - сохранённые карты

    Args:
        subdir: Имя поддиректории (например, 'cache', 'configs', 'logs', 'maps')

    Returns:
        Path: Полный путь к поддиректории

    """
    return get_app_dir() / subdir
