"""Контроллер взаимодействия между моделью и представлением (MVC)."""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import Any

import tomlkit
from dotenv import load_dotenv

from domain.models import DownloadParams
from domain.profiles import (
    ensure_profiles_dir,
    list_profiles,
    load_profile,
    save_profile,
)
from gui.model import MilMapperModel, ModelEvent
from shared.constants import API_KEY_VISIBLE_PREFIX_LEN, WIN32_FILE_ATTRIBUTE_HIDDEN
from shared.diagnostics import log_memory_usage, run_deep_verification
from shared.portable import get_app_dir, is_portable_mode

logger = logging.getLogger(__name__)


def _set_win32_hidden(path: Path, *, hidden: bool) -> None:
    """Установка/снятие атрибута Hidden для файла через Win32 API."""
    if sys.platform == 'win32':
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return
        if hidden:
            attrs |= WIN32_FILE_ATTRIBUTE_HIDDEN
        else:
            attrs &= ~WIN32_FILE_ATTRIBUTE_HIDDEN
        ctypes.windll.kernel32.SetFileAttributesW(str(path), attrs)


class MilMapperController:
    """Основной контроллер приложения в парадигме MVC."""

    def __init__(self, model: MilMapperModel) -> None:
        """Инициализация контроллера ссылкой на модель."""
        self._model = model
        self._api_key: str | None = None
        self._load_api_key()
        logger.info('MilMapperController initialized')

    def update_settings_bulk(self, **kwargs: object) -> None:
        """
        Пакетное обновление нескольких настроек за один вызов модели.

        Обновляет модель одним вызовом update_settings, чтобы сгенерировать
        только одно событие SETTINGS_CHANGED и избежать гонок между View и Model.
        """
        try:
            if kwargs:
                self._model.update_settings(**kwargs)
                logger.debug(f'Обновлены настройки (bulk): {list(kwargs.keys())}')
        except Exception as e:
            error_msg = f'Не удалось обновить настройки (bulk): {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.WARNING_OCCURRED,
                {'warning': error_msg},
            )

    def _load_api_key(self) -> None:
        """
        Загрузка API-ключа из переменных окружения (.env/.secrets.env) для разных
        сценариев запуска. В portable режиме приоритет имеет файл api_key.txt.
        """
        try:
            # Portable режим: проверяем api_key.txt рядом с exe
            if is_portable_mode():
                api_key_file = get_app_dir() / 'api_key.txt'
                if api_key_file.exists():
                    try:
                        with api_key_file.open('r', encoding='utf-8') as f:
                            self._api_key = f.read().strip()
                        if self._api_key:
                            logger.info(
                                'API-ключ успешно загружен из api_key.txt '
                                '(portable режим)'
                            )
                            return
                    except Exception as e:
                        logger.warning(f'Не удалось прочитать api_key.txt: {e}')

            # Каталог установленного приложения (папка с exe при сборке PyInstaller)
            install_dir = Path(sys.argv[0]).resolve().parent
            # Рабочая директория процесса (может отличаться от install_dir)
            cwd = Path.cwd()
            # Корень проекта при разработке (по исходникам)
            repo_root = Path(__file__).resolve().parent.parent.parent

            # Путь в профиле пользователя (на будущее/альтернатива хранения)
            appdata = os.getenv('APPDATA')
            appdata_path = Path(appdata) / 'SK42' if appdata else None

            candidates = [
                install_dir / '.secrets.env',
                install_dir / '.env',
            ]
            if appdata_path is not None:
                candidates.insert(1, appdata_path / '.secrets.env')
                candidates.insert(2, appdata_path / '.env')
            candidates.extend(
                [
                    cwd / '.secrets.env',
                    cwd / '.env',
                    repo_root / '.secrets.env',
                    repo_root / '.env',
                ],
            )

            for p in candidates:
                try:
                    if p and p.exists():
                        load_dotenv(p)
                        break
                except Exception as e:
                    # если конкретный файл проблемный — продолжаем искать
                    logger.debug(f'Не удалось загрузить env файл {p}: {e}')
                    continue

            self._api_key = os.getenv('API_KEY', '').strip()
            if not self._api_key:
                error_msg = 'API-ключ не найден в переменных окружения'
                logger.error(error_msg)
                self._model.notify_observers(
                    ModelEvent.ERROR_OCCURRED,
                    {'error': error_msg},
                )
            else:
                logger.info('API-ключ успешно загружен')

        except Exception as e:
            error_msg = f'Не удалось загрузить API-ключ: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED,
                {'error': error_msg},
            )

    def update_setting(self, field_name: str, value: object) -> None:
        """Точечное обновление одного поля настроек."""
        try:
            self._model.update_settings(**{field_name: value})
            logger.debug(f'Обновлена настройка {field_name} = {value}')
        except Exception as e:
            error_msg = f'Не удалось обновить настройку {field_name}: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.WARNING_OCCURRED,
                {'warning': error_msg},
            )

    def load_profile_by_name(self, profile_name: str) -> None:
        """Загрузка профиля по имени из файла."""
        try:
            settings = load_profile(profile_name)
            self._model.load_profile(profile_name, settings)
            logger.info(f'Профиль загружен: {profile_name}')
        except Exception as e:
            error_msg = f'Не удалось загрузить профиль {profile_name}: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED,
                {'error': error_msg},
            )

    def load_profile_from_path(self, file_path: str) -> None:
        """Загрузка профиля из произвольного TOML файла по полному пути."""
        try:
            settings = load_profile(file_path)
            profile_name = Path(file_path).stem
            self._model.load_profile(profile_name, settings)
            logger.info(f'Профиль загружен из файла: {file_path}')
        except Exception as e:
            error_msg = f'Не удалось открыть профиль {file_path}: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED,
                {'error': error_msg},
            )

    def save_current_profile(self, profile_name: str) -> None:
        """Сохранение текущих настроек в файл профиля."""
        try:
            s = self._model.settings
            # Подробный лог перед сохранением, чтобы отловить проблемы
            # «не те значения сохраняются»
            try:
                logger.info(
                    (
                        "Saving profile '%s' with coords: "
                        'from(xH=%s,xL=%s,yH=%s,yL=%s) '
                        '→ BL(%.3f, %.3f); '
                        'to(xH=%s,xL=%s,yH=%s,yL=%s) → TR(%.3f, %.3f); '
                        'control_point(en=%s, X=%.3f, Y=%.3f)'
                    ),
                    profile_name,
                    getattr(s, 'from_x_high', None),
                    getattr(s, 'from_x_low', None),
                    getattr(s, 'from_y_high', None),
                    getattr(s, 'from_y_low', None),
                    getattr(s, 'bottom_left_x_sk42_gk', 0.0),
                    getattr(s, 'bottom_left_y_sk42_gk', 0.0),
                    getattr(s, 'to_x_high', None),
                    getattr(s, 'to_x_low', None),
                    getattr(s, 'to_y_high', None),
                    getattr(s, 'to_y_low', None),
                    getattr(s, 'top_right_x_sk42_gk', 0.0),
                    getattr(s, 'top_right_y_sk42_gk', 0.0),
                    getattr(s, 'control_point_enabled', False),
                    getattr(s, 'control_point_x_sk42_gk', 0.0),
                    getattr(s, 'control_point_y_sk42_gk', 0.0),
                )
            except Exception:
                logger.debug('Failed to log detailed settings before save')

            save_profile(profile_name, s)
            self._model.save_profile(profile_name)
            logger.info(f'Профиль сохранён: {profile_name}')
        except Exception as e:
            error_msg = f'Не удалось сохранить профиль {profile_name}: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED,
                {'error': error_msg},
            )

    def save_current_profile_as(self, file_path: str) -> str | None:
        """Сохранение текущих настроек в новый файл профиля по заданному пути."""
        try:
            dest_path = Path(file_path)
            if dest_path.suffix.lower() != '.toml':
                dest_path = dest_path.with_suffix('.toml')

            profiles_dir = ensure_profiles_dir()
            # Любой относительный путь сохраняем в каталог профилей пользователя
            if not dest_path.is_absolute():
                final_path = profiles_dir / dest_path.name
            else:
                final_path = dest_path

            # Гарантируем существование каталога
            final_path.parent.mkdir(parents=True, exist_ok=True)
            profile_name = final_path.stem

            s = self._model.settings
            # Подробный лог перед сохранением
            try:
                logger.info(
                    (
                        "Saving profile as '%s' → %s with coords: "
                        'from(xH=%s,xL=%s,yH=%s,yL=%s) → BL(%.3f, %.3f); '
                        'to(xH=%s,xL=%s,yH=%s,yL=%s) → TR(%.3f, %.3f); '
                        'control_point(en=%s, X=%.3f, Y=%.3f)'
                    ),
                    profile_name,
                    str(final_path),
                    getattr(s, 'from_x_high', None),
                    getattr(s, 'from_x_low', None),
                    getattr(s, 'from_y_high', None),
                    getattr(s, 'from_y_low', None),
                    getattr(s, 'bottom_left_x_sk42_gk', 0.0),
                    getattr(s, 'bottom_left_y_sk42_gk', 0.0),
                    getattr(s, 'to_x_high', None),
                    getattr(s, 'to_x_low', None),
                    getattr(s, 'to_y_high', None),
                    getattr(s, 'to_y_low', None),
                    getattr(s, 'top_right_x_sk42_gk', 0.0),
                    getattr(s, 'top_right_y_sk42_gk', 0.0),
                    getattr(s, 'control_point_enabled', False),
                    getattr(s, 'control_point_x_sk42_gk', 0.0),
                    getattr(s, 'control_point_y_sk42_gk', 0.0),
                )
            except Exception:
                logger.debug('Failed to log detailed settings before save-as')

            data = s.model_dump()
            text = tomlkit.dumps(data)
            final_path.write_text(text, encoding='utf-8')

            self._model.save_profile(profile_name)
            logger.info(f'Профиль сохранён как: {final_path}')

        except Exception as e:
            error_msg = f'Не удалось сохранить профиль как {file_path}: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED,
                {'error': error_msg},
            )
            return None
        else:
            return profile_name

    def validate_api_key(self) -> bool:
        """Проверка наличия валидного API-ключа в окружении."""
        return self._api_key is not None and len(self._api_key) > 0

    def get_masked_api_key(self) -> str:
        """Возвращает API-ключ с маскировкой (первые N символов + звёздочки)."""
        if not self._api_key:
            return '(ключ не задан)'
        if len(self._api_key) <= API_KEY_VISIBLE_PREFIX_LEN:
            return self._api_key
        return self._api_key[:API_KEY_VISIBLE_PREFIX_LEN] + '*' * (
            len(self._api_key) - API_KEY_VISIBLE_PREFIX_LEN
        )

    def get_api_key(self) -> str:
        """Возвращает полный API-ключ."""
        return self._api_key or ''

    def save_api_key(self, new_key: str) -> bool:
        """Сохранение нового API-ключа на диск и обновление in-memory."""
        try:
            new_key = new_key.strip()
            if not new_key:
                return False

            if is_portable_mode():
                key_file = get_app_dir() / 'api_key.txt'
                key_file.write_text(new_key, encoding='utf-8')
                logger.info('API-ключ сохранён в api_key.txt (portable режим)')
            else:
                appdata = os.getenv('APPDATA')
                if not appdata:
                    logger.error('Переменная APPDATA не найдена')
                    return False
                secrets_dir = Path(appdata) / 'SK42'
                secrets_dir.mkdir(parents=True, exist_ok=True)
                secrets_file = secrets_dir / '.secrets.env'
                # Снимаем Hidden-атрибут перед записью (Windows)
                if sys.platform == 'win32' and secrets_file.exists():
                    _set_win32_hidden(secrets_file, hidden=False)
                secrets_file.write_text(f'API_KEY={new_key}\n', encoding='utf-8')
                # Возвращаем Hidden-атрибут
                if sys.platform == 'win32':
                    _set_win32_hidden(secrets_file, hidden=True)
                logger.info('API-ключ сохранён в .secrets.env')

            self._api_key = new_key
            os.environ['API_KEY'] = new_key
        except Exception:
            logger.exception('Не удалось сохранить API-ключ')
            return False
        else:
            return True

    def prepare_download_params(self) -> DownloadParams:
        """
        Подготовка параметров для запуска создания карты.

        Выполняет валидацию API-ключа, deep verification,
        вычисление координат центра и размеров,
        переключение модели в состояние «загрузка идёт».

        Raises:
            RuntimeError: если API-ключ недоступен или проверка не пройдена.

        Returns:
            Готовый ``DownloadParams`` для передачи в ``DownloadWorker``.

        """
        if not self.validate_api_key():
            error_msg = 'API-ключ недоступен для загрузки'
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Глубокая проверка перед стартом
        try:
            asyncio.run(
                run_deep_verification(
                    api_key=self._api_key or '',
                    settings=self._model.settings,
                )
            )
        except Exception as e:
            error_msg = f'Проверка перед запуском не пройдена: {e}'
            logger.exception(error_msg)
            raise RuntimeError(error_msg) from e

        self._model.start_download()
        settings = self._model.settings

        center_x = (settings.bottom_left_x_sk42_gk + settings.top_right_x_sk42_gk) / 2
        center_y = (settings.bottom_left_y_sk42_gk + settings.top_right_y_sk42_gk) / 2
        width_m = settings.top_right_x_sk42_gk - settings.bottom_left_x_sk42_gk
        height_m = settings.top_right_y_sk42_gk - settings.bottom_left_y_sk42_gk

        logger.info(
            'Starting download: center=(%s, %s), size=(%sx%s)',
            center_x,
            center_y,
            width_m,
            height_m,
        )
        try:
            logger.info(
                'Starting download with control point (СК-42 ГК): '
                'enabled=%s, X(север)=%.3f, Y(восток)=%.3f; raw Xn=%s, '
                'raw Ye=%s',
                getattr(settings, 'control_point_enabled', False),
                getattr(settings, 'control_point_y_sk42_gk', 0.0),
                getattr(settings, 'control_point_x_sk42_gk', 0.0),
                getattr(settings, 'control_point_y', None),
                getattr(settings, 'control_point_x', None),
            )
        except Exception:
            logger.debug('Failed to log control point settings at start')
        log_memory_usage('before download start')

        return DownloadParams(
            center_x=center_x,
            center_y=center_y,
            width_m=width_m,
            height_m=height_m,
            api_key=self._api_key or '',
            output_path=settings.output_path,
            settings=settings,
        )

    def complete_download(self, *, success: bool, error_msg: str = '') -> None:
        """Обновить модель по завершении загрузки (вызывается из GUI-потока)."""
        log_memory_usage('after download complete')
        if success:
            self._model.complete_download(success=True)
            logger.info('Загрузка завершена успешно')
        else:
            self._model.complete_download(success=False, error_msg=error_msg)
            logger.error('Загрузка не удалась: %s', error_msg)

    def get_available_profiles(self) -> list[str]:
        """Получение списка доступных профилей (по именам файлов TOML)."""
        try:
            names = list_profiles()
        except Exception as e:
            msg = f'Не удалось получить список профилей: {e}'
            logger.exception(msg)
            self._model.notify_observers(
                ModelEvent.WARNING_OCCURRED,
                {'warning': msg},
            )
            return ['default']
        else:
            return names if names else ['default']

    def update_coordinates(self, coords: dict[str, int]) -> None:
        """Обновление координатных настроек (фильтруются только допустимые ключи)."""
        try:
            valid_keys = {
                'from_x_high',
                'from_y_high',
                'to_x_high',
                'to_y_high',
                'from_x_low',
                'from_y_low',
                'to_x_low',
                'to_y_low',
            }

            filtered_coords = {k: v for k, v in coords.items() if k in valid_keys}
            if filtered_coords:
                self._model.update_settings(**filtered_coords)

        except Exception as e:
            error_msg = f'Не удалось обновить координаты: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED,
                {'error': error_msg},
            )

    def update_grid_settings(self, grid_settings: dict[str, Any]) -> None:
        """Обновление настроек сетки (фильтруются только допустимые ключи)."""
        try:
            valid_keys = {
                'grid_width_m',
                'grid_font_size_m',
                'grid_text_margin_m',
                'grid_label_bg_padding_m',
            }

            filtered_settings = {
                k: v for k, v in grid_settings.items() if k in valid_keys
            }
            if filtered_settings:
                self._model.update_settings(**filtered_settings)

        except Exception as e:
            error_msg = f'Не удалось обновить настройки сетки: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED,
                {'error': error_msg},
            )

    def update_output_settings(self, output_settings: dict[str, Any]) -> None:
        """Обновление настроек вывода (фильтруются только допустимые ключи)."""
        try:
            valid_keys = {
                'output_path',
                'mask_opacity',
                'jpeg_quality',
                'brightness',
                'contrast',
                'saturation',
            }

            filtered_settings = {
                k: v for k, v in output_settings.items() if k in valid_keys
            }
            if filtered_settings:
                self._model.update_settings(**filtered_settings)

        except Exception as e:
            error_msg = f'Не удалось обновить настройки вывода: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED,
                {'error': error_msg},
            )
