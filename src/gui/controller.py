"""Контроллер взаимодействия между моделью и представлением (MVC)."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import tomlkit
from dotenv import load_dotenv

from diagnostics import ResourceMonitor, log_memory_usage
from gui.model import MilMapperModel, ModelEvent
from profiles import ensure_profiles_dir, load_profile, save_profile
from service import download_satellite_rectangle

logger = logging.getLogger(__name__)


class MilMapperController:
    """Основной контроллер приложения в парадигме MVC."""

    def __init__(self, model: MilMapperModel) -> None:
        """Инициализация контроллера ссылкой на модель."""
        self._model = model
        self._api_key: str | None = None
        self._load_api_key()
        logger.info('MilMapperController initialized')

    def _load_api_key(self) -> None:
        """Загрузка API-ключа из переменных окружения (.env/.secrets.env) для разных сценариев запуска."""
        try:
            import sys
            # Каталог установленного приложения (папка с exe при сборке PyInstaller)
            install_dir = Path(sys.argv[0]).resolve().parent
            # Рабочая директория процесса (может отличаться от install_dir)
            cwd = Path.cwd()
            # Корень проекта при разработке (по исходникам)
            repo_root = Path(__file__).resolve().parent.parent.parent

            # Путь в профиле пользователя (на будущее/альтернатива хранения)
            appdata = os.getenv('APPDATA')
            appdata_path = Path(appdata) / 'SK42mapper' if appdata else None

            candidates = [
                install_dir / '.secrets.env',
                install_dir / '.env',
            ]
            if appdata_path is not None:
                candidates.insert(1, appdata_path / '.secrets.env')
                candidates.insert(2, appdata_path / '.env')
            candidates.extend([
                cwd / '.secrets.env',
                cwd / '.env',
                repo_root / '.secrets.env',
                repo_root / '.env',
            ])

            for p in candidates:
                try:
                    if p and p.exists():
                        load_dotenv(p)
                        break
                except Exception:
                    # если конкретный файл проблемный — продолжаем искать
                    continue

            self._api_key = os.getenv('API_KEY', '').strip()
            if not self._api_key:
                error_msg = 'API-ключ не найден в переменных окружения'
                logger.error(error_msg)
                self._model.notify_observers(
                    ModelEvent.ERROR_OCCURRED, {'error': error_msg}
                )
            else:
                logger.info('API-ключ успешно загружен')

        except Exception as e:
            error_msg = f'Не удалось загрузить API-ключ: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED, {'error': error_msg}
            )

    def update_setting(self, field_name: str, value: Any) -> None:
        """Точечное обновление одного поля настроек."""
        try:
            self._model.update_settings(**{field_name: value})
            logger.debug(f'Обновлена настройка {field_name} = {value}')
        except Exception as e:
            error_msg = f'Не удалось обновить настройку {field_name}: {e}'
            logger.exception(error_msg)

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
                ModelEvent.ERROR_OCCURRED, {'error': error_msg}
            )

    def save_current_profile(self, profile_name: str) -> None:
        """Сохранение текущих настроек в файл профиля."""
        try:
            save_profile(profile_name, self._model.settings)
            self._model.save_profile(profile_name)
            logger.info(f'Профиль сохранён: {profile_name}')
        except Exception as e:
            error_msg = f'Не удалось сохранить профиль {profile_name}: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED, {'error': error_msg}
            )

    def save_current_profile_as(self, file_path: str) -> str | None:
        """Сохранение текущих настроек в новый файл профиля по заданному пути."""
        try:
            profile_path = Path(file_path)
            if profile_path.suffix.lower() != '.toml':
                profile_path = profile_path.with_suffix('.toml')

            profile_name = profile_path.stem

            profiles_dir = ensure_profiles_dir()

            if not profile_path.is_absolute() and profile_path.parent == Path('..'):
                final_path = profiles_dir / profile_path.name
            else:
                final_path = profile_path

            data = self._model.settings.model_dump()
            text = tomlkit.dumps(data)
            final_path.write_text(text, encoding='utf-8')

            self._model.save_profile(profile_name)
            logger.info(f'Профиль сохранён как: {final_path}')
            return profile_name

        except Exception as e:
            error_msg = f'Не удалось сохранить профиль как {file_path}: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED, {'error': error_msg}
            )
            return None

    def validate_api_key(self) -> bool:
        """Проверка наличия валидного API-ключа в окружении."""
        return self._api_key is not None and len(self._api_key) > 0

    async def start_map_download(self) -> None:
        """Запуск процесса загрузки карты."""
        if not self.validate_api_key():
            error_msg = 'API-ключ недоступен для загрузки'
            logger.error(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED, {'error': error_msg}
            )
            return

        with ResourceMonitor('MAP_DOWNLOAD'):
            try:
                self._model.start_download()
                settings = self._model.settings

                center_x = (
                    settings.bottom_left_x_sk42_gk + settings.top_right_x_sk42_gk
                ) / 2
                center_y = (
                    settings.bottom_left_y_sk42_gk + settings.top_right_y_sk42_gk
                ) / 2
                width_m = settings.top_right_x_sk42_gk - settings.bottom_left_x_sk42_gk
                height_m = settings.top_right_y_sk42_gk - settings.bottom_left_y_sk42_gk

                logger.info(
                    f'Starting download: center=({center_x}, {center_y}), size=({width_m}x{height_m})'
                )
                log_memory_usage('before download start')

                # Start download
                await download_satellite_rectangle(
                    center_x_sk42_gk=center_x,
                    center_y_sk42_gk=center_y,
                    width_m=width_m,
                    height_m=height_m,
                    api_key=self._api_key,
                    output_path=settings.output_path,
                    settings=settings,
                )

                log_memory_usage('after download complete')
                self._model.complete_download(success=True)
                logger.info('Загрузка завершена успешно')

            except Exception as e:
                error_msg = f'Не удалось выполнить загрузку: {e}'
                logger.error(error_msg, exc_info=True)
                self._model.complete_download(success=False, error_msg=error_msg)

    def start_map_download_sync(self) -> None:
        """Запуск загрузки карты в синхронном контексте (обёртка над async)."""
        try:
            asyncio.run(self.start_map_download())
        except Exception as e:
            error_msg = f'Не удалось запустить загрузку: {e}'
            logger.exception(error_msg)
            self._model.complete_download(success=False, error_msg=error_msg)

    def get_available_profiles(self) -> list[str]:
        """Получение списка доступных профилей (по именам файлов TOML)."""
        try:
            profiles_dir = Path(__file__).parent.parent / 'configs' / 'profiles'
            if not profiles_dir.exists():
                return ['default']

            profiles = []
            for file_path in profiles_dir.glob('*.toml'):
                profiles.append(file_path.stem)

            return sorted(profiles) if profiles else ['default']

        except Exception as e:
            logger.exception(f'Не удалось получить список профилей: {e}')
            return ['default']

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
                ModelEvent.ERROR_OCCURRED, {'error': error_msg}
            )

    def update_grid_settings(self, grid_settings: dict[str, Any]) -> None:
        """Обновление настроек сетки (фильтруются только допустимые ключи)."""
        try:
            valid_keys = {
                'grid_width_px',
                'grid_font_size',
                'grid_text_margin',
                'grid_label_bg_padding',
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
                ModelEvent.ERROR_OCCURRED, {'error': error_msg}
            )

    def update_output_settings(self, output_settings: dict[str, Any]) -> None:
        """Обновление настроек вывода (фильтруются только допустимые ключи)."""
        try:
            valid_keys = {'output_path', 'mask_opacity', 'jpeg_quality'}

            filtered_settings = {
                k: v for k, v in output_settings.items() if k in valid_keys
            }
            if filtered_settings:
                self._model.update_settings(**filtered_settings)

        except Exception as e:
            error_msg = f'Не удалось обновить настройки вывода: {e}'
            logger.exception(error_msg)
            self._model.notify_observers(
                ModelEvent.ERROR_OCCURRED, {'error': error_msg}
            )
