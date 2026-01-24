"""Модель и инфраструктура наблюдателей (Observer) для MVC-архитектуры."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from domain.models import MapSettings

logger = logging.getLogger(__name__)


class ModelEvent(str, Enum):
    """События, которые генерирует модель."""

    SETTINGS_CHANGED = 'settings_changed'
    PROFILE_LOADED = 'profile_loaded'
    PROFILE_SAVED = 'profile_saved'
    DOWNLOAD_STARTED = 'download_started'
    DOWNLOAD_PROGRESS = 'download_progress'
    DOWNLOAD_COMPLETED = 'download_completed'
    DOWNLOAD_FAILED = 'download_failed'
    PREVIEW_UPDATED = 'preview_updated'
    WARNING_OCCURRED = 'warning_occurred'
    ERROR_OCCURRED = 'error_occurred'


class EventData(BaseModel):
    """Базовая структура данных события модели."""

    event: ModelEvent
    timestamp: float = Field(default_factory=lambda: __import__('time').time())
    data: dict[str, Any] = Field(default_factory=dict)


class Observer:
    """Базовый интерфейс наблюдателя."""

    def update(self, event_data: EventData) -> None:
        """Обработчик уведомлений модели (должен быть реализован в наследниках)."""
        msg = 'Метод update должен быть реализован в наследнике'
        raise NotImplementedError(msg)


class Observable:
    """Mixin class to add Observer pattern functionality."""

    def __init__(self) -> None:
        self._observers: list[Observer] = []

    def add_observer(self, observer: Observer) -> None:
        """Add an observer to receive notifications."""
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug(f'Added observer: {observer.__class__.__name__}')

    def remove_observer(self, observer: Observer) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
            logger.debug(f'Removed observer: {observer.__class__.__name__}')

    def notify_observers(
        self,
        event: ModelEvent,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Notify all observers of an event."""
        event_data = EventData(event=event, data=data or {})
        logger.debug(f'Notifying {len(self._observers)} observers of {event}')

        for observer in self._observers:
            try:
                observer.update(event_data)
            except Exception as e:
                logger.exception(
                    f'Error notifying observer {observer.__class__.__name__}: {e}',
                )


class ApplicationState(BaseModel):
    """Application state data."""

    current_profile_name: str = 'default'
    is_downloading: bool = False
    download_progress: int = 0
    download_total: int = 0
    download_label: str = ''
    last_error: str | None = None
    preview_image_path: str | None = None

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class MilMapperModel(Observable):
    """Main application model implementing Observer pattern."""

    def __init__(self) -> None:
        super().__init__()
        self._settings = MapSettings(
            from_x_high=54,
            from_y_high=74,
            to_x_high=54,
            to_y_high=74,
            from_x_low=14,
            from_y_low=43,
            to_x_low=23,
            to_y_low=49,
            output_path='../../maps/map.jpg',
            grid_width_m=5.0,
            grid_font_size_m=100.0,
            grid_text_margin_m=50.0,
            grid_label_bg_padding_m=10.0,
            mask_opacity=0.35,
        )
        self._state = ApplicationState()
        logger.info('MilMapperModel initialized')

    @property
    def settings(self) -> MapSettings:
        """Get current map settings."""
        return self._settings

    @property
    def state(self) -> ApplicationState:
        """Get current application state."""
        return self._state

    def update_settings(self, **kwargs: Any) -> None:
        """Update map settings and notify observers."""
        try:
            # Create new settings object with updated values
            current_data = self._settings.model_dump()
            current_data.update(kwargs)
            new_settings = MapSettings(**current_data)

            self._settings = new_settings
            self.notify_observers(
                ModelEvent.SETTINGS_CHANGED,
                {'settings': new_settings},
            )
            logger.debug(f'Settings updated: {list(kwargs.keys())}')

        except Exception as e:
            error_msg = f'Не удалось обновить настройки: {e}'
            logger.exception(error_msg)
            self._state.last_error = error_msg
            self.notify_observers(ModelEvent.ERROR_OCCURRED, {'error': error_msg})

    def load_profile(self, profile_name: str, settings: MapSettings) -> None:
        """Load profile settings and notify observers."""
        try:
            self._settings = settings
            self._state.current_profile_name = profile_name
            self._state.last_error = None

            self.notify_observers(
                ModelEvent.PROFILE_LOADED,
                {'profile_name': profile_name, 'settings': settings},
            )
            logger.info(f'Profile loaded: {profile_name}')

        except Exception as e:
            error_msg = f'Не удалось загрузить профиль {profile_name}: {e}'
            logger.exception(error_msg)
            self._state.last_error = error_msg
            self.notify_observers(ModelEvent.ERROR_OCCURRED, {'error': error_msg})

    def save_profile(self, profile_name: str) -> None:
        """Save current settings to profile and notify observers."""
        try:
            self._state.current_profile_name = profile_name
            self._state.last_error = None

            self.notify_observers(
                ModelEvent.PROFILE_SAVED,
                {'profile_name': profile_name, 'settings': self._settings},
            )
            logger.info(f'Profile saved: {profile_name}')

        except Exception as e:
            error_msg = f'Не удалось сохранить профиль {profile_name}: {e}'
            logger.exception(error_msg)
            self._state.last_error = error_msg
            self.notify_observers(ModelEvent.ERROR_OCCURRED, {'error': error_msg})

    def start_download(self) -> None:
        """Mark download as started and notify observers."""
        self._state.is_downloading = True
        self._state.download_progress = 0
        self._state.download_total = 0
        self._state.last_error = None

        self.notify_observers(ModelEvent.DOWNLOAD_STARTED, {'settings': self._settings})
        logger.info('Download started')

    def update_download_progress(self, done: int, total: int, label: str) -> None:
        """Update download progress and notify observers."""
        self._state.download_progress = done
        self._state.download_total = total
        self._state.download_label = label

        self.notify_observers(
            ModelEvent.DOWNLOAD_PROGRESS,
            {'done': done, 'total': total, 'label': label},
        )

    def complete_download(self, success: bool, error_msg: str | None = None) -> None:
        """Mark download as completed and notify observers."""
        self._state.is_downloading = False
        self._state.last_error = error_msg

        if success:
            self.notify_observers(
                ModelEvent.DOWNLOAD_COMPLETED,
                {'settings': self._settings},
            )
            logger.info('Download completed successfully')
        else:
            self.notify_observers(
                ModelEvent.DOWNLOAD_FAILED,
                {'error': error_msg or 'Неизвестная ошибка'},
            )
            logger.error(f'Download failed: {error_msg}')

    def update_preview(self, image_path: str | None) -> None:
        """Update preview image and notify observers."""
        self._state.preview_image_path = image_path

        self.notify_observers(ModelEvent.PREVIEW_UPDATED, {'image_path': image_path})
        logger.debug(f'Preview updated: {image_path}')
