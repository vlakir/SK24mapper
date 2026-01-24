import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from typing import ClassVar

from shared.constants import DEFAULT_WRITER, SingleLineRenderer

logger = logging.getLogger(__name__)


# Глобальные колбэки для интеграции с GUI (опционально)
class _CbStore:
    progress: Callable[[int, int, str], None] | None = None
    spinner_start: Callable[[str], None] | None = None
    spinner_stop: Callable[[str], None] | None = None
    preview_image: Callable[[object], None] | None = None


def set_progress_callback(cb: Callable[[int, int, str], None] | None) -> None:
    """Устанавливает глобальный колбэк прогресса: (done, total, label)."""
    _CbStore.progress = cb


def set_spinner_callbacks(
    on_start: Callable[[str], None] | None,
    on_stop: Callable[[str], None] | None,
) -> None:
    """Устанавливает глобальные колбэки для старта/остановки спиннера."""
    _CbStore.spinner_start = on_start
    _CbStore.spinner_stop = on_stop


def set_preview_image_callback(cb: Callable[[object], None] | None) -> None:
    """Устанавливает колбэк предпросмотра (получает PIL.Image)."""
    _CbStore.preview_image = cb


def publish_preview_image(img: object) -> bool:
    """
    Публикует изображение предпросмотра в GUI, если колбэк установлен.

    Возвращает True, если колбэк был установлен и вызван без исключений.
    Тип img — PIL.Image.Image (используем object во избежание жёсткой зависимости).
    """
    cb = _CbStore.preview_image
    if cb is not None:
        try:
            cb(img)
        except Exception:
            return False
        else:
            return True
    return False


def cleanup_all_progress_resources() -> None:
    """
    Очистка всех ресурсов прогресса: остановка спиннеров и очистка колбэков.

    Вызывается при закрытии приложения.
    """
    # Останавливаем все активные спиннеры
    _spinner_registry.stop_all()

    # Очищаем все глобальные колбэки
    _CbStore.progress = None
    _CbStore.spinner_start = None
    _CbStore.spinner_stop = None
    _CbStore.preview_image = None


def force_stop_all_spinners() -> None:
    """Принудительная остановка всех активных спиннеров."""
    _spinner_registry.stop_all()


class LiveSpinner:
    """«Крутилка» для операций, у которых нет естественных шагов."""

    frames: ClassVar[list[str]] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(
        self,
        label: str = 'Выполнение',
        interval: float = 0.1,
        writer: SingleLineRenderer | None = None,
    ) -> None:
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._writer = writer or DEFAULT_WRITER
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self._writer.clear_line()
        i = 0
        while not self._stop.is_set():
            msg = f'{self.label}: {self.frames[i % len(self.frames)]}'
            self._writer.write_line(msg)
            time.sleep(self.interval)
            i += 1

    def start(self) -> None:
        # Сообщаем GUI о начале неопределённой операции
        if _CbStore.spinner_start is not None:
            with contextlib.suppress(Exception):
                _CbStore.spinner_start(self.label)
        # Регистрируем спиннер в глобальном реестре
        _spinner_registry.register(self)
        self._th.start()

    def stop(self, final_message: str | None = None) -> None:
        self._stop.set()
        # Ждем завершения потока с таймаутом
        self._th.join(timeout=1.0)
        if self._th.is_alive():
            # Если поток не завершился, помечаем его как daemon для принудительного завершения
            self._th.daemon = True
        # Удаляем спиннер из реестра
        _spinner_registry.unregister(self)
        # Сообщаем GUI о завершении неопределённой операции
        if _CbStore.spinner_stop is not None:
            with contextlib.suppress(Exception):
                _CbStore.spinner_stop(self.label)
        if final_message is not None:
            self._writer.write_line(final_message)


# Глобальный реестр активных LiveSpinner'ов для принудительной остановки
class _SpinnerRegistry:
    def __init__(self) -> None:
        self._active_spinners: list[LiveSpinner] = []
        self._lock = threading.Lock()

    def register(self, spinner: LiveSpinner) -> None:
        """Регистрирует активный спиннер."""
        with self._lock:
            if spinner not in self._active_spinners:
                self._active_spinners.append(spinner)

    def unregister(self, spinner: LiveSpinner) -> None:
        """Удаляет спиннер из реестра."""
        with self._lock:
            if spinner in self._active_spinners:
                self._active_spinners.remove(spinner)

    def stop_all(self) -> None:
        """Принудительно останавливает все активные спиннеры."""
        with self._lock:
            spinners_copy = self._active_spinners.copy()
            self._active_spinners.clear()

        for spinner in spinners_copy:
            try:
                if not spinner._stop.is_set():
                    spinner.stop(final_message=None)
            except Exception as e:
                # Log errors during forced spinner shutdown but don't raise
                logger.debug(f'Error stopping spinner {spinner.label}: {e}')


# Глобальный экземпляр реестра спиннеров
_spinner_registry = _SpinnerRegistry()


class ConsoleProgress:
    """Прогресс-бар для пошаговых операций."""

    def __init__(
        self,
        total: int,
        label: str = 'Прогресс',
        writer: SingleLineRenderer | None = None,
    ) -> None:
        self.total = max(1, int(total))
        self.done = 0
        self.start = time.monotonic()
        self.label = label
        self._writer = writer or DEFAULT_WRITER
        self._lock: asyncio.Lock | None
        try:
            self._lock = asyncio.Lock()
        except Exception:
            self._lock = None
        self._writer.clear_line()
        self._render()  # показать 0%

    def _format_eta(self, remaining: float) -> str:
        if remaining is None or remaining == float('inf'):
            return '--:--'
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f'{h:02d}:{m:02d}:{s:02d}'
        return f'{m:02d}:{s:02d}'

    def _render(self) -> None:
        elapsed = max(1e-6, time.monotonic() - self.start)
        rps = self.done / elapsed
        remaining = (self.total - self.done) / rps if rps > 0 else float('inf')
        bar_len = 30
        filled = int(bar_len * self.done / self.total)
        bar = '█' * filled + '░' * (bar_len - filled)
        msg = (
            f'{self.label}: [{bar}] {self.done}/{self.total} | {rps:4.1f}/s | ETA'
            f' {self._format_eta(remaining)}'
        )
        self._writer.write_line(msg)
        # Сообщаем GUI о прогрессе
        if _CbStore.progress is not None:
            with contextlib.suppress(Exception):
                _CbStore.progress(self.done, self.total, self.label)

    def step_sync(self, n: int = 1) -> None:
        self.done = min(self.total, self.done + n)
        self._render()

    async def step(self, n: int = 1) -> None:
        if self._lock is not None:
            async with self._lock:
                self.done = min(self.total, self.done + n)
                self._render()
        else:
            self.step_sync(n)

    def close(self) -> None:
        # Console output removed - GUI callbacks handle progress display
        pass
