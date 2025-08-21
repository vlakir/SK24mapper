# ------------------------------
# Единая строка прогресса + спиннер (для «монолитных» шагов)
# ------------------------------
import asyncio
import sys
import threading
import time

_PROGRESS_SINGLE_LINE = True  # Рисовать всё в одной строке
_LAST_PROGRESS_LINE_LEN = 0  # Длина последней строки прогресса
_SINGLE_LINE_LOCK = threading.Lock()  # Блокировка на вывод в единую строку


def _clear_line() -> None:
    """Полностью очистить текущую строку прогресса."""
    global _LAST_PROGRESS_LINE_LEN
    with _SINGLE_LINE_LOCK:
        if _PROGRESS_SINGLE_LINE and _LAST_PROGRESS_LINE_LEN > 0:
            sys.stdout.write('\r' + ' ' * _LAST_PROGRESS_LINE_LEN + '\r')
            sys.stdout.flush()
            _LAST_PROGRESS_LINE_LEN = 0


def _write_line(msg: str) -> None:
    """Перерисовать текущую строку прогресса."""
    global _LAST_PROGRESS_LINE_LEN
    with _SINGLE_LINE_LOCK:
        if _PROGRESS_SINGLE_LINE:
            pad = max(0, _LAST_PROGRESS_LINE_LEN - len(msg))
            sys.stdout.write('\r' + msg + (' ' * pad))
        else:
            sys.stdout.write('\r' + msg)
        sys.stdout.flush()
        _LAST_PROGRESS_LINE_LEN = len(msg)


class LiveSpinner:
    """«Крутилка» для операций, у которых нет естественных шагов (поворот, сохранение и т.п.)."""

    frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def __init__(self, label: str = 'Выполнение', interval: float = 0.1):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        _clear_line()
        i = 0
        while not self._stop.is_set():
            msg = f'{self.label}: {self.frames[i % len(self.frames)]}'
            _write_line(msg)
            time.sleep(self.interval)
            i += 1

    def start(self) -> None:
        self._th.start()

    def stop(self, final_message: str | None = None) -> None:
        self._stop.set()
        self._th.join()
        if final_message is not None:
            _write_line(final_message)


class ConsoleProgress:
    """Прогресс-бар для пошаговых операций (загрузка тайлов, склейка, рисование сетки)."""

    def __init__(self, total: int, label: str = 'Прогресс'):
        self.total = max(1, int(total))
        self.done = 0
        self.start = time.monotonic()
        self.label = label
        self._lock: asyncio.Lock | None
        try:
            self._lock = asyncio.Lock()
        except Exception:
            self._lock = None
        _clear_line()
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
        msg = f'{self.label}: [{bar}] {self.done}/{self.total} | {rps:4.1f}/s | ETA {self._format_eta(remaining)}'
        _write_line(msg)

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
        sys.stdout.flush()
