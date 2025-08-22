import contextlib
import threading
import time
import tkinter as tk
import traceback
from collections.abc import Callable
from tkinter import messagebox, ttk

from progress import set_progress_callback, set_spinner_callbacks


class _Animator:
    def __init__(self, root: tk.Misc, progress: ttk.Progressbar) -> None:
        self.root = root
        self.progress = progress
        self.job_id: str | None = None
        self.val: int = 0
        self.direction: int = 1
        self.max_val: int = 100
        self.step: int = 1
        self.interval_ms: int = 40

    def _tick(self, is_active: Callable[[], bool]) -> None:
        # Если спиннеры неактивны — прекращаем анимацию
        if not is_active():
            self.job_id = None
            return
        # Обновляем значение и направление
        v = self.val + self.direction * self.step
        if v >= self.max_val:
            v = self.max_val
            self.direction = -1
        elif v <= 0:
            v = 0
            self.direction = 1
        self.val = v
        with contextlib.suppress(Exception):
            if str(self.progress['mode']) != 'determinate':
                self.progress.config(mode='determinate')
            self.progress.config(maximum=self.max_val)
            self.progress['value'] = v
        self.job_id = self.root.after(self.interval_ms, self._tick, is_active)

    def start_if_needed(self, is_active: Callable[[], bool]) -> None:
        if self.job_id is None:
            self.job_id = self.root.after(self.interval_ms, self._tick, is_active)

    def stop(self) -> None:
        if self.job_id is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self.job_id)
        self.job_id = None


class _UI:
    def __init__(self, on_create_map: Callable[[], None]) -> None:
        self.on_create_map = on_create_map
        self.root = tk.Tk()
        self.root.title('Mil Mapper')

        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        self.btn = tk.Button(frame, text='Создать карту', width=24)
        self.btn.pack(pady=(0, 8))

        self.status_var = tk.StringVar(value='Готов к созданию карты')
        status_lbl = tk.Label(frame, textvariable=self.status_var, anchor='w')
        status_lbl.pack(fill=tk.X)

        self.progress = ttk.Progressbar(
            frame, orient='horizontal', mode='determinate', length=280
        )
        self.progress.pack(fill=tk.X, pady=(6, 0))

        # State
        self.finished: bool = False
        self.spinner_count: int = 0
        self.progress_running: bool = False
        self.last_spinner_stop_ts: float = 0.0
        self.spinner_cooldown_s: float = 0.2
        self.pending_progress: tuple[int, int, str] | None = None
        self.spinner_mode: str | None = None  # 'pingpong' | 'marquee'

        self.animator = _Animator(self.root, self.progress)

        self.btn.config(command=self.handle_click)

    # Thread target
    def _run_task(self) -> None:
        try:
            self.on_create_map()
        except SystemExit as e:
            err_text = str(e) or 'Приложение завершилось'
            self.root.after(0, lambda: self._on_done(ok=False, err_text=err_text))
            return
        except Exception:
            err_text = traceback.format_exc()
            self.root.after(0, lambda: self._on_done(ok=False, err_text=err_text))
            return
        self.root.after(0, lambda: self._on_done(ok=True, err_text=None))

    # Main-thread: finalize
    def _on_done(self, *, ok: bool, err_text: str | None = None) -> None:
        self.finished = True
        with contextlib.suppress(Exception):
            self.progress.stop()
        with contextlib.suppress(Exception):
            self.progress.config(mode='determinate', maximum=100)
            self.progress['value'] = 0
        self.pending_progress = None
        with contextlib.suppress(Exception):
            set_progress_callback(None)
            set_spinner_callbacks(None, None)
        self.btn.config(state=tk.NORMAL)
        if ok:
            self.status_var.set('Готово')
        else:
            self.status_var.set('Ошибка при создании карты')
            messagebox.showerror('Ошибка', err_text or 'Неизвестная ошибка')

    def _apply_progress(self, done: int, total: int, label: str) -> None:
        if self.finished:
            return
        if str(self.progress['mode']) != 'determinate':
            self.progress.config(mode='determinate')
            if self.progress_running:
                with contextlib.suppress(Exception):
                    self.progress.stop()
                self.progress_running = False
        self.progress.config(maximum=total)
        self.progress['value'] = done
        self.status_var.set(f'{label}: {done}/{total}')

    def _flush_pending_if_any(self) -> None:
        if self.finished:
            return
        if self.spinner_count > 0:
            return
        if self.pending_progress is None:
            return
        done, total, label = self.pending_progress
        self.pending_progress = None
        with contextlib.suppress(Exception):
            self._apply_progress(done, total, label)

    # Public button handler
    def handle_click(self) -> None:
        self.btn.config(state=tk.DISABLED)
        self.status_var.set('Создание карты… Подождите…')
        self.finished = False

        set_progress_callback(self.progress_cb)
        set_spinner_callbacks(self.spinner_start_cb, self.spinner_stop_cb)

        self.progress.config(mode='determinate', maximum=100)
        self.progress['value'] = 0

        threading.Thread(target=self._run_task, daemon=True).start()

    # Callbacks passed to progress/spinner
    def progress_cb(self, done: int, total: int, label: str) -> None:
        def _apply() -> None:
            if self.finished:
                return
            recently_stopped = (
                time.time() - self.last_spinner_stop_ts
            ) < self.spinner_cooldown_s
            if self.spinner_count > 0 or recently_stopped:
                self.pending_progress = (done, total, label)
                return
            self.pending_progress = None
            self._apply_progress(done, total, label)

        self.root.after(0, _apply)

    def spinner_start_cb(self, label: str) -> None:
        def _apply() -> None:
            if self.finished:
                return
            self.status_var.set(label)
            self.spinner_count += 1
            if 'Сохранение файла' in label:
                self.spinner_mode = 'marquee'
                with contextlib.suppress(Exception):
                    if str(self.progress['mode']) != 'indeterminate':
                        self.progress.config(mode='indeterminate')
                    self.progress.start(50)
                    self.progress_running = True
            else:
                self.spinner_mode = 'pingpong'
                self.animator.start_if_needed(lambda: self.spinner_count > 0)

        self.root.after(0, _apply)

    def spinner_stop_cb(self, _label: str) -> None:
        def _apply() -> None:
            if self.finished:
                return
            self.spinner_count = max(0, self.spinner_count - 1)
            if self.spinner_count == 0:
                self.last_spinner_stop_ts = time.time()
                if self.spinner_mode == 'pingpong':
                    self.animator.stop()
                elif self.spinner_mode == 'marquee':
                    with contextlib.suppress(Exception):
                        self.progress.stop()
                    self.progress_running = False
                delay_ms = int(self.spinner_cooldown_s * 1000)
                self.root.after(delay_ms, self._flush_pending_if_any)

        self.root.after(0, _apply)

    def mainloop(self) -> None:
        self.root.mainloop()


def run_app(on_create_map: Callable[[], None]) -> None:
    """Запуск GUI-приложения."""
    ui = _UI(on_create_map)
    ui.mainloop()
