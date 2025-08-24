import contextlib
import logging
import sys
import threading
import time
import tkinter as tk
import traceback
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Protocol, cast

from PIL import Image, ImageTk

import constants
from profiles import load_profile, profile_path
from progress import (
    set_preview_image_callback,
    set_progress_callback,
    set_spinner_callbacks,
)

logger = logging.getLogger(__name__)


class _SettingsProto(Protocol):
    bottom_left_x_sk42_gk: float
    bottom_left_y_sk42_gk: float
    top_right_x_sk42_gk: float
    top_right_y_sk42_gk: float


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
        # Ensure the window is not too small so the preview placeholder is visible
        with contextlib.suppress(Exception):
            self.root.minsize(640, 480)

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

        # Preview area (use container to avoid geometry feedback from image size)
        # Give it an initial reasonable height so the placeholder text is visible
        self.preview_container = tk.Frame(frame, height=320)
        self.preview_container.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        # Prevent the container from resizing to fit its children (fixes jitter)
        self.preview_container.pack_propagate(flag=False)

        self.preview_label = tk.Label(
            self.preview_container,
            text='Предпросмотр появится после создания карты',
            anchor='center',
            justify='center',
            relief='sunken',
            bd=1,
            padx=6,
            pady=6,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.preview_img: ImageTk.PhotoImage | None = None
        self.preview_src_img: Image.Image | None = None
        self._resize_job_id: str | None = None
        # Масштабирование предпросмотра при изменении размеров — слушаем контейнер
        self.preview_container.bind('<Configure>', self._on_preview_resize)

        # Флаг: предпросмотр уже показан из памяти
        self._preview_from_memory: bool = False

        # State
        self.finished: bool = False
        self.spinner_count: int = 0
        self.progress_running: bool = False
        self.last_spinner_stop_ts: float = 0.0
        self.spinner_cooldown_s: float = 0.2
        self.pending_progress: tuple[int, int, str] | None = None
        self.spinner_mode: str | None = None  # 'pingpong' | 'marquee'

        self.animator = _Animator(self.root, self.progress)

        # Menu: profile loading
        with contextlib.suppress(Exception):
            menubar = tk.Menu(self.root)
            profile_menu = tk.Menu(menubar, tearoff=0)
            profile_menu.add_command(
                label='Загрузить профиль из файла…', command=self._choose_profile_file
            )
            menubar.add_cascade(label='Профиль', menu=profile_menu)
            self.root.config(menu=menubar)

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
            set_preview_image_callback(None)
        self.btn.config(state=tk.NORMAL)
        if ok:
            self.status_var.set('Готово')
            # Если предпросмотр уже был показан из памяти — не перезагружаем из файла
            if self.preview_src_img is None:
                self._show_preview_if_available()
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
        set_preview_image_callback(self.preview_image_cb)

        self.progress.config(mode='determinate', maximum=100)
        self.progress['value'] = 0

        # Сброс предпросмотра перед новым запуском
        self.preview_img = None
        self.preview_src_img = None
        self._preview_from_memory = False
        # Отменим отложенную подгонку, если она была
        if self._resize_job_id is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self._resize_job_id)
        self._resize_job_id = None
        self.preview_label.config(
            image='',
            text='Создание… Предпросмотр появится после завершения',
        )

        # Обновим настройки модулей на случай, если профиль меняли
        try:
            cur_settings = load_profile(constants.CURRENT_PROFILE)
            self._refresh_modules_with_settings(cur_settings)
        except Exception as e:
            logger.debug('Failed to refresh modules with current profile: %s', e)
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

    def preview_image_cb(self, img_obj: object) -> None:
        """Получает PIL.Image и отображает предпросмотр до сохранения файла."""

        def _apply() -> None:
            if img_obj is None:
                return
            try:
                img = cast('Image.Image', img_obj)
            except Exception:
                return
            # Сохраняем источником предпросмотра и помечаем как показанный из памяти
            self.preview_src_img = img
            self._preview_from_memory = True
            # Рассчитываем текущие доступные размеры и подгоняем
            self.root.update_idletasks()
            max_w = max(200, self.preview_container.winfo_width() or 0)
            max_h = max(200, self.preview_container.winfo_height() or 0)
            max_w = min(max_w, 1000)
            max_h = min(max_h, 800)
            self._resize_preview_to(max_w, max_h)

        self.root.after(0, _apply)

    def mainloop(self) -> None:
        self.root.mainloop()

    def _choose_profile_file(self) -> None:
        """Открыть диалог выбора TOML и применить профиль."""
        # Определяем директорию по умолчанию:
        # 1) Если текущий профиль — путь к файлу и он существует, используем его папку.
        # 2) Иначе — папку, где лежит default.toml.
        # 3) Фолбэк — текущая рабочая директория.
        cur = Path(str(constants.CURRENT_PROFILE))
        if cur.suffix.lower() == '.toml' and cur.exists():
            initialdir = cur.parent
        else:
            default_dir: Path | None = None
            with contextlib.suppress(Exception):
                default_dir = profile_path('default').parent
            initialdir = (
                default_dir if (default_dir and default_dir.exists()) else Path.cwd()
            )

        file_path = filedialog.askopenfilename(
            parent=self.root,
            title='Выберите файл профиля TOML',
            initialdir=str(initialdir),
            filetypes=[('TOML файлы', '*.toml'), ('Все файлы', '*.*')],
        )
        if not file_path:
            return
        try:
            # Пробуем загрузить для валидации
            new_settings = load_profile(file_path)
        except Exception as e:
            messagebox.showerror(
                'Ошибка профиля', f'Не удалось загрузить профиль:\n{e}'
            )
            return

        # Сохраняем выбранный путь в глобальной константе и обновляем модули
        try:
            constants.CURRENT_PROFILE = file_path
            self._refresh_modules_with_settings(new_settings)
            self.status_var.set(f'Профиль загружен: {Path(file_path).name}')
            # Сброс предпросмотра: по требованию превью должно исчезнуть
            # и не показываться до формирования новой карты.
            self.preview_img = None
            self.preview_src_img = None
            self._preview_from_memory = False
            self.preview_label.config(
                image='',
                text='Профиль загружен. Предпросмотр появится после создания карты',
            )
            # Не показываем картинку из старого output_path
            # чтобы избежать «застывшей» картинки
        except Exception as e:
            messagebox.showwarning(
                'Профиль применён частично',
                f'Профиль загружен, но возникла проблема при применении:\n{e}',
            )

    def _refresh_modules_with_settings(self, settings: _SettingsProto) -> None:
        """
        Попробовать обновить загруженные модули глобальными settings.

        Это позволяет использовать новый профиль без перезапуска.
        Обновляются значения в модулях: main, controller, image, topography.
        """
        with contextlib.suppress(Exception):
            mod_controller: Any = sys.modules.get('controller')
            if mod_controller is not None:
                mod_controller.settings = settings
        with contextlib.suppress(Exception):
            mod_image: Any = sys.modules.get('image')
            if mod_image is not None:
                mod_image.settings = settings
        with contextlib.suppress(Exception):
            mod_topography: Any = sys.modules.get('topography')
            if mod_topography is not None:
                mod_topography.settings = settings
                # Пересчёт производных значений
                center_x = (
                    settings.bottom_left_x_sk42_gk + settings.top_right_x_sk42_gk
                ) / 2
                center_y = (
                    settings.bottom_left_y_sk42_gk + settings.top_right_y_sk42_gk
                ) / 2
                width_m = settings.top_right_x_sk42_gk - settings.bottom_left_x_sk42_gk
                height_m = settings.top_right_y_sk42_gk - settings.bottom_left_y_sk42_gk
                mod_topography.center_x_sk42_gk = center_x
                mod_topography.center_y_sk42_gk = center_y
                mod_topography.width_m = width_m
                mod_topography.height_m = height_m

    # Helpers
    def _on_preview_resize(self, event: tk.Event) -> None:
        """
        Изменение размеров области предпросмотра.

        Делает отложенную (debounce) подгонку, чтобы не масштабировать слишком часто.
        """
        if self.preview_src_img is None:
            return
        # Игнорируем слишком маленькие события
        width = max(
            1, int(getattr(event, 'width', self.preview_container.winfo_width()))
        )
        height = max(
            1, int(getattr(event, 'height', self.preview_container.winfo_height()))
        )
        # Отменим предыдущие задания
        if self._resize_job_id is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self._resize_job_id)
            self._resize_job_id = None
        # Небольшая задержка для сглаживания серий событий
        self._resize_job_id = self.root.after(
            60, lambda: self._resize_preview_to(width, height)
        )

    def _resize_preview_to(self, max_w: int, max_h: int) -> None:
        if self.preview_src_img is None:
            return
        # Учитываем внутренние отступы Label (padx/pady) — попытаемся вычесть по 2*6
        try:
            padx = int(self.preview_label.cget('padx'))
            pady = int(self.preview_label.cget('pady'))
        except Exception:
            padx = 0
            pady = 0
        avail_w = max(1, max_w - 2 * padx)
        avail_h = max(1, max_h - 2 * pady)
        src_w, src_h = self.preview_src_img.size
        if src_w <= 0 or src_h <= 0:
            return
        scale = min(avail_w / src_w, avail_h / src_h)
        scale = max(scale, 0.01)
        dst_w = max(1, int(src_w * scale))
        dst_h = max(1, int(src_h * scale))
        with contextlib.suppress(Exception):
            resized = self.preview_src_img.resize(
                (dst_w, dst_h), Image.Resampling.LANCZOS
            )
            self.preview_img = ImageTk.PhotoImage(resized)
            self.preview_label.config(image=self.preview_img, text='')
        self._resize_job_id = None

    def _show_preview_if_available(self) -> None:
        """Показывает предпросмотр карты из пути профиля, если файл существует."""
        if self._preview_from_memory:
            # Уже показали из памяти — ничего не делаем
            return
        try:
            settings = load_profile(constants.CURRENT_PROFILE)
            path = Path(settings.output_path)
            # При необходимости создать абсолютный путь
            path = path if path.is_absolute() else Path.cwd() / path
            if not path.exists():
                # Файл не найден — покажем текст
                self.preview_img = None
                self.preview_src_img = None
                self.preview_label.config(
                    image='',
                    text=f'Файл не найден:\n{path}',
                )
                return

            # Определим желаемый размер предпросмотра исходя из текущей ширины
            self.root.update_idletasks()
            max_w = max(200, self.preview_container.winfo_width() or 0)
            max_h = max(200, self.preview_container.winfo_height() or 0)
            # Ограничим разумным максимумом
            max_w = min(max_w, 1000)
            max_h = min(max_h, 800)

            img = Image.open(path)
            self.preview_src_img = img
            # Немедленно подгоним под текущие размеры
            self._resize_preview_to(max_w, max_h)
        except Exception as e:  # Поймаем и покажем как текст
            self.preview_img = None
            self.preview_src_img = None
            self.preview_label.config(
                image='',
                text=f'Не удалось показать предпросмотр:\n{e}',
            )


def run_app(on_create_map: Callable[[], None]) -> None:
    """Запуск GUI-приложения."""
    ui = _UI(on_create_map)
    ui.mainloop()
