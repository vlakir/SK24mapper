import contextlib
import logging
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Literal, Protocol, cast

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

        # Init UI and state
        self._normalize_popup_fonts()
        self._build_widgets()
        self._init_state()
        self._create_menu()

        self.btn.config(command=self.handle_click)
        self.save_btn.config(command=self._handle_save_click)

    def _normalize_popup_fonts(self) -> None:
        with contextlib.suppress(Exception):
            for fname in (
                'TkMessageFont',
                'TkDefaultFont',
                'TkHeadingFont',
                'TkMenuFont',
                'TkTextFont',
                'TkCaptionFont',
                'TkSmallCaptionFont',
                'TkTooltipFont',
                'TkIconFont',
                'TkFixedFont',
            ):
                with contextlib.suppress(Exception):
                    f = tkfont.nametofont(fname)
                    f.configure(weight='normal')
                    f.configure(slant='roman')

    def _build_widgets(self) -> None:
        # Флаг наличия несохранённых изменений (предпросмотра)
        self._unsaved = False
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)
        buttons_row = tk.Frame(frame)
        buttons_row.pack(fill=tk.X, pady=(0, 8))
        self.btn = tk.Button(buttons_row, text='Создать карту', width=24)
        self.btn.pack(side=tk.LEFT)
        self.save_btn = tk.Button(
            buttons_row, text='Сохранить карту…', width=24, state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.status_var = tk.StringVar(value='Готов к созданию карты')
        status_lbl = tk.Label(frame, textvariable=self.status_var, anchor='w')
        status_lbl.pack(fill=tk.X)
        self.progress = ttk.Progressbar(
            frame, orient='horizontal', mode='determinate', length=280
        )
        self.progress.pack(fill=tk.X, pady=(6, 0))

        # ----- Панель отображения настроек профиля -----
        info_panel = tk.Frame(frame)
        info_panel.pack(fill=tk.X, pady=(8, 0))

        # Левый столбец: координаты углов
        corners_col = tk.Frame(info_panel)
        corners_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

        bl_group = ttk.LabelFrame(corners_col, text='Левый нижний угол карты')
        bl_group.pack(fill=tk.X, padx=(0, 8))
        # Переменные для координат — строковые (две цифры 00..99)
        self._from_x_high_var = tk.StringVar()
        self._from_x_low_var = tk.StringVar()
        self._from_y_high_var = tk.StringVar()
        self._from_y_low_var = tk.StringVar()

        self._edit_controls: list[tk.Widget] = []

        def _make_coord_row(parent: tk.Misc, title: str, high_var: tk.StringVar, low_var: tk.StringVar) -> None:
            row = tk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=title, width=2, anchor='w').pack(side=tk.LEFT)
            vcmd = (self.root.register(self._validate_coord_input), '%P')
            e1 = ttk.Spinbox(
                row,
                from_=0,
                to=99,
                increment=1,
                textvariable=high_var,
                width=3,
                justify='right',
                wrap=False,
                state='normal',
                validate='key',
                validatecommand=vcmd,
                command=lambda v=high_var: self._normalize_coord_var(v),
            )
            e1.pack(side=tk.LEFT)
            try:
                e1.bind('<FocusOut>', lambda _e, v=high_var: self._normalize_coord_var(v))
                e1.bind('<Return>', lambda _e, v=high_var: self._normalize_coord_var(v))
            except Exception:
                pass
            self._edit_controls.append(e1)
            tk.Label(row, text='.').pack(side=tk.LEFT)
            e2 = ttk.Spinbox(
                row,
                from_=0,
                to=99,
                increment=1,
                textvariable=low_var,
                width=3,
                justify='right',
                wrap=False,
                state='normal',
                validate='key',
                validatecommand=vcmd,
                command=lambda v=low_var: self._normalize_coord_var(v),
            )
            e2.pack(side=tk.LEFT)
            try:
                e2.bind('<FocusOut>', lambda _e, v=low_var: self._normalize_coord_var(v))
                e2.bind('<Return>', lambda _e, v=low_var: self._normalize_coord_var(v))
            except Exception:
                pass
            self._edit_controls.append(e2)

        _make_coord_row(bl_group, 'X', self._from_x_high_var, self._from_x_low_var)
        _make_coord_row(bl_group, 'Y', self._from_y_high_var, self._from_y_low_var)

        tr_group = ttk.LabelFrame(corners_col, text='Правый верхний угол карты')
        tr_group.pack(fill=tk.X, padx=(0, 8), pady=(6, 0))
        self._to_x_high_var = tk.StringVar()
        self._to_x_low_var = tk.StringVar()
        self._to_y_high_var = tk.StringVar()
        self._to_y_low_var = tk.StringVar()
        _make_coord_row(tr_group, 'X', self._to_x_high_var, self._to_x_low_var)
        _make_coord_row(tr_group, 'Y', self._to_y_high_var, self._to_y_low_var)

        # Правый столбец: Сетка
        grid_col = tk.Frame(info_panel)
        grid_col.pack(side=tk.LEFT, fill=tk.X, expand=True)

        grid_group = ttk.LabelFrame(grid_col, text='Настройки координатной сетки')
        grid_group.pack(fill=tk.X)

        self._grid_width_var = tk.IntVar()
        self._grid_font_size_var = tk.IntVar()
        self._grid_text_margin_var = tk.IntVar()
        self._grid_label_bg_padding_var = tk.IntVar()
        self._mask_opacity_var = tk.DoubleVar()
        self._mask_opacity_text = tk.StringVar(value='0.00')

        def _make_labeled_spin(parent: tk.Misc, label: str, var: tk.IntVar, from_: int = 0, to: int = 10000, inc: int = 1) -> None:
            row = tk.Frame(parent)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, anchor='w').pack(side=tk.LEFT)
            sp = tk.Spinbox(row, from_=from_, to=to, increment=inc, textvariable=var, width=8, justify='right')
            sp.pack(side=tk.RIGHT)
            self._edit_controls.append(sp)

        _make_labeled_spin(grid_group, 'Толщина линии, px:', self._grid_width_var, 0, 1000, 1)
        _make_labeled_spin(grid_group, 'Размер шрифта, px:', self._grid_font_size_var, 1, 2000, 1)
        _make_labeled_spin(grid_group, 'Отступ надписей от края карты, px:', self._grid_text_margin_var, 0, 2000, 1)
        _make_labeled_spin(grid_group, 'Поля маски надписи, px:', self._grid_label_bg_padding_var, 0, 500, 1)

        # Слайдер прозрачности маски
        opacity_row = tk.Frame(grid_group)
        opacity_row.pack(fill=tk.X, pady=2)
        tk.Label(opacity_row, text='Прозрачность маски:').pack(side=tk.LEFT)
        self._opacity_scale = ttk.Scale(
            opacity_row,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self._mask_opacity_var,
            command=lambda _v: self._on_opacity_change(),
        )
        # Слайдер занимает всё доступное пространство между надписью и значением
        self._opacity_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._edit_controls.append(self._opacity_scale)
        # Значение прозрачности справа от слайдера, в одном ряду
        opacity_val_lbl = tk.Label(
            opacity_row,
            textvariable=self._mask_opacity_text,
            width=8,
            anchor='e',
            justify='right',
        )
        opacity_val_lbl.pack(side=tk.RIGHT, padx=(10, 0))

        # Контейнер предпросмотра
        self.preview_container = tk.Frame(frame, height=320)
        self.preview_container.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
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
        self.preview_container.bind('<Configure>', self._on_preview_resize)
        self._preview_from_memory: bool = False

        # Заполним панель начальными значениями текущего профиля
        with contextlib.suppress(Exception):
            self._populate_profile_fields()
        # Установим обработчики изменений, чтобы обновлять отображение и внутреннее состояние
        try:
            for var in (
                self._from_x_high_var,
                self._from_x_low_var,
                self._from_y_high_var,
                self._from_y_low_var,
                self._to_x_high_var,
                self._to_x_low_var,
                self._to_y_high_var,
                self._to_y_low_var,
                self._grid_width_var,
                self._grid_font_size_var,
                self._grid_text_margin_var,
                self._grid_label_bg_padding_var,
            ):
                var.trace_add('write', lambda *_args: self._on_field_change())
        except Exception:
            pass

    def _init_state(self) -> None:
        self.finished: bool = False
        self.spinner_count: int = 0
        self.progress_running: bool = False
        self.last_spinner_stop_ts: float = 0.0
        self.spinner_cooldown_s: float = 0.2
        self.pending_progress: tuple[int, int, str] | None = None
        self.spinner_mode: Literal['pingpong', 'marquee'] | None = None
        self.animator = _Animator(self.root, self.progress)
        # Текущие настройки (обновляются из профиля и из полей ввода)
        self._current_settings: Any | None = None
        # Флаг временного подавления обработчиков изменений полей
        self._suspend_traces: int = 0
        # Отложенная задача для debounce обновления настроек из полей
        self._field_change_job_id: str | None = None

    def _create_menu(self) -> None:
        with contextlib.suppress(Exception):
            menubar = tk.Menu(self.root)
            profile_menu = tk.Menu(menubar, tearoff=0)
            profile_menu.add_command(
                label='Загрузить профиль из файла…', command=self._choose_profile_file
            )
            menubar.add_cascade(label='Профиль', menu=profile_menu)
            self.root.config(menu=menubar)
            self.menubar = menubar
            self.profile_menu = profile_menu

    def _populate_profile_fields(self, settings: _SettingsProto | Any | None = None) -> None:
        """Заполнить поля панели актуальными значениями профиля."""
        try:
            if settings is None:
                settings = load_profile(constants.CURRENT_PROFILE)
        except Exception:
            return
        try:
            # На время программного обновления значений подавляем обработчики изменений
            prev_suspend = getattr(self, '_suspend_traces', 0)
            setattr(self, '_suspend_traces', prev_suspend + 1)
            # Координаты (старшие, младшие) — в диапазоне 0..99, без ведущих нулей
            def _clamp_0_99(v: int | None) -> int:
                try:
                    v_int = 0 if v is None else int(v)
                except Exception:
                    v_int = 0
                if v_int < 0:
                    v_int = 0
                if v_int > 99:
                    v_int = 99
                return v_int

            def _set_strvar(var: tk.StringVar, val: int) -> None:
                s = str(val)
                if var.get() != s:
                    var.set(s)

            _set_strvar(self._from_x_high_var, _clamp_0_99(getattr(settings, 'from_x_high', 0)))
            _set_strvar(self._from_x_low_var, _clamp_0_99(getattr(settings, 'from_x_low', 0)))
            _set_strvar(self._from_y_high_var, _clamp_0_99(getattr(settings, 'from_y_high', 0)))
            _set_strvar(self._from_y_low_var, _clamp_0_99(getattr(settings, 'from_y_low', 0)))

            _set_strvar(self._to_x_high_var, _clamp_0_99(getattr(settings, 'to_x_high', 0)))
            _set_strvar(self._to_x_low_var, _clamp_0_99(getattr(settings, 'to_x_low', 0)))
            _set_strvar(self._to_y_high_var, _clamp_0_99(getattr(settings, 'to_y_high', 0)))
            _set_strvar(self._to_y_low_var, _clamp_0_99(getattr(settings, 'to_y_low', 0)))

            # Параметры сетки
            def _set_intvar(var: tk.IntVar, val: int) -> None:
                try:
                    if int(var.get()) != int(val):
                        var.set(int(val))
                except Exception:
                    var.set(int(val))

            _set_intvar(self._grid_width_var, int(getattr(settings, 'grid_width_px', 0)))
            _set_intvar(self._grid_font_size_var, int(getattr(settings, 'grid_font_size', 0)))
            _set_intvar(self._grid_text_margin_var, int(getattr(settings, 'grid_text_margin', 0)))
            _set_intvar(self._grid_label_bg_padding_var, int(getattr(settings, 'grid_label_bg_padding', 0)))
            try:
                opacity = float(getattr(settings, 'mask_opacity', 0.0))
            except Exception:
                opacity = 0.0
            self._mask_opacity_var.set(opacity)
            with contextlib.suppress(Exception):
                self._opacity_scale.set(opacity)
            # Текстовое отображение
            self._mask_opacity_text.set(f"{opacity:.2f}")
            # Сохраним текущие настройки
            self._current_settings = settings  # type: ignore[assignment]
        except Exception:
            # Если что-то пойдёт не так — не падаем UI
            pass
        finally:
            try:
                cur_suspend = getattr(self, '_suspend_traces', 1)
                setattr(self, '_suspend_traces', max(0, cur_suspend - 1))
            except Exception:
                pass

    def _on_opacity_change(self) -> None:
        try:
            v = float(self._mask_opacity_var.get())
        except Exception:
            v = 0.0
        v = max(0.0, min(1.0, v))
        self._mask_opacity_var.set(v)
        self._mask_opacity_text.set(f"{v:.2f}")
        # Не навязываем немедленное обновление модулей; сбор будет при запуске

    def _validate_coord_input(self, new_val: str) -> bool:
        """Валидация ввода координаты по ключевым нажатиям: допускаем '' или до 2 цифр."""
        try:
            if new_val == "":
                return True
            if len(new_val) > 2:
                return False
            if not new_val.isdigit():
                return False
            # Промежуточные значения вроде '0' или '9' допустимы
            return True
        except Exception:
            return False

    def _normalize_coord_var(self, var: tk.StringVar) -> None:
        """Нормализует значение var в диапазон 0..99 (без ведущих нулей) с минимальными перезаписями."""
        try:
            s = var.get()
            if s is None:
                s = ""
            s = str(s).strip()
            if s == "":
                v = 0
            else:
                try:
                    v = int(s)
                except Exception:
                    v = 0
            if v < 0:
                v = 0
            if v > 99:
                v = 99
            new_s = str(v)
            if s != new_s:
                var.set(new_s)
        except Exception:
            with contextlib.suppress(Exception):
                var.set("0")


    def _on_field_change(self) -> None:
        # Подавление реакций во время программных обновлений
        if getattr(self, '_suspend_traces', 0) > 0:
            return
        # Debounce: откладываем перерасчёт настроек, отменяя предыдущий
        if getattr(self, '_field_change_job_id', None) is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self._field_change_job_id)
            self._field_change_job_id = None

        def _apply_changes() -> None:
            # Не нормализуем координаты во время набора, когда поле временно пусто,
            # чтобы не сбрасывать его в "0".
            try:
                coord_vars: tuple[tk.StringVar, ...] = (
                    self._from_x_high_var,
                    self._from_x_low_var,
                    self._from_y_high_var,
                    self._from_y_low_var,
                    self._to_x_high_var,
                    self._to_x_low_var,
                    self._to_y_high_var,
                    self._to_y_low_var,
                )
                for v in coord_vars:
                    s = v.get()
                    if s is None or s == "":
                        return  # пользователь ещё вводит значение — не пересобирать настройки
            except Exception:
                pass
            with contextlib.suppress(Exception):
                settings = self._build_settings_from_fields()
                if settings is not None:
                    self._current_settings = settings

        # Небольшая задержка для сглаживания серий нажатий клавиш
        self._field_change_job_id = self.root.after(120, _apply_changes)

    def _build_settings_from_fields(self) -> Any | None:
        """Собирает объект настроек из текущих значений полей. Возвращает None при ошибке.
        Не изменяет значения полей (без var.set), чтобы избежать каскадных trace-обработчиков.
        """
        try:
            from model import MapSettings  # локальный импорт, чтобы избежать циклов
            # Базовые значения, которые не редактируются в этой панели
            output_path = getattr(self._current_settings, 'output_path', None)
            if output_path is None:
                # Попробуем получить из загруженного профиля
                with contextlib.suppress(Exception):
                    output_path = load_profile(constants.CURRENT_PROFILE).output_path
            if output_path is None:
                output_path = '../maps/map.png'

            # Локальные парсеры без модификации виджетов
            def _parse_0_99(var: tk.StringVar) -> int:
                try:
                    s = str(var.get() or '').strip()
                    if s == '':
                        return 0
                    v = int(s)
                except Exception:
                    v = 0
                if v < 0:
                    v = 0
                if v > 99:
                    v = 99
                return v

            settings = MapSettings(
                from_x_high=_parse_0_99(self._from_x_high_var),
                from_y_high=_parse_0_99(self._from_y_high_var),
                to_x_high=_parse_0_99(self._to_x_high_var),
                to_y_high=_parse_0_99(self._to_y_high_var),
                from_x_low=_parse_0_99(self._from_x_low_var),
                from_y_low=_parse_0_99(self._from_y_low_var),
                to_x_low=_parse_0_99(self._to_x_low_var),
                to_y_low=_parse_0_99(self._to_y_low_var),
                output_path=str(output_path),
                grid_width_px=int(self._grid_width_var.get()),
                grid_font_size=int(self._grid_font_size_var.get()),
                grid_text_margin=int(self._grid_text_margin_var.get()),
                grid_label_bg_padding=int(self._grid_label_bg_padding_var.get()),
                mask_opacity=float(self._mask_opacity_var.get()),
            )
            return settings
        except Exception as e:
            logger.debug('Failed to build settings from fields: %s', e)
            return None

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
        # Вернём элементы управления (включая меню) в активное состояние
        self._set_all_controls_enabled(enabled=True)
        self.btn.config(state=tk.NORMAL)
        # Разрешим сохранение, если есть предпросмотр
        self.save_btn.config(
            state=(tk.NORMAL if self.preview_src_img is not None else tk.DISABLED)
        )
        if ok:
            self.status_var.set('Готово')
            # Если предпросмотр уже был показан из памяти — не перезагружаем из файла
            if self.preview_src_img is None:
                self._show_preview_if_available()
                # После показа из файла также включим сохранение, если удалось
                self.save_btn.config(
                    state=(
                        tk.NORMAL if self.preview_src_img is not None else tk.DISABLED
                    )
                )
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

    def handle_click(self) -> None:
        # Предупреждение об несохранённой карте
        if self.preview_src_img is not None and self._unsaved:
            proceed = messagebox.askyesno(
                'Несохранённая карта',
                'У вас есть несохранённая карта. Продолжить без сохранения?',
                parent=self.root,
            )
            if not proceed:
                return

        # На время загрузки и обработки карты делаем меню и элементы
        # управления неактивными
        self._set_all_controls_enabled(enabled=False)
        self.btn.config(state=tk.DISABLED)
        self.status_var.set('Создание карты… Подождите…')
        self.finished = False

        set_progress_callback(self.progress_cb)
        set_spinner_callbacks(self.spinner_start_cb, self.spinner_stop_cb)
        set_preview_image_callback(self.preview_image_cb)
        # Пока идёт создание — блокируем сохранение
        self.save_btn.config(state=tk.DISABLED)

        self.progress.config(mode='determinate', maximum=100)
        self.progress['value'] = 0

        # Сброс предпросмотра перед новым запуском
        self.preview_img = None
        self.preview_src_img = None
        self._preview_from_memory = False
        # Сбросим флаг несохранённости только после подтверждения пользователем выше
        self._unsaved = False
        self.save_btn.config(state=tk.DISABLED)
        # Отменим отложенную подгонку, если она была
        if self._resize_job_id is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self._resize_job_id)
        self._resize_job_id = None
        self.preview_label.config(
            image='',
            text='Создание… Предпросмотр появится после завершения',
        )

        # Соберём настройки из полей и применим их к модулям
        try:
            cur_settings = self._build_settings_from_fields()
            if cur_settings is not None:
                self._refresh_modules_with_settings(cur_settings)
        except Exception as e:
            logger.debug('Failed to build/apply settings from fields: %s', e)
        threading.Thread(target=self._run_task, daemon=True).start()

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
            # Отметим, что есть несохранённые изменения
            self._unsaved = True
            # Разрешим сохранение
            self.save_btn.config(state=tk.NORMAL)
            # Рассчитываем текущие доступные размеры и подгоняем
            self.root.update_idletasks()
            max_w = max(200, self.preview_container.winfo_width() or 0)
            max_h = max(200, self.preview_container.winfo_height() or 0)
            max_w = min(max_w, 1000)
            max_h = min(max_h, 800)
            self._resize_preview_to(max_w, max_h)

        self.root.after(0, _apply)

    def _set_menu_enabled(self, menu: tk.Menu, *, enabled: bool) -> None:
        state: Literal['normal', 'active', 'disabled'] = (
            'normal' if enabled else 'disabled'
        )
        with contextlib.suppress(Exception):
            end_index = menu.index('end')
            if end_index is not None:
                for i in range(end_index + 1):
                    with contextlib.suppress(Exception):
                        menu.entryconfig(i, state=state)

    def _set_all_controls_enabled(self, *, enabled: bool) -> None:
        state: Literal['normal', 'active', 'disabled'] = (
            'normal' if enabled else 'disabled'
        )
        # Buttons
        with contextlib.suppress(Exception):
            self.btn.config(state=state)
        with contextlib.suppress(Exception):
            self.save_btn.config(state=state)
        # Editable widgets (spinboxes, scale)
        for w in getattr(self, '_edit_controls', []):
            with contextlib.suppress(Exception):
                if isinstance(w, ttk.Scale):
                    if enabled:
                        w.state(['!disabled'])
                    else:
                        w.state(['disabled'])
                elif isinstance(w, ttk.Spinbox):
                    if enabled:
                        # Спинбоксы координат делаем редактируемыми вручную
                        w.state(['!disabled'])
                    else:
                        w.state(['disabled'])
                else:
                    w.configure(state=('normal' if enabled else 'disabled'))
        # Menubar cascades and submenu items
        menubar = getattr(self, 'menubar', None)
        if isinstance(menubar, tk.Menu):
            with contextlib.suppress(Exception):
                end_index = menubar.index('end')
                if end_index is not None:
                    for i in range(end_index + 1):
                        with contextlib.suppress(Exception):
                            menubar.entryconfig(i, state=state)
        profile_menu = getattr(self, 'profile_menu', None)
        if isinstance(profile_menu, tk.Menu):
            self._set_menu_enabled(profile_menu, enabled=enabled)

    def mainloop(self) -> None:
        self.root.mainloop()

    def _handle_save_click(self) -> None:
        """Сохранить текущую карту по выбору пользователя (GUI)."""
        if self.preview_src_img is None:
            messagebox.showinfo('Сохранение карты', 'Нет изображения для сохранения.')
            return
        # Директория по умолчанию: maps/ относительно корня проекта (если есть),
        # иначе — текущая
        repo_root = Path(__file__).resolve().parent.parent  # src/gui/.. -> src
        repo_root = repo_root.parent  # -> project root
        default_dir = repo_root / 'maps'
        initialdir = str(default_dir if default_dir.exists() else repo_root)
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title='Сохранить карту',
            initialdir=initialdir,
            initialfile='',
            defaultextension='.png',
            filetypes=[('PNG изображение', '*.png'), ('Все файлы', '*.*')],
        )
        if not file_path:
            return

        # На время сохранения делаем все элементы управления неактивными
        self._set_all_controls_enabled(enabled=False)

        # Запускаем анимацию до начала сохранения
        with contextlib.suppress(Exception):
            if str(self.progress['mode']) != 'indeterminate':
                self.progress.config(mode='indeterminate')
            self.status_var.set('Сохранение файла…')
            self.progress.start(50)

        # Захватываем ссылку на изображение, чтобы типизатор знал, что оно не None
        img = self.preview_src_img
        if img is None:
            # На всякий случай, хотя выше уже проверяли
            return

        def _save_worker(fp: str, _img: Image.Image) -> None:
            err: Exception | None = None
            try:
                # Сохранение файла в фоне, без обращений к Tkinter из этого потока
                _img.save(fp)
            except Exception as e:
                err = e

            def _on_done() -> None:
                # Останавливаем анимацию и восстанавливаем прогрессбар
                with contextlib.suppress(Exception):
                    self.progress.stop()
                    self.progress.config(mode='determinate', maximum=100)
                    self.progress['value'] = 0
                # Вернём элементы управления в активное состояние
                self._set_all_controls_enabled(enabled=True)
                if err is None:
                    # Помечаем как сохранённое и отключаем кнопку сохранения
                    self._unsaved = False
                    self.status_var.set(f'Карта сохранена: {Path(fp).name}')
                    with contextlib.suppress(Exception):
                        self.save_btn.config(state=tk.DISABLED)
                else:
                    # В случае ошибки разрешим повторную попытку сохранения
                    self.status_var.set('Ошибка сохранения')
                    messagebox.showerror(
                        'Ошибка сохранения', f'Не удалось сохранить файл:\n{err}'
                    )
                    with contextlib.suppress(Exception):
                        self.save_btn.config(state=tk.NORMAL)

            self.root.after(0, _on_done)

        threading.Thread(
            target=_save_worker, args=(file_path, img), daemon=True
        ).start()

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
            self._unsaved = False
            self.save_btn.config(state=tk.DISABLED)
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
        # Обновим панель значений профиля
        with contextlib.suppress(Exception):
            self._populate_profile_fields(settings)

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
            self.preview_src_img = cast('Image.Image', img)
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
