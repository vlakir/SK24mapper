import contextlib
import logging
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import traceback
import asyncio
import socket
import errno
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Literal, Protocol, cast

import aiohttp

from PIL import Image, ImageTk
import tomlkit

import constants
from model import MapSettings
from profiles import load_profile, profile_path
from progress import (
    set_preview_image_callback,
    set_progress_callback,
    set_spinner_callbacks,
)

MAX_COORD_DIGITS = 2

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
        # Remember default window minsize to restore later when showing preview
        self._default_min_w = 640
        self._default_min_h = 480
        # Hide window during build to avoid initial oversized height on first show
        with contextlib.suppress(Exception):
            self.root.withdraw()

        # Init UI and state
        self._normalize_popup_fonts()
        self._build_widgets()
        self._init_state()
        self._create_menu()

        self.btn.config(command=self.handle_click)
        self.save_btn.config(command=self._handle_save_click)
        with contextlib.suppress(Exception):
            self.profile_load_btn.config(command=self._choose_profile_file)
        with contextlib.suppress(Exception):
            self.profile_save_btn.config(command=self._handle_save_profile_click)

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
        # Keep a reference to the main content frame for sizing calculations
        self._main_frame = frame
        # Заголовок/кнопки/статус/прогресс
        self._build_header(frame)
        # Панель настроек профиля (координаты и сетка)
        self._build_info_panel(frame)
        # Контейнер предпросмотра
        self._build_preview(frame)
        # Ensure preview is hidden initially and shrink window to content
        with contextlib.suppress(Exception):
            self._set_preview_visible(False)
        # After building widgets and hiding preview, shrink window to content and show it
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            minw = int(getattr(self, '_default_min_w', 640))
            # Allow minimal height to be fully determined by content
            self.root.minsize(minw, 1)
        with contextlib.suppress(Exception):
            # Ask Tk to fit window to requested sizes
            self.root.geometry('')
        with contextlib.suppress(Exception):
            # Show the window already shrunk to content height
            self.root.deiconify()
        # Начальная подстановка значений и подписка на изменения
        self._post_build_setup()

    def _build_header(self, frame: tk.Misc) -> None:
        buttons_row = tk.Frame(frame)
        buttons_row.pack(fill=tk.X, pady=(0, 8))
        self.btn = tk.Button(buttons_row, text='Создать карту', width=24)
        self.btn.pack(side=tk.LEFT)
        self.save_btn = tk.Button(
            buttons_row, text='Сохранить карту…', width=24, state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=(8, 0))
        # Кнопки работы с профилем
        self.profile_load_btn = tk.Button(
            buttons_row, text='Загрузить профиль…', width=20
        )
        self.profile_load_btn.pack(side=tk.LEFT, padx=(16, 0))
        self.profile_save_btn = tk.Button(
            buttons_row, text='Сохранить профиль…', width=20
        )
        self.profile_save_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.status_var = tk.StringVar(value='Готов к созданию карты')
        status_lbl = tk.Label(frame, textvariable=self.status_var, anchor='w')
        status_lbl.pack(fill=tk.X)
        self.progress = ttk.Progressbar(
            frame, orient='horizontal', mode='determinate', length=280
        )
        self.progress.pack(fill=tk.X, pady=(6, 0))

    def _make_normalize_cmd(self, var: tk.StringVar) -> Callable[[], None]:
        def _cmd() -> None:
            self._normalize_coord_var(var)

        return _cmd

    def _make_event_normalize(self, var: tk.StringVar) -> Callable[[tk.Event], None]:
        def _handler(_e: tk.Event) -> None:
            self._normalize_coord_var(var)

        return _handler

    def _make_coord_row(
        self, parent: tk.Misc, title: str, high_var: tk.StringVar, low_var: tk.StringVar
    ) -> None:
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
            command=self._make_normalize_cmd(high_var),
        )
        e1.pack(side=tk.LEFT)
        with contextlib.suppress(Exception):
            e1.bind('<FocusOut>', self._make_event_normalize(high_var))
            e1.bind('<Return>', self._make_event_normalize(high_var))
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
            command=self._make_normalize_cmd(low_var),
        )
        e2.pack(side=tk.LEFT)
        with contextlib.suppress(Exception):
            e2.bind('<FocusOut>', self._make_event_normalize(low_var))
            e2.bind('<Return>', self._make_event_normalize(low_var))
        self._edit_controls.append(e2)

    def _make_labeled_spin(
        self,
        parent: tk.Misc,
        label: str,
        var: tk.IntVar,
        limits: tuple[int, int] = (0, 10000),
    ) -> None:
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, anchor='w').pack(side=tk.LEFT)
        sp = tk.Spinbox(
            row,
            from_=limits[0],
            to=limits[1],
            increment=1,
            textvariable=var,
            width=8,
            justify='right',
        )
        sp.pack(side=tk.RIGHT)
        self._edit_controls.append(sp)

    def _build_info_panel(self, frame: tk.Misc) -> None:
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
        self._make_coord_row(bl_group, 'X', self._from_x_high_var, self._from_x_low_var)
        self._make_coord_row(bl_group, 'Y', self._from_y_high_var, self._from_y_low_var)

        tr_group = ttk.LabelFrame(corners_col, text='Правый верхний угол карты')
        tr_group.pack(fill=tk.X, padx=(0, 8), pady=(6, 0))
        self._to_x_high_var = tk.StringVar()
        self._to_x_low_var = tk.StringVar()
        self._to_y_high_var = tk.StringVar()
        self._to_y_low_var = tk.StringVar()
        self._make_coord_row(tr_group, 'X', self._to_x_high_var, self._to_x_low_var)
        self._make_coord_row(tr_group, 'Y', self._to_y_high_var, self._to_y_low_var)

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
        # PNG compression UI state
        self._png_compress_var = tk.IntVar()
        self._png_compress_text = tk.StringVar(value='6')

        self._make_labeled_spin(
            grid_group, 'Толщина линии, px:', self._grid_width_var, (0, 1000)
        )
        self._make_labeled_spin(
            grid_group, 'Размер шрифта, px:', self._grid_font_size_var, (1, 2000)
        )
        self._make_labeled_spin(
            grid_group,
            'Отступ надписей от края карты, px:',
            self._grid_text_margin_var,
            (0, 2000),
        )
        self._make_labeled_spin(
            grid_group,
            'Поля маски надписи, px:',
            self._grid_label_bg_padding_var,
            (0, 500),
        )

        # Слайдер прозрачности маски
        opacity_row = tk.Frame(grid_group)
        opacity_row.pack(fill=tk.X, pady=2)
        tk.Label(opacity_row, text='Интенсивность засветки:').pack(side=tk.LEFT)
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

        # Слайдер степени сжатия PNG
        compr_row = tk.Frame(grid_group)
        compr_row.pack(fill=tk.X, pady=2)
        tk.Label(compr_row, text='Степень сжатия:').pack(side=tk.LEFT)
        self._png_compress_scale = ttk.Scale(
            compr_row,
            from_=0,
            to=9,
            orient='horizontal',
            variable=self._png_compress_var,
            command=lambda _v: self._on_png_compress_change(),
        )
        self._png_compress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._edit_controls.append(self._png_compress_scale)
        compr_val_lbl = tk.Label(
            compr_row,
            textvariable=self._png_compress_text,
            width=8,
            anchor='e',
            justify='right',
        )
        compr_val_lbl.pack(side=tk.RIGHT, padx=(10, 0))

    def _build_preview(self, frame: tk.Misc) -> None:
        self.preview_container = tk.Frame(frame, height=320)
        # Start hidden: keep pack options for later, but do not pack yet
        self._preview_pack_opts = dict(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.preview_container.pack_propagate(flag=False)
        # Switch to grid layout inside preview_container to properly dock scrollbars
        with contextlib.suppress(Exception):
            self.preview_container.grid_rowconfigure(0, weight=1)
            self.preview_container.grid_columnconfigure(0, weight=1)
        # Scrollbars (hidden by default)
        self.v_scroll = ttk.Scrollbar(self.preview_container, orient='vertical', command=self._on_vscroll)
        self.h_scroll = ttk.Scrollbar(self.preview_container, orient='horizontal', command=self._on_hscroll)
        # Corner spacer (used when both scrollbars visible)
        self._corner_spacer = tk.Frame(self.preview_container, width=1, height=1)
        # Label for rendering preview
        self.preview_label = tk.Label(
            self.preview_container,
            text='',  # No placeholder text; the entire area is hidden when no image
            anchor='center',
            justify='center',
            relief='sunken',
            bd=1,
            padx=6,
            pady=6,
        )
        # Make label focusable so MouseWheel can be delivered reliably on Windows
        with contextlib.suppress(Exception):
            self.preview_label.configure(takefocus=1)
        # Place widgets in grid; initially keep scrollbars and corner hidden
        with contextlib.suppress(Exception):
            self.preview_label.grid(row=0, column=0, sticky='nsew')
            # Do not grid scrollbars yet; they will be shown via _update_scrollbars
            # Ensure they are removed if gridded previously
            try:
                self.v_scroll.grid_remove()
                self.h_scroll.grid_remove()
                self._corner_spacer.grid_remove()
            except Exception:
                pass
        # Track visibility state
        self._h_scroll_visible = False
        self._v_scroll_visible = False
        self._preview_container_visible = False
        self.preview_img: ImageTk.PhotoImage | None = None
        self.preview_src_img: Image.Image | None = None
        self._resize_job_id: str | None = None
        self.preview_container.bind('<Configure>', self._on_preview_resize)
        self._preview_from_memory: bool = False
        # Whether we've adjusted the preview container to match image aspect on first show
        self._preview_aspect_applied: bool = False
        # Preview interactive state
        self._preview_zoom_factor: float = 1.0  # relative to fit scale
        self._preview_center_x: float | None = None  # in source image coords
        self._preview_center_y: float | None = None
        self._preview_fit_scale: float = 1.0  # computed based on available area
        self._preview_avail_w: int = 1
        self._preview_avail_h: int = 1
        self._preview_is_panning: bool = False
        self._preview_last_mouse_x: int = 0
        self._preview_last_mouse_y: int = 0
        # Render scheduling and quality
        self._render_job_id: str | None = None
        # Wheel handling state (to avoid duplicate events)
        self._wheel_global_active: bool = False
        self._wheel_label_bound: bool = True
        self._hq_job_id: str | None = None
        self._is_interacting: bool = False
        try:
            self._resample_quality = Image.Resampling.LANCZOS
        except Exception:
            self._resample_quality = Image.BILINEAR
        # Mouse bindings for zoom and pan
        try:
            # Wheel zoom (Windows/Mac)
            self.preview_label.bind('<MouseWheel>', self._on_preview_wheel)
            # On Windows/Tk, wheel events often require focus; also hook global wheel while hovering
            self.preview_label.bind('<Enter>', self._on_preview_enter)
            self.preview_label.bind('<Leave>', self._on_preview_leave)
            # Wheel zoom (Linux)
            self.preview_label.bind('<Button-4>', lambda e: self._on_preview_wheel(self._make_wheel_event(e, delta=120)))
            self.preview_label.bind('<Button-5>', lambda e: self._on_preview_wheel(self._make_wheel_event(e, delta=-120)))
            # Pan with left mouse button
            self.preview_label.bind('<ButtonPress-1>', self._on_preview_button_press)
            self.preview_label.bind('<B1-Motion>', self._on_preview_mouse_move)
            self.preview_label.bind('<ButtonRelease-1>', self._on_preview_button_release)
        except Exception:
            pass

    def _post_build_setup(self) -> None:
        # Заполним панель начальными значениями текущего профиля
        with contextlib.suppress(Exception):
            self._populate_profile_fields()
        # Установим обработчики изменений,
        # чтобы обновлять отображение и внутреннее состояние
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
        except Exception as e:
            logger.debug('Trace add failed: %s', e)

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
        # Автозакрытие меню: хранение job-id для мониторинга курсора по каждому меню
        self._menu_autoclose_jobs: dict[int, str] = {}

    def _enable_menu_autoclose(self, menu: tk.Menu) -> None:
        """Enable robust auto-close when the mouse leaves the dropdown menu.

        Implementation details:
        - Start a small polling loop when the menu is mapped (shown) to check the pointer.
        - If the pointer is outside this menu and its cascaded submenus, unpost it.
        - Stop polling when the menu is unmapped (closed) to avoid leaks.
        - Works for any tk.Menu; call this for every dropdown you add in future.
        """
        # Helper: collect this menu and all its cascaded submenus (recursively)
        def _collect_related(m: tk.Menu, acc: set[int]) -> None:
            mid = id(m)
            if mid in acc:
                return
            acc.add(mid)
            try:
                end_index = m.index('end')
            except Exception:
                end_index = None
            if end_index is None:
                return
            for i in range(end_index + 1):
                with contextlib.suppress(Exception):
                    if str(m.type(i)) == 'cascade':
                        sub_name = m.entrycget(i, 'menu')
                        if sub_name:
                            with contextlib.suppress(Exception):
                                sub = m.nametowidget(sub_name)
                                if isinstance(sub, tk.Menu):
                                    _collect_related(sub, acc)

        def _is_pointer_inside_any_related() -> bool:
            # Get current pointer coords and containing widget
            try:
                x, y = self.root.winfo_pointerxy()
                w = self.root.winfo_containing(x, y)
            except Exception:
                return False
            if not isinstance(w, tk.Misc):
                return False
            # We only keep open if pointer is inside this menu or its cascaded submenus
            related: set[int] = set()
            _collect_related(menu, related)
            # If the widget under pointer is a Menu and belongs to related set — OK
            if isinstance(w, tk.Menu) and id(w) in related:
                return True
            # Otherwise — outside
            return False

        # Start/stop monitor on Map/Unmap
        def _start_monitor(_e: tk.Event | None = None) -> None:
            # Avoid duplicate monitors per menu
            key = id(menu)
            if key in self._menu_autoclose_jobs:
                return
            def _tick() -> None:
                key_inner = id(menu)
                if not getattr(menu, 'winfo_ismapped', lambda: False)():
                    # Menu already closed; stop.
                    jid = self._menu_autoclose_jobs.pop(key_inner, None)
                    return
                if not _is_pointer_inside_any_related():
                    with contextlib.suppress(Exception):
                        menu.unpost()
                    self._menu_autoclose_jobs.pop(key_inner, None)
                    return
                # schedule next tick
                try:
                    jid2 = self.root.after(80, _tick)
                    self._menu_autoclose_jobs[key_inner] = jid2
                except Exception:
                    self._menu_autoclose_jobs.pop(key_inner, None)
            # First schedule
            try:
                jid = self.root.after(120, _tick)
                self._menu_autoclose_jobs[key] = jid
            except Exception:
                pass

        def _stop_monitor(_e: tk.Event | None = None) -> None:
            key = id(menu)
            jid = self._menu_autoclose_jobs.pop(key, None)
            if jid is not None:
                with contextlib.suppress(Exception):
                    self.root.after_cancel(jid)

        with contextlib.suppress(Exception):
            menu.bind('<Map>', _start_monitor, add='+')
        with contextlib.suppress(Exception):
            menu.bind('<Unmap>', _stop_monitor, add='+')

    def _create_menu(self) -> None:
        with contextlib.suppress(Exception):
            menubar = tk.Menu(self.root)
            # Файл
            file_menu = tk.Menu(menubar, tearoff=0)
            # Автозакрытие меню при уходе курсора
            self._enable_menu_autoclose(file_menu)
            # Пункт сохранения карты — аналог одноимённой кнопки
            file_menu.add_command(
                label='Сохранить карту…', command=self._handle_save_click, state=tk.DISABLED
            )
            # Индекс пункта сохранения (для дальнейшего управления состоянием)
            try:
                self.file_menu_save_index = file_menu.index('end')
            except Exception:
                self.file_menu_save_index = 0
            # Пункт загрузки профиля
            file_menu.add_separator()
            file_menu.add_command(
                label='Загрузить профиль…', command=self._choose_profile_file
            )
            # Пункт сохранения профиля
            file_menu.add_command(
                label='Сохранить профиль…', command=self._handle_save_profile_click
            )
            menubar.add_cascade(label='Файл', menu=file_menu)

            # Настройка меню приложения
            self.root.config(menu=menubar)
            self.menubar = menubar
            self.file_menu = file_menu

    def _populate_profile_fields(
        self, settings: _SettingsProto | object | None = None
    ) -> None:
        """Заполнить поля панели актуальными значениями профиля."""
        try:
            if settings is None:
                settings = load_profile(constants.CURRENT_PROFILE)
        except Exception:
            return
        # На время программного обновления значений подавляем обработчики изменений
        prev_suspend = getattr(self, '_suspend_traces', 0)
        self._suspend_traces = prev_suspend + 1
        try:
            self._apply_coords_from_settings(settings)
            self._apply_grid_from_settings(settings)
            # Прозрачность маски
            with contextlib.suppress(Exception):
                opacity = float(getattr(settings, 'mask_opacity', 0.0))
            if not isinstance(opacity, float):
                opacity = 0.0
            self._mask_opacity_var.set(opacity)
            with contextlib.suppress(Exception):
                self._opacity_scale.set(opacity)
            self._mask_opacity_text.set(f'{opacity:.2f}')
            # PNG compress level
            try:
                compr = int(getattr(settings, 'png_compress_level', 6))
            except Exception:
                compr = 6
            compr = max(0, min(9, compr))
            self._png_compress_var.set(compr)
            with contextlib.suppress(Exception):
                self._png_compress_scale.set(compr)
            self._png_compress_text.set(str(compr))
            # Сохраним текущие настройки
            self._current_settings = settings  # type: ignore[assignment]
        except Exception as e:
            # Если что-то пойдёт не так — не падаем UI
            logger.debug('Failed to populate fields: %s', e)
        finally:
            with contextlib.suppress(Exception):
                cur_suspend = getattr(self, '_suspend_traces', 1)
                self._suspend_traces = max(0, cur_suspend - 1)

    def _on_opacity_change(self) -> None:
        try:
            v = float(self._mask_opacity_var.get())
        except Exception:
            v = 0.0
        v = max(0.0, min(1.0, v))
        self._mask_opacity_var.set(v)
        self._mask_opacity_text.set(f'{v:.2f}')
        # Не навязываем немедленное обновление модулей; сбор будет при запуске

    def _on_png_compress_change(self) -> None:
        try:
            v = int(self._png_compress_var.get())
        except Exception:
            v = 6
        v = max(0, min(9, v))
        self._png_compress_var.set(v)
        self._png_compress_text.set(str(v))
        # Не навязываем немедленное обновление модулей; сбор будет при запуске

    def _clamp_0_99(self, v: int | None) -> int:
        try:
            v_int = 0 if v is None else int(v)
        except Exception:
            v_int = 0
        v_int = max(v_int, 0)
        return min(v_int, 99)

    def _set_strvar(self, var: tk.StringVar, val: int) -> None:
        s = str(val)
        if var.get() != s:
            var.set(s)

    def _set_intvar(self, var: tk.IntVar, val: int) -> None:
        try:
            if int(var.get()) != int(val):
                var.set(int(val))
        except Exception:
            var.set(int(val))

    def _apply_coords_from_settings(self, settings: object) -> None:
        self._set_strvar(
            self._from_x_high_var, self._clamp_0_99(getattr(settings, 'from_x_high', 0))
        )
        self._set_strvar(
            self._from_x_low_var, self._clamp_0_99(getattr(settings, 'from_x_low', 0))
        )
        self._set_strvar(
            self._from_y_high_var, self._clamp_0_99(getattr(settings, 'from_y_high', 0))
        )
        self._set_strvar(
            self._from_y_low_var, self._clamp_0_99(getattr(settings, 'from_y_low', 0))
        )
        self._set_strvar(
            self._to_x_high_var, self._clamp_0_99(getattr(settings, 'to_x_high', 0))
        )
        self._set_strvar(
            self._to_x_low_var, self._clamp_0_99(getattr(settings, 'to_x_low', 0))
        )
        self._set_strvar(
            self._to_y_high_var, self._clamp_0_99(getattr(settings, 'to_y_high', 0))
        )
        self._set_strvar(
            self._to_y_low_var, self._clamp_0_99(getattr(settings, 'to_y_low', 0))
        )

    def _apply_grid_from_settings(self, settings: object) -> None:
        self._set_intvar(
            self._grid_width_var, int(getattr(settings, 'grid_width_px', 0))
        )
        self._set_intvar(
            self._grid_font_size_var, int(getattr(settings, 'grid_font_size', 0))
        )
        self._set_intvar(
            self._grid_text_margin_var, int(getattr(settings, 'grid_text_margin', 0))
        )
        self._set_intvar(
            self._grid_label_bg_padding_var,
            int(getattr(settings, 'grid_label_bg_padding', 0)),
        )

    def _validate_coord_input(self, new_val: str) -> bool:
        """
        Валидация ввода координаты по ключевым нажатиям.

        Допускаем пустую строку или до MAX_COORD_DIGITS цифр.
        """
        try:
            if new_val == '':
                return True
            if len(new_val) > MAX_COORD_DIGITS:
                return False
            # Промежуточные значения вроде '0' или '9' допустимы
            return new_val.isdigit()
        except Exception:
            return False

    def _normalize_coord_var(self, var: tk.StringVar) -> None:
        """
        Нормализует значение var в диапазон 0..99 (без ведущих нулей).

        Стремимся к минимальным перезаписям.
        """
        try:
            s = var.get()
            if s is None:
                s = ''
            s = str(s).strip()
            if s == '':
                v = 0
            else:
                try:
                    v = int(s)
                except Exception:
                    v = 0
            v = max(v, 0)
            v = min(v, 99)
            new_s = str(v)
            if s != new_s:
                var.set(new_s)
        except Exception:
            with contextlib.suppress(Exception):
                var.set('0')

    def _on_field_change(self) -> None:
        # Подавление реакций во время программных обновлений
        if getattr(self, '_suspend_traces', 0) > 0:
            return
        # Debounce: откладываем перерасчёт настроек, отменяя предыдущий
        if getattr(self, '_field_change_job_id', None) is not None:
            job_id = self._field_change_job_id
            if job_id is not None:
                with contextlib.suppress(Exception):
                    self.root.after_cancel(job_id)
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
                    if s is None or s == '':
                        # пользователь ещё вводит значение — не пересобирать настройки
                        return
            except Exception as e:
                logger.debug('Field change pre-check failed: %s', e)
            with contextlib.suppress(Exception):
                settings = self._build_settings_from_fields()
                if settings is not None:
                    self._current_settings = settings

        # Небольшая задержка для сглаживания серий нажатий клавиш
        self._field_change_job_id = self.root.after(120, _apply_changes)

    def _build_settings_from_fields(self) -> MapSettings | None:
        """
        Собрать объект настроек из текущих значений полей.

        Возвращает None при ошибке. Не изменяет значения полей (без var.set),
        чтобы избежать каскадных trace-обработчиков.
        """
        try:
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
                v = max(v, 0)
                return min(v, 99)

            return MapSettings(
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
                png_compress_level=int(self._png_compress_var.get()),
            )
        except Exception as e:
            logger.debug('Failed to build settings from fields: %s', e)
            return None

    def _run_task(self) -> None:
        def _classify_exception(exc: BaseException) -> str:
            """Возвращает пользовательское сообщение по типу ошибки."""
            # Собираем цепочку исключений для анализа
            def _iter_chain(e: BaseException):
                seen: set[int] = set()
                cur: BaseException | None = e
                while cur is not None and id(cur) not in seen:
                    seen.add(id(cur))
                    yield cur
                    nxt = cur.__cause__ or cur.__context__
                    cur = nxt if isinstance(nxt, BaseException) else None

            # 1) Недействительный/заблокированный API ключ: 401/403
            for ex in _iter_chain(exc):
                if isinstance(ex, aiohttp.ClientResponseError):
                    sc = getattr(ex, 'status', None)
                    if sc in (401, 403):
                        return 'Неверный или заблокированный ключ API.'
                msg = str(ex).lower()
                if 'доступ запрещ' in msg or 'http 401' in msg or 'http 403' in msg or 'unauthorized' in msg or 'forbidden' in msg:
                    return 'Неверный или заблокированный ключ API.'

            # 2) Проблемы соединения/таймаута/интернета
            net_errnos = {errno.ECONNREFUSED, errno.ETIMEDOUT, errno.ENETUNREACH, errno.EHOSTUNREACH}
            for ex in _iter_chain(exc):
                if isinstance(ex, (asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientConnectorError, aiohttp.ClientOSError)):
                    return 'Невозможно соединиться с сервером. Проверьте интернет-соединение.'
                if isinstance(ex, socket.gaierror):
                    return 'Невозможно соединиться с сервером. Проверьте интернет-соединение.'
                if isinstance(ex, OSError) and getattr(ex, 'errno', None) in net_errnos:
                    return 'Невозможно соединиться с сервером. Проверьте интернет-соединение.'
                # Строковые признаки
                m = str(ex).lower()
                if any(s in m for s in ['timed out', 'timeout', 'connection reset', 'temporary failure in name resolution', 'cannot connect', 'failed to establish a new connection']):
                    return 'Невозможно соединиться с сервером. Проверьте интернет-соединение.'

            # 3) Прочие ошибки
            return 'Ошибка при обработке задачи. Обратитесь к разработчику.'

        try:
            self.on_create_map()
        except SystemExit as e:
            err_text = str(e) or 'Приложение завершилось'
            self.root.after(0, lambda: self._on_done(ok=False, err_text=err_text))
            return
        except Exception as e:
            # Выводим трейс в консоль/лог, но не показываем пользователю
            logging.exception('Ошибка при создании карты')
            user_msg = _classify_exception(e)
            self.root.after(0, lambda: self._on_done(ok=False, err_text=user_msg))
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
        with contextlib.suppress(Exception):
            fm = getattr(self, 'file_menu', None)
            idx = getattr(self, 'file_menu_save_index', 0)
            if isinstance(fm, tk.Menu):
                fm.entryconfig(idx, state=(tk.NORMAL if self.preview_src_img is not None else tk.DISABLED))
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
                with contextlib.suppress(Exception):
                    fm = getattr(self, 'file_menu', None)
                    idx = getattr(self, 'file_menu_save_index', 0)
                    if isinstance(fm, tk.Menu):
                        fm.entryconfig(idx, state=(tk.NORMAL if self.preview_src_img is not None else tk.DISABLED))
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

        # Сначала валидируем введённые координаты и размеры участка
        try:
            cur_settings = self._build_settings_from_fields()
        except Exception as e:
            messagebox.showerror('Ошибка', f'Не удалось прочитать параметры: {e}')
            return
        if cur_settings is None:
            messagebox.showerror('Ошибка', 'Не удалось прочитать параметры')
            return
        try:
            # Проверка относительного положения углов (строго левее и ниже)
            bl_x = cur_settings.bottom_left_x_sk42_gk
            bl_y = cur_settings.bottom_left_y_sk42_gk
            tr_x = cur_settings.top_right_x_sk42_gk
            tr_y = cur_settings.top_right_y_sk42_gk
            if not (bl_x < tr_x and bl_y < tr_y):
                messagebox.showwarning(
                    'Неверные координаты',
                    'Левый нижний угол карты должен быть всегда ниже и левее правого верхнего.',
                )
                return
            # Проверка предельного размера стороны участка
            width_m = tr_x - bl_x
            height_m = tr_y - bl_y
            max_side_m = float(getattr(constants, 'MAX_SIDE_SIZE', 20)) * 1000.0
            if width_m > max_side_m or height_m > max_side_m:
                # Сформируем текст с указанием превышения
                width_km = width_m / 1000.0
                height_km = height_m / 1000.0
                messagebox.showwarning(
                    'Слишком большой участок',
                    (
                        'Общая площадь участка не должна превышать '
                        f'{int(getattr(constants, "MAX_SIDE_SIZE", 20))} км по любой стороне.\n'
                        f'Текущие размеры: ширина ≈ {width_km:.2f} км, высота ≈ {height_km:.2f} км.'
                    ),
                )
                return
        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка при проверке параметров: {e}')
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
        with contextlib.suppress(Exception):
            fm = getattr(self, 'file_menu', None)
            idx = getattr(self, 'file_menu_save_index', 0)
            if isinstance(fm, tk.Menu):
                fm.entryconfig(idx, state=tk.DISABLED)

        self.progress.config(mode='determinate', maximum=100)
        self.progress['value'] = 0

        # Сброс предпросмотра перед новым запуском
        self.preview_img = None
        self.preview_src_img = None
        self._preview_from_memory = False
        # Сбросим флаг несохранённости только после подтверждения пользователем выше
        self._unsaved = False
        self.save_btn.config(state=tk.DISABLED)
        with contextlib.suppress(Exception):
            fm = getattr(self, 'file_menu', None)
            idx = getattr(self, 'file_menu_save_index', 0)
            if isinstance(fm, tk.Menu):
                fm.entryconfig(idx, state=tk.DISABLED)
        # Отменим отложенную подгонку, если она была
        if self._resize_job_id is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self._resize_job_id)
        self._resize_job_id = None
        # Hide preview while processing; no image yet
        self._set_preview_visible(False)

        # Соберём настройки из полей и применим их к модулям
        try:
            # cur_settings уже собран и проверен выше
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
            # Reset interactive preview state for new image
            self._preview_center_x = None
            self._preview_center_y = None
            self._preview_zoom_factor = 1.0
            # Reset first-aspect flag so we can adjust container once for this image
            self._preview_aspect_applied = False
            # Отметим, что есть несохранённые изменения
            self._unsaved = True
            # Разрешим сохранение
            self.save_btn.config(state=tk.NORMAL)
            with contextlib.suppress(Exception):
                fm = getattr(self, 'file_menu', None)
                idx = getattr(self, 'file_menu_save_index', 0)
                if isinstance(fm, tk.Menu):
                    fm.entryconfig(idx, state=tk.NORMAL)
            # Ensure the preview container is visible for rendering
            self._set_preview_visible(True)
            # Рассчитываем текущие доступные размеры и подгоняем
            self.root.update_idletasks()
            # Первичная подгонка: сделать окно предпросмотра с той же пропорцией, что и карта
            self._adjust_preview_container_to_image_aspect()
            max_w = max(200, self.preview_container.winfo_width() or 0)
            max_h = max(200, self.preview_container.winfo_height() or 0)
            max_w = min(max_w, 1000)
            max_h = min(max_h, 900)
            # Выполним подгонку на следующем цикле событий, чтобы не блокировать UI
            try:
                self.root.after(0, lambda: self._resize_preview_to(max_w, max_h))
            except Exception:
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
        with contextlib.suppress(Exception):
            self.profile_load_btn.config(state=state)
        with contextlib.suppress(Exception):
            self.profile_save_btn.config(state=state)
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
        file_menu = getattr(self, 'file_menu', None)
        if isinstance(file_menu, tk.Menu):
            self._set_menu_enabled(file_menu, enabled=enabled)

    # ===== Preview visibility helpers =====
    def _set_preview_visible(self, visible: bool) -> None:
        """Show or hide the preview container as a whole.
        When hidden, also shrink the main window vertically to fit the remaining
        content so there is no empty space. When shown, restore default minsize.
        Idempotent.
        """
        cur = bool(getattr(self, '_preview_container_visible', False))
        if visible and not cur:
            with contextlib.suppress(Exception):
                self.preview_container.pack(**getattr(self, '_preview_pack_opts', {}))
            self._preview_container_visible = True
            # Restore default minimum window height (for preview usage)
            with contextlib.suppress(Exception):
                minw = int(getattr(self, '_default_min_w', 640))
                minh = int(getattr(self, '_default_min_h', 480))
                self.root.minsize(minw, minh)
        elif not visible and cur:
            # Hide the preview container
            with contextlib.suppress(Exception):
                self.preview_container.pack_forget()
            self._preview_container_visible = False
            # Allow shrinking below the default min-height
            with contextlib.suppress(Exception):
                minw = int(getattr(self, '_default_min_w', 640))
                # Reduce the minimal height so the window can shrink to content
                self.root.minsize(minw, 1)
            # Let Tk recompute requested sizes and then fit window to content
            def _shrink() -> None:
                with contextlib.suppress(Exception):
                    self.root.update_idletasks()
                    # Clearing geometry string asks Tk to fit to requested size
                    # of contained widgets (no extra empty space).
                    self.root.geometry('')
            # If window not mapped yet, shrink immediately; else schedule
            try:
                if not self.root.winfo_ismapped():
                    _shrink()
                else:
                    self.root.after(0, _shrink)
            except Exception:
                _shrink()

    def _set_initial_preview_square(self) -> None:
        """Ensure the preview area is square on first app launch.
        Sets container height equal to its current width and enlarges the main
        window vertically if needed so the square actually fits. Safe to call
        if already visible.
        """
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        try:
            cw = int(self.preview_container.winfo_width() or 0)
        except Exception:
            cw = 0
        if cw <= 0:
            cw = 600
        target_h = max(1, cw)
        # Capture current sizes to compute delta for the main window height
        try:
            cur_cont_h = int(self.preview_container.winfo_height() or 0)
        except Exception:
            cur_cont_h = 0
        try:
            cur_root_w = int(self.root.winfo_width() or 0)
            cur_root_h = int(self.root.winfo_height() or 0)
            root_x = int(self.root.winfo_x() or 0)
            root_y = int(self.root.winfo_y() or 0)
        except Exception:
            cur_root_w = 0
            cur_root_h = 0
            root_x = 0
            root_y = 0
        with contextlib.suppress(Exception):
            # Lock expansion briefly to enforce height
            self.preview_container.pack_configure(expand=False)
            self.preview_container.config(height=target_h)
        # Grow the main window height if necessary so the square fits
        try:
            if cur_cont_h > 0 and cur_root_h > 0:
                delta = max(0, target_h - cur_cont_h)
                if delta > 0:
                    new_h = cur_root_h + delta
                    self.root.geometry(f"{max(1, cur_root_w)}x{max(1, new_h)}+{root_x}+{root_y}")
        except Exception:
            pass
        def _restore_expand() -> None:
            with contextlib.suppress(Exception):
                self.preview_container.pack_configure(expand=True)
        with contextlib.suppress(Exception):
            try:
                self.root.after(200, _restore_expand)
            except Exception:
                _restore_expand()

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
            defaultextension='.jpg',
            filetypes=[('JPEG изображение', '*.jpg *.jpeg *.JPG *.JPEG'), ('Все файлы', '*.*')],
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
                save_kwargs: dict[str, object] = {}
                try:
                    suffix = Path(fp).suffix.lower()
                    lvl = int(self._png_compress_var.get())
                    lvl = max(0, min(9, lvl))
                    if suffix == '.png':
                        save_kwargs = {'compress_level': lvl}
                    elif suffix in ('.jpg', '.jpeg'):
                        quality = max(10, min(95, 95 - lvl * 7))
                        save_kwargs = {
                            'format': 'JPEG',
                            'quality': quality,
                            'subsampling': 0,
                            'optimize': True,
                            'progressive': True,
                            'exif': b'',
                        }
                except Exception:
                    save_kwargs = {}
                _img.save(fp, **save_kwargs)
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
                        fm = getattr(self, 'file_menu', None)
                        idx = getattr(self, 'file_menu_save_index', 0)
                        if isinstance(fm, tk.Menu):
                            fm.entryconfig(idx, state=tk.DISABLED)
                else:
                    # В случае ошибки разрешим повторную попытку сохранения
                    self.status_var.set('Ошибка сохранения')
                    messagebox.showerror(
                        'Ошибка сохранения', f'Не удалось сохранить файл:\n{err}'
                    )
                    with contextlib.suppress(Exception):
                        self.save_btn.config(state=tk.NORMAL)
                        fm = getattr(self, 'file_menu', None)
                        idx = getattr(self, 'file_menu_save_index', 0)
                        if isinstance(fm, tk.Menu):
                            fm.entryconfig(idx, state=tk.NORMAL)

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
            with contextlib.suppress(Exception):
                fm = getattr(self, 'file_menu', None)
                idx = getattr(self, 'file_menu_save_index', 0)
                if isinstance(fm, tk.Menu):
                    fm.entryconfig(idx, state=tk.DISABLED)
            # Hide preview area entirely until a new map is created
            self._set_preview_visible(False)
            # Не показываем картинку из старого output_path
            # чтобы избежать «застывшей» картинки
        except Exception as e:
            messagebox.showwarning(
                'Профиль применён частично',
                f'Профиль загружен, но возникла проблема при применении:\n{e}',
            )

    def _handle_save_profile_click(self) -> None:
        """Сохранить текущий профиль (MapSettings) в TOML."""
        # Сначала пытаемся собрать настройки из текущих полей, чтобы учесть изменения
        settings: MapSettings | None = None
        with contextlib.suppress(Exception):
            settings = self._build_settings_from_fields()
        if settings is None:
            # Фолбэк: берём последние известные настройки
            with contextlib.suppress(Exception):
                cs = getattr(self, '_current_settings', None)
                if isinstance(cs, MapSettings):
                    settings = cs
        if settings is None:
            # Ещё один фолбэк — пробуем загрузить текущий профиль
            try:
                settings = load_profile(constants.CURRENT_PROFILE)
            except Exception as e:
                messagebox.showerror('Сохранение профиля', f'Нет данных профиля для сохранения:\n{e}')
                return

        # Определяем каталог и предлагаемое имя файла
        curp = Path(str(constants.CURRENT_PROFILE))
        if curp.suffix.lower() == '.toml':
            initialdir = curp.parent
            initialfile = curp.name
        else:
            # Если указано имя без .toml — возьмём стандартную папку профилей
            try:
                initialdir = profile_path('default').parent
            except Exception:
                initialdir = Path.cwd()
            # Предложим текущее имя + .toml
            stem = str(constants.CURRENT_PROFILE).strip() or 'profile'
            initialfile = f'{stem}.toml'

        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            title='Сохранить профиль как TOML',
            initialdir=str(initialdir),
            initialfile=str(initialfile),
            defaultextension='.toml',
            filetypes=[('TOML файлы', '*.toml'), ('Все файлы', '*.*')],
        )
        if not file_path:
            return

        # Формируем TOML и сохраняем
        try:
            data = settings.model_dump()
            text = tomlkit.dumps(data)
            Path(file_path).write_text(text, encoding='utf-8')
        except Exception as e:
            messagebox.showerror('Сохранение профиля', f'Не удалось сохранить профиль:\n{e}')
            return

        # Обновим текущий профиль и сообщим пользователю
        try:
            constants.CURRENT_PROFILE = file_path
            self.status_var.set(f'Профиль сохранён: {Path(file_path).name}')
        except Exception:
            pass

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

    def _make_wheel_event(self, e: tk.Event, *, delta: int) -> tk.Event:
        # Helper to normalize Linux button-4/5 to MouseWheel-like event
        try:
            e.delta = delta  # type: ignore[attr-defined]
        except Exception:
            pass
        return e

    def _adjust_preview_container_to_image_aspect(self) -> None:
        """On first image show, make the preview container square: height equals width.
        Also enlarge the main window vertically if needed so the square area fits.
        """
        if self._preview_aspect_applied:
            return
        # If container is hidden, pack it temporarily to measure width
        if not getattr(self, '_preview_container_visible', False):
            with contextlib.suppress(Exception):
                self.preview_container.pack(**getattr(self, '_preview_pack_opts', {}))
            self._preview_container_visible = True
        # Measure current container width
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        try:
            cw = int(self.preview_container.winfo_width() or 0)
        except Exception:
            cw = 0
        if cw <= 0:
            cw = 600  # fallback width
        target_h = max(1, cw)
        # Capture current sizes to compute delta for the main window height
        try:
            cur_cont_h = int(self.preview_container.winfo_height() or 0)
        except Exception:
            cur_cont_h = 0
        try:
            cur_root_w = int(self.root.winfo_width() or 0)
            cur_root_h = int(self.root.winfo_height() or 0)
            root_x = int(self.root.winfo_x() or 0)
            root_y = int(self.root.winfo_y() or 0)
        except Exception:
            cur_root_w = 0
            cur_root_h = 0
            root_x = 0
            root_y = 0
        # Apply square height to container first
        with contextlib.suppress(Exception):
            self.preview_container.pack_configure(expand=False)
            self.preview_container.config(height=target_h)
        # If the window is too short to accommodate the new container height,
        # increase the window height by the required delta to ensure a square area.
        try:
            if cur_cont_h > 0 and cur_root_h > 0:
                delta = max(0, target_h - cur_cont_h)
                if delta > 0:
                    new_h = cur_root_h + delta
                    self.root.geometry(f"{max(1, cur_root_w)}x{max(1, new_h)}+{root_x}+{root_y}")
        except Exception:
            pass
        # Restore expansion shortly after first rendering is scheduled
        def _restore_expand() -> None:
            with contextlib.suppress(Exception):
                self.preview_container.pack_configure(expand=True)
        with contextlib.suppress(Exception):
            try:
                self.root.after(200, _restore_expand)
            except Exception:
                _restore_expand()
        self._preview_aspect_applied = True

    # ===== Preview interaction helpers =====
    def _cursor_pos_in_preview(self) -> tuple[int, int]:
        """Get mouse position relative to preview image area (label inner rect).
        Accounts for borderwidth, highlightthickness, and padding.
        Works reliably even for bind_all handlers on Windows.
        """
        try:
            x_root, y_root = self.root.winfo_pointerxy()
            # Widget outer origin in root coords
            wx = self.preview_label.winfo_rootx()
            wy = self.preview_label.winfo_rooty()
            # Inner offsets (left/top) inside label
            try:
                padx = int(self.preview_label.cget('padx'))
                pady = int(self.preview_label.cget('pady'))
            except Exception:
                padx = 0
                pady = 0
            try:
                bd = int(self.preview_label.cget('bd'))
            except Exception:
                bd = 0
            try:
                ht = int(self.preview_label.cget('highlightthickness'))
            except Exception:
                ht = 0
            inner_left = wx + bd + ht + padx
            inner_top = wy + bd + ht + pady
            lx = x_root - inner_left
            ly = y_root - inner_top
        except Exception:
            lx = self._preview_avail_w // 2
            ly = self._preview_avail_h // 2
        # clamp to available area
        aw = max(0, int(self._preview_avail_w))
        ah = max(0, int(self._preview_avail_h))
        lx = max(0, min(int(lx), max(0, aw - 1)))
        ly = max(0, min(int(ly), max(0, ah - 1)))
        return lx, ly

    def _enter_interactive_mode(self) -> None:
        self._is_interacting = True
        try:
            self._resample_quality = Image.Resampling.BILINEAR
        except Exception:
            self._resample_quality = Image.BILINEAR
        # cancel pending HQ finalize if any
        if self._hq_job_id is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self._hq_job_id)
            self._hq_job_id = None

    def _schedule_render(self, delay_ms: int = 12) -> None:
        if self._render_job_id is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self._render_job_id)
        try:
            self._render_job_id = self.root.after(delay_ms, self._render_preview_current)
        except Exception:
            self._render_job_id = None
            self._render_preview_current()

    def _schedule_hq_render(self, delay_ms: int = 160) -> None:
        def _apply() -> None:
            self._is_interacting = False
            try:
                self._resample_quality = Image.Resampling.LANCZOS
            except Exception:
                self._resample_quality = Image.BILINEAR
            self._render_preview_current()
            self._hq_job_id = None
        if self._hq_job_id is not None:
            with contextlib.suppress(Exception):
                self.root.after_cancel(self._hq_job_id)
        try:
            self._hq_job_id = self.root.after(delay_ms, _apply)
        except Exception:
            self._hq_job_id = None
            _apply()

    def _on_preview_enter(self, _e: tk.Event) -> None:
        # Ensure focus and capture wheel globally while hovering over preview
        with contextlib.suppress(Exception):
            self.preview_label.focus_set()
        # Avoid duplicate handlers: disable label-level wheel binding while global is active
        with contextlib.suppress(Exception):
            self.preview_label.unbind('<MouseWheel>')
            self.preview_label.unbind('<Button-4>')
            self.preview_label.unbind('<Button-5>')
            self._wheel_label_bound = False
        with contextlib.suppress(Exception):
            # Capture wheel globally (Windows/macOS) and Linux Button-4/5 while hovering
            self.root.bind_all('<MouseWheel>', self._on_preview_wheel)
            self.root.bind_all('<Button-4>', lambda e: self._on_preview_wheel(self._make_wheel_event(e, delta=120)))
            self.root.bind_all('<Button-5>', lambda e: self._on_preview_wheel(self._make_wheel_event(e, delta=-120)))
            self._wheel_global_active = True

    def _on_preview_leave(self, _e: tk.Event) -> None:
        # Release global wheel capture when cursor leaves preview
        with contextlib.suppress(Exception):
            self.root.unbind_all('<MouseWheel>')
            self.root.unbind_all('<Button-4>')
            self.root.unbind_all('<Button-5>')
            self._wheel_global_active = False
        # Restore label-level wheel binding for fallback (for all platforms)
        with contextlib.suppress(Exception):
            self.preview_label.bind('<MouseWheel>', self._on_preview_wheel)
            self.preview_label.bind('<Button-4>', lambda e: self._on_preview_wheel(self._make_wheel_event(e, delta=120)))
            self.preview_label.bind('<Button-5>', lambda e: self._on_preview_wheel(self._make_wheel_event(e, delta=-120)))
            self._wheel_label_bound = True

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
        # Measure label's inner (content) size precisely: subtract border, highlight and padding
        # Prefer actual label size; fall back to provided container size
        lw = int(self.preview_label.winfo_width() or 0)
        lh = int(self.preview_label.winfo_height() or 0)
        if lw <= 1 or lh <= 1:
            lw, lh = int(max_w), int(max_h)
        try:
            padx = int(self.preview_label.cget('padx'))
            pady = int(self.preview_label.cget('pady'))
        except Exception:
            padx = 0
            pady = 0
        try:
            bd = int(self.preview_label.cget('bd'))
        except Exception:
            bd = 0
        try:
            ht = int(self.preview_label.cget('highlightthickness'))
        except Exception:
            ht = 0
        inner_w = max(1, lw - 2 * (bd + ht) - 2 * padx)
        inner_h = max(1, lh - 2 * (bd + ht) - 2 * pady)
        self._preview_avail_w = inner_w
        self._preview_avail_h = inner_h
        src_w, src_h = self.preview_src_img.size
        if src_w <= 0 or src_h <= 0:
            return
        # Fit scale must be defined by width only (height can grow up to screen bounds)
        fit_scale = max(0.01, inner_w / src_w)
        self._preview_fit_scale = fit_scale
        # Initialize center if needed
        if self._preview_center_x is None or self._preview_center_y is None:
            self._preview_center_x = src_w / 2
            self._preview_center_y = src_h / 2
            self._preview_zoom_factor = 1.0
        self._render_preview_current()
        self._resize_job_id = None

    def _render_preview_current(self) -> None:
        if self.preview_src_img is None:
            return
        avail_w = max(1, int(self._preview_avail_w))
        avail_h = max(1, int(self._preview_avail_h))
        src_w, src_h = self.preview_src_img.size
        if src_w <= 0 or src_h <= 0:
            return
        # final scale in screen px per source px
        final_scale = max(0.01, self._preview_fit_scale * max(0.05, min(self._preview_zoom_factor, 100.0)))
        # desired viewport size in source coords (float)
        crop_w_f = max(1.0, float(avail_w) / float(final_scale))
        crop_h_f = max(1.0, float(avail_h) / float(final_scale))
        # center in source coords (do NOT clamp to preserve anchor behavior near edges)
        cx = float(self._preview_center_x or (src_w / 2))
        cy = float(self._preview_center_y or (src_h / 2))
        # desired viewport (may go outside the source bounds)
        left_f = cx - crop_w_f / 2.0
        top_f = cy - crop_h_f / 2.0
        right_f = left_f + crop_w_f
        bottom_f = top_f + crop_h_f
        # intersection with source image
        inter_left_f = max(0.0, left_f)
        inter_top_f = max(0.0, top_f)
        inter_right_f = min(float(src_w), right_f)
        inter_bottom_f = min(float(src_h), bottom_f)
        # If intersection is empty, show blank canvas
        if inter_right_f <= inter_left_f or inter_bottom_f <= inter_top_f:
            with contextlib.suppress(Exception):
                canvas = Image.new('RGB', (avail_w, avail_h), (240, 240, 240))
                self.preview_img = ImageTk.PhotoImage(canvas)
                self.preview_label.config(image=self.preview_img, text='')
            return
        # Convert intersection to integer box for cropping
        left = int(round(inter_left_f))
        top = int(round(inter_top_f))
        right = int(round(inter_right_f))
        bottom = int(round(inter_bottom_f))
        # Compute destination placement within the viewport
        dx = int(round((inter_left_f - left_f) * final_scale))
        dy = int(round((inter_top_f - top_f) * final_scale))
        dest_w = max(1, int(round((right - left) * final_scale)))
        dest_h = max(1, int(round((bottom - top) * final_scale)))
        with contextlib.suppress(Exception):
            cropped = self.preview_src_img.crop((left, top, right, bottom))
            resized = cropped.resize((dest_w, dest_h), self._resample_quality)
            # Create canvas and paste the resized patch at offset; out-of-bounds area remains blank
            canvas = Image.new('RGB', (avail_w, avail_h), (240, 240, 240))
            # PIL allows negative offsets; it will clip automatically
            canvas.paste(resized, (dx, dy))
            self.preview_img = ImageTk.PhotoImage(canvas)
            self.preview_label.config(image=self.preview_img, text='')
        # Update scrollbars based on new geometry
        with contextlib.suppress(Exception):
            self._update_scrollbars()

    def _on_preview_wheel(self, event: tk.Event) -> None:
        if self.preview_src_img is None:
            return
        # Normalize delta by sign only (works for wheels and touchpads on Windows)
        raw = 0
        try:
            raw = int(getattr(event, 'delta', 0))
        except Exception:
            raw = 0
        if raw == 0:
            return
        sign = 1 if raw > 0 else -1
        # If trying to zoom out below fit, ignore to avoid jumpy center due to crop capping
        if sign < 0 and self._preview_zoom_factor <= 1.0 + 1e-6:
            return
        # Cursor position relative to label, robust under bind_all
        x, y = self._cursor_pos_in_preview()
        # Compute current final scale and the source point under cursor
        avail_w = self._preview_avail_w
        avail_h = self._preview_avail_h
        src_w, src_h = self.preview_src_img.size
        final_scale_before = max(0.01, self._preview_fit_scale * max(0.05, min(self._preview_zoom_factor, 100.0)))
        cx = float(self._preview_center_x or (src_w / 2))
        cy = float(self._preview_center_y or (src_h / 2))
        src_px = cx + (x - avail_w / 2) / final_scale_before
        src_py = cy + (y - avail_h / 2) / final_scale_before
        # Adjust zoom factor
        zoom_step = 1.12 if sign > 0 else (1.0 / 1.12)
        new_zoom = self._preview_zoom_factor * zoom_step
        # Clamp zoom factor to [1.0 .. 20.0] relative to fit
        new_zoom = max(1.0, min(new_zoom, 20.0))
        self._preview_zoom_factor = new_zoom
        # Compute new center to keep anchor under cursor stable
        final_scale_after = max(0.01, self._preview_fit_scale * self._preview_zoom_factor)
        new_cx = src_px - (x - avail_w / 2) / final_scale_after
        new_cy = src_py - (y - avail_h / 2) / final_scale_after
        self._preview_center_x = new_cx
        self._preview_center_y = new_cy
        # Schedule coalesced renders with fast filter during interaction
        self._enter_interactive_mode()
        self._schedule_render()
        self._schedule_hq_render()

    def _on_preview_button_press(self, event: tk.Event) -> None:
        self._preview_is_panning = True
        try:
            self._preview_last_mouse_x = int(event.x)  # type: ignore[attr-defined]
            self._preview_last_mouse_y = int(event.y)  # type: ignore[attr-defined]
        except Exception:
            self._preview_last_mouse_x = 0
            self._preview_last_mouse_y = 0

    def _on_preview_mouse_move(self, event: tk.Event) -> None:
        if not self._preview_is_panning or self.preview_src_img is None:
            return
        try:
            x = int(event.x)  # type: ignore[attr-defined]
            y = int(event.y)  # type: ignore[attr-defined]
        except Exception:
            return
        dx = x - self._preview_last_mouse_x
        dy = y - self._preview_last_mouse_y
        self._preview_last_mouse_x = x
        self._preview_last_mouse_y = y
        # Translate drag in screen pixels to source coords delta
        final_scale = max(0.01, self._preview_fit_scale * self._preview_zoom_factor)
        if final_scale <= 0:
            return
        # Move content with mouse: dragging right moves image right -> center shifts negatively
        self._preview_center_x = (self._preview_center_x or 0.0) - dx / final_scale
        self._preview_center_y = (self._preview_center_y or 0.0) - dy / final_scale
        # Schedule coalesced renders
        self._enter_interactive_mode()
        self._schedule_render()
        self._schedule_hq_render()

    def _on_preview_button_release(self, _event: tk.Event) -> None:
        self._preview_is_panning = False
        # finalize with HQ render shortly after release
        self._schedule_hq_render(140)
        # update scrollbars position one more time
        with contextlib.suppress(Exception):
            self._update_scrollbars()

    # ===== Scrollbar integration =====
    def _update_scrollbars(self) -> None:
        if self.preview_src_img is None:
            # hide both
            if self._h_scroll_visible:
                with contextlib.suppress(Exception):
                    self.h_scroll.pack_forget()
                self._h_scroll_visible = False
            if self._v_scroll_visible:
                with contextlib.suppress(Exception):
                    self.v_scroll.pack_forget()
                self._v_scroll_visible = False
            return
        # Geometry
        avail_w = max(1, int(self._preview_avail_w))
        avail_h = max(1, int(self._preview_avail_h))
        src_w, src_h = self.preview_src_img.size
        s = max(0.01, self._preview_fit_scale * max(0.05, min(self._preview_zoom_factor, 100.0)))
        content_w = float(src_w) * s
        content_h = float(src_h) * s
        need_h = content_w > (avail_w + 1)
        need_v = content_h > (avail_h + 1)
        # Show/hide scrollbars as needed
        if need_v and not self._v_scroll_visible:
            with contextlib.suppress(Exception):
                self.v_scroll.grid(row=0, column=1, sticky='ns')
            self._v_scroll_visible = True
        if not need_v and self._v_scroll_visible:
            with contextlib.suppress(Exception):
                self.v_scroll.grid_remove()
            self._v_scroll_visible = False
        if need_h and not self._h_scroll_visible:
            with contextlib.suppress(Exception):
                self.h_scroll.grid(row=1, column=0, sticky='ew')
            self._h_scroll_visible = True
        if not need_h and self._h_scroll_visible:
            with contextlib.suppress(Exception):
                self.h_scroll.grid_remove()
            self._h_scroll_visible = False
        # Corner spacer when both scrollbars visible
        if self._h_scroll_visible and self._v_scroll_visible:
            with contextlib.suppress(Exception):
                self._corner_spacer.grid(row=1, column=1, sticky='nsew')
        else:
            with contextlib.suppress(Exception):
                self._corner_spacer.grid_remove()
        # Re-measure label inner area might have changed due to (un)packing scrollbars
        try:
            prev_aw = int(self._preview_avail_w)
            prev_ah = int(self._preview_avail_h)
            lw = int(self.preview_label.winfo_width() or 0)
            lh = int(self.preview_label.winfo_height() or 0)
            padx = int(self.preview_label.cget('padx'))
            pady = int(self.preview_label.cget('pady'))
            bd = int(self.preview_label.cget('bd'))
            ht = int(self.preview_label.cget('highlightthickness'))
            inner_w = max(1, lw - 2 * (bd + ht) - 2 * padx)
            inner_h = max(1, lh - 2 * (bd + ht) - 2 * pady)
            avail_w = inner_w
            avail_h = inner_h
            self._preview_avail_w = avail_w
            self._preview_avail_h = avail_h
            if avail_w != prev_aw or avail_h != prev_ah:
                # schedule re-render with new viewport size
                self._enter_interactive_mode()
                self._schedule_render()
        except Exception:
            pass
        # Compute current viewport left/top in screen px relative to content
        crop_w_f = float(avail_w) / s
        crop_h_f = float(avail_h) / s
        cx = float(self._preview_center_x or (src_w / 2))
        cy = float(self._preview_center_y or (src_h / 2))
        left_raw_px = (cx - crop_w_f / 2.0) * s
        top_raw_px = (cy - crop_h_f / 2.0) * s
        max_left_px = max(0.0, content_w - avail_w)
        max_top_px = max(0.0, content_h - avail_h)
        left_px = min(max(0.0, left_raw_px), max_left_px)
        top_px = min(max(0.0, top_raw_px), max_top_px)
        # Configure scrollbar thumbs [lo, hi]
        if self._h_scroll_visible:
            total_w = max(1.0, content_w)
            lo = 0.0 if total_w <= 0 else left_px / total_w
            hi = (left_px + avail_w) / total_w
            hi = max(lo, min(1.0, hi))
            with contextlib.suppress(Exception):
                self.h_scroll.set(lo, hi)
        if self._v_scroll_visible:
            total_h = max(1.0, content_h)
            lo = 0.0 if total_h <= 0 else top_px / total_h
            hi = (top_px + avail_h) / total_h
            hi = max(lo, min(1.0, hi))
            with contextlib.suppress(Exception):
                self.v_scroll.set(lo, hi)

    def _on_hscroll(self, *args: object) -> None:
        if self.preview_src_img is None:
            return
        # Compute geometry and scale
        avail_w = max(1, int(self._preview_avail_w))
        src_w = self.preview_src_img.size[0]
        s = max(0.01, self._preview_fit_scale * max(0.05, min(self._preview_zoom_factor, 100.0)))
        content_w = float(src_w) * s
        crop_w_f = float(avail_w) / s
        # Current left in screen px
        cx = float(self._preview_center_x or (src_w / 2))
        left_px = (cx - crop_w_f / 2.0) * s
        max_left_px = max(0.0, content_w - avail_w)
        try:
            if len(args) >= 1 and args[0] == 'moveto':
                f = float(args[1]) if len(args) > 1 else 0.0
                left_px = max(0.0, min(max_left_px, f * max_left_px))
            elif len(args) >= 3 and args[0] == 'scroll':
                n = int(args[1])
                what = str(args[2])
                step = 40.0 if what == 'units' else float(avail_w) * 0.9
                left_px = max(0.0, min(max_left_px, left_px + n * step))
        except Exception:
            return
        # Update center from left_px
        left_src = left_px / s
        new_cx = left_src + crop_w_f / 2.0
        self._preview_center_x = new_cx
        # Schedule re-render
        self._enter_interactive_mode()
        self._schedule_render()
        self._schedule_hq_render()

    def _on_vscroll(self, *args: object) -> None:
        if self.preview_src_img is None:
            return
        avail_h = max(1, int(self._preview_avail_h))
        src_h = self.preview_src_img.size[1]
        s = max(0.01, self._preview_fit_scale * max(0.05, min(self._preview_zoom_factor, 100.0)))
        content_h = float(src_h) * s
        crop_h_f = float(avail_h) / s
        cy = float(self._preview_center_y or (src_h / 2))
        top_px = (cy - crop_h_f / 2.0) * s
        max_top_px = max(0.0, content_h - avail_h)
        try:
            if len(args) >= 1 and args[0] == 'moveto':
                f = float(args[1]) if len(args) > 1 else 0.0
                top_px = max(0.0, min(max_top_px, f * max_top_px))
            elif len(args) >= 3 and args[0] == 'scroll':
                n = int(args[1])
                what = str(args[2])
                step = 40.0 if what == 'units' else float(avail_h) * 0.9
                top_px = max(0.0, min(max_top_px, top_px + n * step))
        except Exception:
            return
        top_src = top_px / s
        new_cy = top_src + crop_h_f / 2.0
        self._preview_center_y = new_cy
        self._enter_interactive_mode()
        self._schedule_render()
        self._schedule_hq_render()

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
                # Файл не найден — скрываем превью
                self.preview_img = None
                self.preview_src_img = None
                self._set_preview_visible(False)
                return

            # Ensure container is visible before sizing
            self._set_preview_visible(True)
            # Определим желаемый размер предпросмотра исходя из текущей ширины
            self.root.update_idletasks()
            max_w = max(200, self.preview_container.winfo_width() or 0)
            max_h = max(200, self.preview_container.winfo_height() or 0)
            # Ограничим разумным максимумом
            max_w = min(max_w, 1000)
            max_h = min(max_h, 800)

            img = Image.open(path)
            self.preview_src_img = cast('Image.Image', img)
            # Reset interactive preview state for new image/file
            self._preview_center_x = None
            self._preview_center_y = None
            self._preview_zoom_factor = 1.0
            # Reset first-aspect flag so new image can adjust container once
            self._preview_aspect_applied = False
            # Подгонка размеров контейнера под пропорции картинки при первом показе
            self._adjust_preview_container_to_image_aspect()
            # Немедленно подгоним под текущие размеры
            self._resize_preview_to(max_w, max_h)
        except Exception as e:  # Ошибка — скрыть превью
            self.preview_img = None
            self.preview_src_img = None
            self._set_preview_visible(False)


def run_app(on_create_map: Callable[[], None]) -> None:
    """Запуск GUI-приложения."""
    ui = _UI(on_create_map)
    ui.mainloop()
