import contextlib
import threading
import time
import tkinter as tk
import traceback
from collections.abc import Callable
from tkinter import messagebox, ttk

from progress import set_progress_callback, set_spinner_callbacks


def run_app(on_create_map: Callable[[], None]) -> None:
    """
    Запускает простое GUI-приложение Tkinter с одной кнопкой
    "Создать карту", которая вызывает переданный колбэк on_create_map().

    Колбэк выполняется в отдельном потоке, чтобы не блокировать GUI.
    """
    root = tk.Tk()
    root.title('Mil Mapper')

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    btn = tk.Button(frame, text='Создать карту', width=24)
    btn.pack(pady=(0, 8))

    status_var = tk.StringVar(value='Готов к созданию карты')
    status_lbl = tk.Label(frame, textvariable=status_var, anchor='w')
    status_lbl.pack(fill=tk.X)

    # Прогресс-бар
    progress = ttk.Progressbar(
        frame, orient='horizontal', mode='determinate', length=280
    )
    progress.pack(fill=tk.X, pady=(6, 0))

    # Флаг завершения текущей задачи, чтобы игнорировать поздние обновления
    finished = {'done': False}

    def on_done(ok: bool, err_text: str | None = None) -> None:
        # Эта функция вызывается в главном потоке через root.after
        # Помечаем, что задача завершена — игнорировать все поздние обновления
        finished['done'] = True
        with contextlib.suppress(Exception):
            progress.stop()
        # Сбрасываем прогресс-бар полностью, чтобы не оставался закрашенный сегмент
        try:
            progress.config(mode='determinate', maximum=100)
            progress['value'] = 0
        except Exception:
            pass
        # Очистить буфер отложенного прогресса
        try:
            pending_progress['data'] = None  # type: ignore[name-defined]
        except Exception:
            pass
        # Отписываем GUI от колбэков прогресса/спиннера на всякий случай
        try:
            set_progress_callback(None)
            set_spinner_callbacks(None, None)
        except Exception:
            pass
        btn.config(state=tk.NORMAL)
        if ok:
            status_var.set('Готово')
        else:
            status_var.set('Ошибка при создании карты')
            messagebox.showerror('Ошибка', err_text or 'Неизвестная ошибка')

    def run_task() -> None:
        try:
            on_create_map()
        except SystemExit as e:
            # Сообщаем пользователю причину завершения (например, отсутствует API_KEY)
            err_text = str(e) or 'Приложение завершилось'
            root.after(0, on_done, False, err_text)
            return
        except Exception:
            err_text = traceback.format_exc()
            root.after(0, on_done, False, err_text)
            return
        root.after(0, on_done, True, None)

    def handle_click() -> None:
        btn.config(state=tk.DISABLED)
        status_var.set('Создание карты… Подождите…')

        # Сбрасываем флаг завершения на случай повторного запуска
        finished['done'] = False

        # Локальное состояние для управления режимами прогресс-бара
        spinner_active = {'count': 0}  # используем dict, чтобы замкнуть по ссылке
        progress_running = {'on': False}
        last_spinner_stop_ts = {'t': 0.0}
        spinner_cooldown_s = (
            0.2  # игнорировать детерминированные обновления сразу после спиннера
        )
        pending_progress = {
            'data': None
        }  # буфер последнего проигнорированного прогресса
        spinner_mode = {'type': None}  # 'pingpong' или 'marquee' (для сохранения файла)

        # Состояние анимации «вперёд-назад» для проблемных этапов (спиннеров)
        anim = {'job': None, 'val': 0, 'dir': 1}
        anim_max = 100
        anim_step = 1
        anim_interval_ms = 40

        def _apply_progress(done: int, total: int, label: str) -> None:
            if finished['done']:
                return
            # Режим: определённый прогресс
            if str(progress['mode']) != 'determinate':
                progress.config(mode='determinate')
                if progress_running['on']:
                    with contextlib.suppress(Exception):
                        progress.stop()
                    progress_running['on'] = False
            progress.config(maximum=total)
            progress['value'] = done
            status_var.set(f'{label}: {done}/{total}')

        def _flush_pending_if_any() -> None:
            # Вызывается после кулдауна, чтобы дорисовать первый 0/total
            if finished['done']:
                return
            if spinner_active['count'] > 0:
                return
            data = pending_progress['data']
            if data is None:
                return
            done, total, label = data
            pending_progress['data'] = None
            with contextlib.suppress(Exception):
                _apply_progress(done, total, label)

        def _animate_pingpong() -> None:
            # Если всё уже завершено — прекращаем анимацию
            if finished['done']:
                anim['job'] = None
                return
            # Если спиннеры неактивны — прекращаем анимацию
            if spinner_active['count'] <= 0:
                anim['job'] = None
                return
            # Обновляем значение и направление
            v = anim['val'] + anim['dir'] * anim_step
            if v >= anim_max:
                v = anim_max
                anim['dir'] = -1
            elif v <= 0:
                v = 0
                anim['dir'] = 1
            anim['val'] = v
            # Рисуем «сегмент» путем изменения value в determinate-режиме
            try:
                if str(progress['mode']) != 'determinate':
                    progress.config(mode='determinate')
                progress.config(maximum=anim_max)
                progress['value'] = v
            except Exception:
                pass
            # Планируем следующий тик
            anim['job'] = root.after(anim_interval_ms, _animate_pingpong)

        def _start_animation_if_needed() -> None:
            if anim['job'] is None:
                anim['job'] = root.after(anim_interval_ms, _animate_pingpong)

        # Колбэк прогресса (из фонового потока) -> обновления в GUI-потоке
        def progress_cb(done: int, total: int, label: str) -> None:
            def _apply() -> None:
                if finished['done']:
                    return
                # Если активен спиннер или кулдаун — сохраняем прогресс в буфер
                if (
                    spinner_active['count'] > 0
                    or (time.time() - last_spinner_stop_ts['t']) < spinner_cooldown_s
                ):
                    pending_progress['data'] = (done, total, label)
                    return
                # Иначе применяем сразу
                pending_progress['data'] = None
                _apply_progress(done, total, label)

            root.after(0, _apply)

        # Колбэки спиннера -> запустить/остановить индикатор в режиме indeterminate
        def spinner_start_cb(label: str) -> None:
            def _apply() -> None:
                if finished['done']:
                    return
                status_var.set(label)
                spinner_active['count'] += 1
                # Для этапа сохранения используем «маркировку» (один сегмент ttk)
                if 'Сохранение файла' in label:
                    spinner_mode['type'] = 'marquee'
                    try:
                        if str(progress['mode']) != 'indeterminate':
                            progress.config(mode='indeterminate')
                        progress.start(50)  # стандартная индикация одного сегмента
                    except Exception:
                        pass
                else:
                    spinner_mode['type'] = 'pingpong'
                    # Запускаем нашу анимацию «вперёд-назад» при первом входе в спиннер
                    _start_animation_if_needed()

            root.after(0, _apply)

        def spinner_stop_cb(_label: str) -> None:
            def _apply() -> None:
                if finished['done']:
                    return
                # Уменьшаем вложенность спиннеров и останавливаем анимацию
                # только когда все спиннеры завершены
                spinner_active['count'] = max(0, spinner_active['count'] - 1)
                if spinner_active['count'] == 0:
                    last_spinner_stop_ts['t'] = time.time()
                    if spinner_mode['type'] == 'pingpong':
                        # Останавливаем пинг-понг анимацию
                        if anim['job'] is not None:
                            with contextlib.suppress(Exception):
                                root.after_cancel(anim['job'])
                            anim['job'] = None
                    elif spinner_mode['type'] == 'marquee':
                        # Останавливаем стандартный индикатор ttk
                        with contextlib.suppress(Exception):
                            progress.stop()
                    # Планируем принудительную дорисовку буферизованного прогресса
                    root.after(int(spinner_cooldown_s * 1000), _flush_pending_if_any)

            root.after(0, _apply)

        # Регистрируем колбэки перед запуском задачи
        set_progress_callback(progress_cb)
        set_spinner_callbacks(spinner_start_cb, spinner_stop_cb)

        # Сбрасываем прогресс-бар
        progress.config(mode='determinate', maximum=100)
        progress['value'] = 0

        threading.Thread(target=run_task, daemon=True).start()

    btn.config(command=handle_click)

    root.mainloop()
