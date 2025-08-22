import threading
import traceback
import tkinter as tk
from tkinter import messagebox
from typing import Callable


def run_app(on_create_map: Callable[[], None]) -> None:
    """
    Запускает простое GUI-приложение Tkinter с одной кнопкой
    "Создать карту", которая вызывает переданный колбэк on_create_map().

    Колбэк выполняется в отдельном потоке, чтобы не блокировать GUI.
    """

    root = tk.Tk()
    root.title("Mil Mapper")

    frame = tk.Frame(root, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    btn = tk.Button(frame, text="Создать карту", width=24)
    btn.pack(pady=(0, 8))

    status_var = tk.StringVar(value="Готов к созданию карты")
    status_lbl = tk.Label(frame, textvariable=status_var, anchor="w")
    status_lbl.pack(fill=tk.X)

    def on_done(ok: bool, err_text: str | None = None) -> None:
        # Эта функция вызывается в главном потоке через root.after
        btn.config(state=tk.NORMAL)
        if ok:
            status_var.set("Готово")
        else:
            status_var.set("Ошибка при создании карты")
            messagebox.showerror("Ошибка", err_text or "Неизвестная ошибка")

    def run_task() -> None:
        try:
            on_create_map()
        except SystemExit as e:
            # Сообщаем пользователю причину завершения (например, отсутствует API_KEY)
            err_text = str(e) or "Приложение завершилось"
            root.after(0, on_done, False, err_text)
            return
        except Exception:
            err_text = traceback.format_exc()
            root.after(0, on_done, False, err_text)
            return
        root.after(0, on_done, True, None)

    def handle_click() -> None:
        btn.config(state=tk.DISABLED)
        status_var.set("Создание карты… Подождите…")
        threading.Thread(target=run_task, daemon=True).start()

    btn.config(command=handle_click)

    root.mainloop()
