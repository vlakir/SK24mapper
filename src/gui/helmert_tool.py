from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QClipboard
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyproj import CRS, Transformer
from constants import (
    GK_FALSE_EASTING,
    GK_ZONE_WIDTH_DEG,
    GK_ZONE_CM_OFFSET_DEG,
    GK_ZONE_X_PREFIX_DIV,
    MAX_GK_ZONE,
    WGS84_CODE,
)


@dataclass
class Helmert2DResult:
    dx: float  # meters
    dy: float  # meters
    rot_as: float  # arcseconds (rotation about Z)
    ds_ppm: float  # ppm (scale minus 1)

    def to_profile_text(self) -> str:
        # Map to 7-parameter set for profile usage: dz=rx=ry=0
        return (
            "# СК-42 → WGS84 (Helmert, м / угл. сек / ppm)\n"
            f"helmert_dx = {self.dx:.6f}\n"
            f"helmert_dy = {self.dy:.6f}\n"
            f"helmert_dz = {0.0:.6f}\n"
            f"helmert_rx_as = {0.0:.6f}\n"
            f"helmert_ry_as = {0.0:.6f}\n"
            f"helmert_rz_as = {self.rot_as:.6f}\n"
            f"helmert_ds_ppm = {self.ds_ppm:.6f}\n"
        )


def estimate_helmert_2d(p_src: np.ndarray, p_dst: np.ndarray) -> Helmert2DResult:
    """
    Оценка 2D преобразования подобия (Хельмерт-4):
    x' = a*x - b*y + tx
    y' = b*x + a*y + ty
    где a = s*cosα, b = s*sinα, s ≈ 1 + ds.

    Вход:
    - p_src: (N,2) — точки в исходной системе (СК-42, военный вариант, Гаусс–Крюгер, метры)
    - p_dst: (N,2) — соответствующие точки в целевой системе (WGS84, спроецированные в метры той же плоскости)
    Требуется N ≥ 2, рекомендуется N ≥ 6.
    """
    if p_src.shape != p_dst.shape or p_src.shape[1] != 2:
        raise ValueError("Формат точек должен быть (N,2), p_src и p_dst одинаковых размеров")
    n = p_src.shape[0]
    if n < 2:
        raise ValueError("Не менее двух точек")

    # Построим систему на 2N уравнений относительно [a, b, tx, ty]
    A = np.zeros((2 * n, 4), dtype=float)
    L = np.zeros((2 * n, 1), dtype=float)

    for i in range(n):
        x, y = p_src[i]
        X, Y = p_dst[i]
        # x' = a*x - b*y + tx
        A[2 * i, 0] = x
        A[2 * i, 1] = -y
        A[2 * i, 2] = 1.0
        A[2 * i, 3] = 0.0
        L[2 * i, 0] = X
        # y' = b*x + a*y + ty
        A[2 * i + 1, 0] = y
        A[2 * i + 1, 1] = x
        A[2 * i + 1, 2] = 0.0
        A[2 * i + 1, 3] = 1.0
        L[2 * i + 1, 0] = Y

    # МНК решение
    x_hat, *_ = np.linalg.lstsq(A, L, rcond=None)
    a = float(x_hat[0, 0])
    b = float(x_hat[1, 0])
    tx = float(x_hat[2, 0])
    ty = float(x_hat[3, 0])

    s = math.hypot(a, b)
    alpha = math.atan2(b, a)  # radians
    rot_as = math.degrees(alpha) * 3600.0
    ds_ppm = (s - 1.0) * 1e6

    return Helmert2DResult(dx=tx, dy=ty, rot_as=rot_as, ds_ppm=ds_ppm)



def _determine_zone_from_x(eastings: np.ndarray) -> int:
    """Определить 6° зону Гаусса–Крюгера по восточным X (в метрах)."""
    if eastings.size == 0:
        raise ValueError("Нет значений X для определения зоны")
    center_x = float(np.median(eastings))
    zone = int(center_x // GK_ZONE_X_PREFIX_DIV)
    if zone < 1 or zone > MAX_GK_ZONE:
        zone = max(
            1,
            min(
                MAX_GK_ZONE,
                int((center_x - GK_FALSE_EASTING) // GK_ZONE_X_PREFIX_DIV) + 1,
            ),
        )
    return zone


def _build_wgs84_gk_crs(zone: int) -> CRS:
    """Построить TM‑плоскость (аналог ГК) на WGS84 для заданной зоны."""
    lon0 = zone * GK_ZONE_WIDTH_DEG - GK_ZONE_CM_OFFSET_DEG
    proj4 = (
        f"+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 "
        f"+x_0={GK_FALSE_EASTING} +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs"
    )
    return CRS.from_proj4(proj4)


class HelmertToolWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Helmert 2D (СК-42 военный → WGS84) — оценка по 6 точкам")
        self._init_ui()

    def _init_ui(self) -> None:
        cw = QWidget(self)
        self.setCentralWidget(cw)
        v = QVBoxLayout(cw)

        hint = QLabel(
            "Введите 6 соответствующих точек: слева — СК-42 (военный порядок: X — северинг, Y — восточинг, м), справа — WGS84 в градусах (широта, долгота).\n"
            "Утилита автоматически спроецирует WGS84 в метры зоны СК‑42 и оценит 2D Хельмерта (dx, dy, поворот rz, масштаб ds)."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.table = QTableWidget(6, 4, self)
        self.table.setHorizontalHeaderLabels([
            "СК-42 X (военный, северинг м)",
            "СК-42 Y (военный, восточинг м)",
            "WGS84 lat (°)",
            "WGS84 lon (°)",
        ])
        v.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить CSV", self)
        self.btn_calc = QPushButton("Рассчитать", self)
        self.btn_copy = QPushButton("Копировать в профиль", self)
        self.btn_copy.setEnabled(False)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_calc)
        btn_row.addWidget(self.btn_copy)
        v.addLayout(btn_row)

        self.lbl_result = QLabel("Результат: —")
        v.addWidget(self.lbl_result)
        # Диагностика: остатки по точкам и RMS
        self.lbl_diag = QLabel("")
        self.lbl_diag.setWordWrap(True)
        v.addWidget(self.lbl_diag)

        self.btn_load.clicked.connect(self._on_load_csv)
        self.btn_calc.clicked.connect(self._on_calc)
        self.btn_copy.clicked.connect(self._on_copy)

        # Menu: clear, quit
        act_clear = QAction("Очистить", self)
        act_clear.triggered.connect(self._clear)
        self.menuBar().addAction(act_clear)
        act_load = QAction("Загрузить CSV", self)
        act_load.triggered.connect(self._on_load_csv)
        self.menuBar().addAction(act_load)
        act_quit = QAction("Выход", self)
        act_quit.triggered.connect(self.close)
        self.menuBar().addAction(act_quit)

    def _clear(self) -> None:
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                self.table.setItem(r, c, QTableWidgetItem(""))
        self.lbl_result.setText("Результат: —")
        self.lbl_diag.setText("")
        self.btn_copy.setEnabled(False)

    def _read_points(self) -> Tuple[np.ndarray, np.ndarray]:
        src: List[Tuple[float, float]] = []
        dst: List[Tuple[float, float]] = []
        for r in range(self.table.rowCount()):
            try:
                x1 = float(self._text(r, 0))
                y1 = float(self._text(r, 1))
                x2 = float(self._text(r, 2))
                y2 = float(self._text(r, 3))
            except ValueError:
                continue
            src.append((x1, y1))
            dst.append((x2, y2))
        if len(src) < 2:
            raise ValueError("Введите минимум две корректные пары точек")
        return np.array(src, dtype=float), np.array(dst, dtype=float)

    def _text(self, r: int, c: int) -> str:
        item = self.table.item(r, c)
        return item.text().strip() if item is not None else ""

    def _on_calc(self) -> None:
        try:
            # Считываем: слева СК-42 (военный порядок: X=northing, Y=easting), справа WGS84 (lat, lon)
            p_src_m_mil, p_dst_deg_latlon = self._read_points()
            # Приводим СК-42 к геодезическому порядку (easting, northing)
            p_src_en = np.column_stack([p_src_m_mil[:, 1], p_src_m_mil[:, 0]])
            # Определяем зону по восточным X (т.е. по Y военном)
            zone = _determine_zone_from_x(p_src_en[:, 0])
            # Удаляем миллионный префикс зоны из восточинг (военный формат включает zone*1e6)
            p_src_en[:, 0] = p_src_en[:, 0] - zone * GK_ZONE_X_PREFIX_DIV
            # Географическая WGS84 (EPSG:4326)
            crs_wgs84_geog = CRS.from_epsg(WGS84_CODE)
            # WGS84 TM (аналог ГК) для той же зоны
            crs_wgs84_gk = _build_wgs84_gk_crs(zone)
            t_wgs_to_wgs_gk = Transformer.from_crs(
                crs_wgs84_geog, crs_wgs84_gk, always_xy=True
            )
            # Преобразуем градусы в метры: вход (lat, lon) → pyproj ждёт (lon, lat)
            lats = p_dst_deg_latlon[:, 0]
            lons = p_dst_deg_latlon[:, 1]
            Xw, Yw = t_wgs_to_wgs_gk.transform(lons, lats)
            p_dst_m = np.column_stack([Xw, Yw])
            # Оцениваем 2D-Хельмерта по двум метрическим наборам
            res = estimate_helmert_2d(p_src_en, p_dst_m)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(e))
            return
        self._last_result = res
        # Сохраняем последнюю зону для отображения
        self._last_zone = zone
        # Диагностика: вычислим остатки и RMS
        s = 1.0 + (res.ds_ppm / 1e6)
        alpha = math.radians(res.rot_as / 3600.0)
        a = s * math.cos(alpha)
        b = s * math.sin(alpha)
        x = p_src_en[:, 0]
        y = p_src_en[:, 1]
        x_hat = a * x - b * y + res.dx
        y_hat = b * x + a * y + res.dy
        rx = p_dst_m[:, 0] - x_hat
        ry = p_dst_m[:, 1] - y_hat
        r2 = rx**2 + ry**2
        # RMS
        rms_x = float(np.sqrt(np.mean(rx**2)))
        rms_y = float(np.sqrt(np.mean(ry**2)))
        rms_2d = float(np.sqrt(np.mean(r2)))
        # Проверка зоны по долготе (ожидаемая зона из WGS84 lon)
        # Берём медиану долгот из введённых значений
        try:
            median_lon = float(np.median(lons))
            lon360 = (median_lon % 360.0 + 360.0) % 360.0
            zone_by_lon = int(lon360 // 6.0) + 1
        except Exception:
            zone_by_lon = None
        # Проверка "миллионного" префикса и смешанных зон слева
        zones_in_src = set(int(float(val) // GK_ZONE_X_PREFIX_DIV) for val in p_src_m_mil[:, 1])
        warnings: list[str] = []
        if zone_by_lon is not None and zone_by_lon != zone:
            warnings.append(f"Предупреждение: зона по долготам WGS84 = {zone_by_lon}, по СК-42 = {zone}. Проверьте порядок X/Y (военный) и соответствие зоны.")
        if len(zones_in_src) > 1:
            warnings.append(f"Предупреждение: обнаружены разные префиксы зон в исходных СК-42 easting (Y военный): {sorted(zones_in_src)}. Нельзя смешивать точки из разных зон.")
        if abs(res.dx) > 100000 or abs(res.dy) > 100000:
            warnings.append("Предупреждение: очень большие dx/dy (>|100 км|). Возможна проблема с миллионным префиксом зоны или несоответствие проекций.")
        # Сформируем текст диагностики
        lines: list[str] = []
        lines.append(f"RMS: X = {rms_x:.3f} м; Y = {rms_y:.3f} м; 2D = {rms_2d:.3f} м")
        for i in range(rx.size):
            lines.append(f"Точка {i+1}: rX = {rx[i]:.3f} м; rY = {ry[i]:.3f} м; |r| = {math.sqrt(r2[i]):.3f} м")
        if warnings:
            lines.append("\n" + " \n".join(warnings))
        diag_text = "\n".join(lines)
        self.lbl_diag.setText(diag_text)
        self.lbl_result.setText(
            f"Результат: dx = {res.dx:.6f} м; dy = {res.dy:.6f} м; rz = {res.rot_as:.6f}''; ds = {res.ds_ppm:.6f} ppm | зона: {zone}"
        )
        self.btn_copy.setEnabled(True)

    def _on_copy(self) -> None:
        res: Helmert2DResult | None = getattr(self, "_last_result", None)
        if res is None:
            return
        text = res.to_profile_text()
        cb: QClipboard = QApplication.clipboard()
        cb.setText(text)
        QMessageBox.information(self, "Скопировано", "Параметры скопированы в буфер обмена")

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите CSV с точками",
            "",
            "CSV files (*.csv);;Все файлы (*.*)",
        )
        if not path:
            return
        try:
            rows: list[tuple[float, float, float, float]] = []
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    line = ln.strip()
                    if not line or line.startswith("#") or line.startswith("//"):
                        continue
                    # Нормализуем разделители: запятая/точка с запятой/таб/пробел
                    line = line.replace(";", " ").replace(",", " ")
                    parts = [p for p in line.split() if p]
                    if len(parts) < 4:
                        continue
                    try:
                        x_mil = float(parts[0])
                        y_mil = float(parts[1])
                        lat = float(parts[2])
                        lon = float(parts[3])
                    except ValueError:
                        continue
                    rows.append((x_mil, y_mil, lat, lon))
            if not rows:
                raise ValueError("В файле не найдено ни одной валидной строки из 4 чисел")
            self._set_table_rows(rows)
            QMessageBox.information(self, "Загрузка завершена", f"Загружено строк: {len(rows)}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки CSV", str(e))

    def _set_table_rows(self, rows: list[tuple[float, float, float, float]]) -> None:
        # Очистить таблицу
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                self.table.setItem(r, c, QTableWidgetItem(""))
        # Заполнить столбцы
        max_rows = min(len(rows), self.table.rowCount())
        for r in range(max_rows):
            vals = rows[r]
            for c in range(4):
                item = QTableWidgetItem(str(vals[c]))
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)


def main() -> None:
    app = QApplication()
    w = HelmertToolWindow()
    w.resize(800, 400)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
