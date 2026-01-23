from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QClipboard
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from constants import (
    COORD_DIMENSIONS,
    CSV_COLUMNS_REQUIRED,
    EPSG_SK42_GK_BASE,
    GK_FALSE_EASTING,
    GK_ZONE_CM_OFFSET_DEG,
    GK_ZONE_WIDTH_DEG,
    GK_ZONE_X_PREFIX_DIV,
    MAX_ROTATION_ARCSEC,
    MAX_SCALE_PPM,
    MAX_TRANSLATION_M,
    MAX_GK_ZONE,
    MIN_POINTS_FOR_HELMERT,
    MIN_POINT_PAIRS,
    SK42_CODE,
    WGS84_CODE,
)

logger = logging.getLogger(__name__)

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger.setLevel(logging.INFO)


@dataclass
class Helmert7Result:
    dx: float  # meters
    dy: float  # meters
    dz: float  # meters
    rx_as: float  # arcseconds (rotation about X)
    ry_as: float  # arcseconds (rotation about Y)
    rz_as: float  # arcseconds (rotation about Z)
    ds_ppm: float  # ppm (scale minus 1)

    def to_profile_text(self) -> str:
        return (
            '# СК-42 → WGS84 (Bursa–Wolf 7 параметров; единицы: м / угл. сек / ppm)\n'
            f'helmert_dx = {self.dx:.6f}\n'
            f'helmert_dy = {self.dy:.6f}\n'
            f'helmert_dz = {self.dz:.6f}\n'
            f'helmert_rx_as = {self.rx_as:.6f}\n'
            f'helmert_ry_as = {self.ry_as:.6f}\n'
            f'helmert_rz_as = {self.rz_as:.6f}\n'
            f'helmert_ds_ppm = {self.ds_ppm:.6f}\n'
        )


def estimate_helmert_7p(xs: np.ndarray, xw: np.ndarray) -> Helmert7Result:
    """
    Оценка 7 параметров Bursa–Wolf между двумя наборами геоцентрических координат.

    xs — (N,3) точки в СК-42 (Krassovsky) геоцентрических XYZ (метры)
    xw — (N,3) соответствующие точки в WGS84 геоцентрических XYZ (метры)
    Возвращает dx,dy,dz (м), rx,ry,rz (угл.сек), ds_ppm (ppm).
    """
    if xs.shape != xw.shape or xs.shape[1] != COORD_DIMENSIONS:
        msg = 'Формат точек должен быть (N,3), xs и xw одинаковых размеров'
        raise ValueError(msg)
    n = xs.shape[0]
    if n < MIN_POINTS_FOR_HELMERT:
        msg = 'Для оценки 7 параметров нужно не менее трёх точек'
        raise ValueError(msg)

    matrix_a = np.zeros((3 * n, 7), dtype=float)
    vector_l = np.zeros((3 * n, 1), dtype=float)
    for i in range(n):
        x_s, y_s, z_s = xs[i]
        x_w, y_w, z_w = xw[i]
        # Уравнения разностей (WGS - SK42)
        # Xw - Xs = dX + (-rz*Ys + ry*Zs) + m*Xs
        # Yw - Ys = dY + ( rz*Xs - rx*Zs) + m*Ys
        # Zw - Zs = dZ + (-ry*Xs + rx*Ys) + m*Zs
        r = 3 * i
        # Для X
        matrix_a[r, 0] = 1.0  # dX
        matrix_a[r, 1] = 0.0  # dY
        matrix_a[r, 2] = 0.0  # dZ
        matrix_a[r, 3] = 0.0  # rx
        matrix_a[r, 4] = z_s  # ry
        matrix_a[r, 5] = -y_s  # rz
        matrix_a[r, 6] = x_s  # m
        vector_l[r, 0] = x_w - x_s
        # Для Y
        matrix_a[r + 1, 0] = 0.0
        matrix_a[r + 1, 1] = 1.0
        matrix_a[r + 1, 2] = 0.0
        matrix_a[r + 1, 3] = -z_s  # rx
        matrix_a[r + 1, 4] = 0.0  # ry
        matrix_a[r + 1, 5] = x_s  # rz
        matrix_a[r + 1, 6] = y_s  # m
        vector_l[r + 1, 0] = y_w - y_s
        # Для Z
        matrix_a[r + 2, 0] = 0.0
        matrix_a[r + 2, 1] = 0.0
        matrix_a[r + 2, 2] = 1.0
        matrix_a[r + 2, 3] = y_s  # rx
        matrix_a[r + 2, 4] = -x_s  # ry
        matrix_a[r + 2, 5] = 0.0  # rz
        matrix_a[r + 2, 6] = z_s  # m
        vector_l[r + 2, 0] = z_w - z_s

    x_hat, *_ = np.linalg.lstsq(matrix_a, vector_l, rcond=None)
    dx = float(x_hat[0, 0])
    dy = float(x_hat[1, 0])
    dz = float(x_hat[2, 0])
    rx_rad = float(x_hat[3, 0])
    ry_rad = float(x_hat[4, 0])
    rz_rad = float(x_hat[5, 0])
    m = float(x_hat[6, 0])

    as_factor = 3600.0 * (180.0 / math.pi)
    rx_as = rx_rad * as_factor
    ry_as = ry_rad * as_factor
    rz_as = rz_rad * as_factor
    ds_ppm = m * 1e6

    return Helmert7Result(
        dx=dx, dy=dy, dz=dz, rx_as=rx_as, ry_as=ry_as, rz_as=rz_as, ds_ppm=ds_ppm
    )


def _determine_zone_from_x(eastings: np.ndarray) -> int:
    """Определить 6° зону Гаусса–Крюгера по восточным X (в метрах)."""
    if eastings.size == 0:
        msg = 'Нет значений X для определения зоны'
        raise ValueError(msg)
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
        f'+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 '
        f'+x_0={GK_FALSE_EASTING} +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs'
    )
    return CRS.from_proj4(proj4)


class HelmertToolWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            'Helmert 7-параметрический (СК-42 ↔ WGS84) — оценка по контрольным точкам'
        )
        self._init_ui()

    def _init_ui(self) -> None:
        cw = QWidget(self)
        self.setCentralWidget(cw)
        v = QVBoxLayout(cw)

        hint = QLabel(
            'Введите минимум 3 соответствующие точки: слева — СК-42 Гаусса–Крюгера (военный порядок: X — северинг, Y — восточинг, м), справа — WGS84 (широта, долгота).\n'
            'Утилита восстановит географические координаты СК‑42, пересчитает пары в геоцентрические XYZ и оценит 7 параметров Bursa–Wolf (dx, dy, dz; rx, ry, rz; ds).'
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.table = QTableWidget(12, 4, self)
        self.table.setHorizontalHeaderLabels(
            [
                'СК-42 X (военный, северинг м)',
                'СК-42 Y (военный, восточинг м)',
                'WGS84 lat (°)',
                'WGS84 lon (°)',
            ]
        )
        v.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton('Загрузить CSV', self)
        self.btn_calc = QPushButton('Рассчитать', self)
        self.btn_copy = QPushButton('Копировать в профиль', self)
        self.btn_copy.setEnabled(False)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_calc)
        btn_row.addWidget(self.btn_copy)
        v.addLayout(btn_row)

        self.lbl_result = QLabel('Результат: —')
        v.addWidget(self.lbl_result)
        # Диагностика: остатки по точкам и RMS
        self.lbl_diag = QLabel('')
        self.lbl_diag.setWordWrap(True)
        v.addWidget(self.lbl_diag)

        self.btn_load.clicked.connect(self._on_load_csv)
        self.btn_calc.clicked.connect(self._on_calc)
        self.btn_copy.clicked.connect(self._on_copy)

        # Menu: clear, quit
        act_clear = QAction('Очистить', self)
        act_clear.triggered.connect(self._clear)
        self.menuBar().addAction(act_clear)
        act_load = QAction('Загрузить CSV', self)
        act_load.triggered.connect(self._on_load_csv)
        self.menuBar().addAction(act_load)
        act_quit = QAction('Выход', self)
        act_quit.triggered.connect(self.close)
        self.menuBar().addAction(act_quit)

    def _clear(self) -> None:
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                self.table.setItem(r, c, QTableWidgetItem(''))
        self.lbl_result.setText('Результат: —')
        self.lbl_diag.setText('')
        self.btn_copy.setEnabled(False)

    def _read_points(self) -> tuple[np.ndarray, np.ndarray]:
        src: list[tuple[float, float]] = []
        dst: list[tuple[float, float]] = []
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
        if len(src) < MIN_POINT_PAIRS:
            msg = 'Введите минимум две корректные пары точек'
            raise ValueError(msg)
        return np.array(src, dtype=float), np.array(dst, dtype=float)

    def _text(self, r: int, c: int) -> str:
        item = self.table.item(r, c)
        return item.text().strip() if item is not None else ''

    def _on_calc(self) -> None:
        try:
            # Считываем: слева СК-42 (военный порядок: X=northing, Y=easting), справа WGS84 (lat, lon)
            p_src_m_mil, p_dst_deg_latlon = self._read_points()
            # Приводим СК-42 к геодезическому порядку (easting, northing)
            p_src_en_base = np.column_stack([p_src_m_mil[:, 1], p_src_m_mil[:, 0]])
            # Определяем зону по восточным X (т.е. по Y военном)
            zone_from_x = _determine_zone_from_x(p_src_en_base[:, 0])
            # Определяем зону по долготам WGS84 (медиана)
            try:
                p_dst_deg_latlon[:, 0]
                lons_all = p_dst_deg_latlon[:, 1]
                median_lon = float(np.median(lons_all))
                lon360 = (median_lon % 360.0 + 360.0) % 360.0
                zone_by_lon = int(lon360 // 6.0) + 1
            except Exception:
                zone_by_lon = None
            zone = zone_by_lon if (zone_by_lon is not None) else zone_from_x
            if zone_by_lon is not None and zone_by_lon != zone_from_x:
                logger.warning(
                    'Зоны расходятся: по X=%s, по долготе=%s. Выбрана по WGS84=%s',
                    zone_from_x,
                    zone_by_lon,
                    zone,
                )

            logger.info('Начало оценки 7-параметров: зона=%s', zone)

            # CRS и трансформеры (для выбранной зоны)
            crs_wgs84_geog = CRS.from_epsg(WGS84_CODE)
            crs_sk42_geog = CRS.from_epsg(SK42_CODE)
            crs_wgs84_gk = _build_wgs84_gk_crs(zone)
            crs_sk42_gk = _build_sk42_gk_crs(zone)
            t_wgs_to_wgs_gk = Transformer.from_crs(
                crs_wgs84_geog, crs_wgs84_gk, always_xy=True
            )
            t_sk42_from_gk = Transformer.from_crs(
                crs_sk42_gk, crs_sk42_geog, always_xy=True
            )
            # Геоцентрические
            crs_sk42_geocent = CRS.from_proj4(
                '+proj=geocent +a=6378245.0 +rf=298.3 +units=m +no_defs'
            )
            crs_wgs84_geocent = CRS.from_proj4(
                '+proj=geocent +datum=WGS84 +units=m +no_defs'
            )
            t_sk42_geog_to_xyz = Transformer.from_crs(
                crs_sk42_geog, crs_sk42_geocent, always_xy=True
            )
            t_wgs_geog_to_xyz = Transformer.from_crs(
                crs_wgs84_geog, crs_wgs84_geocent, always_xy=True
            )
            t_wgs_xyz_to_geog = Transformer.from_crs(
                crs_wgs84_geocent, crs_wgs84_geog, always_xy=True
            )

            lats = p_dst_deg_latlon[:, 0]
            lons = p_dst_deg_latlon[:, 1]

            def _solve_variant(
                use_prefix: bool,
            ) -> dict[str, float | Helmert7Result | np.ndarray | bool]:
                p_src_en = p_src_en_base.copy()
                if use_prefix:
                    p_src_en[:, 0] = p_src_en[:, 0] - zone * GK_ZONE_X_PREFIX_DIV
                # TM (для диагностики 2D)
                x_w_tm, y_w_tm = t_wgs_to_wgs_gk.transform(lons, lats)
                # СК-42 ГК -> СК-42 географические
                lng_sk42, lat_sk42 = t_sk42_from_gk.transform(
                    p_src_en[:, 0], p_src_en[:, 1]
                )
                # Географические -> геоцентрические (h=0)
                zeros = np.zeros_like(lng_sk42)
                x_s, y_s, z_s = t_sk42_geog_to_xyz.transform(lng_sk42, lat_sk42, zeros)
                zeros_w = np.zeros_like(lons)
                x_w, y_w, z_w = t_wgs_geog_to_xyz.transform(lons, lats, zeros_w)
                xs = np.column_stack([x_s, y_s, z_s])
                xw = np.column_stack([x_w, y_w, z_w])
                res_local = estimate_helmert_7p(xs, xw)
                # Применим параметры
                mloc = res_local.ds_ppm / 1e6
                rxloc = math.radians(res_local.rx_as / 3600.0)
                ryloc = math.radians(res_local.ry_as / 3600.0)
                rzloc = math.radians(res_local.rz_as / 3600.0)
                x_p = x_s + res_local.dx + (-rzloc * y_s + ryloc * z_s) + mloc * x_s
                y_p = y_s + res_local.dy + (rzloc * x_s - rxloc * z_s) + mloc * y_s
                z_p = z_s + res_local.dz + (-ryloc * x_s + rxloc * y_s) + mloc * z_s
                lon_pred, lat_pred, _hp = t_wgs_xyz_to_geog.transform(x_p, y_p, z_p)
                x_p_tm, y_p_tm = t_wgs_to_wgs_gk.transform(lon_pred, lat_pred)
                rx2d = x_w_tm - x_p_tm
                ry2d = y_w_tm - y_p_tm
                r2 = rx2d**2 + ry2d**2
                rms_x_loc = float(np.sqrt(np.mean(rx2d**2)))
                rms_y_loc = float(np.sqrt(np.mean(ry2d**2)))
                rms_2d_loc = float(np.sqrt(np.mean(r2)))
                r3x = x_w - x_p
                r3y = y_w - y_p
                r3z = z_w - z_p
                rms_3d_loc = float(np.sqrt(np.mean(r3x**2 + r3y**2 + r3z**2)))
                return {
                    'res': res_local,
                    'rms_2d': rms_2d_loc,
                    'rms_x': rms_x_loc,
                    'rms_y': rms_y_loc,
                    'rms_3d': rms_3d_loc,
                    'rx2d': rx2d,
                    'ry2d': ry2d,
                    'r2': r2,
                    'used_prefix': use_prefix,
                }

            out_with = _solve_variant(True)
            out_without = _solve_variant(False)
            rms_2d_with = out_with['rms_2d']
            rms_2d_without = out_without['rms_2d']
            assert isinstance(rms_2d_with, float)
            assert isinstance(rms_2d_without, float)
            chosen = out_with if rms_2d_with <= rms_2d_without else out_without
            res = chosen['res']
            assert isinstance(res, Helmert7Result)
            rms_2d = chosen['rms_2d']
            assert isinstance(rms_2d, float)
            rms_x = chosen['rms_x']
            assert isinstance(rms_x, float)
            rms_y = chosen['rms_y']
            assert isinstance(rms_y, float)
            rms_3d = chosen['rms_3d']
            assert isinstance(rms_3d, float)
            rx2d = chosen['rx2d']
            assert isinstance(rx2d, np.ndarray)
            ry2d = chosen['ry2d']
            assert isinstance(ry2d, np.ndarray)
            r2 = chosen['r2']
            assert isinstance(r2, np.ndarray)
            used_prefix = chosen['used_prefix']
            assert isinstance(used_prefix, bool)
            logger.info(
                'Гипотезы: с префиксом RMS2D=%.3f, без префикса RMS2D=%.3f. Выбрана: %s',
                out_with['rms_2d'],
                out_without['rms_2d'],
                'с префиксом' if used_prefix else 'без префикса',
            )
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', str(e))
            return

        self._last_result = res
        self._last_zone = zone

        # Предупреждения и диагностика
        try:
            median_lon = float(np.median(lons))
            lon360 = (median_lon % 360.0 + 360.0) % 360.0
            zone_by_lon = int(lon360 // 6.0) + 1
        except Exception:
            zone_by_lon = None
        zones_in_src = {
            int(float(val) // GK_ZONE_X_PREFIX_DIV) for val in p_src_m_mil[:, 1]
        }
        warnings: list[str] = []
        if zone_by_lon is not None and zone_by_lon != zone:
            warnings.append(
                f'Предупреждение: зона по долготам WGS84 = {zone_by_lon}, по СК-42 = {zone}. Проверьте порядок X/Y (военный) и соответствие зоны.'
            )
        if len(zones_in_src) > 1:
            warnings.append(
                f'Предупреждение: обнаружены разные префиксы зон в исходных СК-42 easting (Y военный): {sorted(zones_in_src)}. Нельзя смешивать точки из разных зон.'
            )
        # Выбранная гипотеза по миллионному префиксу
        warnings.append(
            f'Выбран вариант нормализации восточинга: {"с миллионным префиксом (вычтен zone*1e6)" if used_prefix else "без миллионного префикса"}.'
        )

        # Санити‑чек реалистичности параметров
        insane = (
            abs(res.dx) > MAX_TRANSLATION_M
            or abs(res.dy) > MAX_TRANSLATION_M
            or abs(res.dz) > MAX_TRANSLATION_M
            or abs(res.rx_as) > MAX_ROTATION_ARCSEC
            or abs(res.ry_as) > MAX_ROTATION_ARCSEC
            or abs(res.rz_as) > MAX_ROTATION_ARCSEC
            or abs(res.ds_ppm) > MAX_SCALE_PPM
        )
        if insane:
            warnings.append(
                'Оценённые параметры выглядят НЕРЕАЛИСТИЧНО. Проверьте исходные данные: зона ГК, порядок столбцов X/Y и наличие миллионного префикса восточинга.'
            )
            logger.warning(
                'Нереалистичные параметры: dx=%.3f dy=%.3f dz=%.3f rx=%.3f" ry=%.3f" rz=%.3f" ds=%.3fppm',
                res.dx,
                res.dy,
                res.dz,
                res.rx_as,
                res.ry_as,
                res.rz_as,
                res.ds_ppm,
            )

        # Диагностика
        lines: list[str] = []
        lines.append(
            f'RMS (TM WGS84): X = {rms_x:.3f} м; Y = {rms_y:.3f} м; 2D = {rms_2d:.3f} м'
        )
        lines.append(f'RMS (геоцентрические XYZ): {rms_3d:.3f} м')
        for i in range(rx2d.size):
            lines.append(
                f'Точка {i + 1}: rX = {rx2d[i]:.3f} м; rY = {ry2d[i]:.3f} м; |r| = {math.sqrt(r2[i]):.3f} м'
            )
        if warnings:
            lines.append('\n' + ' \n'.join(warnings))
        diag_text = '\n'.join(lines)
        self.lbl_diag.setText(diag_text)
        self.lbl_result.setText(
            'Результат: '
            f'dx={res.dx:.6f} м, dy={res.dy:.6f} м, dz={res.dz:.6f} м; '
            f"rx={res.rx_as:.6f}'', ry={res.ry_as:.6f}'', rz={res.rz_as:.6f}''; "
            f'ds={res.ds_ppm:.6f} ppm | зона: {zone}'
        )
        self.btn_copy.setEnabled(not insane)
        if insane:
            QMessageBox.warning(
                self,
                'Нереалистичные параметры',
                'Параметры выглядят нереалистично. Копирование отключено. Проверьте данные и попробуйте другую выборку точек.',
            )
        logger.info(
            'Оценка завершена: RMS2D=%.3f м, RMS3D=%.3f м; параметры: dx=%.3f dy=%.3f dz=%.3f rx=%.3f" ry=%.3f" rz=%.3f" ds=%.3fppm',
            rms_2d,
            rms_3d,
            res.dx,
            res.dy,
            res.dz,
            res.rx_as,
            res.ry_as,
            res.rz_as,
            res.ds_ppm,
        )

    def _on_copy(self) -> None:
        res: Helmert7Result | None = getattr(self, '_last_result', None)
        if res is None:
            return
        text = res.to_profile_text()
        cb: QClipboard = QApplication.clipboard()
        cb.setText(text)
        QMessageBox.information(
            self, 'Скопировано', 'Параметры скопированы в буфер обмена'
        )

    def _on_load_csv(self) -> None:
        def _raise_no_valid_rows() -> None:
            msg = 'В файле не найдено ни одной валидной строки из 4 чисел'
            raise ValueError(msg)

        path, _ = QFileDialog.getOpenFileName(
            self,
            'Выберите CSV с точками',
            '',
            'CSV files (*.csv);;Все файлы (*.*)',
        )
        if not path:
            return
        try:
            rows: list[tuple[float, float, float, float]] = []
            with Path(path).open(encoding='utf-8') as f:
                for ln in f:
                    line = ln.strip()
                    if not line or line.startswith(('#', '//')):
                        continue
                    # Нормализуем разделители: запятая/точка с запятой/таб/пробел
                    line = line.replace(';', ' ').replace(',', ' ')
                    parts = [p for p in line.split() if p]
                    if len(parts) < CSV_COLUMNS_REQUIRED:
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
                _raise_no_valid_rows()
            self._set_table_rows(rows)
            QMessageBox.information(
                self, 'Загрузка завершена', f'Загружено строк: {len(rows)}'
            )
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка загрузки CSV', str(e))

    def _set_table_rows(self, rows: list[tuple[float, float, float, float]]) -> None:
        # Очистить таблицу
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                self.table.setItem(r, c, QTableWidgetItem(''))
        # Заполнить столбцы
        max_rows = min(len(rows), self.table.rowCount())
        for r in range(max_rows):
            vals = rows[r]
            for c in range(4):
                item = QTableWidgetItem(str(vals[c]))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(r, c, item)


def main() -> None:
    app = QApplication()
    w = HelmertToolWindow()
    w.resize(800, 400)
    w.show()
    app.exec()


def _build_sk42_gk_crs(zone: int) -> CRS:
    """Построить CRS СК-42 / Гаусса–Крюгера для заданной зоны (EPSG:284xx)."""
    try:
        return CRS.from_epsg(EPSG_SK42_GK_BASE + zone)
    except Exception:
        lon0 = zone * GK_ZONE_WIDTH_DEG - GK_ZONE_CM_OFFSET_DEG
        proj4 = (
            f'+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 '
            f'+x_0={GK_FALSE_EASTING} +y_0=0 +ellps=krass +units=m +no_defs +type=crs'
        )
        return CRS.from_proj4(proj4)


if __name__ == '__main__':
    main()
