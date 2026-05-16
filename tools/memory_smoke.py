"""
Memory smoke test for the GUI.

Запускает MainWindow в offscreen-режиме, прогоняет реалистичный
workflow (builds + interactive recompute + map-type cycling + alpha
slider) с замером RSS на каждой контрольной точке. Печатает таблицу
дельт и сохраняет CSV.

Использование:
    .venv/bin/python tools/memory_smoke.py [--iterations N] [--no-warmup]

Exit code:
    0 — все пост-warmup дельты в пределах MAX_RSS_DELTA_MB.
    1 — обнаружен подозрительный рост RSS, см. таблицу.
    2 — конфигурационная ошибка (нет API_KEY, не удалось создать окно).

Это НЕ pytest-test: standalone diagnostic. Если стабилизируется и
оказывается полезным, переносим в tests/integration/ под pytest-qt.
"""

from __future__ import annotations

import os

# RLIMIT_AS и offscreen Qt платформу ставим ДО любых импортов Qt и
# numpy/cv2, иначе limit не подхватится наследованием на child threads.
os.environ.setdefault('MALLOC_ARENA_MAX', '2')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# Worker subprocess использует mp.spawn (см. main.py); тут устанавливаем
# то же ПЕРЕД импортами Qt/numpy/cv2, иначе fork после OpenMP-init →
# «Terminating: fork() called from a process already using GNU OpenMP».
import multiprocessing as _mp  # noqa: E402

_mp.set_start_method('spawn', force=True)

import argparse  # noqa: E402
import csv  # noqa: E402
import gc  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from contextlib import suppress  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dotenv import load_dotenv  # noqa: E402

for _env_name in ('.secrets.env', '.env'):
    _env = REPO_ROOT / _env_name
    if _env.exists():
        load_dotenv(_env, override=False)
        break

from shared.constants import MEMORY_RLIMIT_RATIO  # noqa: E402
from shared.memory_limit import apply_rlimit_as, compact_c_heap  # noqa: E402

# RLIMIT_AS как в production GUI process.
apply_rlimit_as(MEMORY_RLIMIT_RATIO, component='memory_smoke')

import psutil  # noqa: E402
from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

# Monkey-patch модальные диалоги — в offscreen Qt без пользователя
# они блокируют main thread навсегда (нечем нажать OK). Локализовано
# через py-spy dump: после build #N в scenario D _on_download
# _finished встал на QMessageBox.warning(self._download_warnings) →
# 5 минут до timeout. У реального пользователя диалоги работают,
# это чисто тестовое окружение.
_dialog_calls: list[tuple[str, str]] = []


def _silent_dialog(_parent, title: str = '', text: str = '', *_args, **_kwargs) -> int:
    _dialog_calls.append((str(title), str(text)[:200]))
    return QMessageBox.StandardButton.Ok


for _dlg_method in ('warning', 'critical', 'information', 'question', 'about'):
    setattr(QMessageBox, _dlg_method, staticmethod(_silent_dialog))

from gui.controller import MilMapperController  # noqa: E402
from gui.model import MilMapperModel  # noqa: E402
from gui.view import MainWindow  # noqa: E402
from shared.constants import MapType  # noqa: E402

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# После warmup-фазы (первые N итераций для cold JIT + LRU fill) каждая
# следующая итерация не должна добавлять больше этого к RSS.
MAX_RSS_DELTA_MB = 150.0
# После полного цикла одного сценария RSS не должен превысить baseline + этого.
MAX_SCENARIO_GROWTH_MB = 300.0
# Сколько builds считать warmup'ом (JIT, contour disk cache, etc).
WARMUP_BUILDS = 2
# Таймауты ожидания фоновых workers (мс).
DOWNLOAD_TIMEOUT_MS = 90_000
RECOMPUTE_TIMEOUT_MS = 30_000

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _measure_after_settle() -> float:
    """Force GC + malloc_trim, потом замерить RSS. Применять между шагами."""
    gc.collect()
    compact_c_heap()
    # Дать Qt event loop осесть (закрытые pixmap'ы освобождаются deferred).
    QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    gc.collect()
    return _rss_mb()


def _wait_until(predicate, timeout_ms: int, *, poll_ms: int = 100) -> bool:
    """
    Крутить Qt event loop пока predicate True или истекли timeout_ms.
    Используется вместо pytest-qt's qtbot.waitUntil — мы standalone.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        QApplication.processEvents(
            QEventLoop.ProcessEventsFlag.AllEvents, poll_ms,
        )
        if predicate():
            return True
        time.sleep(poll_ms / 1000.0)
    return False


def _wait_download_done(window: MainWindow) -> bool:
    """Build = когда download_worker None ИЛИ не isRunning."""
    return _wait_until(
        lambda: (
            window._download_worker is None  # noqa: SLF001
            or not window._download_worker.isRunning()  # noqa: SLF001
        ),
        DOWNLOAD_TIMEOUT_MS,
    )


def _wait_recompute_done(window: MainWindow) -> bool:
    """Recompute = когда _rh_worker None или not running."""
    return _wait_until(
        lambda: (
            window._rh_worker is None  # noqa: SLF001
            or not window._rh_worker.isRunning()  # noqa: SLF001
        ),
        RECOMPUTE_TIMEOUT_MS,
    )


def _wait_alpha_apply_done(window: MainWindow) -> bool:
    return _wait_until(
        lambda: (
            window._alpha_apply_worker is None  # noqa: SLF001
            or not window._alpha_apply_worker.isRunning()  # noqa: SLF001
        ),
        RECOMPUTE_TIMEOUT_MS,
    )


def _wait_orphans_drained(window: MainWindow, timeout_ms: int = 10_000) -> bool:
    """orphaned workers закрываются естественно — дать им время."""
    return _wait_until(
        lambda: all(
            not w.isRunning() for w in window._orphaned_workers  # noqa: SLF001
        ),
        timeout_ms,
    )

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _select_map_type(window: MainWindow, mt: MapType) -> None:
    """Set map_type in combo box → triggers _on_map_type_changed."""
    combo = window.map_type_combo
    for idx in range(combo.count()):
        if combo.itemData(idx) == mt.value:
            combo.setCurrentIndex(idx)
            return
    raise RuntimeError(f'MapType {mt} not in combo box')


def _trigger_build(window: MainWindow) -> bool:
    """
    Programmatic click of «Скачать»; True если build реально прошёл.

    Ждём НЕ download_worker.isRunning() (он завершается раньше, чем
    preview_ready signal доедет до _show_preview), а смены
    last_map_metadata — её обновляет _show_preview уже в самом конце
    pipeline.
    """
    meta_before = window._model.state.last_map_metadata  # noqa: SLF001
    window._start_download()  # noqa: SLF001
    return _wait_until(
        lambda: (
            window._model.state.last_map_metadata is not meta_before  # noqa: SLF001
        ),
        DOWNLOAD_TIMEOUT_MS,
    )


def _drag_cp(window: MainWindow, dem_row: int, dem_col: int) -> bool:
    """
    Имитация drag CP: invoke _recompute_coverage_at_click напрямую с
    dem_row/dem_col. low_res_factor=2 → draft → refine через timer.
    """
    # Передаём (px=0, py=0) и dem_row/dem_col — _recompute_coverage_at
    # _click примет их прямо без обратной конверсии.
    window._recompute_coverage_at_click(  # noqa: SLF001
        0.0, 0.0, dem_row=dem_row, dem_col=dem_col, low_res_factor=2,
    )
    if not _wait_recompute_done(window):
        return False
    # Ждём refine через _coverage_refine_timer (700ms) — он pendable.
    # Если успел запуститься, ждём его завершения.
    _wait_until(
        lambda: (
            not window._coverage_refine_timer.isActive()  # noqa: SLF001
            and (
                window._rh_worker is None  # noqa: SLF001
                or not window._rh_worker.isRunning()  # noqa: SLF001
            )
        ),
        2000,
    )
    return _wait_recompute_done(window)


def _change_alpha(window: MainWindow, value_pct: int) -> bool:
    """Set alpha slider value + trigger release."""
    if not hasattr(window, 'radio_horizon_alpha_slider'):
        return True
    window.radio_horizon_alpha_slider.setValue(value_pct)
    # Manually flag + release (без physical mouse).
    window._alpha_needs_recompute = True  # noqa: SLF001
    window._on_alpha_slider_released()  # noqa: SLF001
    return _wait_alpha_apply_done(window)


# ---------------------------------------------------------------------------
# Smoke runner
# ---------------------------------------------------------------------------


def _run_scenario_A_repeated_builds(
    window: MainWindow, record, failures: list[str], n: int,
) -> None:
    """N RH builds same area → RSS должен plateau после WARMUP_BUILDS."""
    print('\n=== A. Repeated RH builds ===')
    _select_map_type(window, MapType.RADIO_HORIZON)
    QApplication.processEvents()
    for i in range(n):
        if not _trigger_build(window):
            failures.append(f'A: build #{i+1} timeout / no metadata change')
            break
        _wait_orphans_drained(window)
        record('A.repeat_RH', f'build #{i+1}')


def _run_scenario_B_drag_cp(window: MainWindow, record, failures: list[str]) -> None:
    """10 drag CP recomputes (low-res preview + refine)."""
    print('\n=== B. Drag CP recompute ===')
    if window._rh_cache.get('dem') is None:  # noqa: SLF001
        print('  (skipped: no RH dem in cache; run scenario A first)')
        return
    dem = window._rh_cache['dem']  # noqa: SLF001
    dh, dw = dem.shape
    targets = [
        (dh // 4, dw // 4), (dh // 4, 3 * dw // 4),
        (3 * dh // 4, dw // 4), (3 * dh // 4, 3 * dw // 4),
        (dh // 2, dw // 2),
    ]
    for i, (r, c) in enumerate(targets * 2):  # 10 drags
        if not _drag_cp(window, r, c):
            failures.append(f'B: drag #{i+1} timeout')
            break
        _wait_orphans_drained(window)
        record('B.drag_CP', f'drag #{i+1}')


def _run_scenario_C_alpha(window: MainWindow, record, failures: list[str]) -> None:
    """Alpha-slider sweep."""
    print('\n=== C. Alpha slider sweep ===')
    for pct in [10, 30, 50, 70, 90]:
        if not _change_alpha(window, pct):
            failures.append(f'C: alpha {pct}% timeout')
            break
        record('C.alpha', f'alpha {pct}%')


def _run_scenario_D_cycle(window: MainWindow, record, failures: list[str]) -> None:
    """Map type cycle — переключение освобождает прошлые caches."""
    print('\n=== D. Map type cycle ===')
    cycle = [
        MapType.HYBRID, MapType.ELEVATION_COLOR,
        MapType.RADIO_HORIZON, MapType.RADAR_COVERAGE, MapType.HYBRID,
    ]
    for i, mt in enumerate(cycle):
        _select_map_type(window, mt)
        QApplication.processEvents()
        if not _trigger_build(window):
            failures.append(f'D: build {mt.value} timeout')
            break
        _wait_orphans_drained(window)
        record('D.cycle', f'{i+1} {mt.value}')


def run_smoke(args: argparse.Namespace) -> int:
    api_key = os.getenv('API_KEY')
    if not api_key:
        print('ERROR: API_KEY not in env (.secrets.env / .env required).')
        return 2

    app = QApplication.instance() or QApplication(sys.argv)
    _ = app  # avoid linter
    model = MilMapperModel()
    # Controller сам грузит API_KEY из env / portable api_key.txt в
    # __init__. Просто проверяем что он подхватился.
    controller = MilMapperController(model)
    if not controller.get_api_key():
        print('ERROR: controller failed to load API key (env loaded? .env present?)')
        return 2
    window = MainWindow(model, controller)

    # Не показываем окно (offscreen), но Qt инициализирует scene.
    QApplication.processEvents()

    if not args.no_warmup:
        print('Numba warmup...', flush=True)
        try:
            from services.radio_horizon import warmup_numba_kernels
            t0 = time.monotonic()
            warmup_numba_kernels()
            print(f'  numba ready in {(time.monotonic() - t0)*1000:.0f} ms')
        except Exception as exc:  # noqa: BLE001
            print(f'  numba warmup skipped: {exc}')

    samples: list[dict] = []
    failures: list[str] = []

    def _record(scenario: str, step: str) -> float:
        rss = _measure_after_settle()
        prev = samples[-1]['rss_mb'] if samples else rss
        delta = rss - prev
        samples.append({
            'scenario': scenario,
            'step': step,
            'rss_mb': rss,
            'delta_mb': delta,
        })
        print(f'  {scenario:<24} {step:<28} RSS={rss:>7.1f} MB  Δ={delta:+7.1f}')
        return rss

    # ─── Baseline ───────────────────────────────────────────────────────
    print('\n=== BASELINE ===')
    baseline_rss = _record('baseline', 'startup')

    # Run selected scenarios. Между сценариями обязательный
    # drain orphans + double-gc — без этого хвосты от предыдущего
    # (orphaned workers, недоосвобождённые Qt объекты) могут
    # портить замеры.
    run_all = args.scenarios == 'all'
    scenarios = [
        ('A', _run_scenario_A_repeated_builds, (args.iterations,)),
        ('B', _run_scenario_B_drag_cp, ()),
        ('C', _run_scenario_C_alpha, ()),
        ('D', _run_scenario_D_cycle, ()),
    ]
    for letter, fn, extra_args in scenarios:
        if not run_all and letter not in args.scenarios:
            continue
        fn(window, _record, failures, *extra_args)
        # Inter-scenario settle: orphans drain + memory compact +
        # Qt event loop pump чтобы deleteLater'ы успели сработать.
        if not _wait_orphans_drained(window, timeout_ms=20_000):
            print(f'  ! scenario {letter}: orphan workers still alive after 20s')
        for _ in range(3):
            QApplication.processEvents(
                QEventLoop.ProcessEventsFlag.AllEvents, 100,
            )
        gc.collect()
        compact_c_heap()

    # ─── Cleanup ────────────────────────────────────────────────────────
    print('\n=== TEARDOWN ===')
    with suppress(Exception):
        window.close()
    QApplication.processEvents()
    _wait_orphans_drained(window, timeout_ms=15_000)
    final_rss = _record('teardown', 'after close')

    # ─── Analysis ───────────────────────────────────────────────────────
    print('\n=== ANALYSIS ===')
    print(f'Baseline RSS: {baseline_rss:.1f} MB')
    print(f'Final RSS:    {final_rss:.1f} MB')
    print(f'Net growth:   {final_rss - baseline_rss:+.1f} MB')

    # Per-scenario оценка роста.
    by_scenario: dict[str, list[float]] = {}
    for s in samples:
        by_scenario.setdefault(s['scenario'], []).append(s['rss_mb'])

    suspect = []
    for scenario, rss_list in by_scenario.items():
        if scenario in ('baseline', 'teardown') or len(rss_list) < 2:
            continue
        post_warmup = rss_list[WARMUP_BUILDS:] if len(rss_list) > WARMUP_BUILDS else rss_list
        if not post_warmup:
            continue
        scenario_growth = post_warmup[-1] - post_warmup[0]
        print(
            f'  {scenario:<24} after warmup: '
            f'{post_warmup[0]:>7.1f} → {post_warmup[-1]:>7.1f} MB '
            f'(Δ={scenario_growth:+.1f})'
        )
        if scenario_growth > MAX_SCENARIO_GROWTH_MB:
            suspect.append(
                f'{scenario}: grew {scenario_growth:+.0f} MB '
                f'> threshold {MAX_SCENARIO_GROWTH_MB:.0f}'
            )
        # Per-step дельта (после warmup).
        max_step_delta = 0.0
        for i in range(WARMUP_BUILDS + 1, len(rss_list)):
            step_delta = rss_list[i] - rss_list[i - 1]
            max_step_delta = max(max_step_delta, step_delta)
        if max_step_delta > MAX_RSS_DELTA_MB:
            suspect.append(
                f'{scenario}: max per-step delta {max_step_delta:+.0f} MB '
                f'> threshold {MAX_RSS_DELTA_MB:.0f}'
            )

    # ─── CSV ────────────────────────────────────────────────────────────
    out_dir = REPO_ROOT / '.memory_smoke'
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f'samples_{int(time.time())}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['scenario', 'step', 'rss_mb', 'delta_mb'])
        writer.writeheader()
        writer.writerows(samples)
    print(f'\nCSV saved: {csv_path}')

    if _dialog_calls:
        print(f'\n=== Intercepted {len(_dialog_calls)} modal dialog(s) ===')
        for title, text in _dialog_calls:
            print(f'  [{title}] {text}')
    if failures:
        print('\n!!! FAILURES:')
        for f in failures:
            print(f'  {f}')
        return 1
    if suspect:
        print('\n!!! SUSPECT GROWTH:')
        for s in suspect:
            print(f'  {s}')
        return 1
    print('\nOK — memory growth within thresholds.')
    return 0


def _setup_logging(log_file: Path, *, quiet: bool) -> None:
    handlers: list[logging.Handler] = []
    if not quiet:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.WARNING)
        handlers.append(sh)
    fh = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    handlers.append(fh)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for h in handlers:
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = handlers


def main() -> int:
    parser = argparse.ArgumentParser(description='Memory smoke test (offscreen GUI)')
    parser.add_argument(
        '--iterations', type=int, default=3,
        help='Number of repeated RH builds in scenario A (default: 3)',
    )
    parser.add_argument(
        '--scenarios', default='A',
        help='Which scenarios to run: subset of "ABCD" or "all" '
             '(default: A — quick smoke ~90s; all = ABCD, ~9 минут).',
    )
    parser.add_argument(
        '--no-warmup', action='store_true',
        help='Skip Numba JIT warmup (faster start, but first build slower)',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress non-WARNING stdout (still saved to log file)',
    )
    parser.add_argument(
        '--log-file', default=str(REPO_ROOT / '.memory_smoke' / 'smoke.log'),
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(exist_ok=True)
    _setup_logging(log_path, quiet=args.quiet)

    try:
        return run_smoke(args)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
