"""
Performance / leak-detection loop for the map build pipeline.

Runs MapDownloadService.download() directly (no GUI, no worker subprocess)
N times per map type, records timings + RSS per build, prints a summary
with leak detection.

Usage:
  .venv/bin/python tools/perf_loop.py --type ELEVATION_COLOR --runs 3
  .venv/bin/python tools/perf_loop.py --type all --runs 5
  .venv/bin/python tools/perf_loop.py --type all --runs 3 --quiet

Map types: HYBRID, SATELLITE, STREETS, OUTDOORS, ELEVATION_COLOR,
ELEVATION_CONTOURS, ELEVATION_HILLSHADE, RADIO_HORIZON, RADAR_COVERAGE,
LINK_PROFILE, NSU_OPTIMIZER, or `all` for the standard suite below.

Output: stdout + a summary table. Output JPEGs are written to
.perf_out/<type>_<run_idx>.jpg unless --no-save is passed.
"""

from __future__ import annotations

import os

# Mirror main.py: cap glibc per-thread arenas before any C extension loads.
# Without this, baseline VMS after a single SATELLITE retina build sits at
# ~6 GB and the 2nd run trips RLIMIT_AS.
os.environ.setdefault('MALLOC_ARENA_MAX', '2')

import argparse  # noqa: E402
import asyncio  # noqa: E402
import gc  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import mean, stdev  # noqa: E402

import psutil  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

# Load API_KEY from .secrets.env / .env. Done BEFORE importing app modules
# so that anything reading env at import time picks it up.
from dotenv import load_dotenv  # noqa: E402

for _env_name in ('.secrets.env', '.env'):
    _env = REPO_ROOT / _env_name
    if _env.exists():
        load_dotenv(_env, override=False)
        break

from domain.profiles import load_profile  # noqa: E402
from services.map_download_service import MapDownloadService  # noqa: E402
from shared.constants import MEMORY_RLIMIT_RATIO, MapType  # noqa: E402
from shared.memory_limit import apply_rlimit_as  # noqa: E402

# Apply RLIMIT_AS to the perf_loop process itself. Without it, choose_safe
# _zoom может пустить z=18 (peak ~7.5GB на 6.6km link_profile) → linux
# OOM-killer прибивает процесс (exit 144) на 13GB-машине. С RLIMIT_AS
# процесс получает MemoryError изнутри pipeline, тестирующий fallback-путь
# вместо системного OOM.
apply_rlimit_as(MEMORY_RLIMIT_RATIO, component='perf_loop')


# Standard test suite — covers every processor branch except the
# placeholders. ELEVATION_HILLSHADE is known to have a visual issue
# (see CLAUDE.md) but still exercises the pipeline.
STANDARD_SUITE: tuple[MapType, ...] = (
    MapType.HYBRID,
    MapType.ELEVATION_COLOR,
    MapType.ELEVATION_CONTOURS,
    MapType.ELEVATION_HILLSHADE,
    MapType.RADIO_HORIZON,
    MapType.RADAR_COVERAGE,
    MapType.LINK_PROFILE,
    MapType.NSU_OPTIMIZER,
)


def _rss_mb(proc: psutil.Process) -> float:
    return proc.memory_info().rss / (1024 * 1024)


def _compact_c_heap() -> None:
    """Linux malloc_trim / Windows HeapCompact — see _process_entry."""
    try:
        if sys.platform.startswith('linux'):
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        elif sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.GetProcessHeap.restype = ctypes.c_void_p
            kernel32.HeapCompact.restype = ctypes.c_size_t
            kernel32.HeapCompact.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            heap = kernel32.GetProcessHeap()
            if heap:
                kernel32.HeapCompact(heap, 0)
    except Exception:  # noqa: BLE001
        pass


async def run_one(
    service: MapDownloadService,
    settings_base,
    map_type: MapType,
    output_dir: Path,
    run_idx: int,
    *,
    proc: psutil.Process,
) -> dict:
    """Run one build, return metrics dict."""
    settings = settings_base.model_copy(update={'map_type': map_type})

    # Center the requested area on the control point (always set in the
    # default profile). Same area as the profile's [from..to] box (6600 m²).
    cx = settings.control_point_x_sk42_gk
    cy = settings.control_point_y_sk42_gk

    output_path = output_dir / f'{map_type.value.lower()}_{run_idx}.jpg'

    rss_before = _rss_mb(proc)
    t_start = time.monotonic()
    error: str | None = None
    try:
        await service.download(
            center_x_sk42_gk=cx,
            center_y_sk42_gk=cy,
            width_m=6600.0,
            height_m=6600.0,
            output_path=str(output_path),
            settings=settings,
        )
        ok = True
    except Exception as e:
        ok = False
        error = f'{type(e).__name__}: {e}'

    elapsed = time.monotonic() - t_start
    rss_after = _rss_mb(proc)

    # Drain background contour-disk-cache save threads before next run /
    # exit. In production the persistent worker lives long enough that
    # daemon threads finish between builds; perf_loop with --runs 1 exits
    # immediately after, killing pending saves and breaking the "warm
    # disk on next launch" measurement.
    try:
        from shared import contour_layer_disk_cache  # noqa: PLC0415
        contour_layer_disk_cache.wait_for_pending_saves(timeout=10.0)
    except ImportError:
        pass

    gc.collect()
    # Mirror the worker's post-build cleanup so perf_loop measures the
    # same baseline the production worker would see between builds.
    _compact_c_heap()
    rss_after_gc = _rss_mb(proc)

    return {
        'map_type': map_type.value,
        'run_idx': run_idx,
        'elapsed_s': elapsed,
        'rss_before_mb': rss_before,
        'rss_after_mb': rss_after,
        'rss_after_gc_mb': rss_after_gc,
        'ok': ok,
        'error': error,
    }


def _detect_leak(rss_after_gc_series: list[float]) -> str | None:
    """
    Detect monotonic RSS growth across runs (cold-cache plateaus after 1-2
    runs in normal operation). Returns description string if leak suspected.
    """
    if len(rss_after_gc_series) < 3:
        return None
    # Compare last build's RSS to build #2's RSS (skip cold #1 which
    # populates caches). If the per-build delta exceeds the threshold,
    # flag it.
    baseline = rss_after_gc_series[1]
    last = rss_after_gc_series[-1]
    n_builds_after_baseline = len(rss_after_gc_series) - 2
    if n_builds_after_baseline < 1:
        return None
    per_build_growth = (last - baseline) / n_builds_after_baseline
    # ±30 MB/build is normal noise; >50 MB is suspicious.
    if per_build_growth > 50.0:
        return f'+{per_build_growth:.0f} MB / build (suspect leak)'
    return None


async def main_async(args: argparse.Namespace) -> int:
    # Profile with map area + control point + (NSU/link) points.
    profile_path = REPO_ROOT / 'configs' / 'profiles' / 'default.toml'
    settings_base = load_profile(str(profile_path))

    api_key = os.getenv('API_KEY')
    if not api_key:
        print('ERROR: API_KEY not found in env (.secrets.env / .env).')
        return 2

    output_dir = REPO_ROOT / '.perf_out'
    output_dir.mkdir(exist_ok=True)

    # Resolve types from args
    if args.type == 'all':
        types: list[MapType] = list(STANDARD_SUITE)
    else:
        try:
            types = [MapType(args.type)]
        except ValueError:
            try:
                types = [MapType[args.type.upper()]]
            except KeyError:
                print(f'ERROR: unknown map type "{args.type}". '
                      f'Valid: {[t.value for t in MapType]} or "all".')
                return 2

    service = MapDownloadService(api_key)
    proc = psutil.Process(os.getpid())

    # Warm up Numba JIT before the loop so first-run timings don't
    # include the one-shot compile cost (~1.5 s cold, ~0.3 s with the
    # on-disk cache). Mirrors what the production persistent worker
    # does at startup.
    try:
        from services.radio_horizon import warmup_numba_kernels
        t0 = time.monotonic()
        warmup_numba_kernels()
        print(f'numba warmup: {(time.monotonic() - t0)*1000:.0f} ms')
    except Exception as exc:  # noqa: BLE001
        print(f'numba warmup skipped: {exc}')

    print(f'\nperf_loop: {len(types)} type(s) × {args.runs} run(s) — '
          f'starting; rss_initial={_rss_mb(proc):.0f} MB\n')

    all_results: list[dict] = []
    for mt in types:
        print(f'--- {mt.value} ---')
        type_results: list[dict] = []
        for run_idx in range(args.runs):
            result = await run_one(
                service, settings_base, mt, output_dir, run_idx, proc=proc,
            )
            type_results.append(result)
            status = 'OK ' if result['ok'] else 'ERR'
            line = (
                f'  [{status}] run#{run_idx}: '
                f'elapsed={result["elapsed_s"]:5.2f}s  '
                f'rss={result["rss_before_mb"]:.0f}→'
                f'{result["rss_after_gc_mb"]:.0f} MB'
            )
            print(line)
            if not result['ok']:
                print(f'    ERROR: {result["error"]}')
        all_results.extend(type_results)

    # ---- Summary table ----
    print('\n=== SUMMARY ===')
    header = (
        f'{"Type":<22} {"OK/N":>6} {"mean":>7} {"std":>6} '
        f'{"min":>6} {"max":>6} {"RSS Δ":>8} {"leak":>20}'
    )
    print(header)
    print('-' * len(header))
    by_type: dict[str, list[dict]] = {}
    for r in all_results:
        by_type.setdefault(r['map_type'], []).append(r)
    exit_code = 0
    for mt_str, runs in by_type.items():
        ok_runs = [r for r in runs if r['ok']]
        if not ok_runs:
            print(f'{mt_str:<22} {"0/" + str(len(runs)):>6} '
                  f'{"--":>7} {"--":>6} {"--":>6} {"--":>6} '
                  f'{"--":>8} {"--":>20}')
            exit_code = 1
            continue
        times = [r['elapsed_s'] for r in ok_runs]
        rss_series = [r['rss_after_gc_mb'] for r in ok_runs]
        rss_delta = rss_series[-1] - ok_runs[0]['rss_before_mb']
        leak_note = _detect_leak(rss_series) or ''
        print(
            f'{mt_str:<22} '
            f'{f"{len(ok_runs)}/{len(runs)}":>6} '
            f'{mean(times):>7.2f} '
            f'{(stdev(times) if len(times) > 1 else 0):>6.2f} '
            f'{min(times):>6.2f} '
            f'{max(times):>6.2f} '
            f'{rss_delta:>+8.0f} '
            f'{leak_note:>20}'
        )
        if leak_note:
            exit_code = max(exit_code, 0)  # warn but don't fail
        if len(ok_runs) < len(runs):
            exit_code = 1

    print()
    return exit_code


def _setup_logging(quiet: bool, log_file: Path) -> None:
    handlers: list[logging.Handler] = []
    if not quiet:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        handlers.append(sh)
    fh = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    handlers.append(fh)
    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    for h in handlers:
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Replace any inherited handlers (pytest/etc.)
    root.handlers = handlers


def main() -> int:
    parser = argparse.ArgumentParser(description='Map build perf loop')
    parser.add_argument(
        '--type', default='all',
        help='Map type (HYBRID, ELEVATION_COLOR, …) or "all".',
    )
    parser.add_argument(
        '--runs', type=int, default=3,
        help='Builds per type (default: 3).',
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress per-stage logs from stdout (still saved to file).',
    )
    parser.add_argument(
        '--log-file',
        default=str(REPO_ROOT / '.perf_out' / 'perf_loop.log'),
        help='Log file path (.perf_out/perf_loop.log by default).',
    )
    args = parser.parse_args()

    log_file = Path(args.log_file)
    log_file.parent.mkdir(exist_ok=True)
    _setup_logging(quiet=args.quiet, log_file=log_file)

    try:
        rc = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print('\nInterrupted.')
        return 130
    return rc


if __name__ == '__main__':
    sys.exit(main())
