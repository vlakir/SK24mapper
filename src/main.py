"""Main entry point for Mil Mapper 2.0 with PySide6 MVC architecture."""

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QLocale
from PySide6.QtGui import QIcon

from diagnostics import log_comprehensive_diagnostics, log_memory_usage
from gui.view import create_application

logger = logging.getLogger(__name__)


def setup_logging() -> tuple[Path, Path]:
    """Configure application logging to LOCALAPPDATA and ensure user dirs.

    Returns:
        Tuple (appdata_base, local_base) for further use.
    """
    import os
    # Determine user profile dirs
    appdata_base = Path(os.getenv('APPDATA') or Path.home() / 'AppData' / 'Roaming') / 'SK42mapper'
    local_base = Path(os.getenv('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local') / 'SK42mapper'
    log_dir = local_base / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'mil_mapper.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding='utf-8'),
        ],
    )
    return appdata_base, local_base


def main() -> int:
    """Main application entry point."""
    appdata_base, local_base = setup_logging()
    logger.info('Starting Mil Mapper 2.0')

    # Ensure user data directories and bootstrap defaults
    try:
        import os
        import shutil
        import sys as _sys
        # Create user dirs
        (appdata_base / 'configs' / 'profiles').mkdir(parents=True, exist_ok=True)
        (appdata_base / 'maps').mkdir(parents=True, exist_ok=True)
        (local_base / '.cache' / 'tiles').mkdir(parents=True, exist_ok=True)
        # Copy default configs if not present
        install_dir = Path(_sys.argv[0]).resolve().parent
        default_cfg_root = install_dir / 'configs'
        if default_cfg_root.exists():
            # Copy files only if missing in user configs
            for src in default_cfg_root.rglob('*'):
                if src.is_file():
                    rel = src.relative_to(default_cfg_root)
                    dst = appdata_base / 'configs' / rel
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copy2(src, dst)
                        except Exception:
                            pass
    except Exception as _e:
        logger.warning(f'User data bootstrap failed: {_e}')

    # Log initial system state
    log_comprehensive_diagnostics('APPLICATION_STARTUP')

    parser = argparse.ArgumentParser(
        description='Mil Mapper 2.0 - создание топографических карт'
    )
    parser.add_argument(
        '--mode',
        choices=['gui'],
        default='gui',
        help='Режим запуска приложения (только gui в версии 2.0)',
    )

    parser.parse_args()

    try:
        # Create PySide6 application
        log_memory_usage('before creating application')
        app, window, model, controller = create_application()
        log_memory_usage('after creating application')

        # Configure application properties
        app.setQuitOnLastWindowClosed(True)

        # Set Russian locale for all Qt dialogs
        QLocale.setDefault(QLocale(QLocale.Language.Russian, QLocale.Country.Russia))

        # Set application icon if available
        import sys as _sys2
        install_dir2 = Path(_sys2.argv[0]).resolve().parent
        icon_path = install_dir2 / 'img' / 'icon.ico'
        if not icon_path.exists():
            # fallback to source path (dev mode)
            icon_path = Path(__file__).parent.parent / 'img' / 'icon.ico'
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))

        # Show main window
        window.show()

        logger.info('Application started successfully')
        log_comprehensive_diagnostics('APPLICATION_READY')

        # Run application event loop
        result = app.exec()

        # Log final state before exit
        log_comprehensive_diagnostics('APPLICATION_SHUTDOWN')
        return result

    except Exception as e:
        logger.error(f'Failed to start application: {e}', exc_info=True)
        log_comprehensive_diagnostics('APPLICATION_ERROR')
        return 1


if __name__ == '__main__':
    sys.exit(main())
