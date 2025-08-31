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


def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mil_mapper.log', encoding='utf-8'),
        ],
    )


def main() -> int:
    """Main application entry point."""
    setup_logging()
    logger.info('Starting Mil Mapper 2.0')

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
