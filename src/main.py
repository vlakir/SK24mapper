import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

import topography
from constants import CURRENT_PROFILE, DESIRED_ZOOM
from controller import download_satellite_rectangle
from gui.app import run_app
from profiles import load_profile


def _load_secrets() -> list[str]:
    """Load secrets from common locations and return the paths that were checked."""
    checked: list[str] = []
    repo_root = Path(__file__).resolve().parent.parent  # src/.. -> project root
    candidates = [
        Path('.secrets.env'),
        Path('.env'),
        repo_root / '.secrets.env',
        repo_root / '.env',
    ]
    for p in candidates:
        checked.append(str(p))
        if p.exists():
            load_dotenv(p)
            break
    return checked


def main() -> None:
    checked_paths = _load_secrets()

    api_key = os.getenv('API_KEY', '').strip()
    if not api_key:
        msg = (
            'Не найден API ключ. Создайте файл .secrets.env или .env\n'
            'с содержимым вида:\n'
            'API_KEY=ваш_ключ\n'
            'Либо задайте переменную окружения API_KEY перед запуском.\n\n'
            'Пути, где выполнялся поиск: \n- ' + '\n- '.join(checked_paths)
        )
        raise SystemExit(msg)

    settings = load_profile(CURRENT_PROFILE)
    asyncio.run(
        download_satellite_rectangle(
            center_x_sk42_gk=topography.center_x_sk42_gk,
            center_y_sk42_gk=topography.center_y_sk42_gk,
            width_m=topography.width_m,
            height_m=topography.height_m,
            api_key=api_key,
            output_path=settings.output_path,
            max_zoom=DESIRED_ZOOM,
        ),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Mil Mapper - создание топографических карт'
    )
    parser.add_argument(
        '--mode',
        choices=['gui', 'console'],
        default='console',
        help='Режим запуска приложения: gui или console (по умолчанию)',
    )

    args = parser.parse_args()

    if args.mode == 'gui':
        run_app(main)
    else:
        main()
