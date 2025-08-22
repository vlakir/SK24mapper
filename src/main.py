import asyncio
import os

from dotenv import load_dotenv

from constants import CURRENT_PROFILE
from controller import download_satellite_rectangle
from profiles import load_profile
from topography import center_x_sk42_gk, center_y_sk42_gk, height_m, width_m

settings = load_profile(CURRENT_PROFILE)


def main() -> None:
    load_dotenv('../.secrets.env')

    api_key = os.getenv('API_KEY', '').strip()
    if not api_key:
        msg = (
            'Не найден API ключ. Создайте файл secrets.env с содержимым вида:\n'
            'API_KEY=ваш_ключ\n'
            'Либо задайте переменную окружения API_KEY перед запуском.'
        )
        raise SystemExit(msg)

    # 3) Запуск конвейера
    asyncio.run(
        download_satellite_rectangle(
            center_x_sk42_gk=center_y_sk42_gk,
            center_y_sk42_gk=center_x_sk42_gk,
            width_m=width_m,
            height_m=height_m,
            api_key=api_key,
            output_path=settings.output_path,
            max_zoom=settings.zoom,
        ),
    )


if __name__ == '__main__':
    main()
