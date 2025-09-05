import os
from pathlib import Path

import tomlkit

from domen import MapSettings


def _user_profiles_dir() -> Path:
    """
    Determine profiles directory.

    1) If <project_root>/configs/profiles exists, use it (useful for portable/run-from-repo setups).
    2) Otherwise, fall back to user APPDATA directory: %APPDATA%/SK42mapper/configs/profiles
       or ~/AppData/Roaming/SK42mapper/configs/profiles when APPDATA is not set.
    """
    # Try project-local configs/profiles
    project_root = Path(__file__).resolve().parent.parent
    local_profiles = project_root / 'configs' / 'profiles'
    if local_profiles.exists():
        return local_profiles

    # Fallback to user-specific APPDATA location
    return (
        Path(os.getenv('APPDATA') or (Path.home() / 'AppData' / 'Roaming'))
        / 'SK42mapper'
        / 'configs'
        / 'profiles'
    )


def ensure_profiles_dir() -> Path:
    # Prefer user APPDATA directory for profiles
    profiles_dir = _user_profiles_dir()
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir


def list_profiles() -> list[str]:
    """Список имён профилей без расширения."""
    folder = ensure_profiles_dir()
    return sorted(p.stem for p in folder.glob('*.toml') if p.is_file())


def profile_path(name: str) -> Path:
    """Путь к файлу профиля по имени."""
    return ensure_profiles_dir() / f'{name}.toml'


def load_profile(name_or_path: str) -> MapSettings:
    """
    Загрузка и валидация профиля TOML -> MapSettings.

    Поддерживает как имя профиля (без .toml) из каталога profiles,
    так и абсолютный/относительный путь до TOML файла.
    """
    p = Path(name_or_path)
    path = (
        p if p.suffix.lower() == '.toml' and p.exists() else profile_path(name_or_path)
    )
    if not path.exists():
        msg = f'Профиль не найден: {path}'
        raise FileNotFoundError(msg)
    text = path.read_text(encoding='utf-8')
    data = tomlkit.parse(text)

    return MapSettings.model_validate(data)  # type: ignore[no-any-return]


def save_profile(name: str, settings: MapSettings) -> Path:
    """Сохранение профиля в TOML (без атомарности и бэкапов)."""
    path = profile_path(name)
    data = settings.model_dump()
    text = tomlkit.dumps(data)
    path.write_text(text, encoding='utf-8')
    return path


def delete_profile(name: str) -> None:
    """Удаление файла профиля, если он существует."""
    path = profile_path(name)
    if path.exists():
        path = profile_path(name)
    if path.exists():
        path.unlink()
