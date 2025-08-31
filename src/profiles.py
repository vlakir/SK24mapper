from pathlib import Path

import tomlkit

from constants import PROFILES_DIR
from domen import MapSettings


def ensure_profiles_dir() -> Path:
    # Resolve path relative to project root
    if Path(PROFILES_DIR).is_absolute():
        profiles_dir = Path(PROFILES_DIR)
    else:
        # Get project root (parent of src directory)
        project_root = Path(__file__).parent.parent
        profiles_dir = project_root / PROFILES_DIR

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
