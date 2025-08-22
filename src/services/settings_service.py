from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import tomlkit

from constants import PROFILES_DIR
from model import MapSettings
from profiles import delete_profile as _delete_profile
from profiles import list_profiles as _list_profiles
from profiles import load_profile as _load_profile
from profiles import profile_path as _profile_path
from profiles import save_profile as _save_profile


class SettingsService:
    """Service layer for managing MapSettings profiles and active selection.

    Stores the active profile name in a .active marker file inside profiles dir.
    """

    NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or PROFILES_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.active_file = self.base_dir / ".active"

    # Profile name utils
    def _validate_name(self, name: str) -> None:
        if not self.NAME_RE.fullmatch(name):
            msg = (
                "Некорректное имя профиля. Используйте латиницу, цифры, '-', '_' и не начинайте с точки; длина до 64."
            )
            raise ValueError(msg)
        if name.startswith('.'):
            raise ValueError("Имя профиля не должно начинаться с точки")

    # Active profile
    def get_active_profile(self) -> str | None:
        if self.active_file.exists():
            name = self.active_file.read_text(encoding='utf-8').strip()
            return name or None
        return None

    def set_active_profile(self, name: str) -> None:
        if not self.exists(name):
            raise FileNotFoundError(f"Профиль не найден: {name}")
        self.active_file.write_text(name, encoding='utf-8')

    # CRUD and queries
    def list_profiles(self) -> list[str]:
        return _list_profiles()

    def exists(self, name: str) -> bool:
        return _profile_path(name).exists()

    def load(self, name: str) -> MapSettings:
        return _load_profile(name)

    def save(self, name: str, data: MapSettings | dict[str, Any]) -> None:
        settings = data if isinstance(data, MapSettings) else MapSettings.model_validate(data)
        _save_profile(name, settings)

    def create(self, name: str, base: MapSettings | None = None) -> None:
        self._validate_name(name)
        if self.exists(name):
            raise FileExistsError(f"Профиль уже существует: {name}")
        if base is None:
            # Try default.toml if exists
            default_path = self.base_dir / "default.toml"
            if default_path.exists():
                text = default_path.read_text(encoding='utf-8')
                data = tomlkit.parse(text)
                base = MapSettings.model_validate(data)
            else:
                # Create from model defaults by minimally valid stub? MapSettings has required fields so we cannot default
                # Use default profile template if present in repo; otherwise raise
                raise ValueError("Не задан базовый профиль и отсутствует default.toml")
        _save_profile(name, base)

    def duplicate(self, src: str, dst: str) -> None:
        self._validate_name(dst)
        if not self.exists(src):
            raise FileNotFoundError(f"Исходный профиль не найден: {src}")
        if self.exists(dst):
            raise FileExistsError(f"Профиль уже существует: {dst}")
        settings = self.load(src)
        _save_profile(dst, settings)

    def rename(self, old: str, new: str) -> None:
        self._validate_name(new)
        if not self.exists(old):
            raise FileNotFoundError(f"Профиль не найден: {old}")
        if self.exists(new):
            raise FileExistsError(f"Профиль уже существует: {new}")
        _profile_path(old).rename(_profile_path(new))
        if self.get_active_profile() == old:
            self.set_active_profile(new)

    def delete(self, name: str) -> None:
        active = self.get_active_profile()
        if active == name:
            raise PermissionError("Нельзя удалить активный профиль. Переключитесь на другой.")
        if not self.exists(name):
            return
        _delete_profile(name)
