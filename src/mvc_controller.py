"""MVC Controller for coordinating between Model and View."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from controller import download_satellite_rectangle
from model import MapSettings
from mvc_model import MilMapperModel, ModelEvent
from profiles import load_profile, save_profile

logger = logging.getLogger(__name__)


class MilMapperController:
    """Main application controller implementing MVC pattern."""
    
    def __init__(self, model: MilMapperModel) -> None:
        """Initialize controller with model reference."""
        self._model = model
        self._api_key: str | None = None
        self._load_api_key()
        logger.info("MilMapperController initialized")
    
    def _load_api_key(self) -> None:
        """Load API key from environment variables."""
        try:
            # Load secrets from common locations
            repo_root = Path(__file__).resolve().parent.parent
            candidates = [
                Path('.secrets.env'),
                Path('.env'),
                repo_root / '.secrets.env',
                repo_root / '.env',
            ]
            
            for p in candidates:
                if p.exists():
                    load_dotenv(p)
                    break
            
            self._api_key = os.getenv('API_KEY', '').strip()
            if not self._api_key:
                error_msg = "API key not found in environment variables"
                logger.error(error_msg)
                self._model.notify_observers(ModelEvent.ERROR_OCCURRED, {"error": error_msg})
            else:
                logger.info("API key loaded successfully")
                
        except Exception as e:
            error_msg = f"Failed to load API key: {e}"
            logger.error(error_msg)
            self._model.notify_observers(ModelEvent.ERROR_OCCURRED, {"error": error_msg})
    
    def update_setting(self, field_name: str, value: Any) -> None:
        """Update a specific setting field."""
        try:
            self._model.update_settings(**{field_name: value})
            logger.debug(f"Updated setting {field_name} = {value}")
        except Exception as e:
            error_msg = f"Failed to update setting {field_name}: {e}"
            logger.error(error_msg)
    
    def load_profile_by_name(self, profile_name: str) -> None:
        """Load profile from file."""
        try:
            settings = load_profile(profile_name)
            self._model.load_profile(profile_name, settings)
            logger.info(f"Loaded profile: {profile_name}")
        except Exception as e:
            error_msg = f"Failed to load profile {profile_name}: {e}"
            logger.error(error_msg)
            self._model.notify_observers(ModelEvent.ERROR_OCCURRED, {"error": error_msg})
    
    def save_current_profile(self, profile_name: str) -> None:
        """Save current settings to profile file."""
        try:
            save_profile(profile_name, self._model.settings)
            self._model.save_profile(profile_name)
            logger.info(f"Saved profile: {profile_name}")
        except Exception as e:
            error_msg = f"Failed to save profile {profile_name}: {e}"
            logger.error(error_msg)
            self._model.notify_observers(ModelEvent.ERROR_OCCURRED, {"error": error_msg})
    
    def validate_api_key(self) -> bool:
        """Check if API key is available."""
        return self._api_key is not None and len(self._api_key) > 0
    
    async def start_map_download(self) -> None:
        """Start the map download process."""
        if not self.validate_api_key():
            error_msg = "API key not available for download"
            logger.error(error_msg)
            self._model.notify_observers(ModelEvent.ERROR_OCCURRED, {"error": error_msg})
            return
        
        try:
            self._model.start_download()
            settings = self._model.settings
            
            # Progress callbacks will be set up by the worker thread
            # to avoid threading issues with Qt UI updates
            
            # Calculate center coordinates and dimensions
            center_x = (settings.bottom_left_x_sk42_gk + settings.top_right_x_sk42_gk) / 2
            center_y = (settings.bottom_left_y_sk42_gk + settings.top_right_y_sk42_gk) / 2
            width_m = settings.top_right_x_sk42_gk - settings.bottom_left_x_sk42_gk
            height_m = settings.top_right_y_sk42_gk - settings.bottom_left_y_sk42_gk
            
            logger.info(f"Starting download: center=({center_x}, {center_y}), size=({width_m}x{height_m})")
            
            # Start download
            await download_satellite_rectangle(
                center_x_sk42_gk=center_x,
                center_y_sk42_gk=center_y,
                width_m=width_m,
                height_m=height_m,
                api_key=self._api_key,
                output_path=settings.output_path,
                settings=settings,
            )
            
            self._model.complete_download(success=True)
            logger.info("Download completed successfully")
            
        except Exception as e:
            error_msg = f"Download failed: {e}"
            logger.error(error_msg, exc_info=True)
            self._model.complete_download(success=False, error_msg=error_msg)
    
    def start_map_download_sync(self) -> None:
        """Start map download in synchronous context."""
        try:
            # Run the async download in a new event loop
            asyncio.run(self.start_map_download())
        except Exception as e:
            error_msg = f"Failed to start download: {e}"
            logger.error(error_msg)
            self._model.complete_download(success=False, error_msg=error_msg)
    
    def get_available_profiles(self) -> list[str]:
        """Get list of available profile names."""
        try:
            profiles_dir = Path(__file__).parent.parent / "configs" / "profiles"
            if not profiles_dir.exists():
                return ["default"]
            
            profiles = []
            for file_path in profiles_dir.glob("*.toml"):
                profiles.append(file_path.stem)
            
            return sorted(profiles) if profiles else ["default"]
            
        except Exception as e:
            logger.error(f"Failed to get available profiles: {e}")
            return ["default"]
    
    def update_coordinates(self, coords: dict[str, int]) -> None:
        """Update coordinate settings."""
        try:
            valid_keys = {
                'from_x_high', 'from_y_high', 'to_x_high', 'to_y_high',
                'from_x_low', 'from_y_low', 'to_x_low', 'to_y_low'
            }
            
            filtered_coords = {k: v for k, v in coords.items() if k in valid_keys}
            if filtered_coords:
                self._model.update_settings(**filtered_coords)
                
        except Exception as e:
            error_msg = f"Failed to update coordinates: {e}"
            logger.error(error_msg)
            self._model.notify_observers(ModelEvent.ERROR_OCCURRED, {"error": error_msg})
    
    def update_grid_settings(self, grid_settings: dict[str, Any]) -> None:
        """Update grid-related settings."""
        try:
            valid_keys = {
                'grid_width_px', 'grid_font_size', 'grid_text_margin',
                'grid_label_bg_padding'
            }
            
            filtered_settings = {k: v for k, v in grid_settings.items() if k in valid_keys}
            if filtered_settings:
                self._model.update_settings(**filtered_settings)
                
        except Exception as e:
            error_msg = f"Failed to update grid settings: {e}"
            logger.error(error_msg)
            self._model.notify_observers(ModelEvent.ERROR_OCCURRED, {"error": error_msg})
    
    def update_output_settings(self, output_settings: dict[str, Any]) -> None:
        """Update output-related settings."""
        try:
            valid_keys = {'output_path', 'mask_opacity', 'png_compress_level'}
            
            filtered_settings = {k: v for k, v in output_settings.items() if k in valid_keys}
            if filtered_settings:
                self._model.update_settings(**filtered_settings)
                
        except Exception as e:
            error_msg = f"Failed to update output settings: {e}"
            logger.error(error_msg)
            self._model.notify_observers(ModelEvent.ERROR_OCCURRED, {"error": error_msg})