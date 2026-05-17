
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from main import setup_logging
from shared.constants import LOG_FSYNC_TO_FILE


class TestMain:
    def test_setup_logging(self, tmp_path, monkeypatch):
        appdata = tmp_path / "appdata"
        localappdata = tmp_path / "localappdata"
        monkeypatch.setenv("APPDATA", str(appdata))
        monkeypatch.setenv("LOCALAPPDATA", str(localappdata))

        appdata_base, local_base = setup_logging()

        assert appdata_base == appdata / "SK42"
        assert local_base == localappdata / "SK42"
        # Log directory is always created
        assert (local_base / "log").is_dir()
        # Log file is only created when fsync file handler is enabled
        if LOG_FSYNC_TO_FILE:
            assert (local_base / "log" / "mil_mapper.log").exists()

