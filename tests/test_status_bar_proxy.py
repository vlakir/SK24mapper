
import pytest
from unittest.mock import MagicMock
from status_bar_proxy import StatusBarProxy
from PySide6.QtCore import Qt

def test_status_bar_proxy_emit():
    mock_status_bar = MagicMock()
    # To test StatusBarProxy without a full QApp, we might have issues with signals, 
    # but let's try a basic test.
    proxy = StatusBarProxy(mock_status_bar)
    
    # We can't easily wait for QueuedConnection without a QEventLoop.
    # But we can change it to DirectConnection for testing.
    proxy.show_message_requested.disconnect(proxy._on_show_message)
    proxy.show_message_requested.connect(proxy._on_show_message, Qt.ConnectionType.DirectConnection)
    
    proxy.show_message("test message", 1000)
    mock_status_bar.showMessage.assert_called_with("test message", 1000)
    
    proxy.show_message("no timeout")
    mock_status_bar.showMessage.assert_called_with("no timeout")
