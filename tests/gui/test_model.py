"""Tests for gui.model — MilMapperModel."""

from unittest.mock import MagicMock

from gui.model import MilMapperModel, ModelEvent


class TestPatchSettingsSilent:
    """Tests for MilMapperModel.patch_settings_silent()."""

    def test_updates_field(self):
        """patch_settings_silent should update the requested field."""
        model = MilMapperModel()
        old_x = model.settings.control_point_x
        new_x = old_x + 1000
        model.patch_settings_silent(control_point_x=new_x)
        assert model.settings.control_point_x == new_x

    def test_does_not_notify_observers(self):
        """patch_settings_silent must NOT emit SETTINGS_CHANGED."""
        model = MilMapperModel()
        observer = MagicMock()
        model.add_observer(observer)
        model.patch_settings_silent(control_point_y=7450000)
        # Observer should NOT have been called with SETTINGS_CHANGED
        for call in observer.call_args_list:
            event_data = call[0][0]
            assert event_data.event != ModelEvent.SETTINGS_CHANGED

    def test_multiple_fields(self):
        """patch_settings_silent should update several fields at once."""
        model = MilMapperModel()
        model.patch_settings_silent(
            link_point_a_x=5420000,
            link_point_a_y=7450000,
        )
        assert model.settings.link_point_a_x == 5420000
        assert model.settings.link_point_a_y == 7450000

    def test_preserves_other_fields(self):
        """patch_settings_silent should not affect untouched fields."""
        model = MilMapperModel()
        original_y = model.settings.control_point_y
        model.patch_settings_silent(control_point_x=5420000)
        assert model.settings.control_point_y == original_y
