"""
Integration tests for GUI components.

These tests verify that GUI components can be instantiated and interact
with the MainController correctly.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from PyQt5.QtWidgets import QApplication
import sys

from src.gui import (
    MainWindow, RecordingDialog, GestureNameDialog,
    ActionSelectionDialog, EditGestureDialog, SettingsDialog
)
from src.main_controller import MainController
from src.models import Config, Gesture, Action, ActionType
from datetime import datetime
import numpy as np


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication instance for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def mock_controller():
    """Create a mock MainController for testing."""
    controller = Mock(spec=MainController)
    controller.config = Config()
    controller.is_running.return_value = False
    controller.get_state.return_value = Mock()
    controller.get_all_gestures.return_value = []
    controller.get_current_match.return_value = None
    controller.process_frame.return_value = None
    return controller


def test_main_window_initialization(qapp, mock_controller):
    """Test that MainWindow can be initialized."""
    window = MainWindow(mock_controller)
    assert window is not None
    assert window.controller == mock_controller
    assert window.windowTitle() == "Gesture Control System"


def test_settings_dialog_initialization(qapp):
    """Test that SettingsDialog can be initialized."""
    config = Config()
    dialog = SettingsDialog(config)
    assert dialog is not None
    assert dialog.config == config


def test_settings_dialog_get_config(qapp):
    """Test that SettingsDialog returns updated config."""
    config = Config()
    dialog = SettingsDialog(config)
    
    # Modify some values
    dialog.threshold_spin.setValue(0.7)
    dialog.duration_spin.setValue(5.0)
    dialog.fps_spin.setValue(60)
    
    # Get updated config
    new_config = dialog.get_config()
    assert new_config.similarity_threshold == 0.7
    assert new_config.recording_duration == 5.0
    assert new_config.fps == 60


def test_action_selection_dialog_launch_app(qapp):
    """Test ActionSelectionDialog for launch app action."""
    dialog = ActionSelectionDialog()
    
    # Select launch app
    dialog.action_type_combo.setCurrentIndex(0)
    dialog.app_path_input.setText("notepad.exe")
    
    action = dialog.get_action()
    assert action.type == ActionType.LAUNCH_APP
    assert action.data['path'] == "notepad.exe"


def test_action_selection_dialog_keystroke(qapp):
    """Test ActionSelectionDialog for keystroke action."""
    dialog = ActionSelectionDialog()
    
    # Select keystroke
    dialog.action_type_combo.setCurrentIndex(1)
    dialog.keys_input.setText("ctrl+c")
    
    action = dialog.get_action()
    assert action.type == ActionType.KEYSTROKE
    assert action.data['keys'] == ['ctrl', 'c']


def test_action_selection_dialog_media_control(qapp):
    """Test ActionSelectionDialog for media control action."""
    dialog = ActionSelectionDialog()
    
    # Select media control
    dialog.action_type_combo.setCurrentIndex(2)
    dialog.media_command_combo.setCurrentText("play_pause")
    
    action = dialog.get_action()
    assert action.type == ActionType.MEDIA_CONTROL
    assert action.data['command'] == "play_pause"


def test_action_selection_dialog_system_control(qapp):
    """Test ActionSelectionDialog for system control action."""
    dialog = ActionSelectionDialog()
    
    # Select system control
    dialog.action_type_combo.setCurrentIndex(3)
    dialog.system_command_combo.setCurrentText("volume_up")
    dialog.amount_spin.setValue(5)
    
    action = dialog.get_action()
    assert action.type == ActionType.SYSTEM_CONTROL
    assert action.data['command'] == "volume_up"
    assert action.data['amount'] == 5


def test_edit_gesture_dialog(qapp):
    """Test EditGestureDialog initialization and get_gesture."""
    gesture = Gesture(
        name="test_gesture",
        embedding=np.random.rand(16),
        action=Action(ActionType.KEYSTROKE, {'keys': ['ctrl', 'c']}),
        created_at=datetime.now()
    )
    
    dialog = EditGestureDialog(gesture)
    assert dialog is not None
    assert dialog.name_input.text() == "test_gesture"
    
    # Modify name
    dialog.name_input.setText("modified_gesture")
    
    updated_gesture = dialog.get_gesture()
    assert updated_gesture.name == "modified_gesture"
    assert np.array_equal(updated_gesture.embedding, gesture.embedding)


def test_main_window_refresh_gesture_list(qapp, mock_controller):
    """Test that gesture list is refreshed correctly."""
    # Create mock gestures
    gestures = [
        Gesture(
            name="gesture1",
            embedding=np.random.rand(16),
            action=Action(ActionType.KEYSTROKE, {'keys': ['ctrl', 'c']}),
            created_at=datetime.now()
        ),
        Gesture(
            name="gesture2",
            embedding=np.random.rand(16),
            action=Action(ActionType.MEDIA_CONTROL, {'command': 'play_pause'}),
            created_at=datetime.now()
        )
    ]
    mock_controller.get_all_gestures.return_value = gestures
    
    window = MainWindow(mock_controller)
    window._refresh_gesture_list()
    
    assert window.gesture_list.count() == 2


def test_main_window_format_action(qapp, mock_controller):
    """Test action formatting for display."""
    window = MainWindow(mock_controller)
    
    # Test launch app
    action = Action(ActionType.LAUNCH_APP, {'path': 'notepad.exe'})
    formatted = window._format_action(action)
    assert "Launch" in formatted
    assert "notepad.exe" in formatted
    
    # Test keystroke
    action = Action(ActionType.KEYSTROKE, {'keys': ['ctrl', 'c']})
    formatted = window._format_action(action)
    assert "Keys" in formatted
    assert "ctrl+c" in formatted
    
    # Test media control
    action = Action(ActionType.MEDIA_CONTROL, {'command': 'play_pause'})
    formatted = window._format_action(action)
    assert "Media" in formatted
    assert "play_pause" in formatted
    
    # Test system control
    action = Action(ActionType.SYSTEM_CONTROL, {'command': 'volume_up'})
    formatted = window._format_action(action)
    assert "System" in formatted
    assert "volume_up" in formatted


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
