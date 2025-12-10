"""
Unit tests for ActionExecutor component.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.action_executor import ActionExecutor
from src.models import Action, ActionType


class TestActionExecutor:
    """Test suite for ActionExecutor class."""
    
    def test_initialization(self):
        """Test that ActionExecutor initializes correctly."""
        executor = ActionExecutor()
        assert executor.keyboard is not None
        assert executor.mouse is not None
        assert executor.system in ['Windows', 'Linux', 'Darwin']
    
    def test_execute_launch_app(self):
        """Test executing a launch app action."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.LAUNCH_APP,
            data={'path': 'notepad.exe'}
        )
        
        with patch('subprocess.Popen') as mock_popen:
            result = executor.execute(action)
            assert mock_popen.called
    
    def test_execute_keystroke(self):
        """Test executing a keystroke action."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.KEYSTROKE,
            data={'keys': ['ctrl', 'c']}
        )
        
        with patch.object(executor.keyboard, 'press') as mock_press, \
             patch.object(executor.keyboard, 'release') as mock_release:
            result = executor.execute(action)
            assert result is True
            assert mock_press.call_count == 2  # ctrl and c
            assert mock_release.call_count == 2
    
    def test_execute_media_control(self):
        """Test executing a media control action."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.MEDIA_CONTROL,
            data={'command': 'play_pause'}
        )
        
        with patch.object(executor.keyboard, 'press') as mock_press, \
             patch.object(executor.keyboard, 'release') as mock_release:
            result = executor.execute(action)
            assert result is True
            assert mock_press.called
            assert mock_release.called
    
    def test_execute_system_control_volume(self):
        """Test executing a system control action for volume."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.SYSTEM_CONTROL,
            data={'command': 'volume_up', 'amount': 2}
        )
        
        with patch.object(executor.keyboard, 'press') as mock_press, \
             patch.object(executor.keyboard, 'release') as mock_release:
            result = executor.execute(action)
            assert result is True
            assert mock_press.call_count == 2  # Called twice for amount=2
            assert mock_release.call_count == 2
    
    def test_execute_unknown_action_type(self):
        """Test that unknown action types are handled gracefully."""
        executor = ActionExecutor()
        # Create a mock action with invalid type
        action = Mock()
        action.type = "INVALID_TYPE"
        action.data = {}
        
        result = executor.execute(action)
        assert result is False
    
    def test_launch_app_missing_path(self):
        """Test that launch_app handles missing path gracefully."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.LAUNCH_APP,
            data={}  # Missing 'path'
        )
        
        result = executor.execute(action)
        assert result is False
    
    def test_simulate_keystroke_empty_keys(self):
        """Test that simulate_keystroke handles empty keys gracefully."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.KEYSTROKE,
            data={'keys': []}
        )
        
        result = executor.execute(action)
        assert result is False
    
    def test_media_control_missing_command(self):
        """Test that media_control handles missing command gracefully."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.MEDIA_CONTROL,
            data={}
        )
        
        result = executor.execute(action)
        assert result is False
    
    def test_system_control_missing_command(self):
        """Test that system_control handles missing command gracefully."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.SYSTEM_CONTROL,
            data={}
        )
        
        result = executor.execute(action)
        assert result is False
    
    def test_error_handling_continues_operation(self):
        """Test that errors are logged and execution continues."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.KEYSTROKE,
            data={'keys': ['ctrl', 'c']}
        )
        
        # Simulate an error during keystroke
        with patch.object(executor.keyboard, 'press', side_effect=Exception("Test error")):
            result = executor.execute(action)
            assert result is False  # Should return False but not crash
    
    def test_keystroke_with_special_keys(self):
        """Test keystroke simulation with special keys."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.KEYSTROKE,
            data={'keys': ['ctrl', 'shift', 'esc']}
        )
        
        with patch.object(executor.keyboard, 'press') as mock_press, \
             patch.object(executor.keyboard, 'release') as mock_release:
            result = executor.execute(action)
            assert result is True
            assert mock_press.call_count == 3
            assert mock_release.call_count == 3
    
    def test_media_control_next(self):
        """Test media control for next track."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.MEDIA_CONTROL,
            data={'command': 'next'}
        )
        
        with patch.object(executor.keyboard, 'press') as mock_press, \
             patch.object(executor.keyboard, 'release') as mock_release:
            result = executor.execute(action)
            assert result is True
    
    def test_media_control_previous(self):
        """Test media control for previous track."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.MEDIA_CONTROL,
            data={'command': 'previous'}
        )
        
        with patch.object(executor.keyboard, 'press') as mock_press, \
             patch.object(executor.keyboard, 'release') as mock_release:
            result = executor.execute(action)
            assert result is True
    
    def test_system_control_switch_window(self):
        """Test system control for window switching."""
        executor = ActionExecutor()
        action = Action(
            type=ActionType.SYSTEM_CONTROL,
            data={'command': 'switch_window'}
        )
        
        with patch.object(executor.keyboard, 'press') as mock_press, \
             patch.object(executor.keyboard, 'release') as mock_release:
            result = executor.execute(action)
            assert result is True
