"""
Tests for execution mode functionality.
"""
import pytest
from src.models import Action, ActionType, ExecutionMode


class TestExecutionMode:
    """Test execution mode enum and action integration."""
    
    def test_execution_mode_enum_values(self):
        """Test that ExecutionMode enum has correct values."""
        assert ExecutionMode.TRIGGER_ONCE.value == "trigger_once"
        assert ExecutionMode.HOLD_REPEAT.value == "hold_repeat"
    
    def test_action_default_execution_mode(self):
        """Test that Action defaults to TRIGGER_ONCE."""
        action = Action(
            type=ActionType.KEYSTROKE,
            data={'keys': ['ctrl', 'c']}
        )
        assert action.execution_mode == ExecutionMode.TRIGGER_ONCE
    
    def test_action_with_trigger_once_mode(self):
        """Test creating action with TRIGGER_ONCE mode."""
        action = Action(
            type=ActionType.LAUNCH_APP,
            data={'path': 'notepad.exe'},
            execution_mode=ExecutionMode.TRIGGER_ONCE
        )
        assert action.execution_mode == ExecutionMode.TRIGGER_ONCE
    
    def test_action_with_hold_repeat_mode(self):
        """Test creating action with HOLD_REPEAT mode."""
        action = Action(
            type=ActionType.MEDIA_CONTROL,
            data={'command': 'volume_up'},
            execution_mode=ExecutionMode.HOLD_REPEAT
        )
        assert action.execution_mode == ExecutionMode.HOLD_REPEAT
    
    def test_action_serialization_with_execution_mode(self):
        """Test that execution mode is preserved in JSON serialization."""
        action = Action(
            type=ActionType.SYSTEM_CONTROL,
            data={'command': 'volume_up', 'amount': 5},
            execution_mode=ExecutionMode.HOLD_REPEAT
        )
        
        json_str = action.to_json()
        restored_action = Action.from_json(json_str)
        
        assert restored_action.execution_mode == ExecutionMode.HOLD_REPEAT
        assert restored_action.type == ActionType.SYSTEM_CONTROL
        assert restored_action.data == {'command': 'volume_up', 'amount': 5}
    
    def test_action_deserialization_defaults_to_trigger_once(self):
        """Test that old actions without execution_mode default to TRIGGER_ONCE."""
        # Simulate old JSON format without execution_mode
        json_str = '{"type": "keystroke", "data": {"keys": ["ctrl", "v"]}}'
        action = Action.from_json(json_str)
        
        assert action.execution_mode == ExecutionMode.TRIGGER_ONCE
        assert action.type == ActionType.KEYSTROKE
