"""
Tests for execution mode persistence in gesture repository.
"""
import pytest
import tempfile
import os
import numpy as np
from datetime import datetime

from src.gesture_repository import GestureRepository
from src.models import Gesture, Action, ActionType, ExecutionMode


class TestExecutionModePersistence:
    """Test that execution mode is properly saved and loaded."""
    
    def test_save_and_load_trigger_once(self):
        """Test saving and loading gesture with TRIGGER_ONCE mode."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        try:
            repo = GestureRepository(db_path)
            
            # Create gesture with TRIGGER_ONCE mode
            gesture = Gesture(
                name="test_trigger",
                embedding=np.random.rand(16),
                action=Action(
                    type=ActionType.KEYSTROKE,
                    data={'keys': ['ctrl', 'c']},
                    execution_mode=ExecutionMode.TRIGGER_ONCE
                ),
                created_at=datetime.now()
            )
            
            # Save and retrieve
            repo.save_gesture(gesture)
            loaded = repo.get_gesture("test_trigger")
            
            assert loaded is not None
            assert loaded.action.execution_mode == ExecutionMode.TRIGGER_ONCE
        
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_save_and_load_hold_repeat(self):
        """Test saving and loading gesture with HOLD_REPEAT mode."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        try:
            repo = GestureRepository(db_path)
            
            # Create gesture with HOLD_REPEAT mode
            gesture = Gesture(
                name="test_hold",
                embedding=np.random.rand(16),
                action=Action(
                    type=ActionType.MEDIA_CONTROL,
                    data={'command': 'volume_up'},
                    execution_mode=ExecutionMode.HOLD_REPEAT
                ),
                created_at=datetime.now()
            )
            
            # Save and retrieve
            repo.save_gesture(gesture)
            loaded = repo.get_gesture("test_hold")
            
            assert loaded is not None
            assert loaded.action.execution_mode == ExecutionMode.HOLD_REPEAT
        
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_get_all_preserves_execution_modes(self):
        """Test that get_all_gestures preserves execution modes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        try:
            repo = GestureRepository(db_path)
            
            # Create gestures with different modes
            gesture1 = Gesture(
                name="trigger_gesture",
                embedding=np.random.rand(16),
                action=Action(
                    type=ActionType.LAUNCH_APP,
                    data={'path': 'notepad.exe'},
                    execution_mode=ExecutionMode.TRIGGER_ONCE
                ),
                created_at=datetime.now()
            )
            
            gesture2 = Gesture(
                name="hold_gesture",
                embedding=np.random.rand(16),
                action=Action(
                    type=ActionType.SYSTEM_CONTROL,
                    data={'command': 'volume_up', 'amount': 1},
                    execution_mode=ExecutionMode.HOLD_REPEAT
                ),
                created_at=datetime.now()
            )
            
            # Save both
            repo.save_gesture(gesture1)
            repo.save_gesture(gesture2)
            
            # Retrieve all
            all_gestures = repo.get_all_gestures()
            
            assert len(all_gestures) == 2
            
            # Find each gesture and check mode
            trigger_g = next(g for g in all_gestures if g.name == "trigger_gesture")
            hold_g = next(g for g in all_gestures if g.name == "hold_gesture")
            
            assert trigger_g.action.execution_mode == ExecutionMode.TRIGGER_ONCE
            assert hold_g.action.execution_mode == ExecutionMode.HOLD_REPEAT
        
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
    
    def test_update_gesture_preserves_execution_mode(self):
        """Test that updating a gesture preserves execution mode."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        try:
            repo = GestureRepository(db_path)
            
            # Create and save original gesture
            original = Gesture(
                name="test_update",
                embedding=np.random.rand(16),
                action=Action(
                    type=ActionType.KEYSTROKE,
                    data={'keys': ['a']},
                    execution_mode=ExecutionMode.HOLD_REPEAT
                ),
                created_at=datetime.now()
            )
            repo.save_gesture(original)
            
            # Update with new data but same mode
            updated = Gesture(
                name="test_update",
                embedding=np.random.rand(16),
                action=Action(
                    type=ActionType.KEYSTROKE,
                    data={'keys': ['b']},
                    execution_mode=ExecutionMode.HOLD_REPEAT
                ),
                created_at=datetime.now()
            )
            repo.update_gesture("test_update", updated)
            
            # Retrieve and verify
            loaded = repo.get_gesture("test_update")
            assert loaded is not None
            assert loaded.action.execution_mode == ExecutionMode.HOLD_REPEAT
            assert loaded.action.data == {'keys': ['b']}
        
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
