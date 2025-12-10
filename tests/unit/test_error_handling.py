"""
Unit tests for error handling and edge cases.

Tests the error handling features added in task 14:
- Webcam reconnection logic
- Storage corruption recovery
- Comprehensive logging
"""
import pytest
import numpy as np
import time
import sqlite3
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

from src.video_capture import VideoCapture
from src.gesture_repository import GestureRepository
from src.models import Gesture, Action, ActionType
from datetime import datetime


class TestWebcamReconnection:
    """Test webcam reconnection logic (Requirement 9.2)."""
    
    def test_reconnection_state_initialization(self):
        """Test that reconnection state is properly initialized."""
        capture = VideoCapture()
        
        assert not capture.is_disconnected()
        assert capture._reconnect_attempts == 0
        assert capture._max_reconnect_attempts == 10
    
    def test_status_callback_registration(self):
        """Test that status callback can be registered."""
        capture = VideoCapture()
        callback = Mock()
        
        capture.set_status_callback(callback)
        assert capture._status_callback == callback
    
    def test_reset_reconnection_state(self):
        """Test that reconnection state can be reset."""
        capture = VideoCapture()
        
        # Simulate disconnection
        capture._disconnected = True
        capture._reconnect_attempts = 5
        capture._last_reconnect_time = time.time()
        
        # Reset
        capture.reset_reconnection_state()
        
        assert not capture._disconnected
        assert capture._reconnect_attempts == 0
        assert capture._last_reconnect_time == 0.0
    
    @patch('cv2.VideoCapture')
    def test_get_frame_detects_disconnection(self, mock_cv2_capture):
        """Test that get_frame detects camera disconnection."""
        # Setup mock
        mock_capture_instance = MagicMock()
        mock_cv2_capture.return_value = mock_capture_instance
        mock_capture_instance.isOpened.return_value = True
        
        # First read succeeds (for start), then fails (disconnection)
        mock_capture_instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # start() test frame
            (False, None)  # get_frame() failure
        ]
        
        capture = VideoCapture()
        callback = Mock()
        capture.set_status_callback(callback)
        
        # Start capture
        assert capture.start()
        
        # Try to get frame - should detect disconnection
        frame = capture.get_frame()
        
        assert frame is None
        assert capture.is_disconnected()
        callback.assert_called()





class TestStorageCorruptionRecovery:
    """Test storage corruption recovery (Requirement 9.5)."""
    
    def test_recovery_callback_registration(self):
        """Test that recovery callback can be registered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            repo = GestureRepository(db_path)
            
            callback = Mock()
            repo.set_recovery_callback(callback)
            
            assert repo._recovery_callback == callback
    
    def test_corrupted_database_recovery(self):
        """Test that corrupted database is recovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # Create a corrupted database file
            with open(db_path, 'w') as f:
                f.write("This is not a valid SQLite database")
            
            # Initialize repository - should recover
            callback = Mock()
            repo = GestureRepository(db_path)
            repo.set_recovery_callback(callback)
            
            # Force reinitialization to trigger recovery
            repo._initialize_database()
            
            # Should be able to use the database now
            gesture = Gesture(
                name="test",
                embedding=np.array([1.0, 2.0, 3.0]),
                action=Action(ActionType.KEYSTROKE, {"keys": ["a"]}),
                created_at=datetime.now()
            )
            
            # This should work without errors
            repo.save_gesture(gesture)
            retrieved = repo.get_gesture("test")
            assert retrieved is not None
    
    def test_database_error_handling_in_get_gesture(self):
        """Test that database errors are handled gracefully in get_gesture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            repo = GestureRepository(db_path)
            
            # Close and corrupt the database
            os.remove(db_path)
            with open(db_path, 'w') as f:
                f.write("corrupted")
            
            # Should return None instead of crashing
            result = repo.get_gesture("nonexistent")
            assert result is None
    
    def test_database_error_handling_in_get_all_gestures(self):
        """Test that database errors are handled gracefully in get_all_gestures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            repo = GestureRepository(db_path)
            
            # Close and corrupt the database
            os.remove(db_path)
            with open(db_path, 'w') as f:
                f.write("corrupted")
            
            # Should return empty list instead of crashing
            result = repo.get_all_gestures()
            assert result == []


class TestComprehensiveLogging:
    """Test comprehensive logging (Requirement 7.5)."""
    
    def test_logging_in_gesture_matcher(self):
        """Test that GestureMatcher logs initialization."""
        from src.gesture_matcher import GestureMatcher
        import logging
        
        with patch('src.gesture_matcher.logger') as mock_logger:
            matcher = GestureMatcher(threshold=0.5, metric="euclidean")
            
            # Should log initialization
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "initialized" in call_args.lower()
            assert "0.5" in call_args
    
    def test_logging_in_gesture_recorder(self):
        """Test that GestureRecorder logs recording events."""
        from src.gesture_recorder import GestureRecorder
        
        with patch('src.gesture_recorder.logger') as mock_logger:
            recorder = GestureRecorder(duration_seconds=3.0)
            
            # Start recording
            recorder.start_recording()
            mock_logger.info.assert_called()
            
            # Add embedding
            recorder.add_embedding(np.array([1.0, 2.0, 3.0]))
            
            # Stop recording
            recorder.stop_recording()
            
            # Should have logged start and stop
            assert mock_logger.info.call_count >= 2
    
    def test_logging_in_embedding_model(self):
        """Test that EmbeddingModel logs training."""
        from src.embedding_model import EmbeddingModel
        
        with patch('src.embedding_model.logger') as mock_logger:
            model = EmbeddingModel(n_components=8)
            
            # Fit the model
            training_data = [np.random.rand(21, 3) for _ in range(20)]
            model.fit(training_data)
            
            # Should log training completion
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0][0]
            assert "fitted" in call_args.lower() or "trained" in call_args.lower()
    
    def test_logging_in_video_capture(self):
        """Test that VideoCapture logs important events."""
        from src.video_capture import VideoCapture
        
        with patch('src.video_capture.logger') as mock_logger:
            capture = VideoCapture()
            
            # Call stop which should log
            capture.stop()
            
            # Should have logged the stop event
            assert mock_logger.info.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
