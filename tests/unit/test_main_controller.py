"""
Unit tests for the MainController component.
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.main_controller import MainController, SystemState
from src.models import Config, Action, ActionType, Gesture


class TestMainControllerInitialization:
    """Test MainController initialization."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        controller = MainController()
        
        assert controller.config is not None
        assert controller.state == SystemState.IDLE
        assert not controller.is_running()
        assert controller.video_capture is not None
        assert controller.hand_detector is not None
        assert controller.preprocessor is not None
        assert controller.embedding_model is not None
        assert controller.gesture_recorder is not None
        assert controller.gesture_matcher is not None
        assert controller.gesture_repository is not None
        assert controller.action_executor is not None
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = Config(
            fps=60,
            similarity_threshold=0.3,
            recording_duration=5.0
        )
        controller = MainController(config)
        
        assert controller.config.fps == 60
        assert controller.config.similarity_threshold == 0.3
        assert controller.config.recording_duration == 5.0


class TestMainControllerStartStop:
    """Test MainController start and stop operations."""
    
    @patch('src.main_controller.VideoCapture')
    def test_start_success(self, mock_video_capture_class):
        """Test successful start."""
        # Mock video capture to return True on start
        mock_video_capture = Mock()
        mock_video_capture.start.return_value = True
        mock_video_capture_class.return_value = mock_video_capture
        
        controller = MainController()
        controller.video_capture = mock_video_capture
        
        result = controller.start()
        
        assert result is True
        assert controller.is_running()
        assert controller.state == SystemState.RECOGNIZING
        mock_video_capture.start.assert_called_once()
    
    @patch('src.main_controller.VideoCapture')
    def test_start_failure(self, mock_video_capture_class):
        """Test start failure when video capture fails."""
        # Mock video capture to return False on start
        mock_video_capture = Mock()
        mock_video_capture.start.return_value = False
        mock_video_capture_class.return_value = mock_video_capture
        
        controller = MainController()
        controller.video_capture = mock_video_capture
        
        result = controller.start()
        
        assert result is False
        assert not controller.is_running()
    
    def test_stop(self):
        """Test stop operation."""
        controller = MainController()
        controller._is_running = True
        controller.state = SystemState.RECOGNIZING
        controller.video_capture = Mock()
        
        controller.stop()
        
        assert not controller.is_running()
        assert controller.state == SystemState.IDLE
        controller.video_capture.stop.assert_called_once()


class TestRecordingMode:
    """Test recording mode operations."""
    
    def test_enter_recording_mode(self):
        """Test entering recording mode."""
        controller = MainController()
        controller._is_running = True
        controller.state = SystemState.RECOGNIZING
        
        result = controller.enter_recording_mode()
        
        assert result is True
        assert controller.state == SystemState.RECORDING
        assert controller.gesture_recorder.is_recording()
    
    def test_enter_recording_mode_with_custom_duration(self):
        """Test entering recording mode with custom duration."""
        controller = MainController()
        controller._is_running = True
        controller.state = SystemState.RECOGNIZING
        
        result = controller.enter_recording_mode(duration=5.0)
        
        assert result is True
        assert controller.gesture_recorder.duration_seconds == 5.0
    
    def test_enter_recording_mode_when_not_running(self):
        """Test entering recording mode when controller not running."""
        controller = MainController()
        controller._is_running = False
        
        result = controller.enter_recording_mode()
        
        assert result is False
        assert controller.state == SystemState.IDLE
    
    def test_save_recorded_gesture(self):
        """Test saving a recorded gesture."""
        controller = MainController()
        controller._is_running = True
        controller.state = SystemState.RECORDING
        
        # Start recording
        controller.gesture_recorder.start_recording()
        
        # Simulate collecting landmarks during recording (cold start scenario)
        landmarks = [np.random.rand(21, 3) for _ in range(10)]
        controller._recording_landmarks = landmarks
        
        # Mock the repository
        controller.gesture_repository = Mock()
        controller.gesture_repository.save_gesture = Mock()
        
        # Create action
        action = Action(type=ActionType.KEYSTROKE, data={'keys': ['ctrl', 'c']})
        
        result = controller.save_recorded_gesture("test_gesture", action)
        
        assert result is True
        assert controller.state == SystemState.RECOGNIZING
        assert "test_gesture" in controller._gesture_prototypes
        assert controller.embedding_model.is_fitted  # Model should be fitted now
        controller.gesture_repository.save_gesture.assert_called_once()
    
    def test_save_recorded_gesture_when_not_recording(self):
        """Test saving gesture when not in recording mode."""
        controller = MainController()
        controller.state = SystemState.RECOGNIZING
        
        action = Action(type=ActionType.KEYSTROKE, data={'keys': ['ctrl', 'c']})
        result = controller.save_recorded_gesture("test_gesture", action)
        
        assert result is False
    
    def test_cancel_recording(self):
        """Test cancelling a recording."""
        controller = MainController()
        controller._is_running = True
        controller.state = SystemState.RECORDING
        controller.gesture_recorder.start_recording()
        
        result = controller.cancel_recording()
        
        assert result is True
        assert controller.state == SystemState.RECOGNIZING
        assert not controller.gesture_recorder.is_recording()
    
    def test_get_recording_progress(self):
        """Test getting recording progress."""
        controller = MainController()
        controller._is_running = True
        controller.state = SystemState.RECORDING
        controller.gesture_recorder.start_recording()
        
        progress = controller.get_recording_progress()
        
        assert 0.0 <= progress <= 1.0


class TestGestureManagement:
    """Test gesture management operations."""
    
    def test_delete_gesture(self):
        """Test deleting a gesture."""
        controller = MainController()
        
        # Add a gesture to cache
        controller._gesture_prototypes["test_gesture"] = np.random.rand(16)
        
        # Mock repository
        controller.gesture_repository = Mock()
        controller.gesture_repository.delete_gesture.return_value = True
        
        result = controller.delete_gesture("test_gesture")
        
        assert result is True
        assert "test_gesture" not in controller._gesture_prototypes
        controller.gesture_repository.delete_gesture.assert_called_once_with("test_gesture")
    
    def test_delete_nonexistent_gesture(self):
        """Test deleting a gesture that doesn't exist."""
        controller = MainController()
        
        # Mock repository
        controller.gesture_repository = Mock()
        controller.gesture_repository.delete_gesture.return_value = False
        
        result = controller.delete_gesture("nonexistent")
        
        assert result is False
    
    def test_update_settings(self):
        """Test updating system settings."""
        controller = MainController()
        controller._is_running = True
        
        # Mock video capture
        controller.video_capture = Mock()
        
        new_config = Config(
            fps=60,
            similarity_threshold=0.3,
            recording_duration=5.0
        )
        
        result = controller.update_settings(new_config)
        
        assert result is True
        assert controller.config.fps == 60
        assert controller.config.similarity_threshold == 0.3
        assert controller.config.recording_duration == 5.0
    
    def test_get_all_gestures(self):
        """Test getting all gestures."""
        controller = MainController()
        
        # Mock repository
        mock_gestures = [
            Gesture(
                name="gesture1",
                embedding=np.random.rand(16),
                action=Action(type=ActionType.KEYSTROKE, data={'keys': ['a']}),
                created_at=datetime.now()
            )
        ]
        controller.gesture_repository = Mock()
        controller.gesture_repository.get_all_gestures.return_value = mock_gestures
        
        gestures = controller.get_all_gestures()
        
        assert len(gestures) == 1
        assert gestures[0].name == "gesture1"
    
    def test_get_gesture(self):
        """Test getting a specific gesture."""
        controller = MainController()
        
        # Mock repository
        mock_gesture = Gesture(
            name="test_gesture",
            embedding=np.random.rand(16),
            action=Action(type=ActionType.KEYSTROKE, data={'keys': ['a']}),
            created_at=datetime.now()
        )
        controller.gesture_repository = Mock()
        controller.gesture_repository.get_gesture.return_value = mock_gesture
        
        gesture = controller.get_gesture("test_gesture")
        
        assert gesture is not None
        assert gesture.name == "test_gesture"


class TestFrameProcessing:
    """Test frame processing operations."""
    
    def test_process_frame_when_not_running(self):
        """Test processing frame when controller not running."""
        controller = MainController()
        controller._is_running = False
        
        result = controller.process_frame()
        
        assert result is None
    
    @patch('src.main_controller.VideoCapture')
    def test_process_frame_no_hands_detected(self, mock_video_capture_class):
        """Test processing frame when no hands detected."""
        controller = MainController()
        controller._is_running = True
        
        # Mock video capture to return a frame
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        controller.video_capture = Mock()
        controller.video_capture.get_frame.return_value = mock_frame
        
        # Mock hand detector to return no hands
        controller.hand_detector = Mock()
        controller.hand_detector.detect.return_value = []
        
        result = controller.process_frame()
        
        assert result is not None
        assert result.shape == mock_frame.shape
