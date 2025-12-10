"""
Unit tests for HandDetector component.
"""
import pytest
import numpy as np
import cv2
from src.hand_detector import HandDetector
from src.models import HandLandmarks


class TestHandDetectorInitialization:
    """Tests for HandDetector initialization."""
    
    def test_init_with_defaults(self):
        """Test HandDetector initializes with default parameters."""
        detector = HandDetector()
        assert detector.max_hands == 1
        assert detector.detection_confidence == 0.7
    
    def test_init_with_custom_params(self):
        """Test HandDetector initializes with custom parameters."""
        detector = HandDetector(max_hands=2, detection_confidence=0.8)
        assert detector.max_hands == 2
        assert detector.detection_confidence == 0.8
    
    def test_init_invalid_max_hands(self):
        """Test HandDetector rejects invalid max_hands."""
        with pytest.raises(ValueError, match="max_hands must be >= 1"):
            HandDetector(max_hands=0)
    
    def test_init_invalid_confidence_low(self):
        """Test HandDetector rejects confidence below 0.0."""
        with pytest.raises(ValueError, match="detection_confidence must be in"):
            HandDetector(detection_confidence=-0.1)
    
    def test_init_invalid_confidence_high(self):
        """Test HandDetector rejects confidence above 1.0."""
        with pytest.raises(ValueError, match="detection_confidence must be in"):
            HandDetector(detection_confidence=1.1)


class TestHandDetectorDetect:
    """Tests for HandDetector.detect() method."""
    
    def test_detect_empty_frame(self):
        """Test detect handles empty frame gracefully."""
        detector = HandDetector()
        empty_frame = np.array([])
        result = detector.detect(empty_frame)
        assert result == []
    
    def test_detect_none_frame(self):
        """Test detect handles None frame gracefully."""
        detector = HandDetector()
        result = detector.detect(None)
        assert result == []
    
    def test_detect_no_hands(self):
        """Test detect returns empty list when no hands present."""
        detector = HandDetector()
        # Create a blank frame (no hands)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank_frame)
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_detect_returns_list(self):
        """Test detect always returns a list."""
        detector = HandDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, list)


class TestHandDetectorDrawLandmarks:
    """Tests for HandDetector.draw_landmarks() method."""
    
    def test_draw_landmarks_empty_frame(self):
        """Test draw_landmarks handles empty frame."""
        detector = HandDetector()
        empty_frame = np.array([])
        landmarks = HandLandmarks(
            points=np.random.rand(21, 3),
            handedness="Right",
            timestamp=0.0
        )
        result = detector.draw_landmarks(empty_frame, landmarks)
        assert result.size == 0
    
    def test_draw_landmarks_none_frame(self):
        """Test draw_landmarks handles None frame."""
        detector = HandDetector()
        landmarks = HandLandmarks(
            points=np.random.rand(21, 3),
            handedness="Right",
            timestamp=0.0
        )
        result = detector.draw_landmarks(None, landmarks)
        assert result is None
    
    def test_draw_landmarks_returns_frame(self):
        """Test draw_landmarks returns a frame with same shape."""
        detector = HandDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create valid normalized landmarks (MediaPipe uses [0, 1] range)
        landmarks = HandLandmarks(
            points=np.random.rand(21, 3) * 0.5 + 0.25,  # Keep in [0.25, 0.75] range
            handedness="Right",
            timestamp=0.0
        )
        result = detector.draw_landmarks(frame, landmarks)
        assert result.shape == frame.shape
    
    def test_draw_landmarks_does_not_modify_original(self):
        """Test draw_landmarks doesn't modify the original frame."""
        detector = HandDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_frame = frame.copy()
        landmarks = HandLandmarks(
            points=np.random.rand(21, 3) * 0.5 + 0.25,
            handedness="Right",
            timestamp=0.0
        )
        detector.draw_landmarks(frame, landmarks)
        # Original frame should be unchanged
        assert np.array_equal(frame, original_frame)
