"""
Unit tests for validation functions.
"""
import pytest
from src.validation import (
    ValidationError,
    validate_gesture_name,
    validate_similarity_threshold,
    validate_recording_duration,
    validate_fps,
    validate_detection_confidence,
    validate_config
)
from src.models import Config


class TestGestureNameValidation:
    """Tests for gesture name validation."""
    
    def test_valid_gesture_name(self):
        """Test that valid gesture names pass validation."""
        # Should not raise any exception
        validate_gesture_name("thumbs_up")
        validate_gesture_name("peace")
        validate_gesture_name("ok sign")
        validate_gesture_name("123")
    
    def test_empty_gesture_name(self):
        """Test that empty gesture names are rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_gesture_name("")
    
    def test_whitespace_only_gesture_name(self):
        """Test that whitespace-only gesture names are rejected."""
        with pytest.raises(ValidationError, match="cannot be whitespace-only"):
            validate_gesture_name("   ")
        
        with pytest.raises(ValidationError, match="cannot be whitespace-only"):
            validate_gesture_name("\t\n")
    
    def test_duplicate_gesture_name(self):
        """Test that duplicate gesture names are rejected."""
        existing_names = ["thumbs_up", "peace", "wave"]
        
        with pytest.raises(ValidationError, match="already exists"):
            validate_gesture_name("thumbs_up", existing_names)
        
        with pytest.raises(ValidationError, match="already exists"):
            validate_gesture_name("peace", existing_names)
    
    def test_non_duplicate_gesture_name(self):
        """Test that non-duplicate names pass validation."""
        existing_names = ["thumbs_up", "peace", "wave"]
        
        # Should not raise any exception
        validate_gesture_name("ok_sign", existing_names)
        validate_gesture_name("fist", existing_names)


class TestSimilarityThresholdValidation:
    """Tests for similarity threshold validation."""
    
    def test_valid_threshold(self):
        """Test that valid thresholds pass validation."""
        validate_similarity_threshold(0.0)
        validate_similarity_threshold(0.5)
        validate_similarity_threshold(1.0)
        validate_similarity_threshold(0.25)
        validate_similarity_threshold(0.99)
    
    def test_threshold_below_zero(self):
        """Test that thresholds below 0 are rejected."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_similarity_threshold(-0.1)
        
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_similarity_threshold(-1.0)
    
    def test_threshold_above_one(self):
        """Test that thresholds above 1 are rejected."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_similarity_threshold(1.1)
        
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_similarity_threshold(2.0)
    
    def test_threshold_non_numeric(self):
        """Test that non-numeric thresholds are rejected."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_similarity_threshold("0.5")
        
        with pytest.raises(ValidationError, match="must be a number"):
            validate_similarity_threshold(None)


class TestRecordingDurationValidation:
    """Tests for recording duration validation."""
    
    def test_valid_duration(self):
        """Test that valid durations pass validation."""
        validate_recording_duration(1.0)
        validate_recording_duration(3.0)
        validate_recording_duration(10.5)
        validate_recording_duration(0.1)
    
    def test_zero_duration(self):
        """Test that zero duration is rejected."""
        with pytest.raises(ValidationError, match="must be greater than 0"):
            validate_recording_duration(0.0)
    
    def test_negative_duration(self):
        """Test that negative durations are rejected."""
        with pytest.raises(ValidationError, match="must be greater than 0"):
            validate_recording_duration(-1.0)
        
        with pytest.raises(ValidationError, match="must be greater than 0"):
            validate_recording_duration(-0.5)
    
    def test_duration_non_numeric(self):
        """Test that non-numeric durations are rejected."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_recording_duration("3.0")
        
        with pytest.raises(ValidationError, match="must be a number"):
            validate_recording_duration(None)


class TestFPSValidation:
    """Tests for FPS validation."""
    
    def test_valid_fps(self):
        """Test that valid FPS values pass validation."""
        validate_fps(30)
        validate_fps(60)
        validate_fps(1)
        validate_fps(120)
    
    def test_zero_fps(self):
        """Test that zero FPS is rejected."""
        with pytest.raises(ValidationError, match="must be greater than 0"):
            validate_fps(0)
    
    def test_negative_fps(self):
        """Test that negative FPS values are rejected."""
        with pytest.raises(ValidationError, match="must be greater than 0"):
            validate_fps(-30)
        
        with pytest.raises(ValidationError, match="must be greater than 0"):
            validate_fps(-1)
    
    def test_fps_non_integer(self):
        """Test that non-integer FPS values are rejected."""
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_fps(30.5)
        
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_fps("30")


class TestDetectionConfidenceValidation:
    """Tests for detection confidence validation."""
    
    def test_valid_confidence(self):
        """Test that valid confidence values pass validation."""
        validate_detection_confidence(0.0)
        validate_detection_confidence(0.5)
        validate_detection_confidence(1.0)
        validate_detection_confidence(0.7)
    
    def test_confidence_below_zero(self):
        """Test that confidence below 0 is rejected."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_detection_confidence(-0.1)
    
    def test_confidence_above_one(self):
        """Test that confidence above 1 is rejected."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            validate_detection_confidence(1.1)
    
    def test_confidence_non_numeric(self):
        """Test that non-numeric confidence values are rejected."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_detection_confidence("0.7")


class TestConfigValidation:
    """Tests for complete configuration validation."""
    
    def test_valid_config(self):
        """Test that valid configuration passes validation."""
        config = Config(
            similarity_threshold=0.5,
            recording_duration=3.0,
            fps=30,
            smoothing_window=5,
            max_hands=1,
            detection_confidence=0.7,
            embedding_dimensions=16
        )
        # Should not raise any exception
        validate_config(config)
    
    def test_invalid_similarity_threshold(self):
        """Test that invalid similarity threshold is caught."""
        config = Config(similarity_threshold=1.5)
        with pytest.raises(ValidationError, match="Similarity threshold"):
            validate_config(config)
    
    def test_invalid_recording_duration(self):
        """Test that invalid recording duration is caught."""
        config = Config(recording_duration=-1.0)
        with pytest.raises(ValidationError, match="Recording duration"):
            validate_config(config)
    
    def test_invalid_fps(self):
        """Test that invalid FPS is caught."""
        config = Config(fps=0)
        with pytest.raises(ValidationError, match="FPS"):
            validate_config(config)
    
    def test_invalid_detection_confidence(self):
        """Test that invalid detection confidence is caught."""
        config = Config(detection_confidence=2.0)
        with pytest.raises(ValidationError, match="Detection confidence"):
            validate_config(config)
    
    def test_invalid_smoothing_window(self):
        """Test that invalid smoothing window is caught."""
        config = Config(smoothing_window=0)
        with pytest.raises(ValidationError, match="Smoothing window"):
            validate_config(config)
    
    def test_invalid_max_hands(self):
        """Test that invalid max hands is caught."""
        config = Config(max_hands=-1)
        with pytest.raises(ValidationError, match="Max hands"):
            validate_config(config)
    
    def test_invalid_embedding_dimensions(self):
        """Test that invalid embedding dimensions is caught."""
        config = Config(embedding_dimensions=0)
        with pytest.raises(ValidationError, match="Embedding dimensions"):
            validate_config(config)
