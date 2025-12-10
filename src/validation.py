"""
Input validation functions for the Gesture Control System.
"""
from typing import List, Optional
from src.models import Config


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


def validate_gesture_name(name: str, existing_names: Optional[List[str]] = None) -> None:
    """
    Validate a gesture name according to requirements.
    
    Requirements validated:
    - Non-empty
    - Not whitespace-only
    - Not a duplicate of existing names
    
    Args:
        name: The gesture name to validate
        existing_names: Optional list of existing gesture names to check for duplicates
        
    Raises:
        ValidationError: If the name is invalid
        
    Requirements: 3.2
    """
    # Check if name is empty
    if not name:
        raise ValidationError("Gesture name cannot be empty")
    
    # Check if name is whitespace-only
    if name.strip() == "":
        raise ValidationError("Gesture name cannot be whitespace-only")
    
    # Check for duplicates if existing names provided
    if existing_names is not None and name in existing_names:
        raise ValidationError(f"Gesture name '{name}' already exists")


def validate_similarity_threshold(threshold: float) -> None:
    """
    Validate similarity threshold value.
    
    Requirements validated:
    - Threshold must be between 0.0 and 1.0 (inclusive)
    
    Args:
        threshold: The similarity threshold value to validate
        
    Raises:
        ValidationError: If the threshold is out of bounds
        
    Requirements: 10.2, 10.5
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError("Similarity threshold must be a number")
    
    if threshold < 0.0 or threshold > 1.0:
        raise ValidationError(f"Similarity threshold must be between 0.0 and 1.0, got {threshold}")


def validate_recording_duration(duration: float) -> None:
    """
    Validate recording duration value.
    
    Requirements validated:
    - Duration must be greater than 0
    
    Args:
        duration: The recording duration in seconds to validate
        
    Raises:
        ValidationError: If the duration is invalid
        
    Requirements: 10.2, 10.5
    """
    if not isinstance(duration, (int, float)):
        raise ValidationError("Recording duration must be a number")
    
    if duration <= 0:
        raise ValidationError(f"Recording duration must be greater than 0, got {duration}")


def validate_fps(fps: int) -> None:
    """
    Validate frames per second value.
    
    Requirements validated:
    - FPS must be greater than 0
    
    Args:
        fps: The frames per second value to validate
        
    Raises:
        ValidationError: If the FPS is invalid
        
    Requirements: 10.2, 10.5
    """
    if not isinstance(fps, int):
        raise ValidationError("FPS must be an integer")
    
    if fps <= 0:
        raise ValidationError(f"FPS must be greater than 0, got {fps}")


def validate_detection_confidence(confidence: float) -> None:
    """
    Validate detection confidence value.
    
    Requirements validated:
    - Confidence must be between 0.0 and 1.0 (inclusive)
    
    Args:
        confidence: The detection confidence value to validate
        
    Raises:
        ValidationError: If the confidence is out of bounds
        
    Requirements: 10.2, 10.5
    """
    if not isinstance(confidence, (int, float)):
        raise ValidationError("Detection confidence must be a number")
    
    if confidence < 0.0 or confidence > 1.0:
        raise ValidationError(f"Detection confidence must be between 0.0 and 1.0, got {confidence}")


def validate_config(config: Config) -> None:
    """
    Validate all configuration parameters.
    
    Args:
        config: The configuration object to validate
        
    Raises:
        ValidationError: If any configuration parameter is invalid
        
    Requirements: 10.2, 10.5
    """
    validate_similarity_threshold(config.similarity_threshold)
    validate_recording_duration(config.recording_duration)
    validate_fps(config.fps)
    validate_detection_confidence(config.detection_confidence)
    
    # Additional validations for other config parameters
    if not isinstance(config.smoothing_window, int) or config.smoothing_window <= 0:
        raise ValidationError(f"Smoothing window must be a positive integer, got {config.smoothing_window}")
    
    if not isinstance(config.max_hands, int) or config.max_hands <= 0:
        raise ValidationError(f"Max hands must be a positive integer, got {config.max_hands}")
    
    if not isinstance(config.embedding_dimensions, int) or config.embedding_dimensions <= 0:
        raise ValidationError(f"Embedding dimensions must be a positive integer, got {config.embedding_dimensions}")
