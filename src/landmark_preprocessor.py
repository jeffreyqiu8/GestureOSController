"""
Landmark preprocessing component for normalization and smoothing.
"""
import numpy as np
import logging
from collections import deque
from typing import Optional

from src.models import HandLandmarks

logger = logging.getLogger(__name__)


class LandmarkPreprocessor:
    """
    Preprocesses hand landmarks through normalization and smoothing.
    
    This component normalizes landmarks to be scale and position invariant,
    and applies temporal smoothing to reduce jitter in the landmark data.
    
    Attributes:
        smoothing_window: Number of frames to use for moving average smoothing
    """
    
    def __init__(self, smoothing_window: int = 5):
        """
        Initialize the LandmarkPreprocessor.
        
        Args:
            smoothing_window: Number of frames for moving average filter (default: 5)
        
        Raises:
            ValueError: If smoothing_window is less than 1
        """
        if smoothing_window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}")
        
        self.smoothing_window = smoothing_window
        self._landmark_history = deque(maxlen=smoothing_window)
        logger.debug(f"LandmarkPreprocessor initialized with smoothing_window={smoothing_window}")
    
    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to be scale and position invariant.
        
        Normalization process:
        1. Center coordinates relative to wrist (landmark 0)
        2. Calculate palm size (distance from wrist to middle finger base)
        3. Scale all coordinates by 1/palm_size
        
        Args:
            landmarks: Array of shape (21, 3) containing landmark coordinates
        
        Returns:
            np.ndarray: Normalized landmarks of shape (21, 3)
        
        Requirements: 2.2, 2.3
        """
        if landmarks.shape != (21, 3):
            raise ValueError(f"Landmarks must have shape (21, 3), got {landmarks.shape}")
        
        # Step 1: Center on wrist (landmark 0)
        wrist = landmarks[0].copy()
        centered = landmarks - wrist
        
        # Step 2: Calculate palm size (distance from wrist to middle finger base - landmark 9)
        middle_finger_base = landmarks[9]
        palm_size = np.linalg.norm(middle_finger_base - wrist)
        
        # Handle edge case where palm_size is very small or zero
        if palm_size < 1e-6:
            # Return centered landmarks without scaling if palm size is too small
            return centered
        
        # Step 3: Scale by 1/palm_size
        normalized = centered / palm_size
        
        return normalized
    
    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing to reduce jitter.
        
        Uses a moving average filter over the last N frames (where N is smoothing_window)
        to smooth the landmark coordinates and reduce noise.
        
        Args:
            landmarks: Array of shape (21, 3) containing landmark coordinates
        
        Returns:
            np.ndarray: Smoothed landmarks of shape (21, 3)
        
        Requirements: 8.1
        """
        if landmarks.shape != (21, 3):
            raise ValueError(f"Landmarks must have shape (21, 3), got {landmarks.shape}")
        
        # Add current landmarks to history
        self._landmark_history.append(landmarks.copy())
        
        # Compute moving average over all frames in history
        smoothed = np.mean(np.array(self._landmark_history), axis=0)
        
        return smoothed
    
    def preprocess(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline: normalization followed by smoothing.
        
        Combines normalize() and smooth() to produce landmarks that are both
        scale/position invariant and temporally smoothed.
        
        Args:
            landmarks: Array of shape (21, 3) containing landmark coordinates
        
        Returns:
            np.ndarray: Preprocessed landmarks of shape (21, 3)
        
        Requirements: 2.2, 2.3, 8.1
        """
        # First normalize
        normalized = self.normalize(landmarks)
        
        # Then smooth
        smoothed = self.smooth(normalized)
        
        return smoothed
    
    def reset(self) -> None:
        """
        Reset the smoothing history.
        
        Clears the internal landmark history buffer. Useful when starting
        a new recording session or when there's a discontinuity in the video stream.
        """
        self._landmark_history.clear()
        logger.debug("Landmark history reset")
