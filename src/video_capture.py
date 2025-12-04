"""
Video capture component for the Gesture Control System.

This module provides the VideoCapture class for managing webcam access
and frame delivery.
"""
import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Manages webcam access and frame delivery.
    
    This class wraps OpenCV's VideoCapture functionality to provide
    a simple interface for starting/stopping video capture and retrieving
    frames at a configurable frame rate.
    
    Attributes:
        _capture: OpenCV VideoCapture object
        _fps: Target frames per second
        _is_running: Flag indicating if capture is active
    """
    
    def __init__(self, camera_index: int = 0, fps: int = 30):
        """
        Initialize the VideoCapture component.
        
        Args:
            camera_index: Index of the camera device (default: 0 for primary webcam)
            fps: Target frames per second (default: 30)
        """
        self._camera_index = camera_index
        self._fps = fps
        self._capture: Optional[cv2.VideoCapture] = None
        self._is_running = False
    
    def start(self) -> bool:
        """
        Initialize the webcam and begin capturing frames.
        
        Returns:
            bool: True if webcam was successfully initialized, False otherwise
        """
        try:
            self._capture = cv2.VideoCapture(self._camera_index)
            
            if not self._capture.isOpened():
                logger.error(f"Failed to open camera at index {self._camera_index}")
                return False
            
            # Set the FPS
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)
            
            # Verify the camera is working by reading a test frame
            ret, _ = self._capture.read()
            if not ret:
                logger.error("Camera opened but failed to read frame")
                self._capture.release()
                self._capture = None
                return False
            
            self._is_running = True
            logger.info(f"Video capture started successfully at {self._fps} FPS")
            return True
            
        except Exception as e:
            logger.exception(f"Error starting video capture: {e}")
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            return False
    
    def stop(self) -> None:
        """
        Stop video capture and release the webcam.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        
        self._is_running = False
        logger.info("Video capture stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Retrieve the current frame from the webcam.
        
        Returns:
            Optional[np.ndarray]: The captured frame as a numpy array (BGR format),
                                 or None if capture is not running or frame read fails
        """
        if not self._is_running or self._capture is None:
            logger.warning("Attempted to get frame while capture is not running")
            return None
        
        try:
            ret, frame = self._capture.read()
            
            if not ret or frame is None:
                logger.warning("Failed to read frame from camera")
                return None
            
            return frame
            
        except Exception as e:
            logger.exception(f"Error reading frame: {e}")
            return None
    
    def is_running(self) -> bool:
        """
        Check if video capture is currently active.
        
        Returns:
            bool: True if capture is running, False otherwise
        """
        return self._is_running
    
    def set_fps(self, fps: int) -> None:
        """
        Set the target frames per second for video capture.
        
        Args:
            fps: Target frames per second (must be > 0)
        
        Raises:
            ValueError: If fps is not positive
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        
        self._fps = fps
        
        # If capture is running, update the FPS setting
        if self._capture is not None and self._is_running:
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)
            logger.info(f"FPS updated to {self._fps}")
    
    def __del__(self):
        """Cleanup: ensure camera is released when object is destroyed."""
        self.stop()
