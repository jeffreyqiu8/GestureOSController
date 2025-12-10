"""
Video capture component for the Gesture Control System.

This module provides the VideoCapture class for managing webcam access
and frame delivery.
"""
import cv2
import numpy as np
import logging
import time
from typing import Optional, Callable

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
        
        # Reconnection state
        self._disconnected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._last_reconnect_time = 0.0
        self._reconnect_backoff_base = 1.0  # Base delay in seconds
        self._status_callback: Optional[Callable[[str], None]] = None
    
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
        
        Automatically attempts reconnection if the camera becomes disconnected.
        
        Returns:
            Optional[np.ndarray]: The captured frame as a numpy array (BGR format),
                                 or None if capture is not running or frame read fails
        
        Requirements: 9.2
        """
        if not self._is_running or self._capture is None:
            logger.warning("Attempted to get frame while capture is not running")
            return None
        
        try:
            ret, frame = self._capture.read()
            
            if not ret or frame is None:
                # Camera disconnection detected
                if not self._disconnected:
                    logger.warning("Camera disconnection detected")
                    self._disconnected = True
                    self._reconnect_attempts = 0
                    self._notify_status("Camera disconnected - attempting reconnection...")
                
                # Attempt reconnection
                if self._attempt_reconnection():
                    # Successfully reconnected, try reading again
                    ret, frame = self._capture.read()
                    if ret and frame is not None:
                        return frame
                
                return None
            
            # Successfully read frame - reset disconnection state if we were disconnected
            if self._disconnected:
                logger.info("Camera reconnected successfully")
                self._disconnected = False
                self._reconnect_attempts = 0
                self._notify_status("Camera reconnected successfully")
            
            return frame
            
        except Exception as e:
            logger.exception(f"Error reading frame: {e}")
            
            # Mark as disconnected and attempt reconnection
            if not self._disconnected:
                self._disconnected = True
                self._reconnect_attempts = 0
                self._notify_status("Camera error - attempting reconnection...")
            
            self._attempt_reconnection()
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
    
    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set a callback function for status notifications.
        
        Args:
            callback: Function that takes a status message string
        """
        self._status_callback = callback
    
    def _notify_status(self, message: str) -> None:
        """
        Notify status change via callback if set.
        
        Args:
            message: Status message to send
        """
        if self._status_callback:
            self._status_callback(message)
    
    def _attempt_reconnection(self) -> bool:
        """
        Attempt to reconnect to the camera with exponential backoff.
        
        Uses exponential backoff strategy: delay = base * (2 ^ attempt_number)
        
        Returns:
            bool: True if reconnection succeeded, False otherwise
        
        Requirements: 9.2
        """
        # Check if we've exceeded max attempts
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self._max_reconnect_attempts}) reached")
            self._notify_status(f"Camera reconnection failed after {self._max_reconnect_attempts} attempts")
            return False
        
        # Calculate backoff delay
        current_time = time.time()
        backoff_delay = self._reconnect_backoff_base * (2 ** self._reconnect_attempts)
        
        # Check if enough time has passed since last attempt
        if current_time - self._last_reconnect_time < backoff_delay:
            # Not enough time has passed, skip this attempt
            return False
        
        # Update reconnection state
        self._reconnect_attempts += 1
        self._last_reconnect_time = current_time
        
        logger.info(f"Attempting camera reconnection (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}, "
                   f"backoff: {backoff_delay:.1f}s)")
        self._notify_status(f"Reconnecting... (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})")
        
        try:
            # Release existing capture if any
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            
            # Try to reinitialize the camera
            self._capture = cv2.VideoCapture(self._camera_index)
            
            if not self._capture.isOpened():
                logger.warning(f"Reconnection attempt {self._reconnect_attempts} failed: camera not opened")
                return False
            
            # Set the FPS
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)
            
            # Verify the camera is working by reading a test frame
            ret, frame = self._capture.read()
            if not ret or frame is None:
                logger.warning(f"Reconnection attempt {self._reconnect_attempts} failed: cannot read frame")
                self._capture.release()
                self._capture = None
                return False
            
            # Success!
            logger.info(f"Camera reconnected successfully on attempt {self._reconnect_attempts}")
            self._disconnected = False
            self._reconnect_attempts = 0
            self._notify_status("Camera reconnected successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error during reconnection attempt {self._reconnect_attempts}: {e}")
            if self._capture is not None:
                self._capture.release()
                self._capture = None
            return False
    
    def is_disconnected(self) -> bool:
        """
        Check if the camera is currently disconnected.
        
        Returns:
            bool: True if disconnected, False otherwise
        """
        return self._disconnected
    
    def reset_reconnection_state(self) -> None:
        """
        Reset the reconnection state.
        
        Useful for manual reconnection attempts or after fixing camera issues.
        """
        self._disconnected = False
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0.0
        logger.info("Reconnection state reset")
    
    def __del__(self):
        """Cleanup: ensure camera is released when object is destroyed."""
        self.stop()
