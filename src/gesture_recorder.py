"""
Gesture recording component for capturing and collecting gesture embeddings.
"""
import time
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class GestureRecorder:
    """
    Manages gesture recording sessions by collecting embeddings over a specified duration.
    
    This component tracks recording progress and collects embeddings from consecutive
    frames during a recording session. The collected embeddings can then be averaged
    to create a prototype vector for gesture recognition.
    
    Attributes:
        duration_seconds: Duration of the recording session in seconds
    
    Requirements: 2.1
    """
    
    def __init__(self, duration_seconds: float = 3.0):
        """
        Initialize the GestureRecorder.
        
        Args:
            duration_seconds: Duration for recording sessions (default: 3.0 seconds)
        
        Raises:
            ValueError: If duration_seconds is not positive
        """
        if duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be positive, got {duration_seconds}")
        
        self.duration_seconds = duration_seconds
        self._is_recording = False
        self._start_time: Optional[float] = None
        self._embeddings: List[np.ndarray] = []
    
    def start_recording(self) -> None:
        """
        Begin a new recording session.
        
        Initializes the recording state, clears any previous embeddings,
        and starts the timer for tracking recording duration.
        
        Requirements: 2.1
        """
        self._is_recording = True
        self._start_time = time.time()
        self._embeddings = []
        logger.info(f"Recording started (duration: {self.duration_seconds}s)")
    
    def add_embedding(self, embedding: np.ndarray) -> None:
        """
        Add an embedding to the current recording session.
        
        This method should be called for each frame during recording to collect
        embeddings. Only adds embeddings if a recording session is active.
        
        Args:
            embedding: Embedding vector to add to the recording
        
        Raises:
            RuntimeError: If called when not recording
        """
        if not self._is_recording:
            raise RuntimeError("Cannot add embedding when not recording. Call start_recording() first.")
        
        self._embeddings.append(embedding.copy())
    
    def stop_recording(self) -> List[np.ndarray]:
        """
        Stop the recording session and return collected embeddings.
        
        Ends the current recording session and returns all embeddings that were
        collected during the session. The embeddings can then be averaged to
        create a prototype vector.
        
        Returns:
            List[np.ndarray]: List of collected embedding vectors
        
        Raises:
            RuntimeError: If called when not recording
        
        Requirements: 2.1
        """
        if not self._is_recording:
            raise RuntimeError("Cannot stop recording when not recording. Call start_recording() first.")
        
        self._is_recording = False
        embeddings = self._embeddings.copy()
        
        logger.info(f"Recording stopped. Collected {len(embeddings)} embeddings")
        
        # Clear internal state
        self._start_time = None
        self._embeddings = []
        
        return embeddings
    
    def is_recording(self) -> bool:
        """
        Check if a recording session is currently active.
        
        Returns:
            bool: True if recording is in progress, False otherwise
        
        Requirements: 2.1
        """
        return self._is_recording
    
    def get_progress(self) -> float:
        """
        Get the current recording progress as a fraction.
        
        Returns a value between 0.0 and 1.0 indicating how much of the
        recording duration has elapsed. Returns 0.0 if not recording.
        
        Returns:
            float: Progress fraction (0.0 to 1.0), or 0.0 if not recording
        
        Requirements: 2.1
        """
        if not self._is_recording or self._start_time is None:
            return 0.0
        
        elapsed = time.time() - self._start_time
        progress = min(elapsed / self.duration_seconds, 1.0)
        
        return progress
    
    def is_complete(self) -> bool:
        """
        Check if the recording duration has been reached.
        
        Returns:
            bool: True if recording duration has elapsed, False otherwise
        """
        if not self._is_recording:
            return False
        
        return self.get_progress() >= 1.0
    
    def get_embedding_count(self) -> int:
        """
        Get the number of embeddings collected so far.
        
        Returns:
            int: Number of embeddings in the current recording session
        """
        return len(self._embeddings)



def compute_prototype_vector(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute a prototype vector by averaging a list of embeddings.
    
    Takes multiple embeddings from a recorded gesture and computes their
    element-wise mean to create a single prototype vector that represents
    the gesture. This prototype is used for matching during recognition.
    
    Args:
        embeddings: List of embedding vectors to average
    
    Returns:
        np.ndarray: Prototype vector (mean of all input embeddings)
    
    Raises:
        ValueError: If embeddings list is empty or embeddings have inconsistent shapes
    
    Requirements: 2.5
    """
    if not embeddings:
        raise ValueError("Cannot compute prototype from empty embeddings list")
    
    # Verify all embeddings have the same shape
    first_shape = embeddings[0].shape
    for i, emb in enumerate(embeddings):
        if emb.shape != first_shape:
            raise ValueError(
                f"All embeddings must have the same shape. "
                f"Expected {first_shape}, got {emb.shape} at index {i}"
            )
    
    # Compute element-wise mean
    prototype = np.mean(embeddings, axis=0)
    
    return prototype
