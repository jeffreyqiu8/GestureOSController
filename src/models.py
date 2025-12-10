"""
Data models for the Gesture Control System.
"""
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import json


class ActionType(Enum):
    """Enumeration of available OS action types."""
    LAUNCH_APP = "launch_app"
    KEYSTROKE = "keystroke"
    MEDIA_CONTROL = "media_control"
    SYSTEM_CONTROL = "system_control"


class ExecutionMode(Enum):
    """Enumeration of action execution modes."""
    TRIGGER_ONCE = "trigger_once"  # Execute once when gesture is first detected
    HOLD_REPEAT = "hold_repeat"    # Execute repeatedly while gesture is held


@dataclass
class Action:
    """
    Represents an OS-level action to be executed.
    
    Attributes:
        type: The type of action (from ActionType enum)
        data: Dictionary containing action-specific parameters
        execution_mode: How the action should be executed (trigger once or hold to repeat)
    """
    type: ActionType
    data: Dict[str, Any]
    execution_mode: ExecutionMode = ExecutionMode.TRIGGER_ONCE
    
    def to_json(self) -> str:
        """
        Serialize action to JSON string.
        
        Returns:
            str: JSON representation of the action
        """
        return json.dumps({
            'type': self.type.value,
            'data': self.data,
            'execution_mode': self.execution_mode.value
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Action':
        """
        Deserialize action from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            Action: Deserialized action object
        """
        obj = json.loads(json_str)
        return cls(
            type=ActionType(obj['type']),
            data=obj['data'],
            execution_mode=ExecutionMode(obj.get('execution_mode', 'trigger_once'))
        )


@dataclass
class HandLandmarks:
    """
    Represents hand landmark data from MediaPipe.
    
    Attributes:
        points: Array of shape (21, 3) containing x, y, z coordinates for 21 landmarks
        handedness: String indicating "Left" or "Right" hand
        timestamp: Timestamp when landmarks were captured
    """
    points: np.ndarray  # Shape: (21, 3)
    handedness: str     # "Left" or "Right"
    timestamp: float
    
    def __post_init__(self):
        """Validate the shape of points array."""
        if self.points.shape != (21, 3):
            raise ValueError(f"Points must have shape (21, 3), got {self.points.shape}")
    
    def flatten(self) -> np.ndarray:
        """
        Flatten the landmarks into a 1D array.
        
        Returns:
            np.ndarray: Flattened array of shape (63,)
        """
        return self.points.flatten()
    
    def get_wrist(self) -> np.ndarray:
        """
        Get the wrist landmark coordinates.
        
        Returns:
            np.ndarray: Wrist coordinates of shape (3,) - landmark point 0
        """
        return self.points[0]
    
    def get_palm_size(self) -> float:
        """
        Calculate palm size as the distance from wrist to middle finger base.
        
        Returns:
            float: Euclidean distance between wrist (landmark 0) and middle finger base (landmark 9)
        """
        wrist = self.points[0]
        middle_finger_base = self.points[9]
        return np.linalg.norm(middle_finger_base - wrist)
    
    @classmethod
    def from_flat_array(cls, flat_array: np.ndarray, handedness: str = "Right", timestamp: Optional[float] = None) -> 'HandLandmarks':
        """
        Create HandLandmarks from a flattened array.
        
        Args:
            flat_array: Array of shape (63,)
            handedness: Hand type ("Left" or "Right")
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            HandLandmarks: New instance
        """
        if flat_array.shape != (63,):
            raise ValueError(f"Flat array must have shape (63,), got {flat_array.shape}")
        
        points = flat_array.reshape(21, 3)
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        return cls(points=points, handedness=handedness, timestamp=timestamp)


@dataclass
class Gesture:
    """
    Represents a stored gesture with its embedding and assigned action.
    
    Attributes:
        name: Unique name for the gesture
        embedding: Numpy array representing the gesture in embedding space (shape: (16,))
        action: The OS action to execute when this gesture is recognized
        created_at: Timestamp when the gesture was created
    """
    name: str
    embedding: np.ndarray
    action: Action
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert gesture to dictionary for serialization.
        
        Returns:
            Dict: Dictionary representation of the gesture
        """
        return {
            'name': self.name,
            'embedding': self.embedding.tolist(),
            'action': {
                'type': self.action.type.value,
                'data': self.action.data,
                'execution_mode': self.action.execution_mode.value
            },
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Gesture':
        """
        Create gesture from dictionary.
        
        Args:
            data: Dictionary containing gesture data
            
        Returns:
            Gesture: Deserialized gesture object
        """
        return cls(
            name=data['name'],
            embedding=np.array(data['embedding']),
            action=Action(
                type=ActionType(data['action']['type']),
                data=data['action']['data'],
                execution_mode=ExecutionMode(data['action'].get('execution_mode', 'trigger_once'))
            ),
            created_at=datetime.fromisoformat(data['created_at'])
        )


@dataclass
class Config:
    """
    Configuration parameters for the Gesture Control System.
    
    Attributes:
        similarity_threshold: Maximum distance for gesture matching (0.0 to 1.0)
        recording_duration: Duration in seconds for gesture recording
        fps: Target frames per second for video capture
        smoothing_window: Number of frames for moving average smoothing
        max_hands: Maximum number of hands to detect
        detection_confidence: Minimum confidence for hand detection (0.0 to 1.0)
        embedding_dimensions: Dimensionality of gesture embeddings
    """
    similarity_threshold: float = 0.5
    recording_duration: float = 3.0
    fps: int = 30
    smoothing_window: int = 5
    max_hands: int = 1
    detection_confidence: float = 0.7
    embedding_dimensions: int = 16
    
    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: File path to save configuration
        """
        config_dict = {
            'similarity_threshold': self.similarity_threshold,
            'recording_duration': self.recording_duration,
            'fps': self.fps,
            'smoothing_window': self.smoothing_window,
            'max_hands': self.max_hands,
            'detection_confidence': self.detection_confidence,
            'embedding_dimensions': self.embedding_dimensions
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            path: File path to load configuration from
            
        Returns:
            Config: Loaded configuration object
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)

