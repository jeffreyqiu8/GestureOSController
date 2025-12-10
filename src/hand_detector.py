"""
Hand detection component using MediaPipe Hands.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional
from datetime import datetime

from src.models import HandLandmarks


class HandDetector:
    """
    Wraps MediaPipe Hands for hand detection and landmark extraction.
    
    This component detects hands in video frames and extracts 21 3D landmark points
    per hand. It provides methods for detection and visualization.
    
    Attributes:
        max_hands: Maximum number of hands to detect
        detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand detection
    """
    
    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.7):
        """
        Initialize the HandDetector with MediaPipe Hands.
        
        Args:
            max_hands: Maximum number of hands to detect (default: 1)
            detection_confidence: Minimum confidence for detection (default: 0.7)
        
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        if max_hands < 1:
            raise ValueError(f"max_hands must be >= 1, got {max_hands}")
        if not 0.0 <= detection_confidence <= 1.0:
            raise ValueError(f"detection_confidence must be in [0.0, 1.0], got {detection_confidence}")
        
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame: np.ndarray) -> List[HandLandmarks]:
        """
        Detect hands and extract landmarks from a video frame.
        
        Processes the frame through MediaPipe Hands to detect hands and extract
        21 3D landmark points for each detected hand.
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
        
        Returns:
            List[HandLandmarks]: List of detected hand landmarks. Empty list if no hands detected.
        
        Requirements: 1.2, 1.3
        """
        if frame is None or frame.size == 0:
            return []
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(frame_rgb)
        
        # Extract landmarks if hands were detected
        detected_hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            timestamp = datetime.now().timestamp()
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Extract 21 landmark points (x, y, z)
                points = np.array([
                    [landmark.x, landmark.y, landmark.z]
                    for landmark in hand_landmarks.landmark
                ])
                
                # Get handedness label ("Left" or "Right")
                hand_label = handedness.classification[0].label
                
                # Create HandLandmarks object
                hand_data = HandLandmarks(
                    points=points,
                    handedness=hand_label,
                    timestamp=timestamp
                )
                detected_hands.append(hand_data)
        
        return detected_hands
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: HandLandmarks) -> np.ndarray:
        """
        Draw hand landmarks and skeleton overlay on a frame.
        
        Renders visual connections between all 21 landmark points to create
        a skeleton overlay on the video feed.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            landmarks: HandLandmarks object containing points to draw
        
        Returns:
            np.ndarray: Frame with landmarks drawn (same shape as input)
        
        Requirements: 1.5
        """
        if frame is None or frame.size == 0:
            return frame
        
        # Create a copy to avoid modifying the original
        output_frame = frame.copy()
        
        height, width, _ = frame.shape
        
        # Draw landmarks as circles
        for point in landmarks.points:
            x = int(point[0] * width)
            y = int(point[1] * height)
            cv2.circle(output_frame, (x, y), 5, (0, 255, 0), -1)
        
        # Draw connections between landmarks
        # MediaPipe hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start_point = landmarks.points[start_idx]
            end_point = landmarks.points[end_idx]
            
            start_x = int(start_point[0] * width)
            start_y = int(start_point[1] * height)
            end_x = int(end_point[0] * width)
            end_y = int(end_point[1] * height)
            
            cv2.line(output_frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        
        return output_frame
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
