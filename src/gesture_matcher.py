"""
Gesture matcher component for matching live embeddings to stored prototypes.
"""
import numpy as np
import logging
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GestureMatch:
    """
    Represents a matched gesture with confidence information.
    
    Attributes:
        gesture_name: Name of the matched gesture
        confidence: Confidence score (inverse of distance, normalized)
        distance: Raw distance value from the matching metric
    """
    gesture_name: str
    confidence: float
    distance: float


class GestureMatcher:
    """
    Matches live gesture embeddings to stored prototype embeddings.
    
    This component compares incoming gesture embeddings against a set of stored
    prototypes using distance metrics (Euclidean or cosine similarity) and applies
    threshold-based matching logic to determine if a gesture is recognized.
    
    Attributes:
        threshold: Maximum distance value for a gesture to be considered a match
        metric: Distance metric to use ("euclidean" or "cosine")
    
    Requirements: 4.2, 4.3, 4.4
    """
    
    def __init__(self, threshold: float, metric: str = "euclidean"):
        """
        Initialize the GestureMatcher.
        
        Args:
            threshold: Maximum distance for gesture matching (lower = stricter)
            metric: Distance metric to use ("euclidean" or "cosine")
        
        Raises:
            ValueError: If threshold is negative or metric is invalid
        """
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        
        if metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Metric must be 'euclidean' or 'cosine', got '{metric}'")
        
        self.threshold = threshold
        self.metric = metric
        logger.info(f"GestureMatcher initialized with threshold={threshold}, metric={metric}")
    
    def match(self, embedding: np.ndarray, prototypes: Dict[str, np.ndarray]) -> Optional[GestureMatch]:
        """
        Match a live embedding against stored prototypes.
        
        Compares the input embedding against all stored prototype embeddings using
        the configured distance metric. Returns a match if the closest prototype
        is within the threshold distance.
        
        Args:
            embedding: Live gesture embedding to match (shape: (n_dimensions,))
            prototypes: Dictionary mapping gesture names to prototype embeddings
        
        Returns:
            Optional[GestureMatch]: Match information if a gesture is recognized,
                                   None if no match is found or prototypes is empty
        
        Requirements: 4.2, 4.3, 4.4
        """
        if not prototypes:
            return None
        
        # Calculate distances to all prototypes
        distances = {}
        for name, prototype in prototypes.items():
            distance = self._calculate_distance(embedding, prototype)
            distances[name] = distance
        
        # Find the closest match
        closest_name = min(distances, key=distances.get)
        closest_distance = distances[closest_name]
        
        # Check if the closest match is within threshold
        if closest_distance <= self.threshold:
            # Calculate confidence as inverse of distance (normalized)
            # For small distances, confidence is high; for large distances, confidence is low
            confidence = 1.0 / (1.0 + closest_distance)
            
            logger.debug(f"Gesture matched: {closest_name} (distance={closest_distance:.4f}, confidence={confidence:.4f})")
            
            return GestureMatch(
                gesture_name=closest_name,
                confidence=confidence,
                distance=closest_distance
            )
        
        logger.debug(f"No match found (closest: {closest_name}, distance={closest_distance:.4f} > threshold={self.threshold})")
        return None
    
    def _calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate distance between two embeddings using the configured metric.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            float: Distance value (lower = more similar)
        
        Raises:
            ValueError: If embeddings have different shapes
        
        Requirements: 4.3
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(
                f"Embeddings must have the same shape. "
                f"Got {embedding1.shape} and {embedding2.shape}"
            )
        
        if self.metric == "euclidean":
            return self._euclidean_distance(embedding1, embedding2)
        elif self.metric == "cosine":
            return self._cosine_distance(embedding1, embedding2)
        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two embeddings.
        
        Euclidean distance is the straight-line distance between two points:
        d = sqrt(sum((x1 - x2)^2))
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            float: Euclidean distance
        
        Requirements: 4.3
        """
        return float(np.linalg.norm(embedding1 - embedding2))
    
    def _cosine_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine distance between two embeddings.
        
        Cosine distance is 1 - cosine similarity, where cosine similarity is:
        similarity = (A · B) / (||A|| * ||B||)
        distance = 1 - similarity
        
        Cosine distance ranges from 0 (identical direction) to 2 (opposite direction).
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            float: Cosine distance
        
        Requirements: 4.3
        """
        # Calculate dot product
        dot_product = np.dot(embedding1, embedding2)
        
        # Calculate magnitudes
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Handle zero vectors
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance for zero vectors
        
        # Calculate cosine similarity
        cosine_similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [-1, 1] to handle numerical errors
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        # Convert to distance (0 = identical, 2 = opposite)
        cosine_distance = 1.0 - cosine_similarity
        
        return float(cosine_distance)
