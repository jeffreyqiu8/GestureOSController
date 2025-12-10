"""
Unit tests for the GestureMatcher component.
"""
import numpy as np
import pytest

from src.gesture_matcher import GestureMatcher, GestureMatch


class TestGestureMatcherInitialization:
    """Tests for GestureMatcher initialization."""
    
    def test_default_initialization(self):
        """Test that GestureMatcher initializes with default metric."""
        matcher = GestureMatcher(threshold=0.5)
        assert matcher.threshold == 0.5
        assert matcher.metric == "euclidean"
    
    def test_custom_metric_euclidean(self):
        """Test initialization with euclidean metric."""
        matcher = GestureMatcher(threshold=0.5, metric="euclidean")
        assert matcher.metric == "euclidean"
    
    def test_custom_metric_cosine(self):
        """Test initialization with cosine metric."""
        matcher = GestureMatcher(threshold=0.5, metric="cosine")
        assert matcher.metric == "cosine"
    
    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Metric must be 'euclidean' or 'cosine'"):
            GestureMatcher(threshold=0.5, metric="invalid")
    
    def test_negative_threshold(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be non-negative"):
            GestureMatcher(threshold=-0.1)
    
    def test_zero_threshold(self):
        """Test that zero threshold is valid."""
        matcher = GestureMatcher(threshold=0.0)
        assert matcher.threshold == 0.0


class TestGestureMatcherEuclideanDistance:
    """Tests for Euclidean distance calculation."""
    
    def test_euclidean_distance_identical_vectors(self):
        """Test that identical vectors have zero distance."""
        matcher = GestureMatcher(threshold=1.0, metric="euclidean")
        
        embedding = np.array([1.0, 2.0, 3.0, 4.0])
        distance = matcher._euclidean_distance(embedding, embedding)
        
        assert distance == pytest.approx(0.0)
    
    def test_euclidean_distance_orthogonal_vectors(self):
        """Test Euclidean distance for orthogonal vectors."""
        matcher = GestureMatcher(threshold=1.0, metric="euclidean")
        
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        
        distance = matcher._euclidean_distance(embedding1, embedding2)
        expected = np.sqrt(2.0)  # sqrt(1^2 + 1^2)
        
        assert distance == pytest.approx(expected)
    
    def test_euclidean_distance_known_values(self):
        """Test Euclidean distance with known values."""
        matcher = GestureMatcher(threshold=1.0, metric="euclidean")
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 6.0, 8.0])
        
        # Distance = sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        distance = matcher._euclidean_distance(embedding1, embedding2)
        expected = np.sqrt(50.0)
        
        assert distance == pytest.approx(expected)
    
    def test_euclidean_distance_symmetry(self):
        """Test that Euclidean distance is symmetric."""
        matcher = GestureMatcher(threshold=1.0, metric="euclidean")
        
        embedding1 = np.random.randn(16)
        embedding2 = np.random.randn(16)
        
        distance1 = matcher._euclidean_distance(embedding1, embedding2)
        distance2 = matcher._euclidean_distance(embedding2, embedding1)
        
        assert distance1 == pytest.approx(distance2)


class TestGestureMatcherCosineDistance:
    """Tests for cosine distance calculation."""
    
    def test_cosine_distance_identical_vectors(self):
        """Test that identical vectors have zero cosine distance."""
        matcher = GestureMatcher(threshold=1.0, metric="cosine")
        
        embedding = np.array([1.0, 2.0, 3.0, 4.0])
        distance = matcher._cosine_distance(embedding, embedding)
        
        assert distance == pytest.approx(0.0)
    
    def test_cosine_distance_opposite_vectors(self):
        """Test that opposite vectors have maximum cosine distance."""
        matcher = GestureMatcher(threshold=1.0, metric="cosine")
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([-1.0, -2.0, -3.0])
        
        distance = matcher._cosine_distance(embedding1, embedding2)
        
        # Cosine similarity = -1, so distance = 1 - (-1) = 2
        assert distance == pytest.approx(2.0)
    
    def test_cosine_distance_orthogonal_vectors(self):
        """Test cosine distance for orthogonal vectors."""
        matcher = GestureMatcher(threshold=1.0, metric="cosine")
        
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        
        distance = matcher._cosine_distance(embedding1, embedding2)
        
        # Cosine similarity = 0, so distance = 1 - 0 = 1
        assert distance == pytest.approx(1.0)
    
    def test_cosine_distance_parallel_vectors(self):
        """Test cosine distance for parallel vectors with different magnitudes."""
        matcher = GestureMatcher(threshold=1.0, metric="cosine")
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([2.0, 4.0, 6.0])  # Same direction, different magnitude
        
        distance = matcher._cosine_distance(embedding1, embedding2)
        
        # Cosine similarity = 1, so distance = 1 - 1 = 0
        assert distance == pytest.approx(0.0)
    
    def test_cosine_distance_zero_vector(self):
        """Test cosine distance with zero vector."""
        matcher = GestureMatcher(threshold=1.0, metric="cosine")
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([0.0, 0.0, 0.0])
        
        distance = matcher._cosine_distance(embedding1, embedding2)
        
        # Should return 1.0 for zero vectors
        assert distance == pytest.approx(1.0)
    
    def test_cosine_distance_symmetry(self):
        """Test that cosine distance is symmetric."""
        matcher = GestureMatcher(threshold=1.0, metric="cosine")
        
        embedding1 = np.random.randn(16)
        embedding2 = np.random.randn(16)
        
        distance1 = matcher._cosine_distance(embedding1, embedding2)
        distance2 = matcher._cosine_distance(embedding2, embedding1)
        
        assert distance1 == pytest.approx(distance2)


class TestGestureMatcherCalculateDistance:
    """Tests for the _calculate_distance method."""
    
    def test_calculate_distance_euclidean(self):
        """Test that _calculate_distance uses Euclidean metric correctly."""
        matcher = GestureMatcher(threshold=1.0, metric="euclidean")
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 6.0, 8.0])
        
        distance = matcher._calculate_distance(embedding1, embedding2)
        expected = matcher._euclidean_distance(embedding1, embedding2)
        
        assert distance == pytest.approx(expected)
    
    def test_calculate_distance_cosine(self):
        """Test that _calculate_distance uses cosine metric correctly."""
        matcher = GestureMatcher(threshold=1.0, metric="cosine")
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 5.0, 6.0])
        
        distance = matcher._calculate_distance(embedding1, embedding2)
        expected = matcher._cosine_distance(embedding1, embedding2)
        
        assert distance == pytest.approx(expected)
    
    def test_calculate_distance_mismatched_shapes(self):
        """Test that mismatched shapes raise ValueError."""
        matcher = GestureMatcher(threshold=1.0)
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([1.0, 2.0, 3.0, 4.0])
        
        with pytest.raises(ValueError, match="must have the same shape"):
            matcher._calculate_distance(embedding1, embedding2)


class TestGestureMatcherMatch:
    """Tests for the match() method."""
    
    def test_match_with_empty_prototypes(self):
        """Test that matching with empty prototypes returns None."""
        matcher = GestureMatcher(threshold=0.5)
        
        embedding = np.random.randn(16)
        prototypes = {}
        
        result = matcher.match(embedding, prototypes)
        assert result is None
    
    def test_match_exact_match_within_threshold(self):
        """Test matching with an exact match within threshold."""
        matcher = GestureMatcher(threshold=0.5, metric="euclidean")
        
        embedding = np.array([1.0, 2.0, 3.0, 4.0])
        prototypes = {
            "gesture1": np.array([1.0, 2.0, 3.0, 4.0]),  # Exact match
            "gesture2": np.array([5.0, 6.0, 7.0, 8.0])
        }
        
        result = matcher.match(embedding, prototypes)
        
        assert result is not None
        assert result.gesture_name == "gesture1"
        assert result.distance == pytest.approx(0.0)
        assert result.confidence > 0.9  # High confidence for exact match
    
    def test_match_closest_gesture_within_threshold(self):
        """Test matching selects the closest gesture within threshold."""
        matcher = GestureMatcher(threshold=2.0, metric="euclidean")
        
        embedding = np.array([1.0, 1.0, 1.0])
        prototypes = {
            "gesture1": np.array([1.5, 1.5, 1.5]),  # Distance ~0.866
            "gesture2": np.array([3.0, 3.0, 3.0]),  # Distance ~3.464
            "gesture3": np.array([1.2, 1.1, 1.0])   # Distance ~0.223 (closest)
        }
        
        result = matcher.match(embedding, prototypes)
        
        assert result is not None
        assert result.gesture_name == "gesture3"
        assert result.distance < 0.3
    
    def test_match_no_match_above_threshold(self):
        """Test that no match is returned when all distances exceed threshold."""
        matcher = GestureMatcher(threshold=0.1, metric="euclidean")
        
        embedding = np.array([1.0, 1.0, 1.0])
        prototypes = {
            "gesture1": np.array([5.0, 5.0, 5.0]),  # Distance ~6.93
            "gesture2": np.array([10.0, 10.0, 10.0])  # Distance ~15.59
        }
        
        result = matcher.match(embedding, prototypes)
        assert result is None
    
    def test_match_at_threshold_boundary(self):
        """Test matching at the exact threshold boundary."""
        threshold = 1.0
        matcher = GestureMatcher(threshold=threshold, metric="euclidean")
        
        embedding = np.array([0.0, 0.0, 0.0])
        # Create a prototype exactly at threshold distance
        prototypes = {
            "gesture1": np.array([1.0, 0.0, 0.0])  # Distance = 1.0
        }
        
        result = matcher.match(embedding, prototypes)
        
        # Should match because distance <= threshold
        assert result is not None
        assert result.gesture_name == "gesture1"
        assert result.distance == pytest.approx(threshold)
    
    def test_match_just_above_threshold(self):
        """Test that matching fails when distance is just above threshold."""
        threshold = 1.0
        matcher = GestureMatcher(threshold=threshold, metric="euclidean")
        
        embedding = np.array([0.0, 0.0, 0.0])
        prototypes = {
            "gesture1": np.array([1.01, 0.0, 0.0])  # Distance = 1.01 > 1.0
        }
        
        result = matcher.match(embedding, prototypes)
        assert result is None
    
    def test_match_with_cosine_metric(self):
        """Test matching with cosine similarity metric."""
        matcher = GestureMatcher(threshold=0.5, metric="cosine")
        
        embedding = np.array([1.0, 2.0, 3.0])
        prototypes = {
            "gesture1": np.array([1.1, 2.1, 3.1]),  # Very similar direction
            "gesture2": np.array([-1.0, -2.0, -3.0])  # Opposite direction
        }
        
        result = matcher.match(embedding, prototypes)
        
        assert result is not None
        assert result.gesture_name == "gesture1"
        assert result.distance < 0.1  # Should be very small
    
    def test_match_confidence_calculation(self):
        """Test that confidence is calculated correctly."""
        matcher = GestureMatcher(threshold=2.0, metric="euclidean")
        
        embedding = np.array([0.0, 0.0, 0.0])
        prototypes = {
            "gesture1": np.array([1.0, 0.0, 0.0])  # Distance = 1.0
        }
        
        result = matcher.match(embedding, prototypes)
        
        assert result is not None
        # Confidence = 1 / (1 + distance) = 1 / (1 + 1.0) = 0.5
        expected_confidence = 1.0 / (1.0 + result.distance)
        assert result.confidence == pytest.approx(expected_confidence)
    
    def test_match_multiple_prototypes(self):
        """Test matching with multiple prototypes."""
        matcher = GestureMatcher(threshold=5.0, metric="euclidean")
        
        embedding = np.array([2.0, 3.0, 4.0])
        prototypes = {
            "swipe_left": np.array([2.1, 3.1, 4.1]),
            "swipe_right": np.array([10.0, 10.0, 10.0]),
            "pinch": np.array([5.0, 5.0, 5.0]),
            "wave": np.array([2.05, 3.05, 4.05])  # Closest
        }
        
        result = matcher.match(embedding, prototypes)
        
        assert result is not None
        assert result.gesture_name == "wave"


class TestGestureMatchDataclass:
    """Tests for the GestureMatch dataclass."""
    
    def test_gesture_match_creation(self):
        """Test creating a GestureMatch instance."""
        match = GestureMatch(
            gesture_name="test_gesture",
            confidence=0.85,
            distance=0.25
        )
        
        assert match.gesture_name == "test_gesture"
        assert match.confidence == 0.85
        assert match.distance == 0.25
    
    def test_gesture_match_attributes(self):
        """Test that GestureMatch has all required attributes."""
        match = GestureMatch(
            gesture_name="wave",
            confidence=0.9,
            distance=0.1
        )
        
        assert hasattr(match, 'gesture_name')
        assert hasattr(match, 'confidence')
        assert hasattr(match, 'distance')


class TestGestureMatcherIntegration:
    """Integration tests for GestureMatcher."""
    
    def test_full_matching_workflow_euclidean(self):
        """Test complete matching workflow with Euclidean distance."""
        matcher = GestureMatcher(threshold=1.5, metric="euclidean")
        
        # Create realistic 16-dimensional embeddings
        prototypes = {
            "thumbs_up": np.random.randn(16),
            "peace_sign": np.random.randn(16),
            "fist": np.random.randn(16)
        }
        
        # Create a test embedding similar to "fist"
        test_embedding = prototypes["fist"] + np.random.randn(16) * 0.1
        
        result = matcher.match(test_embedding, prototypes)
        
        # Should match "fist" since it's the closest
        assert result is not None
        assert result.gesture_name == "fist"
        assert result.distance < 1.0
        assert 0.0 < result.confidence <= 1.0
    
    def test_full_matching_workflow_cosine(self):
        """Test complete matching workflow with cosine distance."""
        matcher = GestureMatcher(threshold=0.3, metric="cosine")
        
        # Create realistic 16-dimensional embeddings
        base_vector = np.random.randn(16)
        prototypes = {
            "gesture_a": base_vector * 1.0,
            "gesture_b": base_vector * 2.0 + np.random.randn(16) * 0.5,
            "gesture_c": np.random.randn(16)
        }
        
        # Test with a vector in the same direction as gesture_a and gesture_b
        test_embedding = base_vector * 1.5
        
        result = matcher.match(test_embedding, prototypes)
        
        # Should match gesture_a or gesture_b (both have similar direction)
        assert result is not None
        assert result.gesture_name in ["gesture_a", "gesture_b"]
        assert result.distance < 0.3
    
    def test_no_match_scenario(self):
        """Test scenario where no gesture matches."""
        matcher = GestureMatcher(threshold=0.01, metric="euclidean")
        
        prototypes = {
            "gesture1": np.array([1.0] * 16),
            "gesture2": np.array([2.0] * 16),
            "gesture3": np.array([3.0] * 16)
        }
        
        # Very different embedding
        test_embedding = np.array([100.0] * 16)
        
        result = matcher.match(test_embedding, prototypes)
        assert result is None
