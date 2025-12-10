"""
Unit tests for LandmarkPreprocessor component.
"""
import pytest
import numpy as np
from src.landmark_preprocessor import LandmarkPreprocessor


class TestLandmarkPreprocessor:
    """Test suite for LandmarkPreprocessor class."""
    
    def test_init_valid_window(self):
        """Test initialization with valid smoothing window."""
        preprocessor = LandmarkPreprocessor(smoothing_window=5)
        assert preprocessor.smoothing_window == 5
    
    def test_init_invalid_window(self):
        """Test initialization with invalid smoothing window."""
        with pytest.raises(ValueError, match="smoothing_window must be >= 1"):
            LandmarkPreprocessor(smoothing_window=0)
    
    def test_normalize_centers_on_wrist(self):
        """Test that normalization centers landmarks on wrist."""
        preprocessor = LandmarkPreprocessor()
        
        # Create simple landmarks where wrist is at (1, 1, 1)
        landmarks = np.ones((21, 3))
        landmarks[0] = [1, 1, 1]  # Wrist
        landmarks[9] = [2, 1, 1]  # Middle finger base (palm_size = 1)
        
        normalized = preprocessor.normalize(landmarks)
        
        # After centering on wrist, wrist should be at origin
        np.testing.assert_array_almost_equal(normalized[0], [0, 0, 0])
    
    def test_normalize_scales_by_palm_size(self):
        """Test that normalization scales by palm size."""
        preprocessor = LandmarkPreprocessor()
        
        # Create landmarks with known palm size
        landmarks = np.zeros((21, 3))
        landmarks[0] = [0, 0, 0]  # Wrist at origin
        landmarks[9] = [2, 0, 0]  # Middle finger base at distance 2 (palm_size = 2)
        landmarks[5] = [0, 4, 0]  # Another point at distance 4 from wrist
        
        normalized = preprocessor.normalize(landmarks)
        
        # After normalization, middle finger base should be at distance 1
        expected_middle_finger = [1, 0, 0]
        np.testing.assert_array_almost_equal(normalized[9], expected_middle_finger)
        
        # Point at distance 4 should be at distance 2 after scaling by 1/2
        expected_point = [0, 2, 0]
        np.testing.assert_array_almost_equal(normalized[5], expected_point)
    
    def test_normalize_handles_zero_palm_size(self):
        """Test that normalization handles edge case of zero palm size."""
        preprocessor = LandmarkPreprocessor()
        
        # Create landmarks where wrist and middle finger base are at same position
        landmarks = np.ones((21, 3))
        landmarks[0] = [1, 1, 1]  # Wrist
        landmarks[9] = [1, 1, 1]  # Middle finger base (palm_size = 0)
        
        # Should not raise error, just return centered landmarks
        normalized = preprocessor.normalize(landmarks)
        
        # Wrist should still be at origin
        np.testing.assert_array_almost_equal(normalized[0], [0, 0, 0])
    
    def test_normalize_invalid_shape(self):
        """Test that normalize raises error for invalid shape."""
        preprocessor = LandmarkPreprocessor()
        
        invalid_landmarks = np.zeros((20, 3))  # Wrong number of landmarks
        
        with pytest.raises(ValueError, match="Landmarks must have shape"):
            preprocessor.normalize(invalid_landmarks)
    
    def test_smooth_single_frame(self):
        """Test smoothing with a single frame."""
        preprocessor = LandmarkPreprocessor(smoothing_window=3)
        
        landmarks = np.random.rand(21, 3)
        smoothed = preprocessor.smooth(landmarks)
        
        # With only one frame, smoothed should equal input
        np.testing.assert_array_almost_equal(smoothed, landmarks)
    
    def test_smooth_multiple_frames(self):
        """Test smoothing with multiple frames."""
        preprocessor = LandmarkPreprocessor(smoothing_window=3)
        
        # Create three frames with known values
        frame1 = np.ones((21, 3)) * 1.0
        frame2 = np.ones((21, 3)) * 2.0
        frame3 = np.ones((21, 3)) * 3.0
        
        preprocessor.smooth(frame1)
        preprocessor.smooth(frame2)
        smoothed = preprocessor.smooth(frame3)
        
        # Average of [1, 2, 3] should be 2
        expected = np.ones((21, 3)) * 2.0
        np.testing.assert_array_almost_equal(smoothed, expected)
    
    def test_smooth_respects_window_size(self):
        """Test that smoothing only uses frames within window."""
        preprocessor = LandmarkPreprocessor(smoothing_window=2)
        
        # Add three frames, but window is only 2
        frame1 = np.ones((21, 3)) * 1.0
        frame2 = np.ones((21, 3)) * 2.0
        frame3 = np.ones((21, 3)) * 3.0
        
        preprocessor.smooth(frame1)
        preprocessor.smooth(frame2)
        smoothed = preprocessor.smooth(frame3)
        
        # Should only average last 2 frames: [2, 3] -> 2.5
        expected = np.ones((21, 3)) * 2.5
        np.testing.assert_array_almost_equal(smoothed, expected)
    
    def test_smooth_invalid_shape(self):
        """Test that smooth raises error for invalid shape."""
        preprocessor = LandmarkPreprocessor()
        
        invalid_landmarks = np.zeros((20, 3))
        
        with pytest.raises(ValueError, match="Landmarks must have shape"):
            preprocessor.smooth(invalid_landmarks)
    
    def test_preprocess_combines_normalize_and_smooth(self):
        """Test that preprocess applies both normalization and smoothing."""
        preprocessor = LandmarkPreprocessor(smoothing_window=2)
        
        # Create landmarks with known structure
        landmarks1 = np.zeros((21, 3))
        landmarks1[0] = [1, 1, 1]  # Wrist
        landmarks1[9] = [3, 1, 1]  # Middle finger base (palm_size = 2)
        
        landmarks2 = np.zeros((21, 3))
        landmarks2[0] = [2, 2, 2]  # Wrist
        landmarks2[9] = [4, 2, 2]  # Middle finger base (palm_size = 2)
        
        # Process both frames
        preprocessor.preprocess(landmarks1)
        result = preprocessor.preprocess(landmarks2)
        
        # Result should be normalized and smoothed
        # After normalization, both should have wrist at origin and middle finger at [1, 0, 0]
        # After smoothing with window=2, should be average of the two normalized frames
        assert result.shape == (21, 3)
        # Wrist should be at origin after normalization
        np.testing.assert_array_almost_equal(result[0], [0, 0, 0])
    
    def test_reset_clears_history(self):
        """Test that reset clears the smoothing history."""
        preprocessor = LandmarkPreprocessor(smoothing_window=3)
        
        # Add some frames
        frame1 = np.ones((21, 3)) * 1.0
        frame2 = np.ones((21, 3)) * 2.0
        
        preprocessor.smooth(frame1)
        preprocessor.smooth(frame2)
        
        # Reset history
        preprocessor.reset()
        
        # Next frame should not be averaged with previous frames
        frame3 = np.ones((21, 3)) * 3.0
        smoothed = preprocessor.smooth(frame3)
        
        # Should equal frame3 since history was cleared
        np.testing.assert_array_almost_equal(smoothed, frame3)
