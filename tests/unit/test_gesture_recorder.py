"""
Unit tests for the GestureRecorder component.
"""
import pytest
import numpy as np
import time
from src.gesture_recorder import GestureRecorder, compute_prototype_vector


class TestGestureRecorder:
    """Test suite for GestureRecorder class."""
    
    def test_initialization_default_duration(self):
        """Test that GestureRecorder initializes with default duration."""
        recorder = GestureRecorder()
        assert recorder.duration_seconds == 3.0
        assert not recorder.is_recording()
    
    def test_initialization_custom_duration(self):
        """Test that GestureRecorder initializes with custom duration."""
        recorder = GestureRecorder(duration_seconds=5.0)
        assert recorder.duration_seconds == 5.0
    
    def test_initialization_invalid_duration(self):
        """Test that invalid duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            GestureRecorder(duration_seconds=0)
        
        with pytest.raises(ValueError, match="duration_seconds must be positive"):
            GestureRecorder(duration_seconds=-1.0)
    
    def test_start_recording(self):
        """Test that start_recording initializes recording state."""
        recorder = GestureRecorder()
        assert not recorder.is_recording()
        
        recorder.start_recording()
        assert recorder.is_recording()
        assert recorder.get_embedding_count() == 0
    
    def test_add_embedding_during_recording(self):
        """Test adding embeddings during a recording session."""
        recorder = GestureRecorder()
        recorder.start_recording()
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 5.0, 6.0])
        
        recorder.add_embedding(embedding1)
        assert recorder.get_embedding_count() == 1
        
        recorder.add_embedding(embedding2)
        assert recorder.get_embedding_count() == 2
    
    def test_add_embedding_not_recording_raises_error(self):
        """Test that adding embedding when not recording raises RuntimeError."""
        recorder = GestureRecorder()
        embedding = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(RuntimeError, match="Cannot add embedding when not recording"):
            recorder.add_embedding(embedding)
    
    def test_stop_recording_returns_embeddings(self):
        """Test that stop_recording returns collected embeddings."""
        recorder = GestureRecorder()
        recorder.start_recording()
        
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 5.0, 6.0])
        
        recorder.add_embedding(embedding1)
        recorder.add_embedding(embedding2)
        
        embeddings = recorder.stop_recording()
        
        assert len(embeddings) == 2
        assert np.array_equal(embeddings[0], embedding1)
        assert np.array_equal(embeddings[1], embedding2)
        assert not recorder.is_recording()
    
    def test_stop_recording_not_recording_raises_error(self):
        """Test that stopping when not recording raises RuntimeError."""
        recorder = GestureRecorder()
        
        with pytest.raises(RuntimeError, match="Cannot stop recording when not recording"):
            recorder.stop_recording()
    
    def test_stop_recording_clears_state(self):
        """Test that stop_recording clears internal state."""
        recorder = GestureRecorder()
        recorder.start_recording()
        
        embedding = np.array([1.0, 2.0, 3.0])
        recorder.add_embedding(embedding)
        
        recorder.stop_recording()
        
        # Start a new recording - should have clean state
        recorder.start_recording()
        assert recorder.get_embedding_count() == 0
    
    def test_get_progress_not_recording(self):
        """Test that get_progress returns 0.0 when not recording."""
        recorder = GestureRecorder()
        assert recorder.get_progress() == 0.0
    
    def test_get_progress_during_recording(self):
        """Test that get_progress returns correct fraction during recording."""
        recorder = GestureRecorder(duration_seconds=1.0)
        recorder.start_recording()
        
        # Progress should be between 0 and 1
        progress = recorder.get_progress()
        assert 0.0 <= progress <= 1.0
        
        # Wait a bit and check progress increased
        time.sleep(0.1)
        new_progress = recorder.get_progress()
        assert new_progress > progress
    
    def test_get_progress_caps_at_one(self):
        """Test that get_progress never exceeds 1.0."""
        recorder = GestureRecorder(duration_seconds=0.1)
        recorder.start_recording()
        
        time.sleep(0.2)  # Wait longer than duration
        
        progress = recorder.get_progress()
        assert progress == 1.0
    
    def test_is_complete(self):
        """Test that is_complete returns True when duration elapsed."""
        recorder = GestureRecorder(duration_seconds=0.1)
        recorder.start_recording()
        
        assert not recorder.is_complete()
        
        time.sleep(0.15)  # Wait for duration to elapse
        
        assert recorder.is_complete()
    
    def test_is_complete_not_recording(self):
        """Test that is_complete returns False when not recording."""
        recorder = GestureRecorder()
        assert not recorder.is_complete()
    
    def test_embedding_isolation(self):
        """Test that embeddings are copied and not affected by external changes."""
        recorder = GestureRecorder()
        recorder.start_recording()
        
        embedding = np.array([1.0, 2.0, 3.0])
        recorder.add_embedding(embedding)
        
        # Modify original embedding
        embedding[0] = 999.0
        
        # Retrieved embeddings should not be affected
        embeddings = recorder.stop_recording()
        assert embeddings[0][0] == 1.0


class TestComputePrototypeVector:
    """Test suite for compute_prototype_vector function."""
    
    def test_compute_prototype_single_embedding(self):
        """Test prototype computation with a single embedding."""
        embedding = np.array([1.0, 2.0, 3.0])
        prototype = compute_prototype_vector([embedding])
        
        assert np.array_equal(prototype, embedding)
    
    def test_compute_prototype_multiple_embeddings(self):
        """Test prototype computation with multiple embeddings."""
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([3.0, 4.0, 5.0])
        emb3 = np.array([5.0, 6.0, 7.0])
        
        prototype = compute_prototype_vector([emb1, emb2, emb3])
        expected = np.array([3.0, 4.0, 5.0])  # Mean of each dimension
        
        assert np.allclose(prototype, expected)
    
    def test_compute_prototype_empty_list_raises_error(self):
        """Test that empty embeddings list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot compute prototype from empty embeddings list"):
            compute_prototype_vector([])
    
    def test_compute_prototype_inconsistent_shapes_raises_error(self):
        """Test that inconsistent embedding shapes raise ValueError."""
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([4.0, 5.0])  # Different shape
        
        with pytest.raises(ValueError, match="All embeddings must have the same shape"):
            compute_prototype_vector([emb1, emb2])
    
    def test_compute_prototype_multidimensional(self):
        """Test prototype computation with higher-dimensional embeddings."""
        emb1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        emb2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        
        prototype = compute_prototype_vector([emb1, emb2])
        expected = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        
        assert np.allclose(prototype, expected)
    
    def test_compute_prototype_preserves_dtype(self):
        """Test that prototype computation preserves numeric precision."""
        emb1 = np.array([1.5, 2.5, 3.5])
        emb2 = np.array([2.5, 3.5, 4.5])
        
        prototype = compute_prototype_vector([emb1, emb2])
        expected = np.array([2.0, 3.0, 4.0])
        
        assert np.allclose(prototype, expected)
