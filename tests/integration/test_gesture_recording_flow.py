"""
Integration tests for gesture recording flow.
"""
import pytest
import numpy as np
from src.gesture_recorder import GestureRecorder, compute_prototype_vector
from src.embedding_model import EmbeddingModel
from src.landmark_preprocessor import LandmarkPreprocessor
from src.models import HandLandmarks


class TestGestureRecordingFlow:
    """Integration tests for the complete gesture recording workflow."""
    
    def test_complete_recording_workflow(self):
        """Test the complete flow from landmarks to prototype vector."""
        # Setup components
        preprocessor = LandmarkPreprocessor(smoothing_window=3)
        embedding_model = EmbeddingModel(n_components=16)
        recorder = GestureRecorder(duration_seconds=1.0)
        
        # Generate some training data for the embedding model
        training_landmarks = []
        for i in range(20):
            # Create random but valid landmarks
            points = np.random.rand(21, 3)
            training_landmarks.append(points)
        
        # Fit the embedding model
        embedding_model.fit(training_landmarks)
        
        # Start recording
        recorder.start_recording()
        
        # Simulate recording multiple frames
        for i in range(10):
            # Create hand landmarks
            points = np.random.rand(21, 3)
            
            # Preprocess landmarks
            normalized = preprocessor.normalize(points)
            
            # Generate embedding
            embedding = embedding_model.transform(normalized)
            
            # Add to recorder
            recorder.add_embedding(embedding)
        
        # Stop recording and get embeddings
        embeddings = recorder.stop_recording()
        
        # Verify we collected embeddings
        assert len(embeddings) == 10
        
        # Compute prototype vector
        prototype = compute_prototype_vector(embeddings)
        
        # Verify prototype has correct shape
        assert prototype.shape == (16,)
        
        # Verify prototype is the mean of embeddings
        expected_prototype = np.mean(embeddings, axis=0)
        assert np.allclose(prototype, expected_prototype)
    
    def test_recording_with_real_hand_landmarks(self):
        """Test recording with HandLandmarks objects."""
        preprocessor = LandmarkPreprocessor(smoothing_window=2)
        embedding_model = EmbeddingModel(n_components=8)
        recorder = GestureRecorder(duration_seconds=0.5)
        
        # Generate training data
        training_data = [np.random.rand(21, 3) for _ in range(15)]
        embedding_model.fit(training_data)
        
        # Start recording
        recorder.start_recording()
        
        # Simulate capturing frames with HandLandmarks
        for i in range(5):
            # Create HandLandmarks
            points = np.random.rand(21, 3)
            landmarks = HandLandmarks(
                points=points,
                handedness="Right",
                timestamp=float(i)
            )
            
            # Process through pipeline
            normalized = preprocessor.normalize(landmarks.points)
            embedding = embedding_model.transform(normalized)
            recorder.add_embedding(embedding)
        
        # Complete recording
        embeddings = recorder.stop_recording()
        prototype = compute_prototype_vector(embeddings)
        
        # Verify results
        assert len(embeddings) == 5
        assert prototype.shape == (8,)
    
    def test_multiple_recording_sessions(self):
        """Test that multiple recording sessions work independently."""
        embedding_model = EmbeddingModel(n_components=16)
        recorder = GestureRecorder(duration_seconds=0.5)
        
        # Fit model
        training_data = [np.random.rand(21, 3) for _ in range(10)]
        embedding_model.fit(training_data)
        
        # First recording session
        recorder.start_recording()
        for i in range(3):
            embedding = embedding_model.transform(np.random.rand(21, 3))
            recorder.add_embedding(embedding)
        embeddings1 = recorder.stop_recording()
        prototype1 = compute_prototype_vector(embeddings1)
        
        # Second recording session
        recorder.start_recording()
        for i in range(5):
            embedding = embedding_model.transform(np.random.rand(21, 3))
            recorder.add_embedding(embedding)
        embeddings2 = recorder.stop_recording()
        prototype2 = compute_prototype_vector(embeddings2)
        
        # Verify sessions are independent
        assert len(embeddings1) == 3
        assert len(embeddings2) == 5
        assert not np.array_equal(prototype1, prototype2)
