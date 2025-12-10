"""
Unit tests for the EmbeddingModel component.
"""
import numpy as np
import pytest
import tempfile
import os

from src.embedding_model import EmbeddingModel


class TestEmbeddingModelInitialization:
    """Tests for EmbeddingModel initialization."""
    
    def test_default_initialization(self):
        """Test that EmbeddingModel initializes with default parameters."""
        model = EmbeddingModel()
        assert model.n_components == 16
        assert not model.is_fitted
    
    def test_custom_n_components(self):
        """Test initialization with custom n_components."""
        model = EmbeddingModel(n_components=8)
        assert model.n_components == 8
    
    def test_invalid_n_components_too_small(self):
        """Test that n_components < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_components must be between 1 and 63"):
            EmbeddingModel(n_components=0)
    
    def test_invalid_n_components_too_large(self):
        """Test that n_components > 63 raises ValueError."""
        with pytest.raises(ValueError, match="n_components must be between 1 and 63"):
            EmbeddingModel(n_components=64)


class TestEmbeddingModelFit:
    """Tests for the fit() method."""
    
    def test_fit_with_valid_data(self):
        """Test fitting with valid training data."""
        model = EmbeddingModel(n_components=8)
        
        # Create sample training data (20 samples of shape (21, 3))
        training_data = [np.random.randn(21, 3) for _ in range(20)]
        
        model.fit(training_data)
        assert model.is_fitted
    
    def test_fit_with_flattened_data(self):
        """Test fitting with flattened training data."""
        model = EmbeddingModel(n_components=8)
        
        # Create sample training data (20 samples of shape (63,))
        training_data = [np.random.randn(63) for _ in range(20)]
        
        model.fit(training_data)
        assert model.is_fitted
    
    def test_fit_with_mixed_shapes(self):
        """Test fitting with mixed (21,3) and (63,) shapes."""
        model = EmbeddingModel(n_components=8)
        
        training_data = [
            np.random.randn(21, 3),
            np.random.randn(63),
            np.random.randn(21, 3),
            np.random.randn(63)
        ]
        
        model.fit(training_data)
        assert model.is_fitted
    
    def test_fit_with_empty_data(self):
        """Test that fitting with empty data raises ValueError."""
        model = EmbeddingModel()
        
        with pytest.raises(ValueError, match="training_data cannot be empty"):
            model.fit([])
    
    def test_fit_with_invalid_shape(self):
        """Test that fitting with invalid shape raises ValueError."""
        model = EmbeddingModel()
        
        training_data = [np.random.randn(10, 5)]  # Invalid shape
        
        with pytest.raises(ValueError, match="must have shape"):
            model.fit(training_data)
    
    def test_fit_with_fewer_samples_than_components(self):
        """Test fitting when n_samples < n_components."""
        model = EmbeddingModel(n_components=16)
        
        # Only 5 samples, but 16 components requested
        training_data = [np.random.randn(21, 3) for _ in range(5)]
        
        # Should still work by reducing components
        model.fit(training_data)
        assert model.is_fitted


class TestEmbeddingModelTransform:
    """Tests for the transform() method."""
    
    def test_transform_with_fitted_model(self):
        """Test transforming landmarks with a fitted model."""
        model = EmbeddingModel(n_components=8)
        
        # Fit the model
        training_data = [np.random.randn(21, 3) for _ in range(20)]
        model.fit(training_data)
        
        # Transform a single landmark
        landmarks = np.random.randn(21, 3)
        embedding = model.transform(landmarks)
        
        # Check output shape
        assert embedding.shape == (8,)
    
    def test_transform_with_flattened_input(self):
        """Test transforming flattened landmarks."""
        model = EmbeddingModel(n_components=8)
        
        # Fit the model
        training_data = [np.random.randn(63) for _ in range(20)]
        model.fit(training_data)
        
        # Transform a flattened landmark
        landmarks = np.random.randn(63)
        embedding = model.transform(landmarks)
        
        assert embedding.shape == (8,)
    
    def test_transform_without_fitting(self):
        """Test that transforming without fitting raises RuntimeError."""
        model = EmbeddingModel()
        
        landmarks = np.random.randn(21, 3)
        
        with pytest.raises(RuntimeError, match="Model must be fitted before transform"):
            model.transform(landmarks)
    
    def test_transform_with_invalid_shape(self):
        """Test that transforming with invalid shape raises ValueError."""
        model = EmbeddingModel(n_components=8)
        
        # Fit the model
        training_data = [np.random.randn(21, 3) for _ in range(20)]
        model.fit(training_data)
        
        # Try to transform invalid shape
        landmarks = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="must have shape"):
            model.transform(landmarks)


class TestEmbeddingModelPersistence:
    """Tests for save() and load() methods."""
    
    def test_save_and_load(self):
        """Test saving and loading a fitted model."""
        model1 = EmbeddingModel(n_components=8)
        
        # Fit the model
        training_data = [np.random.randn(21, 3) for _ in range(20)]
        model1.fit(training_data)
        
        # Transform a test sample
        test_landmarks = np.random.randn(21, 3)
        embedding1 = model1.transform(test_landmarks)
        
        # Save the model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model1.save(temp_path)
            
            # Load into a new model
            model2 = EmbeddingModel(n_components=8)
            model2.load(temp_path)
            
            # Transform the same test sample
            embedding2 = model2.transform(test_landmarks)
            
            # Embeddings should be identical
            np.testing.assert_array_almost_equal(embedding1, embedding2)
            assert model2.is_fitted
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_save_without_fitting(self):
        """Test that saving without fitting raises RuntimeError."""
        model = EmbeddingModel()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
                model.save(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_load_with_mismatched_n_components(self):
        """Test that loading with mismatched n_components raises ValueError."""
        model1 = EmbeddingModel(n_components=8)
        
        # Fit and save
        training_data = [np.random.randn(21, 3) for _ in range(20)]
        model1.fit(training_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model1.save(temp_path)
            
            # Try to load with different n_components
            model2 = EmbeddingModel(n_components=16)
            
            with pytest.raises(ValueError, match="Loaded model has n_components"):
                model2.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        model = EmbeddingModel()
        
        with pytest.raises(FileNotFoundError):
            model.load("nonexistent_file.pkl")


class TestEmbeddingModelIntegration:
    """Integration tests for the EmbeddingModel."""
    
    def test_full_workflow(self):
        """Test the complete workflow: fit, transform, save, load."""
        # Create and fit model
        model = EmbeddingModel(n_components=16)
        training_data = [np.random.randn(21, 3) for _ in range(50)]
        model.fit(training_data)
        
        # Transform multiple samples
        test_samples = [np.random.randn(21, 3) for _ in range(10)]
        embeddings = [model.transform(sample) for sample in test_samples]
        
        # Verify all embeddings have correct shape
        for embedding in embeddings:
            assert embedding.shape == (16,)
        
        # Save and reload
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            
            new_model = EmbeddingModel(n_components=16)
            new_model.load(temp_path)
            
            # Transform same samples with loaded model
            new_embeddings = [new_model.transform(sample) for sample in test_samples]
            
            # Results should be identical
            for emb1, emb2 in zip(embeddings, new_embeddings):
                np.testing.assert_array_almost_equal(emb1, emb2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
