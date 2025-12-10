"""
Embedding model component for dimensionality reduction of hand landmarks.
"""
import numpy as np
import pickle
import logging
from typing import List, Optional
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Converts preprocessed hand landmarks into low-dimensional embeddings using PCA.
    
    This component reduces the 63-dimensional flattened landmark vector to a
    lower-dimensional embedding space (default: 16 dimensions) to enable efficient
    gesture matching and storage.
    
    Attributes:
        n_components: Number of dimensions in the embedding space
    
    Requirements: 2.4, 8.2, 8.3
    """
    
    def __init__(self, n_components: int = 16):
        """
        Initialize the EmbeddingModel.
        
        Args:
            n_components: Number of dimensions for the embedding space (default: 16)
        
        Raises:
            ValueError: If n_components is less than 1 or greater than 63
        """
        if n_components < 1 or n_components > 63:
            raise ValueError(f"n_components must be between 1 and 63, got {n_components}")
        
        self.n_components = n_components
        self._pca: Optional[PCA] = None
        self._is_fitted = False
    
    def fit(self, training_data: List[np.ndarray]) -> None:
        """
        Train the PCA model on landmark data.
        
        Fits a PCA model to reduce dimensionality from 63 (21 landmarks × 3 coordinates)
        to n_components dimensions. The training data should consist of preprocessed
        (normalized and optionally smoothed) landmark arrays.
        
        Args:
            training_data: List of landmark arrays, each of shape (21, 3) or (63,)
        
        Raises:
            ValueError: If training_data is empty or contains invalid shapes
        
        Requirements: 2.4, 8.2, 8.3
        """
        if not training_data:
            raise ValueError("training_data cannot be empty")
        
        # Convert all training data to flattened format (63,)
        flattened_data = []
        for landmarks in training_data:
            if landmarks.shape == (21, 3):
                flattened = landmarks.flatten()
            elif landmarks.shape == (63,):
                flattened = landmarks
            else:
                raise ValueError(f"Each landmark array must have shape (21, 3) or (63,), got {landmarks.shape}")
            
            flattened_data.append(flattened)
        
        # Convert to 2D array for sklearn: (n_samples, n_features)
        X = np.array(flattened_data)
        
        # Ensure we have enough samples for PCA
        if X.shape[0] < self.n_components:
            # If we have fewer samples than components, reduce n_components
            actual_components = min(self.n_components, X.shape[0])
            self._pca = PCA(n_components=actual_components)
        else:
            self._pca = PCA(n_components=self.n_components)
        
        # Fit the PCA model
        self._pca.fit(X)
        self._is_fitted = True
        
        explained_variance = sum(self._pca.explained_variance_ratio_)
        logger.info(f"PCA model fitted with {len(training_data)} samples. "
                   f"Components: {self._pca.n_components_}, "
                   f"Explained variance: {explained_variance:.2%}")
    
    def transform(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Convert landmarks to embedding vector.
        
        Transforms a single set of preprocessed landmarks into a low-dimensional
        embedding using the fitted PCA model.
        
        Args:
            landmarks: Landmark array of shape (21, 3) or (63,)
        
        Returns:
            np.ndarray: Embedding vector of shape (n_components,)
        
        Raises:
            ValueError: If landmarks have invalid shape
            RuntimeError: If the model has not been fitted yet
        
        Requirements: 2.4, 8.2, 8.3
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform. Call fit() first.")
        
        # Convert to flattened format if needed
        if landmarks.shape == (21, 3):
            flattened = landmarks.flatten()
        elif landmarks.shape == (63,):
            flattened = landmarks
        else:
            raise ValueError(f"Landmarks must have shape (21, 3) or (63,), got {landmarks.shape}")
        
        # Reshape to 2D for sklearn: (1, n_features)
        X = flattened.reshape(1, -1)
        
        # Transform using PCA
        embedding = self._pca.transform(X)
        
        # Return as 1D array
        return embedding.flatten()
    
    def save(self, path: str) -> None:
        """
        Save the fitted PCA model to disk.
        
        Persists the trained PCA model to a file so it can be loaded later
        without retraining. Uses pickle for serialization.
        
        Args:
            path: File path to save the model
        
        Raises:
            RuntimeError: If the model has not been fitted yet
        
        Requirements: 2.4, 8.2, 8.3
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model. Call fit() first.")
        
        model_data = {
            'pca': self._pca,
            'n_components': self.n_components,
            'is_fitted': self._is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Embedding model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load a fitted PCA model from disk.
        
        Restores a previously trained PCA model from a file, allowing the
        embedding model to be used without retraining.
        
        Args:
            path: File path to load the model from
        
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the loaded model has incompatible n_components
        
        Requirements: 2.4, 8.2, 8.3
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Validate that the loaded model matches our configuration
        if model_data['n_components'] != self.n_components:
            raise ValueError(
                f"Loaded model has n_components={model_data['n_components']}, "
                f"but this instance was initialized with n_components={self.n_components}"
            )
        
        self._pca = model_data['pca']
        self._is_fitted = model_data['is_fitted']
        
        logger.info(f"Embedding model loaded from {path}")
    
    @property
    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.
        
        Returns:
            bool: True if the model has been fitted, False otherwise
        """
        return self._is_fitted
