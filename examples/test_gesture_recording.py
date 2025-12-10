"""
Example demonstrating the gesture recording workflow.
"""
import numpy as np
from src.gesture_recorder import GestureRecorder, compute_prototype_vector
from src.embedding_model import EmbeddingModel
from src.landmark_preprocessor import LandmarkPreprocessor


def main():
    """Demonstrate the complete gesture recording workflow."""
    print("=== Gesture Recording Workflow Demo ===\n")
    
    # Setup components
    print("1. Setting up components...")
    preprocessor = LandmarkPreprocessor(smoothing_window=5)
    embedding_model = EmbeddingModel(n_components=16)
    recorder = GestureRecorder(duration_seconds=2.0)
    
    # Generate training data for embedding model
    print("2. Training embedding model...")
    training_data = [np.random.rand(21, 3) for _ in range(30)]
    embedding_model.fit(training_data)
    print(f"   Model trained with {len(training_data)} samples")
    
    # Start recording
    print("\n3. Starting gesture recording...")
    recorder.start_recording()
    print(f"   Recording for {recorder.duration_seconds} seconds")
    print(f"   Is recording: {recorder.is_recording()}")
    
    # Simulate capturing frames
    print("\n4. Capturing frames...")
    num_frames = 15
    for i in range(num_frames):
        # Simulate hand landmarks
        landmarks = np.random.rand(21, 3)
        
        # Preprocess
        normalized = preprocessor.normalize(landmarks)
        
        # Generate embedding
        embedding = embedding_model.transform(normalized)
        
        # Add to recorder
        recorder.add_embedding(embedding)
        
        if (i + 1) % 5 == 0:
            print(f"   Captured {i + 1} frames, Progress: {recorder.get_progress():.1%}")
    
    print(f"   Total embeddings collected: {recorder.get_embedding_count()}")
    
    # Stop recording
    print("\n5. Stopping recording...")
    embeddings = recorder.stop_recording()
    print(f"   Retrieved {len(embeddings)} embeddings")
    print(f"   Is recording: {recorder.is_recording()}")
    
    # Compute prototype
    print("\n6. Computing prototype vector...")
    prototype = compute_prototype_vector(embeddings)
    print(f"   Prototype shape: {prototype.shape}")
    print(f"   Prototype mean: {prototype.mean():.4f}")
    print(f"   Prototype std: {prototype.std():.4f}")
    
    # Verify prototype is the mean
    manual_mean = np.mean(embeddings, axis=0)
    is_correct = np.allclose(prototype, manual_mean)
    print(f"   Prototype equals mean of embeddings: {is_correct}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
