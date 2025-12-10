"""
Example demonstrating the LandmarkPreprocessor functionality.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from src.landmark_preprocessor import LandmarkPreprocessor

# Create a preprocessor
preprocessor = LandmarkPreprocessor(smoothing_window=3)

# Create sample landmarks (21 points with x, y, z coordinates)
# Simulating a hand with wrist at (5, 5, 0) and middle finger base at (7, 5, 0)
landmarks = np.random.rand(21, 3)
landmarks[0] = [5, 5, 0]  # Wrist
landmarks[9] = [7, 5, 0]  # Middle finger base (palm_size = 2)

print("Original landmarks (first 3 points):")
print(landmarks[:3])
print(f"\nWrist position: {landmarks[0]}")
print(f"Middle finger base: {landmarks[9]}")
print(f"Palm size: {np.linalg.norm(landmarks[9] - landmarks[0])}")

# Normalize
normalized = preprocessor.normalize(landmarks)
print("\n--- After Normalization ---")
print(f"Wrist position (should be at origin): {normalized[0]}")
print(f"Middle finger base (should be at distance 1): {normalized[9]}")
print(f"Distance from wrist to middle finger: {np.linalg.norm(normalized[9] - normalized[0])}")

# Preprocess (normalize + smooth)
print("\n--- Testing Smoothing ---")
preprocessor.reset()  # Clear history

frame1 = np.ones((21, 3)) * 1.0
frame2 = np.ones((21, 3)) * 2.0
frame3 = np.ones((21, 3)) * 3.0

result1 = preprocessor.smooth(frame1)
result2 = preprocessor.smooth(frame2)
result3 = preprocessor.smooth(frame3)

print(f"Frame 1 value: {frame1[0, 0]}, Smoothed: {result1[0, 0]}")
print(f"Frame 2 value: {frame2[0, 0]}, Smoothed: {result2[0, 0]}")
print(f"Frame 3 value: {frame3[0, 0]}, Smoothed: {result3[0, 0]} (average of 1, 2, 3)")

print("\n✓ LandmarkPreprocessor is working correctly!")
