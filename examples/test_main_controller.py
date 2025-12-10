"""
Example script demonstrating MainController usage.

This script shows how to:
1. Initialize the MainController
2. Start the system
3. Process frames in a loop
4. Enter recording mode
5. Save gestures
6. Manage gestures
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import time
from datetime import datetime

from src.main_controller import MainController, SystemState
from src.models import Config, Action, ActionType


def main():
    """Demonstrate MainController functionality."""
    
    # Create configuration
    config = Config(
        fps=30,
        similarity_threshold=0.5,
        recording_duration=3.0,
        smoothing_window=5,
        max_hands=1,
        detection_confidence=0.7,
        embedding_dimensions=16
    )
    
    # Initialize controller
    print("Initializing MainController...")
    controller = MainController(config)
    
    # Before we can record gestures, we need to train the embedding model
    # In a real application, you would collect training data first
    # For this example, we'll generate some dummy training data
    print("\nGenerating training data for embedding model...")
    training_landmarks = []
    for _ in range(50):
        # Generate random landmark data (21 points, 3 coordinates each)
        landmarks = np.random.rand(21, 3)
        training_landmarks.append(landmarks)
    
    print("Training embedding model...")
    if controller.train_embedding_model(training_landmarks):
        print("✓ Embedding model trained successfully")
    else:
        print("✗ Failed to train embedding model")
        return
    
    # Start the controller
    print("\nStarting MainController...")
    if not controller.start():
        print("✗ Failed to start controller")
        return
    
    print("✓ Controller started successfully")
    print(f"Current state: {controller.get_state().value}")
    
    # Simulate recording a gesture
    print("\n--- Recording Mode Demo ---")
    print("Entering recording mode...")
    if controller.enter_recording_mode(duration=2.0):
        print("✓ Recording started")
        
        # Simulate collecting embeddings during recording
        start_time = time.time()
        frame_count = 0
        while controller.is_recording_complete() is False:
            # In a real application, this would be done in process_frame()
            # Here we simulate it by adding random embeddings
            if controller.embedding_model.is_fitted:
                dummy_embedding = np.random.rand(16)
                if controller.gesture_recorder.is_recording():
                    controller.gesture_recorder.add_embedding(dummy_embedding)
                    frame_count += 1
            
            progress = controller.get_recording_progress()
            print(f"Recording progress: {progress*100:.1f}%", end='\r')
            time.sleep(0.1)
        
        print(f"\n✓ Recording complete ({frame_count} frames collected)")
        
        # Save the recorded gesture
        action = Action(
            type=ActionType.KEYSTROKE,
            data={'keys': ['ctrl', 'c']}
        )
        
        if controller.save_recorded_gesture("copy_gesture", action):
            print("✓ Gesture saved successfully")
        else:
            print("✗ Failed to save gesture")
    
    # List all gestures
    print("\n--- Gesture Management Demo ---")
    gestures = controller.get_all_gestures()
    print(f"Total gestures: {len(gestures)}")
    for gesture in gestures:
        print(f"  - {gesture.name}: {gesture.action.type.value}")
    
    # Get a specific gesture
    gesture = controller.get_gesture("copy_gesture")
    if gesture:
        print(f"\n✓ Retrieved gesture: {gesture.name}")
        print(f"  Action: {gesture.action.type.value}")
        print(f"  Created: {gesture.created_at}")
    
    # Update settings
    print("\n--- Settings Update Demo ---")
    new_config = Config(
        fps=60,
        similarity_threshold=0.3,
        recording_duration=5.0
    )
    if controller.update_settings(new_config):
        print("✓ Settings updated successfully")
        print(f"  New FPS: {controller.config.fps}")
        print(f"  New threshold: {controller.config.similarity_threshold}")
    
    # Delete gesture
    print("\n--- Gesture Deletion Demo ---")
    if controller.delete_gesture("copy_gesture"):
        print("✓ Gesture deleted successfully")
    else:
        print("✗ Failed to delete gesture")
    
    # Verify deletion
    gestures = controller.get_all_gestures()
    print(f"Total gestures after deletion: {len(gestures)}")
    
    # Stop the controller
    print("\n--- Shutdown ---")
    controller.stop()
    print("✓ Controller stopped")
    print(f"Final state: {controller.get_state().value}")


if __name__ == "__main__":
    main()
