"""
Main controller for the Gesture Control System.

This module orchestrates all components and manages the application flow,
including state management, recording mode, and recognition mode.
"""
import logging
import time
import numpy as np
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime

from src.video_capture import VideoCapture
from src.hand_detector import HandDetector
from src.landmark_preprocessor import LandmarkPreprocessor
from src.embedding_model import EmbeddingModel
from src.gesture_recorder import GestureRecorder, compute_prototype_vector
from src.gesture_matcher import GestureMatcher
from src.gesture_repository import GestureRepository
from src.action_executor import ActionExecutor
from src.models import Config, Gesture, Action, HandLandmarks


logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Enumeration of system states."""
    IDLE = "idle"
    RECORDING = "recording"
    RECOGNIZING = "recognizing"


class MainController:
    """
    Main controller that orchestrates the entire gesture control system.
    
    This class initializes all components, manages the main event loop,
    handles state transitions, and coordinates between recording and
    recognition modes.
    
    Attributes:
        config: System configuration
        state: Current system state (idle, recording, recognizing)
    
    Requirements: 1.1, 4.1
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the MainController with all components.
        
        Args:
            config: Optional configuration object. If None, uses default config.
        
        Requirements: 1.1, 4.1
        """
        # Load or use provided configuration
        self.config = config if config is not None else Config()
        
        # Initialize state
        self.state = SystemState.IDLE
        self._is_running = False
        
        # Initialize components
        logger.info("Initializing MainController components...")
        
        # Video capture
        self.video_capture = VideoCapture(
            camera_index=0,
            fps=self.config.fps
        )
        
        # Hand detection
        self.hand_detector = HandDetector(
            max_hands=self.config.max_hands,
            detection_confidence=self.config.detection_confidence
        )
        
        # Landmark preprocessing
        self.preprocessor = LandmarkPreprocessor(
            smoothing_window=self.config.smoothing_window
        )
        
        # Embedding model
        self.embedding_model = EmbeddingModel(
            n_components=self.config.embedding_dimensions
        )
        
        # Gesture recorder
        self.gesture_recorder = GestureRecorder(
            duration_seconds=self.config.recording_duration
        )
        
        # Gesture matcher
        self.gesture_matcher = GestureMatcher(
            threshold=self.config.similarity_threshold,
            metric="euclidean"
        )
        
        # Gesture repository
        self.gesture_repository = GestureRepository(db_path="gestures.db")
        
        # Action executor
        self.action_executor = ActionExecutor()
        
        # Cache for loaded gesture prototypes
        self._gesture_prototypes: Dict[str, np.ndarray] = {}
        
        # Recording state
        self._recording_embeddings: List[np.ndarray] = []
        self._recording_landmarks: List[np.ndarray] = []  # For cold start
        
        # Recognition state
        self._current_match_name: Optional[str] = None
        self._current_match_confidence: float = 0.0
        self._current_match_distance: float = 0.0
        self._previous_match_name: Optional[str] = None  # Track previous gesture for trigger detection
        
        logger.info("MainController initialized successfully")
    
    def start(self) -> bool:
        """
        Start the main controller and begin processing.
        
        Initializes the webcam, loads stored gestures, and prepares
        the system for operation.
        
        Returns:
            bool: True if started successfully, False otherwise
        
        Requirements: 1.1, 4.1
        """
        logger.info("Starting MainController...")
        
        # Start video capture
        if not self.video_capture.start():
            logger.error("Failed to start video capture")
            return False
        
        # Load all gestures from repository
        self._load_gestures()
        
        self._is_running = True
        self.state = SystemState.RECOGNIZING
        
        logger.info("MainController started successfully")
        return True
    
    def stop(self) -> None:
        """
        Stop the main controller and cleanup resources.
        
        Stops video capture and releases all resources.
        
        Requirements: 1.1, 4.1
        """
        logger.info("Stopping MainController...")
        
        self._is_running = False
        self.video_capture.stop()
        self.state = SystemState.IDLE
        
        logger.info("MainController stopped")
    
    def process_frame(self) -> Optional[np.ndarray]:
        """
        Process a single frame through the pipeline.
        
        This is the main event loop iteration that:
        1. Captures a frame
        2. Detects hands
        3. Preprocesses landmarks
        4. Generates embeddings
        5. Performs recording or recognition based on state
        
        Returns:
            Optional[np.ndarray]: The processed frame with overlays, or None if no frame
        
        Requirements: 1.1, 4.1
        """
        if not self._is_running:
            return None
        
        # Get frame from video capture
        frame = self.video_capture.get_frame()
        if frame is None:
            return None
        
        # Detect hands
        detected_hands = self.hand_detector.detect(frame)
        
        # If no hands detected, return frame as-is
        if not detected_hands:
            return frame
        
        # Select primary hand (use first detected hand)
        primary_hand = self._select_primary_hand(detected_hands)
        
        # Draw landmarks on frame
        frame_with_overlay = self.hand_detector.draw_landmarks(frame, primary_hand)
        
        # Preprocess landmarks
        try:
            preprocessed = self.preprocessor.preprocess(primary_hand.points)
            
            # Handle based on current state
            if self.state == SystemState.RECORDING:
                # During recording, collect preprocessed landmarks
                # If model is fitted, also generate embeddings
                self._recording_landmarks.append(preprocessed.copy())
                
                if self.embedding_model.is_fitted:
                    embedding = self.embedding_model.transform(preprocessed)
                    self._handle_recording(embedding)
                else:
                    # Cold start: collecting landmarks to fit model
                    logger.debug("Collecting landmarks for model training (cold start)")
            
            elif self.state == SystemState.RECOGNIZING:
                # Only recognize if model is fitted
                if self.embedding_model.is_fitted:
                    embedding = self.embedding_model.transform(preprocessed)
                    self._handle_recognition(embedding)
        
        except Exception as e:
            logger.exception(f"Error processing frame: {e}")
        
        return frame_with_overlay
    
    def _select_primary_hand(self, hands: List[HandLandmarks]) -> HandLandmarks:
        """
        Select the primary hand from multiple detected hands.
        
        Uses a consistent selection rule: selects the rightmost hand
        (hand with highest average x coordinate). This ensures that
        the same hand is selected consistently across frames when
        multiple hands are present.
        
        Args:
            hands: List of detected hands
        
        Returns:
            HandLandmarks: The selected primary hand
        
        Requirements: 9.1
        """
        if len(hands) == 1:
            return hands[0]
        
        # Select rightmost hand (highest average x coordinate)
        # This provides a consistent, deterministic selection rule
        rightmost_hand = max(hands, key=lambda h: np.mean(h.points[:, 0]))
        
        # Log when multiple hands are detected for debugging
        if len(hands) > 1:
            logger.debug(f"Multiple hands detected ({len(hands)}), selected rightmost hand ({rightmost_hand.handedness})")
        
        return rightmost_hand
    
    def _load_gestures(self) -> None:
        """
        Load all gestures from the repository into memory.
        
        Populates the gesture prototypes cache for fast matching.
        """
        try:
            gestures = self.gesture_repository.get_all_gestures()
            self._gesture_prototypes = {
                gesture.name: gesture.embedding
                for gesture in gestures
            }
            logger.info(f"Loaded {len(self._gesture_prototypes)} gestures from repository")
        except Exception as e:
            logger.exception(f"Error loading gestures: {e}")
            self._gesture_prototypes = {}
    
    def _handle_recording(self, embedding: np.ndarray) -> None:
        """
        Handle embedding during recording mode.
        
        Args:
            embedding: Current frame's embedding
        """
        if self.gesture_recorder.is_recording():
            self.gesture_recorder.add_embedding(embedding)
    
    def _handle_recognition(self, embedding: np.ndarray) -> None:
        """
        Handle embedding during recognition mode.
        
        Args:
            embedding: Current frame's embedding
        
        Requirements: 4.1, 4.2, 4.4, 4.5
        """
        from src.models import ExecutionMode
        
        # Match against stored gestures
        match = self.gesture_matcher.match(embedding, self._gesture_prototypes)
        
        if match:
            # Update current match state
            self._current_match_name = match.gesture_name
            self._current_match_confidence = match.confidence
            self._current_match_distance = match.distance
            
            # Get the gesture from repository
            gesture = self.gesture_repository.get_gesture(match.gesture_name)
            
            if gesture:
                # Determine if we should execute based on execution mode
                should_execute = False
                
                if gesture.action.execution_mode == ExecutionMode.TRIGGER_ONCE:
                    # Execute only if this is a new gesture (different from previous frame)
                    if match.gesture_name != self._previous_match_name:
                        should_execute = True
                        logger.info(f"Gesture triggered: {match.gesture_name} "
                                   f"(confidence: {match.confidence:.2f}, distance: {match.distance:.4f})")
                
                elif gesture.action.execution_mode == ExecutionMode.HOLD_REPEAT:
                    # Execute every frame while gesture is held
                    should_execute = True
                    logger.debug(f"Gesture held: {match.gesture_name} "
                                f"(confidence: {match.confidence:.2f}, distance: {match.distance:.4f})")
                
                # Execute the action if needed
                if should_execute:
                    success = self.action_executor.execute(gesture.action)
                    if success:
                        logger.debug(f"Action executed for gesture: {match.gesture_name}")
                    else:
                        logger.warning(f"Action execution failed for gesture: {match.gesture_name}")
            
            # Update previous match for next frame
            self._previous_match_name = match.gesture_name
        else:
            # No match found - reset state
            self._current_match_name = None
            self._current_match_confidence = 0.0
            self._current_match_distance = 0.0
            self._previous_match_name = None
    
    def is_running(self) -> bool:
        """
        Check if the controller is running.
        
        Returns:
            bool: True if running, False otherwise
        """
        return self._is_running
    
    def get_state(self) -> SystemState:
        """
        Get the current system state.
        
        Returns:
            SystemState: Current state
        """
        return self.state
    
    def enter_recording_mode(self, duration: Optional[float] = None) -> bool:
        """
        Enter recording mode to capture a new gesture.
        
        Starts a recording session that will collect embeddings from
        consecutive frames for the specified duration.
        
        Args:
            duration: Optional recording duration in seconds. If None, uses config default.
        
        Returns:
            bool: True if recording started successfully, False otherwise
        
        Requirements: 2.1, 3.1
        """
        if not self._is_running:
            logger.error("Cannot enter recording mode: controller not running")
            return False
        
        if self.state == SystemState.RECORDING:
            logger.warning("Already in recording mode")
            return False
        
        # Update recording duration if provided
        if duration is not None:
            self.gesture_recorder.duration_seconds = duration
        
        # Reset preprocessor smoothing history for clean recording
        self.preprocessor.reset()
        
        # Clear previous recording data
        self._recording_landmarks = []
        
        # Start recording
        self.gesture_recorder.start_recording()
        self.state = SystemState.RECORDING
        
        logger.info(f"Entered recording mode (duration: {self.gesture_recorder.duration_seconds}s)")
        return True
    
    def save_recorded_gesture(self, name: str, action: Action) -> bool:
        """
        Save a recorded gesture with its name and assigned action.
        
        Stops the current recording session, computes the prototype vector
        from collected embeddings, and stores the gesture in the repository.
        Handles cold start by fitting the embedding model if needed.
        
        Args:
            name: Unique name for the gesture
            action: OS action to assign to this gesture
        
        Returns:
            bool: True if gesture saved successfully, False otherwise
        
        Requirements: 2.1, 3.1, 3.4
        """
        if self.state != SystemState.RECORDING:
            logger.error("Cannot save gesture: not in recording mode")
            return False
        
        if not self.gesture_recorder.is_recording():
            logger.error("Cannot save gesture: recorder not active")
            return False
        
        try:
            # Stop recording
            self.gesture_recorder.stop_recording()
            
            # Check if we collected landmarks
            if not self._recording_landmarks:
                logger.error("No landmarks collected during recording")
                self.state = SystemState.RECOGNIZING
                self._recording_landmarks = []
                return False
            
            logger.info(f"Collected {len(self._recording_landmarks)} landmark frames during recording")
            
            # If model not fitted, fit it now with collected landmarks (cold start)
            if not self.embedding_model.is_fitted:
                logger.info("Fitting embedding model with collected landmarks (cold start)")
                self.embedding_model.fit(self._recording_landmarks)
                logger.info("Embedding model fitted successfully")
            
            # Now transform all collected landmarks to embeddings
            embeddings = []
            for landmarks in self._recording_landmarks:
                embedding = self.embedding_model.transform(landmarks)
                embeddings.append(embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings from landmarks")
            
            # Compute prototype vector
            prototype = compute_prototype_vector(embeddings)
            
            # Create gesture object
            gesture = Gesture(
                name=name,
                embedding=prototype,
                action=action,
                created_at=datetime.now()
            )
            
            # Save to repository
            self.gesture_repository.save_gesture(gesture)
            
            # Update in-memory cache
            self._gesture_prototypes[name] = prototype
            
            # Clear recording state
            self._recording_landmarks = []
            
            # Return to recognizing state
            self.state = SystemState.RECOGNIZING
            
            logger.info(f"Gesture '{name}' saved successfully")
            return True
        
        except Exception as e:
            logger.exception(f"Error saving gesture: {e}")
            self.state = SystemState.RECOGNIZING
            self._recording_landmarks = []
            return False
    
    def cancel_recording(self) -> bool:
        """
        Cancel the current recording session without saving.
        
        Returns:
            bool: True if recording was cancelled, False if not in recording mode
        """
        if self.state != SystemState.RECORDING:
            return False
        
        if self.gesture_recorder.is_recording():
            self.gesture_recorder.stop_recording()
        
        # Clear recording state
        self._recording_landmarks = []
        
        self.state = SystemState.RECOGNIZING
        logger.info("Recording cancelled")
        return True
    
    def get_recording_progress(self) -> float:
        """
        Get the current recording progress.
        
        Returns:
            float: Progress as a fraction (0.0 to 1.0), or 0.0 if not recording
        """
        if self.state == SystemState.RECORDING:
            return self.gesture_recorder.get_progress()
        return 0.0
    
    def is_recording_complete(self) -> bool:
        """
        Check if the recording duration has been reached.
        
        Returns:
            bool: True if recording is complete, False otherwise
        """
        if self.state == SystemState.RECORDING:
            return self.gesture_recorder.is_complete()
        return False
    
    def delete_gesture(self, name: str) -> bool:
        """
        Delete a gesture from the system.
        
        Removes the gesture from both the repository and the in-memory cache.
        
        Args:
            name: Name of the gesture to delete
        
        Returns:
            bool: True if gesture was deleted, False if it didn't exist
        
        Requirements: 6.3
        """
        try:
            # Delete from repository
            success = self.gesture_repository.delete_gesture(name)
            
            if success:
                # Remove from in-memory cache
                if name in self._gesture_prototypes:
                    del self._gesture_prototypes[name]
                
                logger.info(f"Gesture '{name}' deleted successfully")
                return True
            else:
                logger.warning(f"Gesture '{name}' not found")
                return False
        
        except Exception as e:
            logger.exception(f"Error deleting gesture '{name}': {e}")
            return False
    
    def update_settings(self, config: Config) -> bool:
        """
        Update system configuration settings.
        
        Applies new configuration values to the system. Some settings
        require component reinitialization.
        
        Args:
            config: New configuration object
        
        Returns:
            bool: True if settings updated successfully, False otherwise
        
        Requirements: 10.3
        """
        try:
            logger.info("Updating system settings...")
            
            # Update FPS if changed
            if config.fps != self.config.fps:
                self.video_capture.set_fps(config.fps)
            
            # Update similarity threshold
            if config.similarity_threshold != self.config.similarity_threshold:
                self.gesture_matcher.threshold = config.similarity_threshold
            
            # Update recording duration
            if config.recording_duration != self.config.recording_duration:
                self.gesture_recorder.duration_seconds = config.recording_duration
            
            # Update smoothing window (requires new preprocessor)
            if config.smoothing_window != self.config.smoothing_window:
                self.preprocessor = LandmarkPreprocessor(
                    smoothing_window=config.smoothing_window
                )
            
            # Update hand detector settings (requires new detector)
            if (config.max_hands != self.config.max_hands or
                config.detection_confidence != self.config.detection_confidence):
                self.hand_detector = HandDetector(
                    max_hands=config.max_hands,
                    detection_confidence=config.detection_confidence
                )
            
            # Update embedding dimensions (requires new model and retraining)
            if config.embedding_dimensions != self.config.embedding_dimensions:
                self.embedding_model = EmbeddingModel(
                    n_components=config.embedding_dimensions
                )
                # Refit if we have gestures
                if self._gesture_prototypes:
                    embeddings = list(self._gesture_prototypes.values())
                    self.embedding_model.fit(embeddings)
            
            # Store new config
            self.config = config
            
            logger.info("Settings updated successfully")
            return True
        
        except Exception as e:
            logger.exception(f"Error updating settings: {e}")
            return False
    
    def get_all_gestures(self) -> List[Gesture]:
        """
        Get all stored gestures.
        
        Returns:
            List[Gesture]: List of all gestures
        """
        try:
            return self.gesture_repository.get_all_gestures()
        except Exception as e:
            logger.exception(f"Error retrieving gestures: {e}")
            return []
    
    def get_gesture(self, name: str) -> Optional[Gesture]:
        """
        Get a specific gesture by name.
        
        Args:
            name: Name of the gesture
        
        Returns:
            Optional[Gesture]: The gesture if found, None otherwise
        """
        try:
            return self.gesture_repository.get_gesture(name)
        except Exception as e:
            logger.exception(f"Error retrieving gesture '{name}': {e}")
            return None
    
    def train_embedding_model(self, training_landmarks: List[np.ndarray]) -> bool:
        """
        Train the embedding model with landmark data.
        
        This should be called before recording gestures to fit the PCA model
        on representative hand landmark data.
        
        Args:
            training_landmarks: List of landmark arrays, each of shape (21, 3) or (63,)
        
        Returns:
            bool: True if training succeeded, False otherwise
        """
        try:
            logger.info(f"Training embedding model with {len(training_landmarks)} samples...")
            self.embedding_model.fit(training_landmarks)
            logger.info("Embedding model trained successfully")
            return True
        except Exception as e:
            logger.exception(f"Error training embedding model: {e}")
            return False
    
    def get_current_match(self) -> Optional[tuple]:
        """
        Get the current gesture match information.
        
        Returns:
            Optional[tuple]: (gesture_name, confidence, distance) if a match exists, None otherwise
        """
        if self._current_match_name:
            return (self._current_match_name, self._current_match_confidence, self._current_match_distance)
        return None
