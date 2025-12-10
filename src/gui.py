"""
PyQt5 GUI for the Gesture Control System.

This module provides the graphical user interface for the gesture control system,
including video display, control buttons, gesture management, and settings.
"""
import sys
import logging
from typing import Optional, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QDialog,
    QDialogButtonBox, QLineEdit, QFormLayout, QComboBox, QProgressBar,
    QMessageBox, QDoubleSpinBox, QSpinBox, QGroupBox, QTextEdit
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
import cv2

from src.main_controller import MainController, SystemState
from src.models import Config, Gesture, Action, ActionType


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main window for the Gesture Control System GUI.
    
    Provides video display, control buttons, gesture list, and status display.
    
    Requirements: 1.5, 5.1, 5.2, 6.1
    """
    
    def __init__(self, controller: MainController):
        """
        Initialize the main window.
        
        Args:
            controller: The MainController instance
        """
        super().__init__()
        self.controller = controller
        self.current_match_name = None
        self.current_match_confidence = 0.0
        
        self.setWindowTitle("Gesture Control System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Video display and controls
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, stretch=2)
        
        # Right panel: Gesture list and status
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
        # Timer for updating video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        
        logger.info("MainWindow initialized")
    
    def _create_left_panel(self) -> QWidget:
        """
        Create the left panel with video display and control buttons.
        
        Returns:
            QWidget: The left panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        layout.addWidget(self.video_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._on_start_clicked)
        button_layout.addWidget(self.start_button)
        
        self.record_button = QPushButton("Record Gesture")
        self.record_button.clicked.connect(self._on_record_clicked)
        self.record_button.setEnabled(False)
        button_layout.addWidget(self.record_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self._on_settings_clicked)
        button_layout.addWidget(self.settings_button)
        
        layout.addLayout(button_layout)
        
        # Status display area
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: Idle")
        status_layout.addWidget(self.status_label)
        
        self.gesture_status_label = QLabel("Recognized Gesture: None")
        status_layout.addWidget(self.gesture_status_label)
        
        self.confidence_label = QLabel("Confidence: N/A")
        status_layout.addWidget(self.confidence_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """
        Create the right panel with gesture list and management buttons.
        
        Returns:
            QWidget: The right panel widget
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Gesture list
        list_label = QLabel("Saved Gestures")
        list_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(list_label)
        
        self.gesture_list = QListWidget()
        self.gesture_list.itemSelectionChanged.connect(self._on_gesture_selected)
        layout.addWidget(self.gesture_list)
        
        # Gesture management buttons
        button_layout = QHBoxLayout()
        
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self._on_edit_clicked)
        self.edit_button.setEnabled(False)
        button_layout.addWidget(self.edit_button)
        
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        self.delete_button.setEnabled(False)
        button_layout.addWidget(self.delete_button)
        
        layout.addLayout(button_layout)
        
        return panel
    
    def _update_frame(self):
        """
        Update the video frame display.
        
        Called by timer to process and display the current frame.
        
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        frame = self.controller.process_frame()
        
        if frame is not None:
            # Add visual feedback overlay
            frame = self._add_visual_feedback(frame)
            
            # Convert frame to QImage and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        
        # Update status
        self._update_status()
    
    def _update_status(self):
        """
        Update the status display based on controller state.
        
        Requirements: 5.1, 5.2, 5.4, 5.5
        """
        state = self.controller.get_state()
        
        if state == SystemState.IDLE:
            self.status_label.setText("Status: Idle")
            self.gesture_status_label.setText("Recognized Gesture: None")
            self.confidence_label.setText("Confidence: N/A")
        elif state == SystemState.RECORDING:
            progress = self.controller.get_recording_progress()
            self.status_label.setText(f"Status: Recording ({progress*100:.0f}%)")
            self.gesture_status_label.setText("Recognized Gesture: Recording...")
            self.confidence_label.setText("Confidence: N/A")
        elif state == SystemState.RECOGNIZING:
            self.status_label.setText("Status: Recognizing")
            
            # Update gesture recognition status
            match_info = self.controller.get_current_match()
            if match_info:
                gesture_name, confidence, distance = match_info
                self.gesture_status_label.setText(f"Recognized Gesture: {gesture_name}")
                self.confidence_label.setText(f"Confidence: {confidence:.1%} (distance: {distance:.4f})")
            else:
                self.gesture_status_label.setText("Recognized Gesture: None")
                self.confidence_label.setText("Confidence: Below threshold")
    
    def _add_visual_feedback(self, frame: np.ndarray) -> np.ndarray:
        """
        Add visual feedback overlay to the frame.
        
        Displays recognized gesture name and confidence score on the video feed.
        
        Args:
            frame: The video frame
            
        Returns:
            np.ndarray: Frame with overlay
            
        Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
        """
        # Get current match information
        match_info = self.controller.get_current_match()
        
        # Prepare overlay text
        if self.controller.get_state() == SystemState.RECORDING:
            progress = self.controller.get_recording_progress()
            text = f"RECORDING: {progress*100:.0f}%"
            color = (0, 0, 255)  # Red for recording
        elif match_info:
            gesture_name, confidence, distance = match_info
            text = f"{gesture_name} ({confidence:.1%})"
            color = (0, 255, 0)  # Green for recognized
        else:
            text = "No gesture recognized"
            color = (128, 128, 128)  # Gray for neutral
        
        # Add text overlay to frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position at top center
        x = (frame.shape[1] - text_width) // 2
        y = 40
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - 10, y - text_height - 10),
            (x + text_width + 10, y + baseline + 10),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        # Add confidence score if available
        if match_info and self.controller.get_state() == SystemState.RECOGNIZING:
            _, confidence, distance = match_info
            confidence_text = f"Distance: {distance:.4f}"
            
            # Position below main text
            (conf_width, conf_height), conf_baseline = cv2.getTextSize(
                confidence_text, font, 0.6, 1
            )
            conf_x = (frame.shape[1] - conf_width) // 2
            conf_y = y + text_height + 30
            
            # Draw background
            cv2.rectangle(
                frame,
                (conf_x - 5, conf_y - conf_height - 5),
                (conf_x + conf_width + 5, conf_y + conf_baseline + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                confidence_text,
                (conf_x, conf_y),
                font,
                0.6,
                color,
                1,
                cv2.LINE_AA
            )
        
        return frame
    
    def _on_start_clicked(self):
        """Handle start button click."""
        if self.controller.start():
            self.timer.start(33)  # ~30 FPS
            self.start_button.setEnabled(False)
            self.record_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self._refresh_gesture_list()
            logger.info("System started")
        else:
            QMessageBox.critical(self, "Error", "Failed to start the system. Check camera connection.")
    
    def _on_stop_clicked(self):
        """Handle stop button click."""
        self.timer.stop()
        self.controller.stop()
        self.start_button.setEnabled(True)
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("Camera Stopped")
        logger.info("System stopped")
    
    def _on_record_clicked(self):
        """Handle record button click."""
        # Show recording dialog
        dialog = RecordingDialog(self.controller, self)
        if dialog.exec_() == QDialog.Accepted:
            self._refresh_gesture_list()
    
    def _on_settings_clicked(self):
        """Handle settings button click."""
        dialog = SettingsDialog(self.controller.config, self)
        if dialog.exec_() == QDialog.Accepted:
            new_config = dialog.get_config()
            self.controller.update_settings(new_config)
            logger.info("Settings updated")
    
    def _on_gesture_selected(self):
        """Handle gesture selection in the list."""
        selected_items = self.gesture_list.selectedItems()
        has_selection = len(selected_items) > 0
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
    
    def _on_edit_clicked(self):
        """Handle edit button click."""
        selected_items = self.gesture_list.selectedItems()
        if not selected_items:
            return
        
        gesture_name = selected_items[0].data(Qt.UserRole)
        gesture = self.controller.get_gesture(gesture_name)
        
        if gesture:
            dialog = EditGestureDialog(gesture, self)
            if dialog.exec_() == QDialog.Accepted:
                updated_gesture = dialog.get_gesture()
                # Update gesture in repository
                self.controller.gesture_repository.update_gesture(gesture_name, updated_gesture)
                self.controller._load_gestures()  # Reload cache
                self._refresh_gesture_list()
                logger.info(f"Gesture '{gesture_name}' updated")
    
    def _on_delete_clicked(self):
        """Handle delete button click."""
        selected_items = self.gesture_list.selectedItems()
        if not selected_items:
            return
        
        gesture_name = selected_items[0].data(Qt.UserRole)
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the gesture '{gesture_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.controller.delete_gesture(gesture_name):
                self._refresh_gesture_list()
                logger.info(f"Gesture '{gesture_name}' deleted")
            else:
                QMessageBox.warning(self, "Error", f"Failed to delete gesture '{gesture_name}'")
    
    def _refresh_gesture_list(self):
        """Refresh the gesture list display."""
        self.gesture_list.clear()
        gestures = self.controller.get_all_gestures()
        
        for gesture in gestures:
            action_desc = self._format_action(gesture.action)
            item_text = f"{gesture.name} → {action_desc}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, gesture.name)
            self.gesture_list.addItem(item)
    
    def _format_action(self, action: Action) -> str:
        """
        Format an action for display.
        
        Args:
            action: The action to format
            
        Returns:
            str: Formatted action description
        """
        from src.models import ExecutionMode
        
        # Format the action type and data
        if action.type == ActionType.LAUNCH_APP:
            action_desc = f"Launch: {action.data.get('path', 'Unknown')}"
        elif action.type == ActionType.KEYSTROKE:
            keys = action.data.get('keys', [])
            action_desc = f"Keys: {'+'.join(keys)}"
        elif action.type == ActionType.MEDIA_CONTROL:
            action_desc = f"Media: {action.data.get('command', 'Unknown')}"
        elif action.type == ActionType.SYSTEM_CONTROL:
            action_desc = f"System: {action.data.get('command', 'Unknown')}"
        else:
            action_desc = "Unknown Action"
        
        # Add execution mode indicator
        mode_indicator = "🔄" if action.execution_mode == ExecutionMode.HOLD_REPEAT else "⚡"
        
        return f"{mode_indicator} {action_desc}"
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.controller.is_running():
            self.controller.stop()
        event.accept()




class RecordingDialog(QDialog):
    """
    Dialog for recording a new gesture.
    
    Shows recording progress, prompts for gesture name, and allows action selection.
    
    Requirements: 3.1, 3.3
    """
    
    def __init__(self, controller: MainController, parent=None):
        """
        Initialize the recording dialog.
        
        Args:
            controller: The MainController instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.controller = controller
        
        self.setWindowTitle("Record Gesture")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel(
            "Position your hand in front of the camera and click 'Start Recording'.\n"
            "Perform your gesture and hold it steady during the recording."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to record")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.start_record_button = QPushButton("Start Recording")
        self.start_record_button.clicked.connect(self._on_start_recording)
        button_layout.addWidget(self.start_record_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        # Timer for updating progress
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_progress)
        
        logger.info("RecordingDialog initialized")
    
    def _on_start_recording(self):
        """Start the recording process."""
        self.start_record_button.setEnabled(False)
        self.status_label.setText("Recording...")
        
        # Start recording in controller
        if self.controller.enter_recording_mode():
            self.timer.start(100)  # Update every 100ms
        else:
            QMessageBox.critical(self, "Error", "Failed to start recording")
            self.reject()
    
    def _update_progress(self):
        """Update the recording progress bar."""
        progress = self.controller.get_recording_progress()
        self.progress_bar.setValue(int(progress * 100))
        
        # Check if recording is complete
        if self.controller.is_recording_complete():
            self.timer.stop()
            self.status_label.setText("Recording complete!")
            self._show_save_dialog()
    
    def _show_save_dialog(self):
        """Show dialog to save the recorded gesture."""
        # Prompt for gesture name
        name_dialog = GestureNameDialog(self.controller, self)
        if name_dialog.exec_() == QDialog.Accepted:
            gesture_name = name_dialog.get_gesture_name()
            
            # Show action selection dialog
            action_dialog = ActionSelectionDialog(self)
            if action_dialog.exec_() == QDialog.Accepted:
                action = action_dialog.get_action()
                
                # Save the gesture
                if self.controller.save_recorded_gesture(gesture_name, action):
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Gesture '{gesture_name}' saved successfully!"
                    )
                    self.accept()
                else:
                    QMessageBox.critical(
                        self,
                        "Error",
                        "Failed to save gesture. Please try again."
                    )
                    self.reject()
            else:
                self.controller.cancel_recording()
                self.reject()
        else:
            self.controller.cancel_recording()
            self.reject()


class GestureNameDialog(QDialog):
    """
    Dialog for entering a gesture name.
    
    Validates that the name is unique and non-empty.
    
    Requirements: 3.1, 3.3
    """
    
    def __init__(self, controller: MainController, parent=None):
        """
        Initialize the gesture name dialog.
        
        Args:
            controller: The MainController instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.controller = controller
        
        self.setWindowTitle("Name Your Gesture")
        self.setModal(True)
        
        layout = QFormLayout(self)
        
        # Name input
        self.name_input = QLineEdit()
        layout.addRow("Gesture Name:", self.name_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
    
    def _on_accept(self):
        """Validate and accept the gesture name."""
        name = self.name_input.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Gesture name cannot be empty.")
            return
        
        # Check if name already exists
        existing_gestures = self.controller.get_all_gestures()
        if any(g.name == name for g in existing_gestures):
            QMessageBox.warning(
                self,
                "Duplicate Name",
                f"A gesture named '{name}' already exists. Please choose a different name."
            )
            return
        
        self.accept()
    
    def get_gesture_name(self) -> str:
        """
        Get the entered gesture name.
        
        Returns:
            str: The gesture name
        """
        return self.name_input.text().strip()


class ActionSelectionDialog(QDialog):
    """
    Dialog for selecting an action to assign to a gesture.
    
    Provides options for different action types and their parameters.
    
    Requirements: 3.3
    """
    
    def __init__(self, parent=None):
        """
        Initialize the action selection dialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.setWindowTitle("Select Action")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Action type selection
        form_layout = QFormLayout()
        
        self.action_type_combo = QComboBox()
        self.action_type_combo.addItem("Launch Application", ActionType.LAUNCH_APP)
        self.action_type_combo.addItem("Keystroke", ActionType.KEYSTROKE)
        self.action_type_combo.addItem("Media Control", ActionType.MEDIA_CONTROL)
        self.action_type_combo.addItem("System Control", ActionType.SYSTEM_CONTROL)
        self.action_type_combo.currentIndexChanged.connect(self._on_action_type_changed)
        form_layout.addRow("Action Type:", self.action_type_combo)
        
        # Execution mode selection
        from src.models import ExecutionMode
        self.execution_mode_combo = QComboBox()
        self.execution_mode_combo.addItem("Trigger Once (on detection)", ExecutionMode.TRIGGER_ONCE)
        self.execution_mode_combo.addItem("Hold to Repeat (while held)", ExecutionMode.HOLD_REPEAT)
        form_layout.addRow("Execution Mode:", self.execution_mode_combo)
        
        # Add help text for execution mode
        mode_help = QLabel(
            "• Trigger Once: Execute action once when gesture is first detected\n"
            "• Hold to Repeat: Execute action repeatedly while gesture is held"
        )
        mode_help.setWordWrap(True)
        mode_help.setStyleSheet("color: gray; font-size: 10px; padding: 5px;")
        form_layout.addRow("", mode_help)
        
        layout.addLayout(form_layout)
        
        # Action-specific parameters (stacked widget would be better, but keeping it simple)
        self.params_widget = QWidget()
        self.params_layout = QFormLayout(self.params_widget)
        layout.addWidget(self.params_widget)
        
        # Initialize with first action type
        self._on_action_type_changed(0)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _on_action_type_changed(self, index):
        """Handle action type selection change."""
        # Clear previous parameters
        while self.params_layout.count():
            child = self.params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        action_type = self.action_type_combo.currentData()
        
        if action_type == ActionType.LAUNCH_APP:
            self.app_path_input = QLineEdit()
            self.app_path_input.setPlaceholderText("e.g., notepad.exe or /usr/bin/firefox")
            self.params_layout.addRow("Application Path:", self.app_path_input)
        
        elif action_type == ActionType.KEYSTROKE:
            self.keys_input = QLineEdit()
            self.keys_input.setPlaceholderText("e.g., ctrl+c or alt+tab")
            self.params_layout.addRow("Key Combination:", self.keys_input)
            
            help_label = QLabel("Separate keys with '+'. Examples: ctrl+c, alt+f4, win+d")
            help_label.setWordWrap(True)
            help_label.setStyleSheet("color: gray; font-size: 10px;")
            self.params_layout.addRow("", help_label)
        
        elif action_type == ActionType.MEDIA_CONTROL:
            self.media_command_combo = QComboBox()
            self.media_command_combo.addItems([
                "play_pause", "next", "previous", "volume_up", "volume_down", "mute"
            ])
            self.params_layout.addRow("Media Command:", self.media_command_combo)
        
        elif action_type == ActionType.SYSTEM_CONTROL:
            self.system_command_combo = QComboBox()
            self.system_command_combo.addItems([
                "volume_up", "volume_down", "volume_mute", "lock",
                "switch_window", "minimize", "maximize", "close_window"
            ])
            self.params_layout.addRow("System Command:", self.system_command_combo)
            
            self.amount_spin = QSpinBox()
            self.amount_spin.setRange(1, 10)
            self.amount_spin.setValue(1)
            self.params_layout.addRow("Amount (for volume):", self.amount_spin)
    
    def get_action(self) -> Action:
        """
        Get the configured action.
        
        Returns:
            Action: The configured action object
        """
        action_type = self.action_type_combo.currentData()
        execution_mode = self.execution_mode_combo.currentData()
        
        if action_type == ActionType.LAUNCH_APP:
            return Action(
                type=ActionType.LAUNCH_APP,
                data={'path': self.app_path_input.text()},
                execution_mode=execution_mode
            )
        
        elif action_type == ActionType.KEYSTROKE:
            keys_str = self.keys_input.text()
            keys = [k.strip() for k in keys_str.split('+')]
            return Action(
                type=ActionType.KEYSTROKE,
                data={'keys': keys},
                execution_mode=execution_mode
            )
        
        elif action_type == ActionType.MEDIA_CONTROL:
            return Action(
                type=ActionType.MEDIA_CONTROL,
                data={'command': self.media_command_combo.currentText()},
                execution_mode=execution_mode
            )
        
        elif action_type == ActionType.SYSTEM_CONTROL:
            return Action(
                type=ActionType.SYSTEM_CONTROL,
                data={
                    'command': self.system_command_combo.currentText(),
                    'amount': self.amount_spin.value()
                },
                execution_mode=execution_mode
            )


class EditGestureDialog(QDialog):
    """
    Dialog for editing an existing gesture.
    
    Allows changing the gesture name and assigned action.
    
    Requirements: 6.1, 6.2, 6.3
    """
    
    def __init__(self, gesture: Gesture, parent=None):
        """
        Initialize the edit gesture dialog.
        
        Args:
            gesture: The gesture to edit
            parent: Parent widget
        """
        super().__init__(parent)
        self.original_gesture = gesture
        
        self.setWindowTitle(f"Edit Gesture: {gesture.name}")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Name input
        form_layout = QFormLayout()
        self.name_input = QLineEdit(gesture.name)
        form_layout.addRow("Gesture Name:", self.name_input)
        layout.addLayout(form_layout)
        
        # Action selection (reuse ActionSelectionDialog logic)
        action_group = QGroupBox("Action")
        action_layout = QVBoxLayout()
        
        self.action_selector = ActionSelectionDialog(self)
        # Pre-populate with current action
        self._populate_action(gesture.action)
        
        action_layout.addWidget(self.action_selector)
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _populate_action(self, action: Action):
        """Pre-populate the action selector with current action."""
        # Set action type
        for i in range(self.action_selector.action_type_combo.count()):
            if self.action_selector.action_type_combo.itemData(i) == action.type:
                self.action_selector.action_type_combo.setCurrentIndex(i)
                break
        
        # Set action-specific data
        if action.type == ActionType.LAUNCH_APP:
            self.action_selector.app_path_input.setText(action.data.get('path', ''))
        elif action.type == ActionType.KEYSTROKE:
            keys = action.data.get('keys', [])
            self.action_selector.keys_input.setText('+'.join(keys))
        elif action.type == ActionType.MEDIA_CONTROL:
            command = action.data.get('command', '')
            index = self.action_selector.media_command_combo.findText(command)
            if index >= 0:
                self.action_selector.media_command_combo.setCurrentIndex(index)
        elif action.type == ActionType.SYSTEM_CONTROL:
            command = action.data.get('command', '')
            index = self.action_selector.system_command_combo.findText(command)
            if index >= 0:
                self.action_selector.system_command_combo.setCurrentIndex(index)
            amount = action.data.get('amount', 1)
            self.action_selector.amount_spin.setValue(amount)
    
    def get_gesture(self) -> Gesture:
        """
        Get the updated gesture.
        
        Returns:
            Gesture: The updated gesture object
        """
        return Gesture(
            name=self.name_input.text().strip(),
            embedding=self.original_gesture.embedding,
            action=self.action_selector.get_action(),
            created_at=self.original_gesture.created_at
        )


class SettingsDialog(QDialog):
    """
    Dialog for configuring system settings.
    
    Provides input fields for all configurable parameters with validation.
    
    Requirements: 10.1, 10.2, 10.3
    """
    
    def __init__(self, config: Config, parent=None):
        """
        Initialize the settings dialog.
        
        Args:
            config: Current configuration
            parent: Parent widget
        """
        super().__init__(parent)
        self.config = config
        
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QFormLayout(self)
        
        # Similarity threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 3.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(config.similarity_threshold)
        layout.addRow("Similarity Threshold:", self.threshold_spin)
        
        # Recording duration
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 10.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setValue(config.recording_duration)
        self.duration_spin.setSuffix(" seconds")
        layout.addRow("Recording Duration:", self.duration_spin)
        
        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 60)
        self.fps_spin.setValue(config.fps)
        layout.addRow("Camera FPS:", self.fps_spin)
        
        # Smoothing window
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(1, 20)
        self.smoothing_spin.setValue(config.smoothing_window)
        layout.addRow("Smoothing Window:", self.smoothing_spin)
        
        # Max hands
        self.max_hands_spin = QSpinBox()
        self.max_hands_spin.setRange(1, 2)
        self.max_hands_spin.setValue(config.max_hands)
        layout.addRow("Max Hands:", self.max_hands_spin)
        
        # Detection confidence
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(config.detection_confidence)
        layout.addRow("Detection Confidence:", self.confidence_spin)
        
        # Embedding dimensions
        self.embedding_spin = QSpinBox()
        self.embedding_spin.setRange(8, 32)
        self.embedding_spin.setValue(config.embedding_dimensions)
        layout.addRow("Embedding Dimensions:", self.embedding_spin)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
    
    def _on_accept(self):
        """Validate and accept the settings."""
        # Validation is handled by spin box ranges
        self.accept()
    
    def get_config(self) -> Config:
        """
        Get the configured settings.
        
        Returns:
            Config: The new configuration object
        """
        return Config(
            similarity_threshold=self.threshold_spin.value(),
            recording_duration=self.duration_spin.value(),
            fps=self.fps_spin.value(),
            smoothing_window=self.smoothing_spin.value(),
            max_hands=self.max_hands_spin.value(),
            detection_confidence=self.confidence_spin.value(),
            embedding_dimensions=self.embedding_spin.value()
        )


def main():
    """Main entry point for the GUI application."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create controller
    controller = MainController()
    
    # Create and show main window
    window = MainWindow(controller)
    window.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
