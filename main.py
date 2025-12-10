#!/usr/bin/env python3
"""
Gesture Control System - Main Entry Point

This application enables users to control their computer using custom hand gestures
captured via webcam. Users can record personalized gestures, assign them to system
actions, and perform those gestures in real-time to execute the assigned actions.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_logging(debug: bool = False) -> None:
    """
    Configure comprehensive logging for the application.
    
    Logs errors, warnings, and info to file with timestamps and context.
    Console output shows INFO and above, file logs show DEBUG and above.
    
    Args:
        debug: If True, enables DEBUG level logging to console
    
    Requirements: 7.5
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create file handler with DEBUG level (captures everything)
    file_handler = logging.FileHandler(
        log_dir / "gesture_control.log",
        mode='a',  # Append mode
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # Create console handler with INFO or DEBUG level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Gesture Control System - Logging initialized")
    logger.info(f"Log file: {log_dir / 'gesture_control.log'}")
    logger.info(f"Log level - File: DEBUG, Console: {'DEBUG' if debug else 'INFO'}")
    logger.info("=" * 80)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Gesture Control System - Control your computer with hand gestures"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to configuration file (default: config/default_config.json)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def main() -> int:
    """Main application entry point."""
    args = parse_arguments()
    setup_logging(args.debug)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Gesture Control System")
    logger.info(f"Using configuration file: {args.config}")
    
    try:
        # Import GUI components
        from PyQt5.QtWidgets import QApplication
        from src.gui import MainWindow
        from src.main_controller import MainController
        from src.models import Config
        
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            config = Config.load(str(config_path))
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Using default configuration")
            config = Config()
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create controller
        controller = MainController(config)
        
        # Create and show main window
        window = MainWindow(controller)
        window.show()
        
        logger.info("GUI initialized successfully")
        
        # Run application
        return app.exec_()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1
    finally:
        logger.info("Gesture Control System shutting down")


if __name__ == "__main__":
    sys.exit(main())
