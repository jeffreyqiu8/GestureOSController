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
    """Configure logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / "gesture_control.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


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
        # TODO: Initialize MainController and start application
        # This will be implemented in later tasks
        logger.info("Application skeleton initialized successfully")
        logger.info("Full implementation coming in subsequent tasks")
        
        return 0
        
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
