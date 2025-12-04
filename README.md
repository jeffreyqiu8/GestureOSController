# Gesture Control System

A Python application that enables users to control their computer using custom hand gestures captured via webcam.

## Features

- Real-time hand tracking using MediaPipe
- Custom gesture recording and recognition
- Configurable OS-level actions (launch apps, keystrokes, media control, system control)
- PyQt5 graphical user interface
- Machine learning-based gesture matching using PCA embeddings

## Project Structure

```
GestureOSController/
├── src/                    # Source code
│   └── __init__.py
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── property/          # Property-based tests
│   └── conftest.py        # Shared test fixtures
├── config/                 # Configuration files
│   └── default_config.json
├── logs/                   # Application logs (created at runtime)
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application with default settings:
```bash
python main.py
```

Run with custom configuration:
```bash
python main.py --config path/to/config.json
```

Run with debug logging:
```bash
python main.py --debug
```

## Configuration

The default configuration file is located at `config/default_config.json`. You can customize:

- `similarity_threshold`: Maximum distance for gesture matching (0.0 to 1.0)
- `recording_duration`: Duration in seconds for gesture recording
- `fps`: Target frames per second for video capture
- `smoothing_window`: Number of frames for smoothing filter
- `max_hands`: Maximum number of hands to detect
- `detection_confidence`: Confidence threshold for hand detection (0.0 to 1.0)
- `embedding_dimensions`: Dimensionality of gesture embeddings

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

Run only unit tests:
```bash
pytest tests/unit/
```

Run only property-based tests:
```bash
pytest tests/property/
```

## Development Status

This project is currently under development. The basic project structure has been set up, and implementation of core features is in progress.

## Requirements

- Python 3.8 or higher
- Webcam
- Windows/Linux/Mac OS

## License

[Add license information here]

## Contributing

[Add contribution guidelines here]
