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

### Prerequisites

**Important:** MediaPipe requires Python 3.8-3.12. Python 3.13 is not yet supported.

If you have Python 3.13, you can use `pyenv` to install Python 3.12:

**Windows (using pyenv-win):**
```powershell
# Install pyenv-win
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"
./install-pyenv-win.ps1

# Install Python 3.12
pyenv install 3.12.10
pyenv local 3.12.10
```

**Linux/Mac:**
```bash
# Install pyenv (if not already installed)
curl https://pyenv.run | bash

# Install Python 3.12
pyenv install 3.12.10
pyenv local 3.12.10
```

### Setup Steps

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

### Running the GUI Application

Launch the graphical user interface:
```bash
python src/gui.py
```

Or run the main entry point (when implemented):
```bash
python main.py
```

### Using the Application

1. **Start the System**
   - Click the "Start" button to initialize the webcam and begin hand tracking
   - The video feed will display with hand landmarks overlaid

2. **Record a New Gesture**
   - Click "Record Gesture" to begin recording
   - Position your hand in front of the camera
   - Click "Start Recording" in the dialog
   - Perform your gesture and hold it steady for the recording duration (default: 3 seconds)
   - Enter a unique name for your gesture
   - Select an action to assign (Launch App, Keystroke, Media Control, or System Control)
   - Configure the action parameters
   - Click OK to save

3. **Recognize Gestures**
   - Once gestures are recorded, the system automatically recognizes them in real-time
   - Recognized gestures are displayed on the video feed with confidence scores
   - The assigned action is executed when a gesture is recognized

4. **Manage Gestures**
   - View all saved gestures in the right panel
   - Select a gesture to edit or delete
   - Click "Edit" to modify the gesture name or assigned action
   - Click "Delete" to remove a gesture (with confirmation)

5. **Configure Settings**
   - Click "Settings" to adjust system parameters
   - Modify similarity threshold, recording duration, FPS, and other options
   - Changes take effect immediately

6. **Stop the System**
   - Click "Stop" to halt the webcam and gesture recognition

### Command Line Options

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

- **Python 3.8-3.12** (Python 3.13 not yet supported by MediaPipe)
- Webcam
- Windows/Linux/Mac OS

## License

[Add license information here]

## Contributing

[Add contribution guidelines here]
