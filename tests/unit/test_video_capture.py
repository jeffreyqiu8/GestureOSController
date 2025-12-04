"""
Unit tests for the VideoCapture component.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.video_capture import VideoCapture


class TestVideoCapture:
    """Test suite for VideoCapture class."""
    
    def test_initialization(self):
        """Test that VideoCapture initializes with correct default values."""
        vc = VideoCapture()
        assert vc._camera_index == 0
        assert vc._fps == 30
        assert vc._capture is None
        assert vc._is_running is False
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom camera index and FPS."""
        vc = VideoCapture(camera_index=1, fps=60)
        assert vc._camera_index == 1
        assert vc._fps == 60
    
    @patch('cv2.VideoCapture')
    def test_start_success(self, mock_cv2_capture):
        """Test successful webcam initialization."""
        # Setup mock
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2_capture.return_value = mock_capture_instance
        
        # Test
        vc = VideoCapture()
        result = vc.start()
        
        # Verify
        assert result is True
        assert vc.is_running() is True
        mock_cv2_capture.assert_called_once_with(0)
        mock_capture_instance.set.assert_called_once_with(5, 30)  # CAP_PROP_FPS = 5
    
    @patch('cv2.VideoCapture')
    def test_start_failure_camera_not_opened(self, mock_cv2_capture):
        """Test start failure when camera cannot be opened."""
        # Setup mock
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_capture_instance
        
        # Test
        vc = VideoCapture()
        result = vc.start()
        
        # Verify
        assert result is False
        assert vc.is_running() is False
    
    @patch('cv2.VideoCapture')
    def test_start_failure_cannot_read_frame(self, mock_cv2_capture):
        """Test start failure when camera opens but cannot read frames."""
        # Setup mock
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (False, None)
        mock_cv2_capture.return_value = mock_capture_instance
        
        # Test
        vc = VideoCapture()
        result = vc.start()
        
        # Verify
        assert result is False
        assert vc.is_running() is False
        mock_capture_instance.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_stop(self, mock_cv2_capture):
        """Test stopping video capture."""
        # Setup mock
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2_capture.return_value = mock_capture_instance
        
        # Test
        vc = VideoCapture()
        vc.start()
        assert vc.is_running() is True
        
        vc.stop()
        
        # Verify
        assert vc.is_running() is False
        mock_capture_instance.release.assert_called()
    
    @patch('cv2.VideoCapture')
    def test_get_frame_success(self, mock_cv2_capture):
        """Test successful frame retrieval."""
        # Setup mock
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.side_effect = [
            (True, test_frame),  # First call in start()
            (True, test_frame)   # Second call in get_frame()
        ]
        mock_cv2_capture.return_value = mock_capture_instance
        
        # Test
        vc = VideoCapture()
        vc.start()
        frame = vc.get_frame()
        
        # Verify
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
    
    def test_get_frame_when_not_running(self):
        """Test get_frame returns None when capture is not running."""
        vc = VideoCapture()
        frame = vc.get_frame()
        assert frame is None
    
    @patch('cv2.VideoCapture')
    def test_get_frame_read_failure(self, mock_cv2_capture):
        """Test get_frame returns None when frame read fails."""
        # Setup mock
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First call in start()
            (False, None)  # Second call in get_frame()
        ]
        mock_cv2_capture.return_value = mock_capture_instance
        
        # Test
        vc = VideoCapture()
        vc.start()
        frame = vc.get_frame()
        
        # Verify
        assert frame is None
    
    def test_is_running_initially_false(self):
        """Test is_running returns False initially."""
        vc = VideoCapture()
        assert vc.is_running() is False
    
    @patch('cv2.VideoCapture')
    def test_set_fps_when_not_running(self, mock_cv2_capture):
        """Test setting FPS when capture is not running."""
        vc = VideoCapture()
        vc.set_fps(60)
        assert vc._fps == 60
    
    @patch('cv2.VideoCapture')
    def test_set_fps_when_running(self, mock_cv2_capture):
        """Test setting FPS when capture is running."""
        # Setup mock
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2_capture.return_value = mock_capture_instance
        
        # Test
        vc = VideoCapture()
        vc.start()
        vc.set_fps(60)
        
        # Verify
        assert vc._fps == 60
        # Should be called twice: once in start(), once in set_fps()
        assert mock_capture_instance.set.call_count == 2
    
    def test_set_fps_invalid_value(self):
        """Test that set_fps raises ValueError for non-positive FPS."""
        vc = VideoCapture()
        
        with pytest.raises(ValueError, match="FPS must be positive"):
            vc.set_fps(0)
        
        with pytest.raises(ValueError, match="FPS must be positive"):
            vc.set_fps(-10)
