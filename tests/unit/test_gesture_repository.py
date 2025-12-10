"""
Unit tests for GestureRepository.
"""
import pytest
import numpy as np
import tempfile
import os
from datetime import datetime
from pathlib import Path

from src.gesture_repository import GestureRepository
from src.models import Gesture, Action, ActionType


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def repository(temp_db):
    """Create a GestureRepository instance with temporary database."""
    return GestureRepository(temp_db)


@pytest.fixture
def sample_gesture():
    """Create a sample gesture for testing."""
    embedding = np.random.rand(16)
    action = Action(
        type=ActionType.LAUNCH_APP,
        data={"path": "/usr/bin/firefox"}
    )
    return Gesture(
        name="wave",
        embedding=embedding,
        action=action,
        created_at=datetime.now()
    )


def test_initialize_database(temp_db):
    """Test that database is initialized correctly."""
    repo = GestureRepository(temp_db)
    assert os.path.exists(temp_db)


def test_save_gesture(repository, sample_gesture):
    """Test saving a gesture to the database."""
    repository.save_gesture(sample_gesture)
    
    # Verify it was saved
    retrieved = repository.get_gesture(sample_gesture.name)
    assert retrieved is not None
    assert retrieved.name == sample_gesture.name


def test_save_duplicate_gesture_raises_error(repository, sample_gesture):
    """Test that saving a duplicate gesture raises an error."""
    repository.save_gesture(sample_gesture)
    
    # Try to save again with same name
    with pytest.raises(Exception):  # sqlite3.IntegrityError
        repository.save_gesture(sample_gesture)


def test_get_gesture_existing(repository, sample_gesture):
    """Test retrieving an existing gesture."""
    repository.save_gesture(sample_gesture)
    
    retrieved = repository.get_gesture(sample_gesture.name)
    
    assert retrieved is not None
    assert retrieved.name == sample_gesture.name
    assert np.allclose(retrieved.embedding, sample_gesture.embedding)
    assert retrieved.action.type == sample_gesture.action.type
    assert retrieved.action.data == sample_gesture.action.data


def test_get_gesture_nonexistent(repository):
    """Test retrieving a non-existent gesture returns None."""
    result = repository.get_gesture("nonexistent")
    assert result is None


def test_get_all_gestures_empty(repository):
    """Test getting all gestures from empty database."""
    gestures = repository.get_all_gestures()
    assert gestures == []


def test_get_all_gestures_multiple(repository):
    """Test retrieving multiple gestures."""
    # Create multiple gestures
    gestures = []
    for i in range(3):
        gesture = Gesture(
            name=f"gesture_{i}",
            embedding=np.random.rand(16),
            action=Action(type=ActionType.KEYSTROKE, data={"keys": [f"key_{i}"]}),
            created_at=datetime.now()
        )
        gestures.append(gesture)
        repository.save_gesture(gesture)
    
    # Retrieve all
    all_gestures = repository.get_all_gestures()
    
    assert len(all_gestures) == 3
    names = {g.name for g in all_gestures}
    assert names == {"gesture_0", "gesture_1", "gesture_2"}


def test_delete_gesture_existing(repository, sample_gesture):
    """Test deleting an existing gesture."""
    repository.save_gesture(sample_gesture)
    
    # Delete it
    result = repository.delete_gesture(sample_gesture.name)
    
    assert result is True
    assert repository.get_gesture(sample_gesture.name) is None


def test_delete_gesture_nonexistent(repository):
    """Test deleting a non-existent gesture returns False."""
    result = repository.delete_gesture("nonexistent")
    assert result is False


def test_update_gesture_existing(repository, sample_gesture):
    """Test updating an existing gesture."""
    repository.save_gesture(sample_gesture)
    
    # Create updated gesture
    updated_gesture = Gesture(
        name="wave_updated",
        embedding=np.random.rand(16),
        action=Action(type=ActionType.MEDIA_CONTROL, data={"command": "play_pause"}),
        created_at=datetime.now()
    )
    
    # Update
    result = repository.update_gesture(sample_gesture.name, updated_gesture)
    
    assert result is True
    
    # Verify old name doesn't exist
    assert repository.get_gesture(sample_gesture.name) is None
    
    # Verify new name exists
    retrieved = repository.get_gesture("wave_updated")
    assert retrieved is not None
    assert retrieved.name == "wave_updated"
    assert retrieved.action.type == ActionType.MEDIA_CONTROL


def test_update_gesture_nonexistent(repository):
    """Test updating a non-existent gesture returns False."""
    gesture = Gesture(
        name="new_gesture",
        embedding=np.random.rand(16),
        action=Action(type=ActionType.LAUNCH_APP, data={"path": "/bin/app"}),
        created_at=datetime.now()
    )
    
    result = repository.update_gesture("nonexistent", gesture)
    assert result is False


def test_embedding_serialization_preserves_values(repository):
    """Test that embedding values are preserved through save/load cycle."""
    embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    
    gesture = Gesture(
        name="test_embedding",
        embedding=embedding,
        action=Action(type=ActionType.LAUNCH_APP, data={"path": "/test"}),
        created_at=datetime.now()
    )
    
    repository.save_gesture(gesture)
    retrieved = repository.get_gesture("test_embedding")
    
    assert np.allclose(retrieved.embedding, embedding)


def test_action_types_preserved(repository):
    """Test that all action types are preserved correctly."""
    action_types = [
        (ActionType.LAUNCH_APP, {"path": "/app"}),
        (ActionType.KEYSTROKE, {"keys": ["ctrl", "c"]}),
        (ActionType.MEDIA_CONTROL, {"command": "play"}),
        (ActionType.SYSTEM_CONTROL, {"command": "volume_up", "amount": 5})
    ]
    
    for i, (action_type, data) in enumerate(action_types):
        gesture = Gesture(
            name=f"gesture_{i}",
            embedding=np.random.rand(16),
            action=Action(type=action_type, data=data),
            created_at=datetime.now()
        )
        repository.save_gesture(gesture)
        
        retrieved = repository.get_gesture(f"gesture_{i}")
        assert retrieved.action.type == action_type
        assert retrieved.action.data == data
