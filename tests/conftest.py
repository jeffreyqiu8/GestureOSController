"""
Pytest configuration and shared fixtures for the test suite.

This file contains shared test fixtures and Hypothesis strategies that will be
used across unit tests, integration tests, and property-based tests.
"""

import pytest
import numpy as np
from hypothesis import strategies as st


# Hypothesis strategies for property-based testing
# These will be expanded as we implement the data models

@st.composite
def landmark_strategy(draw):
    """Generate valid 21x3 landmark arrays for hand tracking."""
    # Each landmark has x, y, z coordinates normalized between -1 and 1
    landmarks = draw(st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=63,
        max_size=63
    ))
    return np.array(landmarks).reshape(21, 3)


@st.composite
def embedding_strategy(draw):
    """Generate valid 16-dimensional embedding vectors."""
    embedding = draw(st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=16,
        max_size=16
    ))
    return np.array(embedding)


# Pytest fixtures

@pytest.fixture
def sample_landmarks():
    """Provide sample hand landmark data for testing."""
    # Create a simple hand landmark array with known values
    landmarks = np.random.rand(21, 3)
    return landmarks


@pytest.fixture
def sample_embedding():
    """Provide sample embedding vector for testing."""
    return np.random.rand(16)


@pytest.fixture
def temp_config_file(tmp_path):
    """Provide a temporary configuration file path."""
    config_file = tmp_path / "test_config.json"
    return str(config_file)


@pytest.fixture
def temp_database(tmp_path):
    """Provide a temporary database file path."""
    db_file = tmp_path / "test_gestures.db"
    return str(db_file)
