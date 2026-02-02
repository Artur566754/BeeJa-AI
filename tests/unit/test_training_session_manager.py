"""
Unit tests for the TrainingSessionManager.

Tests session creation, lifecycle management, queueing, and state transitions.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import time

from api.database import Database
from api.services.training_session_manager import TrainingSessionManager
from api.services.training_executor import TrainingExecutor
from api.services.metrics_store import MetricsStore
from api.services.log_store import LogStore
from api.models.data_models import TrainingConfig, SessionState


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    db = Database(db_path)
    yield db
    
    # Cleanup
    try:
        db.close()
        time.sleep(0.1)  # Give Windows time to release the file
        if db_path.exists():
            db_path.unlink()
    except PermissionError:
        pass  # Ignore cleanup errors on Windows


@pytest.fixture
def session_manager(temp_db):
    """Create a TrainingSessionManager with temporary database."""
    # Initialize database for this test
    from api.database import init_database
    init_database(temp_db.db_path)
    
    executor = TrainingExecutor()
    
    return TrainingSessionManager(
        database=temp_db,
        executor=executor
    )


@pytest.fixture
def sample_config():
    """Create a sample training configuration."""
    return TrainingConfig(
        model_architecture="test_model",
        dataset_name="test_dataset",
        learning_rate=0.001,
        batch_size=32,
        epochs=2
    )


def test_create_session(session_manager, sample_config):
    """Test creating a new training session."""
    session_id = session_manager.create_session(sample_config)
    
    assert session_id.startswith("sess_")
    
    # Verify session exists in database
    status = session_manager.get_status(session_id)
    assert status is not None
    assert status.session_id == session_id
    assert status.total_epochs == sample_config.epochs


def test_get_status_existing_session(session_manager, sample_config):
    """Test getting status for an existing session."""
    session_id = session_manager.create_session(sample_config)
    
    status = session_manager.get_status(session_id)
    
    assert status is not None
    assert status.session_id == session_id
    assert status.state in [SessionState.QUEUED, SessionState.RUNNING]
    assert status.total_epochs == sample_config.epochs


def test_get_status_nonexistent_session(session_manager):
    """Test getting status for a nonexistent session."""
    status = session_manager.get_status("nonexistent_session")
    assert status is None


def test_session_starts_automatically(session_manager, sample_config):
    """Test that session starts automatically when resources available."""
    session_id = session_manager.create_session(sample_config)
    
    # Give it a moment to start
    time.sleep(0.1)
    
    status = session_manager.get_status(session_id)
    
    # Should be running or completed (if very fast)
    assert status.state in [SessionState.RUNNING, SessionState.COMPLETED]


def test_session_completes_successfully(session_manager, sample_config):
    """Test that a session completes successfully."""
    # Use a very short config
    short_config = TrainingConfig(
        model_architecture="test_model",
        dataset_name="test_dataset",
        learning_rate=0.001,
        batch_size=32,
        epochs=1
    )
    
    session_id = session_manager.create_session(short_config)
    
    # Wait for completion
    max_wait = 10  # seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status = session_manager.get_status(session_id)
        if status.state == SessionState.COMPLETED:
            break
        time.sleep(0.1)
    
    # Verify completion
    status = session_manager.get_status(session_id)
    assert status.state == SessionState.COMPLETED
    assert status.end_time is not None


def test_stop_running_session(session_manager):
    """Test stopping a running session."""
    # Create a long-running session
    long_config = TrainingConfig(
        model_architecture="test_model",
        dataset_name="test_dataset",
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )
    
    session_id = session_manager.create_session(long_config)
    
    # Wait for it to start
    time.sleep(0.2)
    
    # Stop the session
    session_manager.stop_session(session_id)
    
    # Wait a bit for stop to take effect
    time.sleep(0.5)
    
    # Verify it stopped
    status = session_manager.get_status(session_id)
    assert status.state == SessionState.STOPPED


def test_stop_nonrunning_session_raises_error(session_manager, sample_config):
    """Test that stopping a non-running session raises error."""
    session_id = session_manager.create_session(sample_config)
    
    # Wait for completion
    time.sleep(2)
    
    # Try to stop completed session
    with pytest.raises(ValueError, match="not running"):
        session_manager.stop_session(session_id)


def test_get_active_sessions(session_manager):
    """Test getting list of active sessions."""
    # Create a long-running session
    long_config = TrainingConfig(
        model_architecture="test_model",
        dataset_name="test_dataset",
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )
    
    session_id = session_manager.create_session(long_config)
    
    # Wait for it to start
    time.sleep(0.2)
    
    # Get active sessions
    active = session_manager.get_active_sessions()
    
    # Should have at least one active session
    assert len(active) > 0
    assert any(s.session_id == session_id for s in active)
    
    # Cleanup
    session_manager.stop_session(session_id)


def test_can_start_new_session(session_manager):
    """Test checking if new session can be started."""
    # Initially should be able to start
    assert session_manager.can_start_new_session() is True


def test_session_metrics_collected(session_manager, sample_config):
    """Test that session metrics are collected."""
    session_id = session_manager.create_session(sample_config)
    
    # Wait for some metrics to be collected
    time.sleep(1)
    
    # Get metrics
    metrics = session_manager.metrics_store.get_metrics_history(session_id)
    
    # Should have some metrics
    assert len(metrics) > 0


def test_session_logs_collected(session_manager, sample_config):
    """Test that session logs are collected."""
    session_id = session_manager.create_session(sample_config)
    
    # Wait a bit
    time.sleep(0.5)
    
    # Get logs
    logs = session_manager.log_store.get_logs(session_id)
    
    # Should have some logs
    assert len(logs) > 0
    assert any("created" in log.message.lower() for log in logs)
