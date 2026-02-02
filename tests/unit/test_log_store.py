"""
Unit tests for the LogStore service.

Tests log storage, retrieval, and timestamp handling.
"""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from api.database import Database
from api.services.log_store import LogStore
from api.models.data_models import LogLevel


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    db = Database(db_path)
    yield db
    
    # Cleanup
    db.close()
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def log_store(temp_db):
    """Create a LogStore instance with temporary database."""
    return LogStore(temp_db)


def test_append_log_basic(log_store):
    """Test basic log appending."""
    session_id = "test_session_1"
    message = "Test log message"
    
    log_store.append_log(session_id, message, "INFO")
    
    logs = log_store.get_logs(session_id)
    assert len(logs) == 1
    assert logs[0].session_id == session_id
    assert logs[0].message == message
    assert logs[0].level == LogLevel.INFO


def test_append_log_different_levels(log_store):
    """Test appending logs with different severity levels."""
    session_id = "test_session_2"
    
    log_store.append_log(session_id, "Info message", "INFO")
    log_store.append_log(session_id, "Warning message", "WARNING")
    log_store.append_log(session_id, "Error message", "ERROR")
    log_store.append_log(session_id, "Debug message", "DEBUG")
    
    logs = log_store.get_logs(session_id)
    assert len(logs) == 4
    assert logs[0].level == LogLevel.INFO
    assert logs[1].level == LogLevel.WARNING
    assert logs[2].level == LogLevel.ERROR
    assert logs[3].level == LogLevel.DEBUG


def test_append_log_invalid_level(log_store):
    """Test that invalid log levels raise ValueError."""
    session_id = "test_session_3"
    
    with pytest.raises(ValueError, match="Invalid log level"):
        log_store.append_log(session_id, "Test", "INVALID")


def test_get_logs_empty_session(log_store):
    """Test getting logs for a session with no logs."""
    logs = log_store.get_logs("nonexistent_session")
    assert logs == []


def test_get_logs_limit(log_store):
    """Test that get_logs respects the limit parameter."""
    session_id = "test_session_4"
    
    # Add 10 logs
    for i in range(10):
        log_store.append_log(session_id, f"Message {i}", "INFO")
    
    # Get only 5
    logs = log_store.get_logs(session_id, limit=5)
    assert len(logs) == 5


def test_get_logs_chronological_order(log_store):
    """Test that get_logs returns logs in chronological order (oldest first)."""
    session_id = "test_session_5"
    
    messages = ["First", "Second", "Third"]
    for msg in messages:
        log_store.append_log(session_id, msg, "INFO")
    
    logs = log_store.get_logs(session_id)
    assert len(logs) == 3
    assert logs[0].message == "First"
    assert logs[1].message == "Second"
    assert logs[2].message == "Third"


def test_get_logs_since(log_store):
    """Test getting logs since a specific timestamp."""
    import time
    session_id = "test_session_6"
    
    # Add first log
    log_store.append_log(session_id, "Old message", "INFO")
    
    # Small delay to ensure timestamp difference
    time.sleep(0.01)
    
    # Get timestamp
    checkpoint = datetime.now(timezone.utc)
    
    # Small delay to ensure new logs have later timestamp
    time.sleep(0.01)
    
    # Add more logs
    log_store.append_log(session_id, "New message 1", "INFO")
    log_store.append_log(session_id, "New message 2", "INFO")
    
    # Get logs since checkpoint
    logs = log_store.get_logs_since(session_id, checkpoint)
    assert len(logs) == 2
    assert logs[0].message == "New message 1"
    assert logs[1].message == "New message 2"


def test_get_all_logs(log_store):
    """Test getting all logs without limit."""
    session_id = "test_session_7"
    
    # Add many logs
    for i in range(150):
        log_store.append_log(session_id, f"Message {i}", "INFO")
    
    # get_logs has default limit of 100
    limited_logs = log_store.get_logs(session_id)
    assert len(limited_logs) == 100
    
    # get_all_logs should return all
    all_logs = log_store.get_all_logs(session_id)
    assert len(all_logs) == 150


def test_logs_have_timestamps(log_store):
    """Test that all log entries have valid timestamps."""
    session_id = "test_session_8"
    
    before = datetime.now(timezone.utc)
    log_store.append_log(session_id, "Test message", "INFO")
    after = datetime.now(timezone.utc)
    
    logs = log_store.get_logs(session_id)
    assert len(logs) == 1
    
    # Timestamp should be between before and after (with some tolerance)
    assert before <= logs[0].timestamp <= after or \
           (logs[0].timestamp - before).total_seconds() < 1


def test_logs_isolated_by_session(log_store):
    """Test that logs are properly isolated by session ID."""
    session_1 = "session_1"
    session_2 = "session_2"
    
    log_store.append_log(session_1, "Session 1 message", "INFO")
    log_store.append_log(session_2, "Session 2 message", "INFO")
    
    logs_1 = log_store.get_logs(session_1)
    logs_2 = log_store.get_logs(session_2)
    
    assert len(logs_1) == 1
    assert len(logs_2) == 1
    assert logs_1[0].message == "Session 1 message"
    assert logs_2[0].message == "Session 2 message"


def test_delete_logs(log_store):
    """Test deleting logs for a session."""
    session_id = "test_session_9"
    
    # Add logs
    for i in range(5):
        log_store.append_log(session_id, f"Message {i}", "INFO")
    
    # Verify logs exist
    logs = log_store.get_logs(session_id)
    assert len(logs) == 5
    
    # Delete logs
    deleted_count = log_store.delete_logs(session_id)
    assert deleted_count == 5
    
    # Verify logs are gone
    logs = log_store.get_logs(session_id)
    assert len(logs) == 0


def test_delete_logs_nonexistent_session(log_store):
    """Test deleting logs for a session that doesn't exist."""
    deleted_count = log_store.delete_logs("nonexistent_session")
    assert deleted_count == 0
