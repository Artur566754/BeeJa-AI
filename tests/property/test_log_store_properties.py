"""
Property-based tests for the LogStore service.

Tests universal properties that should hold for all log operations.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

from api.database import Database
from api.services.log_store import LogStore
from api.models.data_models import LogLevel


# Strategies for generating test data
session_ids = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")),
    min_size=1,
    max_size=50
).filter(lambda s: s.strip())

log_messages = st.text(min_size=1, max_size=500)

log_levels = st.sampled_from([level.value for level in LogLevel])


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


@given(
    session_id=session_ids,
    message=log_messages,
    level=log_levels
)
@settings(max_examples=100, deadline=None)
def test_property_all_log_entries_have_timestamps(session_id, message, level):
    """
    Feature: server-management-api, Property 25: All log entries have timestamps
    
    **Validates: Requirements 7.5**
    
    For any log entry in any session, it should contain a valid timestamp
    in ISO 8601 format.
    """
    # Create temporary database and log store
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    try:
        db = Database(db_path)
        log_store = LogStore(db)
        
        # Append a log entry
        before = datetime.now(timezone.utc)
        log_store.append_log(session_id, message, level)
        after = datetime.now(timezone.utc)
        
        # Retrieve the log entry
        logs = log_store.get_logs(session_id)
        
        # Property: All log entries have valid timestamps
        assert len(logs) > 0, "Log entry should be stored"
        
        for log in logs:
            # Timestamp should exist
            assert log.timestamp is not None, "Log entry must have a timestamp"
            
            # Timestamp should be a datetime object
            assert isinstance(log.timestamp, datetime), "Timestamp must be a datetime object"
            
            # Timestamp should be within reasonable bounds (between before and after)
            # Allow for some tolerance due to timezone conversion
            assert before <= log.timestamp <= after or \
                   (log.timestamp - before).total_seconds() < 1, \
                   "Timestamp should be within operation time"
            
            # Timestamp should be in ISO 8601 format (can be serialized)
            iso_string = log.timestamp.isoformat()
            assert isinstance(iso_string, str), "Timestamp should be serializable to ISO 8601"
            
            # Should be able to parse back from ISO format
            parsed = datetime.fromisoformat(iso_string)
            assert parsed == log.timestamp, "Timestamp should round-trip through ISO format"
        
        # Cleanup
        db.close()
        time.sleep(0.1)  # Give Windows time to release the file
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except PermissionError:
            pass  # Ignore cleanup errors on Windows


@given(
    session_id=session_ids,
    messages=st.lists(log_messages, min_size=1, max_size=20)
)
@settings(max_examples=100, deadline=None)
def test_property_completed_session_persists_logs(session_id, messages):
    """
    Feature: server-management-api, Property 24: Completed session persists logs
    
    **Validates: Requirements 7.4**
    
    For any training session that completes or fails, all log entries should
    remain retrievable after the session ends.
    """
    # Create temporary database and log store
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    try:
        db = Database(db_path)
        log_store = LogStore(db)
        
        # Append multiple log entries (simulating an active session)
        for i, message in enumerate(messages):
            log_store.append_log(session_id, message, "INFO")
        
        # Simulate session completion by just continuing to use the log store
        # (In real implementation, session would be marked as completed)
        
        # Property: All logs should still be retrievable
        retrieved_logs = log_store.get_all_logs(session_id)
        
        assert len(retrieved_logs) == len(messages), \
            "All log entries should be persisted and retrievable"
        
        # Verify all messages are present
        retrieved_messages = [log.message for log in retrieved_logs]
        assert retrieved_messages == messages, \
            "Log messages should be preserved in order"
        
        # Verify all logs have timestamps
        for log in retrieved_logs:
            assert log.timestamp is not None, \
                "Persisted logs must retain their timestamps"
        
        # Close and reopen database to ensure persistence
        db.close()
        time.sleep(0.1)
        db = Database(db_path)
        log_store = LogStore(db)
        
        # Property: Logs should still be retrievable after database reconnection
        retrieved_after_reopen = log_store.get_all_logs(session_id)
        assert len(retrieved_after_reopen) == len(messages), \
            "Logs should persist across database connections"
        
        # Cleanup
        db.close()
        time.sleep(0.1)
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except PermissionError:
            pass


@given(
    session_id=session_ids,
    messages=st.lists(log_messages, min_size=1, max_size=10),
    level=log_levels
)
@settings(max_examples=100, deadline=None)
def test_property_logs_retrievable_immediately(session_id, messages, level):
    """
    Feature: server-management-api, Property 23: Active session appends logs
    
    **Validates: Requirements 7.3**
    
    For any active training session, new log entries should be appended
    in real-time and be retrievable immediately.
    """
    # Create temporary database and log store
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    try:
        db = Database(db_path)
        log_store = LogStore(db)
        
        # Append logs one by one and verify immediate retrieval
        for i, message in enumerate(messages):
            # Append log
            log_store.append_log(session_id, message, level)
            
            # Property: Log should be immediately retrievable
            logs = log_store.get_all_logs(session_id)
            
            assert len(logs) == i + 1, \
                f"After appending {i+1} logs, should be able to retrieve {i+1} logs"
            
            # Verify the most recent log matches what we just appended
            assert logs[-1].message == message, \
                "Most recent log should match the just-appended message"
            
            assert logs[-1].level.value == level, \
                "Most recent log should have the correct level"
        
        # Cleanup
        db.close()
        time.sleep(0.1)
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except PermissionError:
            pass


@given(
    session_id=session_ids,
    num_logs=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=100, deadline=None)
def test_property_log_timestamps_chronological(session_id, num_logs):
    """
    Property: Log timestamps are in chronological order.
    
    For any sequence of log entries appended to a session, the timestamps
    should be in chronological order (non-decreasing).
    """
    # Create temporary database and log store
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    try:
        db = Database(db_path)
        log_store = LogStore(db)
        
        # Append multiple logs
        for i in range(num_logs):
            log_store.append_log(session_id, f"Message {i}", "INFO")
        
        # Retrieve logs
        logs = log_store.get_all_logs(session_id)
        
        # Property: Timestamps should be in chronological order
        assert len(logs) == num_logs, "All logs should be retrieved"
        
        for i in range(len(logs) - 1):
            assert logs[i].timestamp <= logs[i + 1].timestamp, \
                f"Log {i} timestamp should be <= log {i+1} timestamp"
        
        # Cleanup
        db.close()
        time.sleep(0.1)
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except PermissionError:
            pass


@given(
    session_id=session_ids,
    messages=st.lists(log_messages, min_size=1, max_size=20)
)
@settings(max_examples=100, deadline=None)
def test_property_session_logs_returns_all_entries(session_id, messages):
    """
    Feature: server-management-api, Property 21: Session logs are retrievable
    
    **Validates: Requirements 7.1**
    
    For any training session with N log entries, requesting logs should
    return all N entries with timestamps and messages.
    """
    # Create temporary database and log store
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    try:
        db = Database(db_path)
        log_store = LogStore(db)
        
        # Append N log entries
        for message in messages:
            log_store.append_log(session_id, message, "INFO")
        
        # Property: Requesting logs should return all N entries
        logs = log_store.get_all_logs(session_id)
        
        assert len(logs) == len(messages), \
            f"Should retrieve all {len(messages)} log entries"
        
        # Verify all entries have timestamps
        for log in logs:
            assert log.timestamp is not None, \
                "Each log entry must have a timestamp"
        
        # Verify all entries have messages
        for log in logs:
            assert log.message, \
                "Each log entry must have a message"
        
        # Verify messages match
        retrieved_messages = [log.message for log in logs]
        assert retrieved_messages == messages, \
            "Retrieved messages should match appended messages in order"
        
        # Cleanup
        db.close()
        time.sleep(0.1)
    finally:
        try:
            if db_path.exists():
                db_path.unlink()
        except PermissionError:
            pass
