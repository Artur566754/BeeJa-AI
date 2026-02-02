"""
Unit tests for MetricsStore service.

Tests basic functionality of metrics storage and retrieval.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile

from api.services.metrics_store import MetricsStore
from api.models import Metrics
from api.database import init_database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = init_database(db_path)
        yield db
        db.close()


@pytest.fixture
def metrics_store(temp_db):
    """Create a MetricsStore instance with temporary database."""
    return MetricsStore()


class TestMetricsStore:
    """Unit tests for MetricsStore functionality."""
    
    def test_save_and_retrieve_metrics(self, metrics_store):
        """Test saving and retrieving metrics."""
        session_id = "test_session_1"
        epoch = 0
        timestamp = datetime.now()
        
        metrics = Metrics(
            session_id=session_id,
            epoch=epoch,
            loss=0.5,
            accuracy=0.85,
            val_loss=0.6,
            val_accuracy=0.82,
            timestamp=timestamp
        )
        
        # Save metrics
        metrics_store.save_metrics(session_id, epoch, metrics)
        
        # Retrieve by epoch
        retrieved = metrics_store.get_metrics_by_epoch(session_id, epoch)
        
        assert retrieved is not None
        assert retrieved.session_id == session_id
        assert retrieved.epoch == epoch
        assert retrieved.loss == 0.5
        assert retrieved.accuracy == 0.85
        assert retrieved.val_loss == 0.6
        assert retrieved.val_accuracy == 0.82
    
    def test_get_latest_metrics(self, metrics_store):
        """Test retrieving the latest metrics."""
        session_id = "test_session_2"
        
        # Save multiple epochs
        for epoch in range(5):
            metrics = Metrics(
                session_id=session_id,
                epoch=epoch,
                loss=1.0 - (epoch * 0.1),
                accuracy=0.5 + (epoch * 0.1),
                timestamp=datetime.now()
            )
            metrics_store.save_metrics(session_id, epoch, metrics)
        
        # Get latest should return epoch 4
        latest = metrics_store.get_latest_metrics(session_id)
        
        assert latest is not None
        assert latest.epoch == 4
        assert latest.loss == pytest.approx(0.6, rel=1e-5)
        assert latest.accuracy == pytest.approx(0.9, rel=1e-5)
    
    def test_get_metrics_history(self, metrics_store):
        """Test retrieving all metrics in chronological order."""
        session_id = "test_session_3"
        num_epochs = 10
        
        # Save metrics for multiple epochs
        for epoch in range(num_epochs):
            metrics = Metrics(
                session_id=session_id,
                epoch=epoch,
                loss=1.0 - (epoch * 0.05),
                accuracy=0.5 + (epoch * 0.05),
                timestamp=datetime.now()
            )
            metrics_store.save_metrics(session_id, epoch, metrics)
        
        # Get history
        history = metrics_store.get_metrics_history(session_id)
        
        assert len(history) == num_epochs
        
        # Verify chronological order
        for i, metrics in enumerate(history):
            assert metrics.epoch == i
            assert metrics.loss == pytest.approx(1.0 - (i * 0.05), rel=1e-5)
            assert metrics.accuracy == pytest.approx(0.5 + (i * 0.05), rel=1e-5)
    
    def test_get_metrics_by_epoch_not_found(self, metrics_store):
        """Test retrieving metrics for non-existent epoch."""
        result = metrics_store.get_metrics_by_epoch("nonexistent", 0)
        assert result is None
    
    def test_get_latest_metrics_empty(self, metrics_store):
        """Test retrieving latest metrics when none exist."""
        result = metrics_store.get_latest_metrics("nonexistent")
        assert result is None
    
    def test_get_metrics_history_empty(self, metrics_store):
        """Test retrieving history when no metrics exist."""
        history = metrics_store.get_metrics_history("nonexistent")
        assert history == []
    
    def test_delete_metrics(self, metrics_store):
        """Test deleting all metrics for a session."""
        session_id = "test_session_4"
        
        # Save some metrics
        for epoch in range(3):
            metrics = Metrics(
                session_id=session_id,
                epoch=epoch,
                loss=0.5,
                accuracy=0.8,
                timestamp=datetime.now()
            )
            metrics_store.save_metrics(session_id, epoch, metrics)
        
        # Verify metrics exist
        assert len(metrics_store.get_metrics_history(session_id)) == 3
        
        # Delete metrics
        deleted_count = metrics_store.delete_metrics(session_id)
        assert deleted_count == 3
        
        # Verify metrics are gone
        assert len(metrics_store.get_metrics_history(session_id)) == 0
    
    def test_get_metrics_count(self, metrics_store):
        """Test counting metrics for a session."""
        session_id = "test_session_5"
        
        # Initially should be 0
        assert metrics_store.get_metrics_count(session_id) == 0
        
        # Add some metrics
        for epoch in range(7):
            metrics = Metrics(
                session_id=session_id,
                epoch=epoch,
                loss=0.5,
                accuracy=0.8,
                timestamp=datetime.now()
            )
            metrics_store.save_metrics(session_id, epoch, metrics)
        
        # Count should be 7
        assert metrics_store.get_metrics_count(session_id) == 7
    
    def test_multiple_sessions_isolated(self, metrics_store):
        """Test that metrics for different sessions are isolated."""
        session1 = "session_1"
        session2 = "session_2"
        
        # Save metrics for session 1
        for epoch in range(3):
            metrics = Metrics(
                session_id=session1,
                epoch=epoch,
                loss=0.5,
                accuracy=0.8,
                timestamp=datetime.now()
            )
            metrics_store.save_metrics(session1, epoch, metrics)
        
        # Save metrics for session 2
        for epoch in range(5):
            metrics = Metrics(
                session_id=session2,
                epoch=epoch,
                loss=0.3,
                accuracy=0.9,
                timestamp=datetime.now()
            )
            metrics_store.save_metrics(session2, epoch, metrics)
        
        # Verify isolation
        history1 = metrics_store.get_metrics_history(session1)
        history2 = metrics_store.get_metrics_history(session2)
        
        assert len(history1) == 3
        assert len(history2) == 5
        assert all(m.session_id == session1 for m in history1)
        assert all(m.session_id == session2 for m in history2)
