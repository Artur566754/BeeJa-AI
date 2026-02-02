"""
Property-based tests for MetricsStore service.

Tests universal properties of metrics storage and retrieval using hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime
from pathlib import Path
import tempfile
from contextlib import contextmanager

from api.services.metrics_store import MetricsStore
from api.models import Metrics
from api.database import init_database


# Hypothesis strategies for generating test data

session_id_strategy = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P"))
).filter(lambda s: s.strip())

epoch_strategy = st.integers(min_value=0, max_value=1000)

loss_strategy = st.floats(
    min_value=0.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False
)

accuracy_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False
)


@contextmanager
def create_metrics_store():
    """Context manager to create a temporary database and metrics store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = init_database(db_path)
        try:
            yield MetricsStore()
        finally:
            db.close()


class TestMetricsRetrievalProperty:
    """
    Property tests for metrics retrieval.
    
    **Validates: Requirements 2.2**
    """
    
    @given(
        session_id=session_id_strategy,
        epochs=st.lists(
            st.tuples(
                epoch_strategy,
                loss_strategy,
                accuracy_strategy
            ),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]  # Unique epochs
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_latest_metrics_returns_highest_epoch(self, session_id, epochs):
        """
        Property 7: Metrics request returns latest data.
        
        For any training session with N recorded metric entries,
        requesting metrics should return the entry with the highest epoch number.
        
        **Validates: Requirements 2.2**
        """
        with create_metrics_store() as metrics_store:
            # Save all metrics
            for epoch, loss, accuracy in epochs:
                metrics = Metrics(
                    session_id=session_id,
                    epoch=epoch,
                    loss=loss,
                    accuracy=accuracy,
                    timestamp=datetime.now()
                )
                metrics_store.save_metrics(session_id, epoch, metrics)
            
            # Get latest metrics
            latest = metrics_store.get_latest_metrics(session_id)
            
            # Verify it's the one with the highest epoch
            assert latest is not None
            max_epoch = max(epoch for epoch, _, _ in epochs)
            assert latest.epoch == max_epoch
            
            # Find the corresponding metrics data
            expected_loss, expected_accuracy = next(
                (loss, acc) for ep, loss, acc in epochs if ep == max_epoch
            )
            assert latest.loss == expected_loss
            assert latest.accuracy == expected_accuracy
    
    @given(
        session_id=session_id_strategy,
        epoch=epoch_strategy,
        loss=loss_strategy,
        accuracy=accuracy_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_single_metric_is_latest(self, session_id, epoch, loss, accuracy):
        """
        Property 7: Metrics request returns latest data (single entry case).
        
        For any training session with exactly one metric entry,
        requesting latest metrics should return that entry.
        
        **Validates: Requirements 2.2**
        """
        with create_metrics_store() as metrics_store:
            # Save single metric
            metrics = Metrics(
                session_id=session_id,
                epoch=epoch,
                loss=loss,
                accuracy=accuracy,
                timestamp=datetime.now()
            )
            metrics_store.save_metrics(session_id, epoch, metrics)
            
            # Get latest metrics
            latest = metrics_store.get_latest_metrics(session_id)
            
            # Verify it matches the saved metric
            assert latest is not None
            assert latest.session_id == session_id
            assert latest.epoch == epoch
            assert latest.loss == loss
            assert latest.accuracy == accuracy
    
    @given(session_id=session_id_strategy)
    @settings(max_examples=100, deadline=None)
    def test_no_metrics_returns_none(self, session_id):
        """
        Property 7: Metrics request returns latest data (empty case).
        
        For any session with no recorded metrics,
        requesting latest metrics should return None.
        
        **Validates: Requirements 2.2**
        """
        with create_metrics_store() as metrics_store:
            # Don't save any metrics
            latest = metrics_store.get_latest_metrics(session_id)
            
            # Should return None
            assert latest is None


class TestMetricsHistoryProperty:
    """
    Property tests for metrics history.
    
    **Validates: Requirements 2.3**
    """
    
    @given(
        session_id=session_id_strategy,
        epochs=st.lists(
            st.tuples(
                epoch_strategy,
                loss_strategy,
                accuracy_strategy
            ),
            min_size=1,
            max_size=50,
            unique_by=lambda x: x[0]  # Unique epochs
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_history_returns_all_epochs_in_order(self, session_id, epochs):
        """
        Property 8: Metrics history returns all epochs.
        
        For any training session with N recorded metric entries,
        requesting metrics history should return all N entries in chronological order.
        
        **Validates: Requirements 2.3**
        """
        with create_metrics_store() as metrics_store:
            # Save all metrics
            for epoch, loss, accuracy in epochs:
                metrics = Metrics(
                    session_id=session_id,
                    epoch=epoch,
                    loss=loss,
                    accuracy=accuracy,
                    timestamp=datetime.now()
                )
                metrics_store.save_metrics(session_id, epoch, metrics)
            
            # Get history
            history = metrics_store.get_metrics_history(session_id)
            
            # Verify count matches
            assert len(history) == len(epochs)
            
            # Verify chronological order (sorted by epoch)
            sorted_epochs = sorted(epochs, key=lambda x: x[0])
            for i, (expected_epoch, expected_loss, expected_accuracy) in enumerate(sorted_epochs):
                assert history[i].epoch == expected_epoch
                assert history[i].loss == expected_loss
                assert history[i].accuracy == expected_accuracy
                assert history[i].session_id == session_id
    
    @given(
        session_id=session_id_strategy,
        num_epochs=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_history_count_matches_saved_count(self, session_id, num_epochs):
        """
        Property 8: Metrics history returns all epochs (count verification).
        
        For any training session with N saved metric entries,
        the history should contain exactly N entries.
        
        **Validates: Requirements 2.3**
        """
        with create_metrics_store() as metrics_store:
            # Save N metrics with sequential epochs
            for epoch in range(num_epochs):
                metrics = Metrics(
                    session_id=session_id,
                    epoch=epoch,
                    loss=1.0 - (epoch * 0.01),
                    accuracy=0.5 + (epoch * 0.005),
                    timestamp=datetime.now()
                )
                metrics_store.save_metrics(session_id, epoch, metrics)
            
            # Get history
            history = metrics_store.get_metrics_history(session_id)
            
            # Verify count
            assert len(history) == num_epochs
    
    @given(session_id=session_id_strategy)
    @settings(max_examples=100, deadline=None)
    def test_empty_history_returns_empty_list(self, session_id):
        """
        Property 8: Metrics history returns all epochs (empty case).
        
        For any session with no recorded metrics,
        requesting history should return an empty list.
        
        **Validates: Requirements 2.3**
        """
        with create_metrics_store() as metrics_store:
            # Don't save any metrics
            history = metrics_store.get_metrics_history(session_id)
            
            # Should return empty list
            assert history == []
            assert len(history) == 0


class TestMetricsPerEpochProperty:
    """
    Property tests for metrics recorded per epoch.
    
    **Validates: Requirements 2.4**
    """
    
    @given(
        session_id=session_id_strategy,
        num_epochs=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_completed_training_has_one_metric_per_epoch(
        self, session_id, num_epochs
    ):
        """
        Property 9: Metrics recorded per epoch.
        
        For any training session that completes E epochs,
        the metrics history should contain exactly E metric entries (one per epoch).
        
        **Validates: Requirements 2.4**
        """
        with create_metrics_store() as metrics_store:
            # Simulate a training session completing E epochs
            for epoch in range(num_epochs):
                metrics = Metrics(
                    session_id=session_id,
                    epoch=epoch,
                    loss=1.0 - (epoch * 0.01),
                    accuracy=0.5 + (epoch * 0.005),
                    timestamp=datetime.now()
                )
                metrics_store.save_metrics(session_id, epoch, metrics)
            
            # Get history
            history = metrics_store.get_metrics_history(session_id)
            
            # Verify exactly E entries
            assert len(history) == num_epochs
            
            # Verify each epoch from 0 to E-1 is present exactly once
            epochs_in_history = [m.epoch for m in history]
            expected_epochs = list(range(num_epochs))
            assert sorted(epochs_in_history) == expected_epochs
    
    @given(
        session_id=session_id_strategy,
        epochs=st.lists(
            epoch_strategy,
            min_size=1,
            max_size=50,
            unique=True
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_each_epoch_appears_once(self, session_id, epochs):
        """
        Property 9: Metrics recorded per epoch (uniqueness).
        
        For any training session, each epoch should appear at most once
        in the metrics history.
        
        **Validates: Requirements 2.4**
        """
        with create_metrics_store() as metrics_store:
            # Save metrics for each epoch
            for epoch in epochs:
                metrics = Metrics(
                    session_id=session_id,
                    epoch=epoch,
                    loss=0.5,
                    accuracy=0.8,
                    timestamp=datetime.now()
                )
                metrics_store.save_metrics(session_id, epoch, metrics)
            
            # Get history
            history = metrics_store.get_metrics_history(session_id)
            
            # Verify each epoch appears exactly once
            epochs_in_history = [m.epoch for m in history]
            assert len(epochs_in_history) == len(set(epochs_in_history))
            assert len(epochs_in_history) == len(epochs)
    
    @given(
        session_id=session_id_strategy,
        num_epochs=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_metrics_count_equals_epoch_count(
        self, session_id, num_epochs
    ):
        """
        Property 9: Metrics recorded per epoch (count verification).
        
        For any training session that records metrics for E epochs,
        the total count of metrics should equal E.
        
        **Validates: Requirements 2.4**
        """
        with create_metrics_store() as metrics_store:
            # Save metrics for E epochs
            for epoch in range(num_epochs):
                metrics = Metrics(
                    session_id=session_id,
                    epoch=epoch,
                    loss=0.5,
                    accuracy=0.8,
                    timestamp=datetime.now()
                )
                metrics_store.save_metrics(session_id, epoch, metrics)
            
            # Get count
            count = metrics_store.get_metrics_count(session_id)
            
            # Verify count equals number of epochs
            assert count == num_epochs
