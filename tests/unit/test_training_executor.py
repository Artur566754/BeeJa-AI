"""
Unit tests for the TrainingExecutor service.

Tests training execution, stopping, and callback integration.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from api.services.training_executor import TrainingExecutor
from api.models.data_models import TrainingConfig, Metrics


@pytest.fixture
def training_executor():
    """Create a TrainingExecutor instance."""
    return TrainingExecutor()


@pytest.fixture
def sample_config():
    """Create a sample training configuration."""
    return TrainingConfig(
        model_architecture="test_model",
        dataset_name="test_dataset",
        learning_rate=0.001,
        batch_size=32,
        epochs=3
    )


def test_execute_training_basic(training_executor, sample_config):
    """Test basic training execution with callbacks."""
    metrics_collected = []
    logs_collected = []
    
    def metrics_callback(metrics: Metrics):
        metrics_collected.append(metrics)
    
    def log_callback(message: str, level: str):
        logs_collected.append((message, level))
    
    # Execute training
    training_executor.execute_training(
        "test_session_1",
        sample_config,
        metrics_callback,
        log_callback
    )
    
    # Verify metrics were collected
    assert len(metrics_collected) == sample_config.epochs
    
    # Verify logs were collected
    assert len(logs_collected) > 0
    
    # Verify metrics have correct structure
    for i, metrics in enumerate(metrics_collected):
        assert metrics.session_id == "test_session_1"
        assert metrics.epoch == i + 1
        assert 0 <= metrics.loss <= 1
        assert 0 <= metrics.accuracy <= 1


def test_stop_training(training_executor, sample_config):
    """Test stopping a training session."""
    import threading
    import time
    
    metrics_collected = []
    logs_collected = []
    
    def metrics_callback(metrics: Metrics):
        metrics_collected.append(metrics)
    
    def log_callback(message: str, level: str):
        logs_collected.append((message, level))
    
    # Start training in a thread
    session_id = "test_session_2"
    
    # Use a longer training config
    long_config = TrainingConfig(
        model_architecture="test_model",
        dataset_name="test_dataset",
        learning_rate=0.001,
        batch_size=32,
        epochs=100  # Many epochs
    )
    
    training_thread = threading.Thread(
        target=training_executor.execute_training,
        args=(session_id, long_config, metrics_callback, log_callback)
    )
    training_thread.start()
    
    # Wait a bit then stop
    time.sleep(0.2)
    stopped = training_executor.stop_training(session_id)
    
    # Wait for thread to finish
    training_thread.join(timeout=5)
    
    # Verify stop was successful
    assert stopped is True
    
    # Verify training was stopped early (less than 100 epochs)
    assert len(metrics_collected) < long_config.epochs


def test_stop_nonexistent_session(training_executor):
    """Test stopping a nonexistent session."""
    stopped = training_executor.stop_training("nonexistent_session")
    assert stopped is False


def test_metrics_callback_receives_correct_data(training_executor, sample_config):
    """Test that metrics callback receives properly formatted data."""
    metrics_collected = []
    
    def metrics_callback(metrics: Metrics):
        metrics_collected.append(metrics)
    
    def log_callback(message: str, level: str):
        pass
    
    training_executor.execute_training(
        "test_session_3",
        sample_config,
        metrics_callback,
        log_callback
    )
    
    # Verify all metrics have required fields
    for metrics in metrics_collected:
        assert metrics.session_id
        assert metrics.epoch > 0
        assert metrics.loss >= 0
        assert metrics.accuracy >= 0
        assert metrics.val_loss is not None
        assert metrics.val_accuracy is not None
        assert metrics.timestamp is not None


def test_log_callback_receives_messages(training_executor, sample_config):
    """Test that log callback receives log messages."""
    logs_collected = []
    
    def metrics_callback(metrics: Metrics):
        pass
    
    def log_callback(message: str, level: str):
        logs_collected.append((message, level))
    
    training_executor.execute_training(
        "test_session_4",
        sample_config,
        metrics_callback,
        log_callback
    )
    
    # Verify logs were collected
    assert len(logs_collected) > 0
    
    # Verify log structure
    for message, level in logs_collected:
        assert isinstance(message, str)
        assert level in ["INFO", "WARNING", "ERROR", "DEBUG"]
    
    # Verify key log messages
    messages = [msg for msg, _ in logs_collected]
    assert any("Starting training" in msg for msg in messages)
    assert any("completed" in msg.lower() for msg in messages)


def test_training_failure_logs_error(training_executor):
    """Test that training failures are logged."""
    logs_collected = []
    
    def metrics_callback(metrics: Metrics):
        # Simulate a failure during metrics collection
        raise ValueError("Simulated failure")
    
    def log_callback(message: str, level: str):
        logs_collected.append((message, level))
    
    # Create config
    config = TrainingConfig(
        model_architecture="test_model",
        dataset_name="test_dataset",
        learning_rate=0.001,
        batch_size=32,
        epochs=1
    )
    
    # Execute training (should fail)
    with pytest.raises(ValueError):
        training_executor.execute_training(
            "test_session_5",
            config,
            metrics_callback,
            log_callback
        )
    
    # Verify error was logged
    error_logs = [msg for msg, level in logs_collected if level == "ERROR"]
    assert len(error_logs) > 0
    assert any("failed" in msg.lower() for msg in error_logs)
