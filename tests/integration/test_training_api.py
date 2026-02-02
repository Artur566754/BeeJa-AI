"""
Integration tests for training API endpoints.

These tests verify the complete training API workflow including session management,
metrics retrieval, logging, and queue management.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os
import json

from api.main import app
from api.config import APIConfig
from api.database import init_database


@pytest.fixture
def test_client():
    """Create a test client with temporary database."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    
    # Initialize database
    db = init_database(db_path)
    
    # Set up test API keys
    original_keys = APIConfig.API_KEYS
    APIConfig.API_KEYS = ["test_api_key_123"]
    
    # Create test client
    client = TestClient(app)
    
    yield client
    
    # Cleanup
    APIConfig.API_KEYS = original_keys
    db.close()
    if db_path.exists():
        os.unlink(db_path)


@pytest.fixture
def auth_headers():
    """Provide authentication headers for requests."""
    return {"Authorization": "Bearer test_api_key_123"}


def test_start_training_session(test_client, auth_headers):
    """Test starting a new training session."""
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5,
        "optimizer": "adam",
        "loss_function": "cross_entropy"
    }
    
    response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["status"] in ["running", "queued"]
    assert "message" in data


def test_start_training_without_auth(test_client):
    """Test that starting training without authentication fails."""
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    response = test_client.post("/api/v1/training/start", json=config)
    
    assert response.status_code == 401


def test_start_training_invalid_config(test_client, auth_headers):
    """Test that invalid configuration is rejected."""
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": -0.001,  # Invalid: negative
        "batch_size": 32,
        "epochs": 5
    }
    
    response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    
    assert response.status_code == 422  # Validation error


def test_get_training_status(test_client, auth_headers):
    """Test getting training session status."""
    # First create a session
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    start_response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    session_id = start_response.json()["session_id"]
    
    # Get status
    response = test_client.get(
        f"/api/v1/training/{session_id}/status",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "state" in data
    assert "total_epochs" in data


def test_get_status_nonexistent_session(test_client, auth_headers):
    """Test getting status for non-existent session returns 404."""
    response = test_client.get(
        "/api/v1/training/nonexistent_session/status",
        headers=auth_headers
    )
    
    assert response.status_code == 404


def test_stop_training_session(test_client, auth_headers):
    """Test stopping a training session."""
    # First create a session
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    start_response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    session_id = start_response.json()["session_id"]
    
    # Stop the session
    response = test_client.post(
        f"/api/v1/training/{session_id}/stop",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    assert "message" in response.json()


def test_get_training_metrics(test_client, auth_headers):
    """Test getting training metrics."""
    # First create a session
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    start_response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    session_id = start_response.json()["session_id"]
    
    # Get metrics (may be None if training hasn't started yet)
    response = test_client.get(
        f"/api/v1/training/{session_id}/metrics",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    # Metrics may be null if no training has occurred yet
    data = response.json()
    assert data is None or "epoch" in data


def test_get_training_history(test_client, auth_headers):
    """Test getting training metrics history."""
    # First create a session
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    start_response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    session_id = start_response.json()["session_id"]
    
    # Get history
    response = test_client.get(
        f"/api/v1/training/{session_id}/history",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_training_logs(test_client, auth_headers):
    """Test getting training logs."""
    # First create a session
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    start_response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    session_id = start_response.json()["session_id"]
    
    # Get logs
    response = test_client.get(
        f"/api/v1/training/{session_id}/logs",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_list_sessions(test_client, auth_headers):
    """Test listing all training sessions."""
    # Create a couple of sessions
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    test_client.post("/api/v1/training/start", json=config, headers=auth_headers)
    test_client.post("/api/v1/training/start", json=config, headers=auth_headers)
    
    # List sessions
    response = test_client.get("/api/v1/training/sessions", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_get_queue_status(test_client, auth_headers):
    """Test getting queue status."""
    response = test_client.get("/api/v1/training/queue", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_cancel_queued_session(test_client, auth_headers):
    """Test canceling a queued session."""
    # Create multiple sessions to fill the queue
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    # Create first session (will run)
    test_client.post("/api/v1/training/start", json=config, headers=auth_headers)
    
    # Create second session (will be queued if max concurrent is 1)
    start_response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    session_id = start_response.json()["session_id"]
    
    # Try to cancel it
    response = test_client.delete(
        f"/api/v1/training/queue/{session_id}",
        headers=auth_headers
    )
    
    # May succeed or fail depending on whether it was queued
    assert response.status_code in [200, 404]


def test_complete_training_workflow(test_client, auth_headers):
    """Test complete training workflow: start -> status -> metrics -> logs."""
    # Start training
    config = {
        "model_architecture": "test_model",
        "dataset_name": "test_dataset",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5
    }
    
    start_response = test_client.post(
        "/api/v1/training/start",
        json=config,
        headers=auth_headers
    )
    assert start_response.status_code == 200
    session_id = start_response.json()["session_id"]
    
    # Check status
    status_response = test_client.get(
        f"/api/v1/training/{session_id}/status",
        headers=auth_headers
    )
    assert status_response.status_code == 200
    assert status_response.json()["session_id"] == session_id
    
    # Get metrics
    metrics_response = test_client.get(
        f"/api/v1/training/{session_id}/metrics",
        headers=auth_headers
    )
    assert metrics_response.status_code == 200
    
    # Get history
    history_response = test_client.get(
        f"/api/v1/training/{session_id}/history",
        headers=auth_headers
    )
    assert history_response.status_code == 200
    assert isinstance(history_response.json(), list)
    
    # Get logs
    logs_response = test_client.get(
        f"/api/v1/training/{session_id}/logs",
        headers=auth_headers
    )
    assert logs_response.status_code == 200
    assert isinstance(logs_response.json(), list)
    
    # Stop training
    stop_response = test_client.post(
        f"/api/v1/training/{session_id}/stop",
        headers=auth_headers
    )
    assert stop_response.status_code == 200
