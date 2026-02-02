"""
Property-based tests for API endpoints.

These tests verify universal properties of the API across all possible inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os

from api.main import app
from api.config import APIConfig
from api.database import init_database


@pytest.fixture(scope="module")
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


# Property 15: Non-existent resource returns 404
@given(
    resource_name=st.text(min_size=1, max_size=50).filter(
        lambda s: s.strip() and not s.startswith(".")
    )
)
@settings(max_examples=100, suppress_health_check=[])
def test_nonexistent_model_returns_404(test_client, resource_name):
    """
    Feature: server-management-api, Property 15: Non-existent resource returns 404
    
    For any randomly generated resource name (model or dataset) that doesn't exist,
    operations on that resource should return a 404 error with an appropriate message.
    
    Validates: Requirements 4.5, 5.4
    """
    # Test model download
    response = test_client.get(
        f"/api/v1/models/{resource_name}",
        headers={"Authorization": "Bearer test_api_key_123"}
    )
    
    # Should return 404 for non-existent model
    assert response.status_code == 404, f"Expected 404 for non-existent model, got {response.status_code}"
    
    # Response should be JSON with error message
    data = response.json()
    assert "detail" in data, "Error response should contain 'detail' field"
    assert resource_name in data["detail"] or "not found" in data["detail"].lower(), \
        f"Error message should mention the resource or 'not found': {data['detail']}"


@given(
    resource_name=st.text(min_size=1, max_size=50).filter(
        lambda s: s.strip() and not s.startswith(".")
    )
)
@settings(max_examples=100, suppress_health_check=[])
def test_nonexistent_dataset_returns_404(test_client, resource_name):
    """
    Feature: server-management-api, Property 15: Non-existent resource returns 404
    
    For any randomly generated dataset name that doesn't exist,
    operations on that dataset should return a 404 error with an appropriate message.
    
    Validates: Requirements 5.4
    """
    # Test dataset info
    response = test_client.get(
        f"/api/v1/datasets/{resource_name}",
        headers={"Authorization": "Bearer test_api_key_123"}
    )
    
    # Should return 404 for non-existent dataset
    assert response.status_code == 404, f"Expected 404 for non-existent dataset, got {response.status_code}"
    
    # Response should be JSON with error message
    data = response.json()
    assert "detail" in data, "Error response should contain 'detail' field"
    assert resource_name in data["detail"] or "not found" in data["detail"].lower(), \
        f"Error message should mention the resource or 'not found': {data['detail']}"


@given(
    session_id=st.text(min_size=1, max_size=50).filter(
        lambda s: s.strip() and not s.startswith(".")
    )
)
@settings(max_examples=100, suppress_health_check=[])
def test_nonexistent_session_returns_404(test_client, session_id):
    """
    Feature: server-management-api, Property 10: Non-existent session returns error
    
    For any randomly generated session ID that doesn't exist, requesting metrics
    should return a 404 error with an appropriate message.
    
    Validates: Requirements 2.5
    """
    # Test session status
    response = test_client.get(
        f"/api/v1/training/{session_id}/status",
        headers={"Authorization": "Bearer test_api_key_123"}
    )
    
    # Should return 404 for non-existent session
    assert response.status_code == 404, f"Expected 404 for non-existent session, got {response.status_code}"
    
    # Response should be JSON with error message
    data = response.json()
    assert "detail" in data, "Error response should contain 'detail' field"
    assert session_id in data["detail"] or "not found" in data["detail"].lower(), \
        f"Error message should mention the session or 'not found': {data['detail']}"



# Property 30: All responses are valid JSON
@given(
    endpoint=st.sampled_from([
        "/health",
        "/",
        "/api/v1/system/info",
        "/api/v1/training/sessions",
        "/api/v1/training/queue",
        "/api/v1/models",
        "/api/v1/datasets"
    ])
)
@settings(max_examples=50, suppress_health_check=[])
def test_all_responses_are_valid_json(test_client, endpoint):
    """
    Feature: server-management-api, Property 30: All responses are valid JSON
    
    For any API endpoint and any request (valid or invalid), the response body
    should be valid JSON that can be parsed.
    
    Validates: Requirements 9.1
    """
    # Make request (with auth for protected endpoints)
    if endpoint.startswith("/api/v1"):
        response = test_client.get(endpoint, headers={"Authorization": "Bearer test_api_key_123"})
    else:
        response = test_client.get(endpoint)
    
    # Response should be valid JSON
    try:
        data = response.json()
        assert data is not None, "Response should contain JSON data"
    except Exception as e:
        pytest.fail(f"Response is not valid JSON: {e}")


# Property 31: HTTP status codes match operation result
@given(
    valid_request=st.booleans()
)
@settings(max_examples=50, suppress_health_check=[])
def test_http_status_codes_match_result(test_client, valid_request):
    """
    Feature: server-management-api, Property 31: HTTP status codes match operation result
    
    For any API operation, the HTTP status code should be 200 for success,
    4xx for client errors, and 5xx for server errors.
    
    Validates: Requirements 9.2, 9.3, 9.4
    """
    if valid_request:
        # Valid request should return 2xx
        response = test_client.get("/health")
        assert 200 <= response.status_code < 300, \
            f"Valid request should return 2xx, got {response.status_code}"
    else:
        # Invalid request (no auth) should return 4xx
        response = test_client.get("/api/v1/training/sessions")
        assert 400 <= response.status_code < 500, \
            f"Invalid request should return 4xx, got {response.status_code}"


# Property 32: Error responses have consistent structure
@given(
    endpoint=st.sampled_from([
        "/api/v1/training/nonexistent_session/status",
        "/api/v1/models/nonexistent_model",
        "/api/v1/datasets/nonexistent_dataset"
    ])
)
@settings(max_examples=50, suppress_health_check=[])
def test_error_responses_have_consistent_structure(test_client, endpoint):
    """
    Feature: server-management-api, Property 32: Error responses have consistent structure
    
    For any error response (4xx or 5xx), the JSON body should contain an error
    structure with appropriate fields.
    
    Validates: Requirements 9.5
    """
    # Make request that will fail
    response = test_client.get(endpoint, headers={"Authorization": "Bearer test_api_key_123"})
    
    # Should be an error response
    assert response.status_code >= 400, "Should be an error response"
    
    # Should have JSON body
    data = response.json()
    assert data is not None, "Error response should have JSON body"
    
    # Should have error information (FastAPI uses "detail" field)
    assert "detail" in data, "Error response should contain error information"
    assert isinstance(data["detail"], str), "Error detail should be a string"
    assert len(data["detail"]) > 0, "Error detail should not be empty"
