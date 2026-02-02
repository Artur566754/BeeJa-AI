"""
Property-based tests for authentication manager.

These tests verify universal properties of the authentication system across
all possible inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from contextlib import contextmanager

from api.services.auth_manager import AuthenticationManager
from api.config import APIConfig
from api.database import init_database
from pathlib import Path
import tempfile
import os


@contextmanager
def temp_auth_manager():
    """Context manager to create a temporary auth manager with test database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    
    db = init_database(db_path)
    
    # Set up test API keys
    original_keys = APIConfig.API_KEYS
    APIConfig.API_KEYS = ["test_key_1", "test_key_2", "test_key_3"]
    
    try:
        manager = AuthenticationManager()
        yield manager, db
    finally:
        # Restore original keys
        APIConfig.API_KEYS = original_keys
        db.close()
        if db_path.exists():
            os.unlink(db_path)


# Property 26: Unauthenticated requests rejected
@given(
    endpoint=st.text(min_size=1, max_size=100),
    header_type=st.sampled_from(["none", "empty", "malformed"])
)
@settings(max_examples=100)
def test_unauthenticated_requests_rejected(endpoint, header_type):
    """
    Feature: server-management-api, Property 26: Unauthenticated requests rejected
    
    For any API endpoint, making a request without a valid API key should 
    return a 401 Unauthorized error.
    
    Validates: Requirements 8.1
    """
    with temp_auth_manager() as (auth_manager, _):
        # Create headers without valid authentication
        if header_type == "none":
            headers = {}
        elif header_type == "empty":
            headers = {"Authorization": ""}
        else:  # malformed
            headers = {"Authorization": "InvalidFormat"}
        
        is_authenticated, error_message = auth_manager.authenticate_request(headers, endpoint)
        
        # Should not be authenticated
        assert not is_authenticated, "Request without valid API key should be rejected"
        assert error_message is not None, "Error message should be provided"
        assert "API key" in error_message or "Missing" in error_message or "Invalid" in error_message


# Property 28: Invalid API key returns auth error
@given(
    endpoint=st.text(min_size=1, max_size=100),
    invalid_key=st.text(min_size=1, max_size=100).filter(
        lambda k: k not in ["test_key_1", "test_key_2", "test_key_3"]
    ),
    header_format=st.sampled_from(["bearer", "x-api-key"])
)
@settings(max_examples=100)
def test_invalid_api_key_returns_error(endpoint, invalid_key, header_format):
    """
    Feature: server-management-api, Property 28: Invalid API key returns auth error
    
    For any randomly generated invalid API key, requests using it should 
    return a 401 error with an authentication error message.
    
    Validates: Requirements 8.4
    """
    with temp_auth_manager() as (auth_manager, _):
        # Create headers with invalid API key
        if header_format == "bearer":
            headers = {"Authorization": f"Bearer {invalid_key}"}
        else:
            headers = {"X-API-Key": invalid_key}
        
        is_authenticated, error_message = auth_manager.authenticate_request(headers, endpoint)
        
        # Should not be authenticated
        assert not is_authenticated, "Request with invalid API key should be rejected"
        assert error_message is not None, "Error message should be provided"
        # Error message should indicate authentication failure
        assert any(word in error_message for word in ["Invalid", "expired", "Missing", "API key"]), \
            f"Error message should indicate auth failure, got: {error_message}"


# Property 29: Authentication attempts logged
@given(
    endpoint=st.text(min_size=1, max_size=100),
    use_valid_key=st.booleans(),
    header_format=st.sampled_from(["bearer", "x-api-key", "none"])
)
@settings(max_examples=100)
def test_authentication_attempts_logged(endpoint, use_valid_key, header_format):
    """
    Feature: server-management-api, Property 29: Authentication attempts logged
    
    For any authentication attempt (valid or invalid), an entry should be 
    created in the authentication log with the timestamp, endpoint, and success status.
    
    Validates: Requirements 8.5
    """
    with temp_auth_manager() as (auth_manager, temp_db):
        # Create headers
        if header_format == "none":
            headers = {}
            expected_success = False
        elif use_valid_key:
            if header_format == "bearer":
                headers = {"Authorization": "Bearer test_key_1"}
            else:
                headers = {"X-API-Key": "test_key_1"}
            expected_success = True
        else:
            if header_format == "bearer":
                headers = {"Authorization": "Bearer invalid_key_xyz"}
            else:
                headers = {"X-API-Key": "invalid_key_xyz"}
            expected_success = False
        
        # Get initial log count
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM auth_logs WHERE endpoint = ?", (endpoint,))
            initial_count = cursor.fetchone()[0]
        
        # Make authentication request
        is_authenticated, _ = auth_manager.authenticate_request(headers, endpoint)
        
        # Verify authentication result matches expectation
        assert is_authenticated == expected_success
        
        # Verify log entry was created
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT success, endpoint, timestamp FROM auth_logs WHERE endpoint = ? ORDER BY timestamp DESC LIMIT 1",
                (endpoint,)
            )
            log_entry = cursor.fetchone()
            
            assert log_entry is not None, "Authentication attempt should be logged"
            assert log_entry[0] == (1 if expected_success else 0), "Log should record correct success status"
            assert log_entry[1] == endpoint, "Log should record correct endpoint"
            assert log_entry[2] is not None, "Log should have timestamp"
        
        # Verify count increased
        with temp_db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM auth_logs WHERE endpoint = ?", (endpoint,))
            final_count = cursor.fetchone()[0]
            assert final_count == initial_count + 1, "Log count should increase by 1"


# Additional test: Valid API keys are accepted
@given(
    endpoint=st.text(min_size=1, max_size=100),
    valid_key=st.sampled_from(["test_key_1", "test_key_2", "test_key_3"]),
    header_format=st.sampled_from(["bearer", "x-api-key"])
)
@settings(max_examples=100)
def test_valid_api_keys_accepted(endpoint, valid_key, header_format):
    """
    Feature: server-management-api, Property 27: Authenticated requests processed
    
    For any API endpoint, making a request with a valid API key should 
    process the request normally (not return 401).
    
    Validates: Requirements 8.2
    """
    with temp_auth_manager() as (auth_manager, _):
        # Create headers with valid API key
        if header_format == "bearer":
            headers = {"Authorization": f"Bearer {valid_key}"}
        else:
            headers = {"X-API-Key": valid_key}
        
        is_authenticated, error_message = auth_manager.authenticate_request(headers, endpoint)
        
        # Should be authenticated
        assert is_authenticated, "Request with valid API key should be accepted"
        assert error_message is None, "No error message should be provided for valid authentication"


# Test API key extraction from headers
@given(
    api_key=st.text(min_size=1, max_size=100),
    header_format=st.sampled_from(["bearer", "x-api-key", "bearer_lower", "x-api-key-lower"])
)
@settings(max_examples=100)
def test_api_key_extraction_from_headers(api_key, header_format):
    """
    Test that API keys can be extracted from various header formats.
    """
    with temp_auth_manager() as (auth_manager, _):
        # Create headers in different formats
        if header_format == "bearer":
            headers = {"Authorization": f"Bearer {api_key}"}
        elif header_format == "bearer_lower":
            headers = {"authorization": f"Bearer {api_key}"}
        elif header_format == "x-api-key":
            headers = {"X-API-Key": api_key}
        else:  # x-api-key-lower
            headers = {"x-api-key": api_key}
        
        extracted_key = auth_manager.get_api_key_from_header(headers)
        
        assert extracted_key == api_key, "API key should be correctly extracted from headers"
