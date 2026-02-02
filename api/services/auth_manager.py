"""
Authentication manager for the Server Management API.

This module handles API key validation, header parsing, and authentication logging.
"""

import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict

from api.config import APIConfig
from api.database import get_database


class AuthenticationManager:
    """Manages API key authentication and logging."""
    
    def __init__(self):
        """Initialize the authentication manager."""
        self.db = get_database()
        self.valid_api_keys = set(APIConfig.API_KEYS)
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            True if the API key is valid, False otherwise
        """
        if not api_key:
            return False
        
        return api_key in self.valid_api_keys
    
    def get_api_key_from_header(self, headers: Dict[str, str]) -> Optional[str]:
        """
        Extract API key from request headers.
        
        Supports both "Authorization: Bearer <key>" and "X-API-Key: <key>" formats.
        
        Args:
            headers: Dictionary of HTTP headers
            
        Returns:
            The API key if found, None otherwise
        """
        # Try Authorization header with Bearer token
        auth_header = headers.get("authorization") or headers.get("Authorization")
        if auth_header:
            # Check if it starts with "Bearer " (case-insensitive)
            if auth_header.lower().startswith("bearer "):
                # Extract everything after "Bearer "
                return auth_header[7:]  # len("Bearer ") = 7
        
        # Try X-API-Key header
        api_key = headers.get("x-api-key") or headers.get("X-API-Key")
        if api_key:
            return api_key
        
        return None
    
    def log_auth_attempt(
        self, 
        api_key: str, 
        success: bool, 
        endpoint: str
    ) -> None:
        """
        Log an authentication attempt to the database.
        
        Args:
            api_key: The API key used (will be hashed for storage)
            success: Whether the authentication was successful
            endpoint: The endpoint being accessed
        """
        # Hash the API key for security (don't store plain text)
        api_key_hash = self._hash_api_key(api_key)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO auth_logs (api_key_hash, success, endpoint, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (api_key_hash, success, endpoint, datetime.now(timezone.utc))
            )
    
    def _hash_api_key(self, api_key: str) -> str:
        """
        Hash an API key for secure storage.
        
        Args:
            api_key: The API key to hash
            
        Returns:
            SHA256 hash of the API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def authenticate_request(
        self, 
        headers: Dict[str, str], 
        endpoint: str
    ) -> tuple[bool, Optional[str]]:
        """
        Authenticate a request and log the attempt.
        
        Args:
            headers: Request headers
            endpoint: The endpoint being accessed
            
        Returns:
            Tuple of (is_authenticated, error_message)
        """
        api_key = self.get_api_key_from_header(headers)
        
        if not api_key:
            self.log_auth_attempt("", False, endpoint)
            return False, "Missing API key. Provide via Authorization: Bearer <key> or X-API-Key: <key> header"
        
        is_valid = self.validate_api_key(api_key)
        self.log_auth_attempt(api_key, is_valid, endpoint)
        
        if not is_valid:
            return False, "Invalid or expired API key"
        
        return True, None


# Global authentication manager instance
_auth_manager: Optional[AuthenticationManager] = None


def get_auth_manager() -> AuthenticationManager:
    """
    Get the global authentication manager instance.
    
    Returns:
        AuthenticationManager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager
