"""
Configuration management for the Server Management API.

This module handles loading and managing configuration settings including
API keys, file paths, and default values.
"""

import os
from pathlib import Path
from typing import List, Optional


class APIConfig:
    """Configuration settings for the Server Management API."""
    
    # API Server Settings
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # API Keys (comma-separated in environment variable)
    API_KEYS: List[str] = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []
    
    # Database Settings
    DATABASE_PATH: Path = Path(os.getenv("DATABASE_PATH", "api_data.db"))
    
    # File Storage Paths
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "models"))
    DATASETS_DIR: Path = Path(os.getenv("DATASETS_DIR", "datasets"))
    
    # Training Defaults
    DEFAULT_OPTIMIZER: str = "adam"
    DEFAULT_LOSS_FUNCTION: str = "cross_entropy"
    MAX_CONCURRENT_SESSIONS: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "1"))
    
    # System Monitoring
    MONITOR_REFRESH_INTERVAL: int = 5  # seconds
    
    # Session Management
    SESSION_CLEANUP_DAYS: int = 7  # days to keep completed sessions
    
    # Validation Ranges
    MIN_LEARNING_RATE: float = 0.0001
    MAX_LEARNING_RATE: float = 1.0
    MIN_BATCH_SIZE: int = 1
    MAX_BATCH_SIZE: int = 256
    MIN_EPOCHS: int = 1
    MAX_EPOCHS: int = 1000
    
    # Valid Options
    VALID_OPTIMIZERS: List[str] = ["adam", "sgd", "rmsprop", "adamw"]
    VALID_LOSS_FUNCTIONS: List[str] = ["cross_entropy", "mse", "mae", "bce"]
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate configuration settings."""
        if not cls.API_KEYS:
            raise ValueError(
                "No API keys configured. Set API_KEYS environment variable."
            )
        
        if cls.MAX_CONCURRENT_SESSIONS < 1:
            raise ValueError("MAX_CONCURRENT_SESSIONS must be at least 1")
        
        cls.ensure_directories()


# Initialize configuration on module import
APIConfig.ensure_directories()
