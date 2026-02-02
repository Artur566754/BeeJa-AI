"""Unit tests for configuration management."""

import os
import pytest
from pathlib import Path

from api.config import APIConfig


class TestAPIConfig:
    """Test API configuration management."""
    
    def test_default_values(self):
        """Test that default configuration values are set."""
        assert APIConfig.HOST == "0.0.0.0" or APIConfig.HOST == os.getenv("API_HOST", "0.0.0.0")
        assert APIConfig.PORT == 8000 or APIConfig.PORT == int(os.getenv("API_PORT", "8000"))
        assert APIConfig.DEFAULT_OPTIMIZER == "adam"
        assert APIConfig.DEFAULT_LOSS_FUNCTION == "cross_entropy"
        assert APIConfig.MAX_CONCURRENT_SESSIONS >= 1
    
    def test_validation_ranges(self):
        """Test that validation ranges are properly defined."""
        assert APIConfig.MIN_LEARNING_RATE == 0.0001
        assert APIConfig.MAX_LEARNING_RATE == 1.0
        assert APIConfig.MIN_BATCH_SIZE == 1
        assert APIConfig.MAX_BATCH_SIZE == 256
        assert APIConfig.MIN_EPOCHS == 1
        assert APIConfig.MAX_EPOCHS == 1000
    
    def test_valid_options(self):
        """Test that valid options lists are defined."""
        assert "adam" in APIConfig.VALID_OPTIMIZERS
        assert "sgd" in APIConfig.VALID_OPTIMIZERS
        assert "cross_entropy" in APIConfig.VALID_LOSS_FUNCTIONS
        assert "mse" in APIConfig.VALID_LOSS_FUNCTIONS
    
    def test_ensure_directories(self):
        """Test that ensure_directories creates required directories."""
        # This should not raise an error
        APIConfig.ensure_directories()
        
        # Verify directories exist
        assert APIConfig.MODELS_DIR.exists()
        assert APIConfig.DATASETS_DIR.exists()
    
    def test_paths_are_path_objects(self):
        """Test that path configurations are Path objects."""
        assert isinstance(APIConfig.DATABASE_PATH, Path)
        assert isinstance(APIConfig.MODELS_DIR, Path)
        assert isinstance(APIConfig.DATASETS_DIR, Path)
