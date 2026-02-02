"""
Unit tests for the ModelRegistry service.

Tests model storage, retrieval, and deletion operations.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil

from api.database import Database
from api.services.model_registry import ModelRegistry


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
def temp_models_dir():
    """Create a temporary directory for model storage."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def model_registry(temp_db, temp_models_dir):
    """Create a ModelRegistry instance with temporary database and directory."""
    return ModelRegistry(temp_db, temp_models_dir)


def test_list_models_empty(model_registry):
    """Test listing models when registry is empty."""
    models = model_registry.list_models()
    assert models == []


def test_save_model_basic(model_registry):
    """Test basic model saving."""
    model_name = "test_model"
    model_data = b"fake model data"
    
    model_info = model_registry.save_model(model_name, model_data)
    
    assert model_info.name == model_name
    assert model_info.size_mb > 0
    assert Path(model_info.file_path).exists()


def test_save_model_duplicate_raises_error(model_registry):
    """Test that saving a duplicate model raises ValueError."""
    model_name = "test_model"
    model_data = b"fake model data"
    
    model_registry.save_model(model_name, model_data)
    
    with pytest.raises(ValueError, match="already exists"):
        model_registry.save_model(model_name, model_data)


def test_list_models_returns_saved_models(model_registry):
    """Test that list_models returns all saved models."""
    # Save multiple models
    model_registry.save_model("model1", b"data1")
    model_registry.save_model("model2", b"data2")
    model_registry.save_model("model3", b"data3")
    
    models = model_registry.list_models()
    
    assert len(models) == 3
    model_names = {m.name for m in models}
    assert model_names == {"model1", "model2", "model3"}


def test_get_model_path_existing(model_registry):
    """Test getting path for an existing model."""
    model_name = "test_model"
    model_data = b"fake model data"
    
    model_info = model_registry.save_model(model_name, model_data)
    path = model_registry.get_model_path(model_name)
    
    assert path is not None
    assert path == Path(model_info.file_path)
    assert path.exists()


def test_get_model_path_nonexistent(model_registry):
    """Test getting path for a nonexistent model."""
    path = model_registry.get_model_path("nonexistent_model")
    assert path is None


def test_get_model_info_existing(model_registry):
    """Test getting info for an existing model."""
    model_name = "test_model"
    model_data = b"fake model data"
    
    saved_info = model_registry.save_model(model_name, model_data)
    retrieved_info = model_registry.get_model_info(model_name)
    
    assert retrieved_info is not None
    assert retrieved_info.name == saved_info.name
    assert retrieved_info.size_mb == saved_info.size_mb
    assert retrieved_info.file_path == saved_info.file_path


def test_get_model_info_nonexistent(model_registry):
    """Test getting info for a nonexistent model."""
    info = model_registry.get_model_info("nonexistent_model")
    assert info is None


def test_model_exists(model_registry):
    """Test checking if a model exists."""
    model_name = "test_model"
    
    assert not model_registry.model_exists(model_name)
    
    model_registry.save_model(model_name, b"data")
    
    assert model_registry.model_exists(model_name)


def test_delete_model_existing(model_registry):
    """Test deleting an existing model."""
    model_name = "test_model"
    model_data = b"fake model data"
    
    model_info = model_registry.save_model(model_name, model_data)
    file_path = Path(model_info.file_path)
    
    # Verify file exists
    assert file_path.exists()
    
    # Delete model
    deleted = model_registry.delete_model(model_name)
    
    assert deleted is True
    assert not model_registry.model_exists(model_name)
    assert not file_path.exists()


def test_delete_model_nonexistent(model_registry):
    """Test deleting a nonexistent model."""
    deleted = model_registry.delete_model("nonexistent_model")
    assert deleted is False


def test_get_model_data(model_registry):
    """Test reading model data."""
    model_name = "test_model"
    model_data = b"fake model data with some content"
    
    model_registry.save_model(model_name, model_data)
    
    retrieved_data = model_registry.get_model_data(model_name)
    
    assert retrieved_data == model_data


def test_get_model_data_nonexistent(model_registry):
    """Test reading data for nonexistent model."""
    data = model_registry.get_model_data("nonexistent_model")
    assert data is None


def test_model_size_calculation(model_registry):
    """Test that model size is calculated correctly."""
    model_name = "test_model"
    # Create 1 MB of data
    model_data = b"x" * (1024 * 1024)
    
    model_info = model_registry.save_model(model_name, model_data)
    
    # Size should be approximately 1 MB
    assert 0.99 <= model_info.size_mb <= 1.01


def test_model_timestamp(model_registry):
    """Test that model has a valid creation timestamp."""
    model_name = "test_model"
    model_data = b"fake model data"
    
    before = datetime.now(timezone.utc)
    model_info = model_registry.save_model(model_name, model_data)
    after = datetime.now(timezone.utc)
    
    # Timestamp should be between before and after
    assert before <= model_info.created_at <= after or \
           (model_info.created_at - before).total_seconds() < 1


def test_models_isolated_by_name(model_registry):
    """Test that models are properly isolated by name."""
    model_registry.save_model("model1", b"data1")
    model_registry.save_model("model2", b"data2")
    
    data1 = model_registry.get_model_data("model1")
    data2 = model_registry.get_model_data("model2")
    
    assert data1 == b"data1"
    assert data2 == b"data2"
