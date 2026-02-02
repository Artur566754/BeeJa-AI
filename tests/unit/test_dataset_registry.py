"""
Unit tests for the DatasetRegistry service.

Tests dataset storage, retrieval, metadata extraction, and deletion operations.
"""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import shutil
import json

from api.database import Database
from api.services.dataset_registry import DatasetRegistry


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
def temp_datasets_dir():
    """Create a temporary directory for dataset storage."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def dataset_registry(temp_db, temp_datasets_dir):
    """Create a DatasetRegistry instance with temporary database and directory."""
    return DatasetRegistry(temp_db, temp_datasets_dir)


def test_list_datasets_empty(dataset_registry):
    """Test listing datasets when registry is empty."""
    datasets = dataset_registry.list_datasets()
    assert datasets == []


def test_save_dataset_basic(dataset_registry):
    """Test basic dataset saving."""
    dataset_name = "test_dataset"
    dataset_data = b"fake dataset data"
    
    dataset_info = dataset_registry.save_dataset(dataset_name, dataset_data)
    
    assert dataset_info.name == dataset_name
    assert dataset_info.size_mb > 0
    assert Path(dataset_info.file_path).exists()


def test_save_dataset_duplicate_raises_error(dataset_registry):
    """Test that saving a duplicate dataset raises ValueError."""
    dataset_name = "test_dataset"
    dataset_data = b"fake dataset data"
    
    dataset_registry.save_dataset(dataset_name, dataset_data)
    
    with pytest.raises(ValueError, match="already exists"):
        dataset_registry.save_dataset(dataset_name, dataset_data)


def test_list_datasets_returns_saved_datasets(dataset_registry):
    """Test that list_datasets returns all saved datasets."""
    # Save multiple datasets
    dataset_registry.save_dataset("dataset1", b"data1")
    dataset_registry.save_dataset("dataset2", b"data2")
    dataset_registry.save_dataset("dataset3", b"data3")
    
    datasets = dataset_registry.list_datasets()
    
    assert len(datasets) == 3
    dataset_names = {d.name for d in datasets}
    assert dataset_names == {"dataset1", "dataset2", "dataset3"}


def test_get_dataset_info_existing(dataset_registry):
    """Test getting info for an existing dataset."""
    dataset_name = "test_dataset"
    dataset_data = b"fake dataset data"
    
    saved_info = dataset_registry.save_dataset(dataset_name, dataset_data)
    retrieved_info = dataset_registry.get_dataset_info(dataset_name)
    
    assert retrieved_info is not None
    assert retrieved_info.name == saved_info.name
    assert retrieved_info.size_mb == saved_info.size_mb
    assert retrieved_info.file_path == saved_info.file_path


def test_get_dataset_info_nonexistent(dataset_registry):
    """Test getting info for a nonexistent dataset."""
    info = dataset_registry.get_dataset_info("nonexistent_dataset")
    assert info is None


def test_dataset_exists(dataset_registry):
    """Test checking if a dataset exists."""
    dataset_name = "test_dataset"
    
    assert not dataset_registry.dataset_exists(dataset_name)
    
    dataset_registry.save_dataset(dataset_name, b"data")
    
    assert dataset_registry.dataset_exists(dataset_name)


def test_delete_dataset_existing(dataset_registry):
    """Test deleting an existing dataset."""
    dataset_name = "test_dataset"
    dataset_data = b"fake dataset data"
    
    dataset_info = dataset_registry.save_dataset(dataset_name, dataset_data)
    file_path = Path(dataset_info.file_path)
    
    # Verify file exists
    assert file_path.exists()
    
    # Delete dataset
    deleted = dataset_registry.delete_dataset(dataset_name)
    
    assert deleted is True
    assert not dataset_registry.dataset_exists(dataset_name)
    assert not file_path.exists()


def test_delete_dataset_nonexistent(dataset_registry):
    """Test deleting a nonexistent dataset."""
    deleted = dataset_registry.delete_dataset("nonexistent_dataset")
    assert deleted is False


def test_get_dataset_path(dataset_registry):
    """Test getting dataset file path."""
    dataset_name = "test_dataset"
    dataset_data = b"fake dataset data"
    
    dataset_info = dataset_registry.save_dataset(dataset_name, dataset_data)
    path = dataset_registry.get_dataset_path(dataset_name)
    
    assert path is not None
    assert path == Path(dataset_info.file_path)
    assert path.exists()


def test_get_dataset_path_nonexistent(dataset_registry):
    """Test getting path for nonexistent dataset."""
    path = dataset_registry.get_dataset_path("nonexistent_dataset")
    assert path is None


def test_json_format_detection(dataset_registry):
    """Test JSON format detection and metadata extraction."""
    dataset_name = "json_dataset"
    json_data = json.dumps([
        {"feature1": 1, "feature2": 2},
        {"feature1": 3, "feature2": 4},
        {"feature1": 5, "feature2": 6}
    ]).encode('utf-8')
    
    dataset_info = dataset_registry.save_dataset(dataset_name, json_data)
    
    assert dataset_info.format == "json"
    assert dataset_info.sample_count == 3


def test_csv_format_detection(dataset_registry):
    """Test CSV format detection and metadata extraction."""
    dataset_name = "csv_dataset"
    csv_data = b"col1,col2,col3\n1,2,3\n4,5,6\n7,8,9"
    
    dataset_info = dataset_registry.save_dataset(dataset_name, csv_data)
    
    assert dataset_info.format == "csv"
    assert dataset_info.sample_count == 3
    assert dataset_info.dimensions == "3"


def test_dataset_size_calculation(dataset_registry):
    """Test that dataset size is calculated correctly."""
    dataset_name = "test_dataset"
    # Create 1 MB of data
    dataset_data = b"x" * (1024 * 1024)
    
    dataset_info = dataset_registry.save_dataset(dataset_name, dataset_data)
    
    # Size should be approximately 1 MB
    assert 0.99 <= dataset_info.size_mb <= 1.01


def test_dataset_timestamp(dataset_registry):
    """Test that dataset has a valid creation timestamp."""
    dataset_name = "test_dataset"
    dataset_data = b"fake dataset data"
    
    before = datetime.now(timezone.utc)
    dataset_info = dataset_registry.save_dataset(dataset_name, dataset_data)
    after = datetime.now(timezone.utc)
    
    # Timestamp should be between before and after
    assert before <= dataset_info.created_at <= after or \
           (dataset_info.created_at - before).total_seconds() < 1


def test_datasets_isolated_by_name(dataset_registry):
    """Test that datasets are properly isolated by name."""
    dataset_registry.save_dataset("dataset1", b"data1")
    dataset_registry.save_dataset("dataset2", b"data2")
    
    info1 = dataset_registry.get_dataset_info("dataset1")
    info2 = dataset_registry.get_dataset_info("dataset2")
    
    assert info1.name == "dataset1"
    assert info2.name == "dataset2"
    assert info1.file_path != info2.file_path
