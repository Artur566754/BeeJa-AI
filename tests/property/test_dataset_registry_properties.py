"""
Property-based tests for the DatasetRegistry service.

Tests universal properties that should hold for all dataset operations.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
import shutil
import time
import json

from api.database import Database
from api.services.dataset_registry import DatasetRegistry


# Strategies for generating test data
dataset_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=50
).filter(lambda s: s.strip() and not s.startswith("."))

dataset_data = st.binary(min_size=1, max_size=10000)


def create_temp_registry():
    """Create a temporary dataset registry for testing."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    # Create temporary datasets directory
    datasets_dir = Path(tempfile.mkdtemp())
    
    db = Database(db_path)
    registry = DatasetRegistry(db, datasets_dir)
    
    return registry, db, db_path, datasets_dir


def cleanup_registry(db, db_path, datasets_dir):
    """Clean up temporary registry resources."""
    try:
        db.close()
        time.sleep(0.1)
    except:
        pass
    
    try:
        if db_path.exists():
            db_path.unlink()
    except:
        pass
    
    try:
        if datasets_dir.exists():
            shutil.rmtree(datasets_dir, ignore_errors=True)
    except:
        pass


@given(
    dataset_names_list=st.lists(dataset_names, min_size=0, max_size=10, unique=True)
)
@settings(max_examples=100, deadline=None)
def test_property_dataset_list_returns_complete_metadata(dataset_names_list):
    """
    Feature: server-management-api, Property 16: Dataset list returns complete metadata
    
    **Validates: Requirements 5.1**
    
    For any set of N datasets in the registry, listing datasets should
    return exactly N entries with complete metadata (name, size, sample count,
    format, dimensions, creation date).
    """
    registry, db, db_path, datasets_dir = create_temp_registry()
    
    try:
        # Save N datasets
        for dataset_name in dataset_names_list:
            registry.save_dataset(dataset_name, b"test data")
        
        # Property: List should return exactly N datasets
        datasets = registry.list_datasets()
        
        assert len(datasets) == len(dataset_names_list), \
            f"Should return exactly {len(dataset_names_list)} datasets"
        
        # Verify all datasets have complete metadata
        for dataset in datasets:
            assert dataset.name, "Dataset must have a name"
            assert dataset.size_mb >= 0, "Dataset must have a valid size"
            assert dataset.sample_count >= 0, "Dataset must have a sample count"
            assert dataset.format, "Dataset must have a format"
            # dimensions can be None for some formats
            assert dataset.created_at is not None, "Dataset must have a creation date"
            assert dataset.file_path, "Dataset must have a file path"
        
        # Verify all saved datasets are in the list
        listed_names = {d.name for d in datasets}
        expected_names = set(dataset_names_list)
        assert listed_names == expected_names, \
            "Listed datasets should match saved datasets"
    
    finally:
        cleanup_registry(db, db_path, datasets_dir)


@given(
    dataset_name=dataset_names
)
@settings(max_examples=100, deadline=None)
def test_property_dataset_info_returns_correct_metadata(dataset_name):
    """
    Feature: server-management-api, Property 17: Dataset info returns correct metadata
    
    **Validates: Requirements 5.2**
    
    For any dataset in the registry, requesting its information should
    return metadata that matches the dataset's actual properties
    (format, dimensions, sample count).
    """
    registry, db, db_path, datasets_dir = create_temp_registry()
    
    try:
        # Create a JSON dataset with known properties
        json_data = json.dumps([
            {"x": 1, "y": 2},
            {"x": 3, "y": 4},
            {"x": 5, "y": 6}
        ]).encode('utf-8')
        
        # Save dataset
        saved_info = registry.save_dataset(dataset_name, json_data)
        
        # Property: Retrieved info should match saved info
        retrieved_info = registry.get_dataset_info(dataset_name)
        
        assert retrieved_info is not None, \
            "Should be able to retrieve dataset info"
        
        assert retrieved_info.name == saved_info.name, \
            "Name should match"
        
        assert retrieved_info.format == saved_info.format, \
            "Format should match"
        
        assert retrieved_info.sample_count == saved_info.sample_count, \
            "Sample count should match"
        
        assert retrieved_info.dimensions == saved_info.dimensions, \
            "Dimensions should match"
        
        assert retrieved_info.size_mb == saved_info.size_mb, \
            "Size should match"
        
        assert retrieved_info.file_path == saved_info.file_path, \
            "File path should match"
        
        # Property: Metadata should reflect actual dataset properties
        assert retrieved_info.format == "json", \
            "Format should be correctly detected as JSON"
        
        assert retrieved_info.sample_count == 3, \
            "Sample count should be correctly extracted"
    
    finally:
        cleanup_registry(db, db_path, datasets_dir)


@given(
    dataset_name=dataset_names,
    dataset_data_bytes=dataset_data
)
@settings(max_examples=100, deadline=None)
def test_property_dataset_upload_stores_with_metadata(dataset_name, dataset_data_bytes):
    """
    Feature: server-management-api, Property 18: Dataset upload stores with metadata
    
    **Validates: Requirements 5.3**
    
    For any valid dataset file uploaded, it should be stored in the registry
    with correctly extracted metadata (size, format, sample count).
    """
    registry, db, db_path, datasets_dir = create_temp_registry()
    
    try:
        # Upload dataset
        dataset_info = registry.save_dataset(dataset_name, dataset_data_bytes)
        
        # Property: Dataset should be stored with metadata
        assert dataset_info.name == dataset_name, \
            "Dataset should have correct name"
        
        assert dataset_info.size_mb > 0, \
            "Dataset should have positive size"
        
        assert dataset_info.format is not None, \
            "Dataset should have a format"
        
        assert dataset_info.sample_count >= 0, \
            "Dataset should have non-negative sample count"
        
        assert dataset_info.file_path, \
            "Dataset should have a file path"
        
        assert dataset_info.created_at is not None, \
            "Dataset should have a creation timestamp"
        
        # Property: File should exist on disk
        file_path = Path(dataset_info.file_path)
        assert file_path.exists(), \
            "Dataset file should exist on disk"
        
        # Property: File content should match uploaded data
        file_content = file_path.read_bytes()
        assert file_content == dataset_data_bytes, \
            "File content should match uploaded data"
        
        # Property: Dataset should be retrievable
        retrieved_info = registry.get_dataset_info(dataset_name)
        assert retrieved_info is not None, \
            "Dataset should be retrievable after upload"
        
        assert retrieved_info.name == dataset_name, \
            "Retrieved dataset should have correct name"
    
    finally:
        cleanup_registry(db, db_path, datasets_dir)


@given(
    dataset_name=dataset_names
)
@settings(max_examples=100, deadline=None)
def test_property_nonexistent_dataset_returns_none(dataset_name):
    """
    Feature: server-management-api, Property 15: Non-existent resource returns 404
    
    **Validates: Requirements 5.4**
    
    For any randomly generated dataset name that doesn't exist, operations
    on that dataset should return None (which maps to 404 in API layer).
    """
    registry, db, db_path, datasets_dir = create_temp_registry()
    
    try:
        # Ensure dataset doesn't exist
        assume(not registry.dataset_exists(dataset_name))
        
        # Property: All operations should return None/False for nonexistent dataset
        assert registry.get_dataset_info(dataset_name) is None, \
            "get_dataset_info should return None for nonexistent dataset"
        
        assert registry.get_dataset_path(dataset_name) is None, \
            "get_dataset_path should return None for nonexistent dataset"
        
        assert not registry.dataset_exists(dataset_name), \
            "dataset_exists should return False for nonexistent dataset"
        
        # Property: Deleting nonexistent dataset should return False
        deleted = registry.delete_dataset(dataset_name)
        assert deleted is False, \
            "Deleting nonexistent dataset should return False"
    
    finally:
        cleanup_registry(db, db_path, datasets_dir)


@given(
    dataset_names_list=st.lists(dataset_names, min_size=2, max_size=5, unique=True)
)
@settings(max_examples=100, deadline=None)
def test_property_datasets_isolated_by_name(dataset_names_list):
    """
    Property: Datasets are properly isolated by name.
    
    For any set of datasets with different names, operations on one dataset
    should not affect other datasets.
    """
    assume(len(dataset_names_list) >= 2)
    
    registry, db, db_path, datasets_dir = create_temp_registry()
    
    try:
        # Save multiple datasets with different data
        dataset_data_map = {}
        for i, dataset_name in enumerate(dataset_names_list):
            data = f"data_{i}".encode()
            registry.save_dataset(dataset_name, data)
            dataset_data_map[dataset_name] = data
        
        # Property: Each dataset should have its own metadata
        for dataset_name in dataset_names_list:
            info = registry.get_dataset_info(dataset_name)
            assert info is not None, \
                f"Dataset {dataset_name} should exist"
            assert info.name == dataset_name, \
                f"Dataset should have correct name"
        
        # Property: Deleting one dataset shouldn't affect others
        first_dataset = dataset_names_list[0]
        registry.delete_dataset(first_dataset)
        
        # Other datasets should still exist
        for dataset_name in dataset_names_list[1:]:
            assert registry.dataset_exists(dataset_name), \
                f"Dataset {dataset_name} should still exist after deleting {first_dataset}"
            
            info = registry.get_dataset_info(dataset_name)
            assert info is not None, \
                f"Dataset {dataset_name} info should still be retrievable"
    
    finally:
        cleanup_registry(db, db_path, datasets_dir)
