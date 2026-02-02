"""
Property-based tests for the ModelRegistry service.

Tests universal properties that should hold for all model operations.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
import shutil
import time

from api.database import Database
from api.services.model_registry import ModelRegistry


# Strategies for generating test data
model_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=50
).filter(lambda s: s.strip() and not s.startswith("."))

model_data = st.binary(min_size=1, max_size=10000)


def create_temp_registry():
    """Create a temporary model registry for testing."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    
    # Create temporary models directory
    models_dir = Path(tempfile.mkdtemp())
    
    db = Database(db_path)
    registry = ModelRegistry(db, models_dir)
    
    return registry, db, db_path, models_dir


def cleanup_registry(db, db_path, models_dir):
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
        if models_dir.exists():
            shutil.rmtree(models_dir, ignore_errors=True)
    except:
        pass


@given(
    model_names_list=st.lists(model_names, min_size=0, max_size=10, unique=True)
)
@settings(max_examples=100, deadline=None)
def test_property_model_list_returns_all_models(model_names_list):
    """
    Feature: server-management-api, Property 12: Model list returns all models
    
    **Validates: Requirements 4.1**
    
    For any set of N models in the model registry, listing models should
    return exactly N entries with complete information (name, size, creation date).
    """
    registry, db, db_path, models_dir = create_temp_registry()
    
    try:
        # Save N models
        for model_name in model_names_list:
            registry.save_model(model_name, b"test data")
        
        # Property: List should return exactly N models
        models = registry.list_models()
        
        assert len(models) == len(model_names_list), \
            f"Should return exactly {len(model_names_list)} models"
        
        # Verify all models have complete information
        for model in models:
            assert model.name, "Model must have a name"
            assert model.size_mb >= 0, "Model must have a valid size"
            assert model.created_at is not None, "Model must have a creation date"
            assert model.file_path, "Model must have a file path"
        
        # Verify all saved models are in the list
        listed_names = {m.name for m in models}
        expected_names = set(model_names_list)
        assert listed_names == expected_names, \
            "Listed models should match saved models"
    
    finally:
        cleanup_registry(db, db_path, models_dir)


@given(
    model_name=model_names,
    model_data_bytes=model_data
)
@settings(max_examples=100, deadline=None)
def test_property_model_download_round_trip(model_name, model_data_bytes):
    """
    Feature: server-management-api, Property 13: Model download round-trip
    
    **Validates: Requirements 4.2, 4.3**
    
    For any model file uploaded to the registry, downloading it by name
    should return file data identical to the original upload.
    """
    registry, db, db_path, models_dir = create_temp_registry()
    
    try:
        # Upload model
        model_info = registry.save_model(model_name, model_data_bytes)
        
        # Property: Downloaded data should match uploaded data
        downloaded_data = registry.get_model_data(model_name)
        
        assert downloaded_data is not None, \
            "Should be able to download uploaded model"
        
        assert downloaded_data == model_data_bytes, \
            "Downloaded data should be identical to uploaded data"
        
        # Verify model info is correct
        assert model_info.name == model_name, \
            "Model info should have correct name"
        
        # Verify file exists on disk
        file_path = Path(model_info.file_path)
        assert file_path.exists(), \
            "Model file should exist on disk"
        
        # Verify file content matches
        file_content = file_path.read_bytes()
        assert file_content == model_data_bytes, \
            "File content on disk should match uploaded data"
    
    finally:
        cleanup_registry(db, db_path, models_dir)


@given(
    model_name=model_names,
    model_data_bytes=model_data
)
@settings(max_examples=100, deadline=None)
def test_property_model_deletion_removes_all_traces(model_name, model_data_bytes):
    """
    Feature: server-management-api, Property 14: Model deletion removes all traces
    
    **Validates: Requirements 4.4**
    
    For any existing model in the registry, deleting it should remove both
    the database entry and the file from the filesystem, and subsequent
    operations on that model should return 404 errors (None in service layer).
    """
    registry, db, db_path, models_dir = create_temp_registry()
    
    try:
        # Save model
        model_info = registry.save_model(model_name, model_data_bytes)
        file_path = Path(model_info.file_path)
        
        # Verify model exists
        assert registry.model_exists(model_name), \
            "Model should exist before deletion"
        assert file_path.exists(), \
            "Model file should exist before deletion"
        
        # Delete model
        deleted = registry.delete_model(model_name)
        
        # Property: Deletion should succeed
        assert deleted is True, \
            "Deletion should return True for existing model"
        
        # Property: Model should no longer exist in database
        assert not registry.model_exists(model_name), \
            "Model should not exist in database after deletion"
        
        # Property: File should be removed from filesystem
        assert not file_path.exists(), \
            "Model file should be removed from filesystem"
        
        # Property: Subsequent operations should return None/False
        assert registry.get_model_path(model_name) is None, \
            "get_model_path should return None for deleted model"
        
        assert registry.get_model_info(model_name) is None, \
            "get_model_info should return None for deleted model"
        
        assert registry.get_model_data(model_name) is None, \
            "get_model_data should return None for deleted model"
        
        # Property: Model should not appear in list
        models = registry.list_models()
        model_names_in_list = {m.name for m in models}
        assert model_name not in model_names_in_list, \
            "Deleted model should not appear in model list"
    
    finally:
        cleanup_registry(db, db_path, models_dir)


@given(
    model_name=model_names
)
@settings(max_examples=100, deadline=None)
def test_property_nonexistent_model_returns_none(model_name):
    """
    Feature: server-management-api, Property 15: Non-existent resource returns 404
    
    **Validates: Requirements 4.5**
    
    For any randomly generated model name that doesn't exist, operations
    on that model should return None (which maps to 404 in API layer).
    """
    registry, db, db_path, models_dir = create_temp_registry()
    
    try:
        # Ensure model doesn't exist
        assume(not registry.model_exists(model_name))
        
        # Property: All operations should return None/False for nonexistent model
        assert registry.get_model_path(model_name) is None, \
            "get_model_path should return None for nonexistent model"
        
        assert registry.get_model_info(model_name) is None, \
            "get_model_info should return None for nonexistent model"
        
        assert registry.get_model_data(model_name) is None, \
            "get_model_data should return None for nonexistent model"
        
        assert not registry.model_exists(model_name), \
            "model_exists should return False for nonexistent model"
        
        # Property: Deleting nonexistent model should return False
        deleted = registry.delete_model(model_name)
        assert deleted is False, \
            "Deleting nonexistent model should return False"
    
    finally:
        cleanup_registry(db, db_path, models_dir)


@given(
    model_name=model_names,
    model_data_bytes=model_data
)
@settings(max_examples=100, deadline=None)
def test_property_duplicate_model_raises_error(model_name, model_data_bytes):
    """
    Property: Saving a model with duplicate name raises ValueError.
    
    For any model name that already exists in the registry, attempting
    to save another model with the same name should raise ValueError.
    """
    registry, db, db_path, models_dir = create_temp_registry()
    
    try:
        # Save first model
        registry.save_model(model_name, model_data_bytes)
        
        # Property: Saving duplicate should raise ValueError
        try:
            registry.save_model(model_name, b"different data")
            assert False, "Should have raised ValueError for duplicate model"
        except ValueError as e:
            assert "already exists" in str(e).lower(), \
                "Error message should indicate model already exists"
    
    finally:
        cleanup_registry(db, db_path, models_dir)


@given(
    model_names_list=st.lists(model_names, min_size=2, max_size=5, unique=True)
)
@settings(max_examples=100, deadline=None)
def test_property_models_isolated_by_name(model_names_list):
    """
    Property: Models are properly isolated by name.
    
    For any set of models with different names, operations on one model
    should not affect other models.
    """
    assume(len(model_names_list) >= 2)
    
    registry, db, db_path, models_dir = create_temp_registry()
    
    try:
        # Save multiple models with different data
        model_data_map = {}
        for i, model_name in enumerate(model_names_list):
            data = f"data_{i}".encode()
            registry.save_model(model_name, data)
            model_data_map[model_name] = data
        
        # Property: Each model should have its own data
        for model_name, expected_data in model_data_map.items():
            retrieved_data = registry.get_model_data(model_name)
            assert retrieved_data == expected_data, \
                f"Model {model_name} should have its own data"
        
        # Property: Deleting one model shouldn't affect others
        first_model = model_names_list[0]
        registry.delete_model(first_model)
        
        # Other models should still exist
        for model_name in model_names_list[1:]:
            assert registry.model_exists(model_name), \
                f"Model {model_name} should still exist after deleting {first_model}"
            
            retrieved_data = registry.get_model_data(model_name)
            assert retrieved_data == model_data_map[model_name], \
                f"Model {model_name} data should be unchanged"
    
    finally:
        cleanup_registry(db, db_path, models_dir)
