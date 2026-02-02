"""
Model Registry service for the Server Management API.

This module provides storage and management of trained model files with metadata.
Models are stored in the file system with metadata indexed in SQLite.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import os

from api.database import Database, get_database
from api.models.data_models import ModelInfo
from api.config import APIConfig


class ModelRegistry:
    """
    Manages storage and retrieval of trained model files.
    
    Provides methods to list, save, retrieve, and delete models with
    automatic metadata management.
    """
    
    def __init__(
        self,
        database: Optional[Database] = None,
        models_dir: Optional[Path] = None
    ):
        """
        Initialize the ModelRegistry.
        
        Args:
            database: Database instance. If None, uses global database.
            models_dir: Directory for model storage. If None, uses config default.
        """
        self.db = database or get_database()
        self.models_dir = models_dir or APIConfig.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def list_models(self) -> List[ModelInfo]:
        """
        List all models in the registry.
        
        Returns:
            List of ModelInfo objects with metadata for all models
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, size_mb, file_path, created_at
                FROM models
                ORDER BY created_at DESC
                """
            )
            
            rows = cursor.fetchall()
            
            models = []
            for row in rows:
                models.append(ModelInfo(
                    name=row['name'],
                    size_mb=row['size_mb'],
                    file_path=row['file_path'],
                    created_at=datetime.fromisoformat(row['created_at'])
                ))
            
            return models
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get the file path for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to model file, or None if model doesn't exist
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_path FROM models WHERE name = ?",
                (model_name,)
            )
            
            row = cursor.fetchone()
            if row:
                return Path(row['file_path'])
            return None
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get metadata for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo object, or None if model doesn't exist
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, size_mb, file_path, created_at
                FROM models
                WHERE name = ?
                """,
                (model_name,)
            )
            
            row = cursor.fetchone()
            if row:
                return ModelInfo(
                    name=row['name'],
                    size_mb=row['size_mb'],
                    file_path=row['file_path'],
                    created_at=datetime.fromisoformat(row['created_at'])
                )
            return None
    
    def save_model(
        self,
        model_name: str,
        file_data: bytes,
        timestamp: Optional[datetime] = None
    ) -> ModelInfo:
        """
        Save a model file to the registry.
        
        Args:
            model_name: Name for the model
            file_data: Binary model file data
            timestamp: Optional creation timestamp (defaults to now)
            
        Returns:
            ModelInfo object with saved model metadata
            
        Raises:
            ValueError: If model name already exists
        """
        # Check if model already exists
        if self.model_exists(model_name):
            raise ValueError(f"Model '{model_name}' already exists")
        
        # Generate file path with timestamp
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp_str}.pth"
        file_path = self.models_dir / filename
        
        # Write file to disk
        file_path.write_bytes(file_data)
        
        # Calculate size in MB
        size_mb = len(file_data) / (1024 * 1024)
        
        # Store metadata in database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO models (name, size_mb, file_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (model_name, size_mb, str(file_path), timestamp.isoformat())
            )
        
        return ModelInfo(
            name=model_name,
            size_mb=size_mb,
            file_path=str(file_path),
            created_at=timestamp
        )
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from the registry.
        
        Removes both the database entry and the file from the filesystem.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if model was deleted, False if model didn't exist
        """
        # Get model path before deleting from database
        model_path = self.get_model_path(model_name)
        
        if model_path is None:
            return False
        
        # Delete from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM models WHERE name = ?",
                (model_name,)
            )
            deleted = cursor.rowcount > 0
        
        # Delete file from filesystem if it exists
        if deleted and model_path.exists():
            try:
                model_path.unlink()
            except OSError:
                # Log error but don't fail the operation
                pass
        
        return deleted
    
    def model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in the registry.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model exists, False otherwise
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM models WHERE name = ? LIMIT 1",
                (model_name,)
            )
            return cursor.fetchone() is not None
    
    def get_model_data(self, model_name: str) -> Optional[bytes]:
        """
        Read and return the binary data for a model file.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Binary model file data, or None if model doesn't exist
        """
        model_path = self.get_model_path(model_name)
        
        if model_path is None or not model_path.exists():
            return None
        
        return model_path.read_bytes()
