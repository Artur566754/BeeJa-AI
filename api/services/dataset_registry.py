"""
Dataset Registry service for the Server Management API.

This module provides storage and management of dataset files with metadata extraction.
Datasets are stored in the file system with metadata indexed in SQLite.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import json
import struct

from api.database import Database, get_database
from api.models.data_models import DatasetInfo
from api.config import APIConfig


class DatasetRegistry:
    """
    Manages storage and retrieval of dataset files with metadata.
    
    Provides methods to list, save, retrieve datasets with automatic
    metadata extraction for format, dimensions, and sample count.
    """
    
    def __init__(
        self,
        database: Optional[Database] = None,
        datasets_dir: Optional[Path] = None
    ):
        """
        Initialize the DatasetRegistry.
        
        Args:
            database: Database instance. If None, uses global database.
            datasets_dir: Directory for dataset storage. If None, uses config default.
        """
        self.db = database or get_database()
        self.datasets_dir = datasets_dir or APIConfig.DATASETS_DIR
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self) -> List[DatasetInfo]:
        """
        List all datasets in the registry.
        
        Returns:
            List of DatasetInfo objects with metadata for all datasets
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, size_mb, sample_count, format, dimensions, 
                       file_path, created_at
                FROM datasets
                ORDER BY created_at DESC
                """
            )
            
            rows = cursor.fetchall()
            
            datasets = []
            for row in rows:
                datasets.append(DatasetInfo(
                    name=row['name'],
                    size_mb=row['size_mb'],
                    sample_count=row['sample_count'],
                    format=row['format'],
                    dimensions=row['dimensions'],
                    file_path=row['file_path'],
                    created_at=datetime.fromisoformat(row['created_at'])
                ))
            
            return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """
        Get metadata for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DatasetInfo object, or None if dataset doesn't exist
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT name, size_mb, sample_count, format, dimensions,
                       file_path, created_at
                FROM datasets
                WHERE name = ?
                """,
                (dataset_name,)
            )
            
            row = cursor.fetchone()
            if row:
                return DatasetInfo(
                    name=row['name'],
                    size_mb=row['size_mb'],
                    sample_count=row['sample_count'],
                    format=row['format'],
                    dimensions=row['dimensions'],
                    file_path=row['file_path'],
                    created_at=datetime.fromisoformat(row['created_at'])
                )
            return None
    
    def save_dataset(
        self,
        dataset_name: str,
        file_data: bytes,
        timestamp: Optional[datetime] = None
    ) -> DatasetInfo:
        """
        Save a dataset file to the registry with metadata extraction.
        
        Args:
            dataset_name: Name for the dataset
            file_data: Binary dataset file data
            timestamp: Optional creation timestamp (defaults to now)
            
        Returns:
            DatasetInfo object with saved dataset metadata
            
        Raises:
            ValueError: If dataset name already exists
        """
        # Check if dataset already exists
        if self.dataset_exists(dataset_name):
            raise ValueError(f"Dataset '{dataset_name}' already exists")
        
        # Generate file path with timestamp
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Extract metadata from file data
        metadata = self._extract_metadata(file_data)
        
        # Determine file extension based on format
        ext = self._get_file_extension(metadata['format'])
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_{timestamp_str}{ext}"
        file_path = self.datasets_dir / filename
        
        # Write file to disk
        file_path.write_bytes(file_data)
        
        # Calculate size in MB
        size_mb = len(file_data) / (1024 * 1024)
        
        # Store metadata in database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO datasets 
                (name, size_mb, sample_count, format, dimensions, file_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_name,
                    size_mb,
                    metadata['sample_count'],
                    metadata['format'],
                    metadata['dimensions'],
                    str(file_path),
                    timestamp.isoformat()
                )
            )
        
        return DatasetInfo(
            name=dataset_name,
            size_mb=size_mb,
            sample_count=metadata['sample_count'],
            format=metadata['format'],
            dimensions=metadata['dimensions'],
            file_path=str(file_path),
            created_at=timestamp
        )
    
    def dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset exists in the registry.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if dataset exists, False otherwise
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM datasets WHERE name = ? LIMIT 1",
                (dataset_name,)
            )
            return cursor.fetchone() is not None
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset from the registry.
        
        Removes both the database entry and the file from the filesystem.
        
        Args:
            dataset_name: Name of the dataset to delete
            
        Returns:
            True if dataset was deleted, False if dataset didn't exist
        """
        # Get dataset info before deleting
        dataset_info = self.get_dataset_info(dataset_name)
        
        if dataset_info is None:
            return False
        
        # Delete from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM datasets WHERE name = ?",
                (dataset_name,)
            )
            deleted = cursor.rowcount > 0
        
        # Delete file from filesystem if it exists
        if deleted:
            file_path = Path(dataset_info.file_path)
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError:
                    # Log error but don't fail the operation
                    pass
        
        return deleted
    
    def get_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """
        Get the file path for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to dataset file, or None if dataset doesn't exist
        """
        dataset_info = self.get_dataset_info(dataset_name)
        if dataset_info:
            return Path(dataset_info.file_path)
        return None
    
    def _extract_metadata(self, file_data: bytes) -> dict:
        """
        Extract metadata from dataset file data.
        
        Attempts to detect format and extract sample count and dimensions.
        
        Args:
            file_data: Binary dataset file data
            
        Returns:
            Dictionary with 'format', 'sample_count', and 'dimensions' keys
        """
        # Try to detect format
        format_type = self._detect_format(file_data)
        
        # Extract metadata based on format
        if format_type == 'json':
            return self._extract_json_metadata(file_data)
        elif format_type == 'csv':
            return self._extract_csv_metadata(file_data)
        elif format_type == 'numpy':
            return self._extract_numpy_metadata(file_data)
        else:
            # Unknown format - use defaults
            return {
                'format': 'binary',
                'sample_count': 0,
                'dimensions': None
            }
    
    def _detect_format(self, file_data: bytes) -> str:
        """
        Detect the format of dataset file data.
        
        Args:
            file_data: Binary dataset file data
            
        Returns:
            Format string: 'json', 'csv', 'numpy', or 'unknown'
        """
        # Check for JSON
        try:
            json.loads(file_data.decode('utf-8'))
            return 'json'
        except:
            pass
        
        # Check for CSV (simple heuristic: contains commas and newlines)
        try:
            text = file_data.decode('utf-8')
            if ',' in text and '\n' in text:
                return 'csv'
        except:
            pass
        
        # Check for NumPy array (magic number)
        if file_data.startswith(b'\x93NUMPY'):
            return 'numpy'
        
        return 'unknown'
    
    def _extract_json_metadata(self, file_data: bytes) -> dict:
        """Extract metadata from JSON dataset."""
        try:
            data = json.loads(file_data.decode('utf-8'))
            
            # Assume JSON is either array of samples or dict with 'data' key
            if isinstance(data, list):
                sample_count = len(data)
                dimensions = None
                if sample_count > 0 and isinstance(data[0], (list, dict)):
                    if isinstance(data[0], list):
                        dimensions = str(len(data[0]))
            elif isinstance(data, dict) and 'data' in data:
                samples = data['data']
                sample_count = len(samples) if isinstance(samples, list) else 0
                dimensions = data.get('dimensions')
            else:
                sample_count = 1
                dimensions = None
            
            return {
                'format': 'json',
                'sample_count': sample_count,
                'dimensions': dimensions
            }
        except:
            return {
                'format': 'json',
                'sample_count': 0,
                'dimensions': None
            }
    
    def _extract_csv_metadata(self, file_data: bytes) -> dict:
        """Extract metadata from CSV dataset."""
        try:
            text = file_data.decode('utf-8')
            lines = text.strip().split('\n')
            
            # Count rows (excluding header)
            sample_count = max(0, len(lines) - 1)
            
            # Count columns from first data row
            if len(lines) > 1:
                columns = len(lines[1].split(','))
                dimensions = str(columns)
            else:
                dimensions = None
            
            return {
                'format': 'csv',
                'sample_count': sample_count,
                'dimensions': dimensions
            }
        except:
            return {
                'format': 'csv',
                'sample_count': 0,
                'dimensions': None
            }
    
    def _extract_numpy_metadata(self, file_data: bytes) -> dict:
        """Extract metadata from NumPy array file."""
        try:
            # Parse NumPy .npy format header
            # This is a simplified parser - real implementation would use numpy
            # For now, return basic metadata
            return {
                'format': 'numpy',
                'sample_count': 0,  # Would need numpy to parse properly
                'dimensions': None
            }
        except:
            return {
                'format': 'numpy',
                'sample_count': 0,
                'dimensions': None
            }
    
    def _get_file_extension(self, format_type: str) -> str:
        """Get file extension for a format type."""
        extensions = {
            'json': '.json',
            'csv': '.csv',
            'numpy': '.npy',
            'binary': '.dat'
        }
        return extensions.get(format_type, '.dat')
