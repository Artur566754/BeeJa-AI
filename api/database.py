"""
Database management for the Server Management API.

This module handles SQLite database connection, table creation, and schema management.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
import threading

from api.config import APIConfig


class Database:
    """SQLite database manager with connection pooling and schema management."""
    
    _local = threading.local()
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses config default.
        """
        self.db_path = db_path or APIConfig.DATABASE_PATH
        self._ensure_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.
        
        Yields:
            SQLite connection object
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _ensure_database(self) -> None:
        """Ensure database file exists and create tables if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.create_tables()
    
    def create_tables(self) -> None:
        """Create all required database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    current_epoch INTEGER,
                    total_epochs INTEGER NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    error_message TEXT,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    loss REAL NOT NULL,
                    accuracy REAL NOT NULL,
                    val_loss REAL,
                    val_accuracy REAL,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    UNIQUE(session_id, epoch)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_session 
                ON metrics(session_id)
            """)
            
            # Logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_session 
                ON logs(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
                ON logs(timestamp)
            """)
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    name TEXT PRIMARY KEY,
                    size_mb REAL NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            
            # Datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    name TEXT PRIMARY KEY,
                    size_mb REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    format TEXT NOT NULL,
                    dimensions TEXT,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)
            
            # Auth logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS auth_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key_hash TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    endpoint TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_auth_timestamp 
                ON auth_logs(timestamp)
            """)
            
            conn.commit()
    
    def drop_all_tables(self) -> None:
        """Drop all tables. WARNING: This deletes all data!"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            tables = [
                'auth_logs', 'logs', 'metrics', 'sessions', 
                'models', 'datasets'
            ]
            
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            
            conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Global database instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """
    Get the global database instance.
    
    Returns:
        Database instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def init_database(db_path: Optional[Path] = None) -> Database:
    """
    Initialize the global database instance.
    
    Args:
        db_path: Optional custom database path
        
    Returns:
        Database instance
    """
    global _db_instance
    _db_instance = Database(db_path)
    return _db_instance
