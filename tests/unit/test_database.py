"""Unit tests for database management."""

import pytest
import tempfile
from pathlib import Path

from api.database import Database, init_database


class TestDatabase:
    """Test database initialization and table creation."""
    
    def test_database_creation(self):
        """Test that database file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            assert db_path.exists()
            db.close()
    
    def test_tables_created(self):
        """Test that all required tables are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check that all tables exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' 
                    ORDER BY name
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                expected_tables = [
                    'auth_logs', 'datasets', 'logs', 
                    'metrics', 'models', 'sessions'
                ]
                
                for table in expected_tables:
                    assert table in tables, f"Table {table} not found"
            
            db.close()
    
    def test_indexes_created(self):
        """Test that indexes are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check that indexes exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' 
                    ORDER BY name
                """)
                indexes = [row[0] for row in cursor.fetchall()]
                
                expected_indexes = [
                    'idx_metrics_session',
                    'idx_logs_session',
                    'idx_logs_timestamp',
                    'idx_auth_timestamp'
                ]
                
                for index in expected_indexes:
                    assert index in indexes, f"Index {index} not found"
            
            db.close()
    
    def test_connection_context_manager(self):
        """Test that connection context manager works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            # Test successful transaction
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions 
                    (session_id, config_json, state, total_epochs, created_at)
                    VALUES (?, ?, ?, ?, datetime('now'))
                """, ("test_session", "{}", "queued", 10))
            
            # Verify data was committed
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT session_id FROM sessions")
                result = cursor.fetchone()
                assert result is not None
                assert result[0] == "test_session"
            
            db.close()
    
    def test_connection_rollback_on_error(self):
        """Test that connection rolls back on error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            # Test failed transaction
            try:
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO sessions 
                        (session_id, config_json, state, total_epochs, created_at)
                        VALUES (?, ?, ?, ?, datetime('now'))
                    """, ("test_session", "{}", "queued", 10))
                    
                    # Force an error
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            # Verify data was rolled back
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sessions")
                count = cursor.fetchone()[0]
                assert count == 0
            
            db.close()
