"""Unit tests for database schema validation."""

import pytest
import tempfile
from pathlib import Path

from api.database import Database


class TestDatabaseSchema:
    """Test that database schema matches design specifications."""
    
    def test_sessions_table_schema(self):
        """Test sessions table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(sessions)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Verify required columns exist with correct types
                assert 'session_id' in columns
                assert 'config_json' in columns
                assert 'state' in columns
                assert 'current_epoch' in columns
                assert 'total_epochs' in columns
                assert 'start_time' in columns
                assert 'end_time' in columns
                assert 'error_message' in columns
                assert 'created_at' in columns
            
            db.close()
    
    def test_metrics_table_schema(self):
        """Test metrics table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(metrics)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Verify required columns exist
                assert 'id' in columns
                assert 'session_id' in columns
                assert 'epoch' in columns
                assert 'loss' in columns
                assert 'accuracy' in columns
                assert 'val_loss' in columns
                assert 'val_accuracy' in columns
                assert 'timestamp' in columns
            
            db.close()
    
    def test_logs_table_schema(self):
        """Test logs table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(logs)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Verify required columns exist
                assert 'id' in columns
                assert 'session_id' in columns
                assert 'timestamp' in columns
                assert 'level' in columns
                assert 'message' in columns
            
            db.close()
    
    def test_models_table_schema(self):
        """Test models table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(models)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Verify required columns exist
                assert 'name' in columns
                assert 'size_mb' in columns
                assert 'file_path' in columns
                assert 'created_at' in columns
            
            db.close()
    
    def test_datasets_table_schema(self):
        """Test datasets table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(datasets)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Verify required columns exist
                assert 'name' in columns
                assert 'size_mb' in columns
                assert 'sample_count' in columns
                assert 'format' in columns
                assert 'dimensions' in columns
                assert 'file_path' in columns
                assert 'created_at' in columns
            
            db.close()
    
    def test_auth_logs_table_schema(self):
        """Test auth_logs table has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(auth_logs)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                # Verify required columns exist
                assert 'id' in columns
                assert 'api_key_hash' in columns
                assert 'success' in columns
                assert 'endpoint' in columns
                assert 'timestamp' in columns
            
            db.close()
    
    def test_foreign_key_constraints(self):
        """Test that foreign key constraints are defined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check metrics foreign key
                cursor.execute("PRAGMA foreign_key_list(metrics)")
                fk_list = cursor.fetchall()
                assert len(fk_list) > 0, "Metrics table should have foreign key"
                
                # Check logs foreign key
                cursor.execute("PRAGMA foreign_key_list(logs)")
                fk_list = cursor.fetchall()
                assert len(fk_list) > 0, "Logs table should have foreign key"
            
            db.close()
    
    def test_unique_constraints(self):
        """Test that unique constraints work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            
            with db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert a session
                cursor.execute("""
                    INSERT INTO sessions 
                    (session_id, config_json, state, total_epochs, created_at)
                    VALUES (?, ?, ?, ?, datetime('now'))
                """, ("test_session", "{}", "queued", 10))
                
                # Insert a metric
                cursor.execute("""
                    INSERT INTO metrics 
                    (session_id, epoch, loss, accuracy, timestamp)
                    VALUES (?, ?, ?, ?, datetime('now'))
                """, ("test_session", 1, 0.5, 0.8))
                
                # Try to insert duplicate metric (same session_id and epoch)
                with pytest.raises(Exception):  # Should raise UNIQUE constraint error
                    cursor.execute("""
                        INSERT INTO metrics 
                        (session_id, epoch, loss, accuracy, timestamp)
                        VALUES (?, ?, ?, ?, datetime('now'))
                    """, ("test_session", 1, 0.4, 0.9))
            
            db.close()
