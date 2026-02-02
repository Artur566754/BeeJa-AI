"""
Metrics Store service for the Server Management API.

This module provides storage and retrieval of training metrics using SQLite backend.
Metrics are stored per epoch for each training session with efficient indexed queries.
"""

from datetime import datetime
from typing import List, Optional
import sqlite3

from api.models import Metrics
from api.database import get_database


class MetricsStore:
    """
    Manages storage and retrieval of training metrics.
    
    Provides methods to save metrics per epoch and retrieve metrics history
    for training sessions. Uses SQLite with indexed queries for efficiency.
    """
    
    def __init__(self):
        """Initialize the metrics store with database connection."""
        self.db = get_database()
    
    def save_metrics(self, session_id: str, epoch: int, metrics: Metrics) -> None:
        """
        Save metrics for a specific epoch of a training session.
        
        Args:
            session_id: Unique identifier for the training session
            epoch: Epoch number (0-indexed)
            metrics: Metrics object containing loss, accuracy, etc.
            
        Raises:
            sqlite3.IntegrityError: If metrics for this session/epoch already exist
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO metrics (
                    session_id, epoch, loss, accuracy, 
                    val_loss, val_accuracy, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                epoch,
                metrics.loss,
                metrics.accuracy,
                metrics.val_loss,
                metrics.val_accuracy,
                metrics.timestamp.isoformat()
            ))
            
            conn.commit()
    
    def get_latest_metrics(self, session_id: str) -> Optional[Metrics]:
        """
        Get the most recent metrics for a training session.
        
        Returns the metrics entry with the highest epoch number.
        
        Args:
            session_id: Unique identifier for the training session
            
        Returns:
            Metrics object for the latest epoch, or None if no metrics exist
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id, epoch, loss, accuracy, 
                       val_loss, val_accuracy, timestamp
                FROM metrics
                WHERE session_id = ?
                ORDER BY epoch DESC
                LIMIT 1
            """, (session_id,))
            
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return Metrics(
                session_id=row[0],
                epoch=row[1],
                loss=row[2],
                accuracy=row[3],
                val_loss=row[4],
                val_accuracy=row[5],
                timestamp=datetime.fromisoformat(row[6])
            )
    
    def get_metrics_history(self, session_id: str) -> List[Metrics]:
        """
        Get all metrics for a training session in chronological order.
        
        Returns all metric entries ordered by epoch number.
        
        Args:
            session_id: Unique identifier for the training session
            
        Returns:
            List of Metrics objects ordered by epoch (earliest to latest)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id, epoch, loss, accuracy, 
                       val_loss, val_accuracy, timestamp
                FROM metrics
                WHERE session_id = ?
                ORDER BY epoch ASC
            """, (session_id,))
            
            rows = cursor.fetchall()
            
            return [
                Metrics(
                    session_id=row[0],
                    epoch=row[1],
                    loss=row[2],
                    accuracy=row[3],
                    val_loss=row[4],
                    val_accuracy=row[5],
                    timestamp=datetime.fromisoformat(row[6])
                )
                for row in rows
            ]
    
    def get_metrics_by_epoch(self, session_id: str, epoch: int) -> Optional[Metrics]:
        """
        Get metrics for a specific epoch of a training session.
        
        Args:
            session_id: Unique identifier for the training session
            epoch: Epoch number to retrieve
            
        Returns:
            Metrics object for the specified epoch, or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_id, epoch, loss, accuracy, 
                       val_loss, val_accuracy, timestamp
                FROM metrics
                WHERE session_id = ? AND epoch = ?
            """, (session_id, epoch))
            
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return Metrics(
                session_id=row[0],
                epoch=row[1],
                loss=row[2],
                accuracy=row[3],
                val_loss=row[4],
                val_accuracy=row[5],
                timestamp=datetime.fromisoformat(row[6])
            )
    
    def delete_metrics(self, session_id: str) -> int:
        """
        Delete all metrics for a training session.
        
        Useful for cleanup when a session is deleted.
        
        Args:
            session_id: Unique identifier for the training session
            
        Returns:
            Number of metric entries deleted
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM metrics
                WHERE session_id = ?
            """, (session_id,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            return deleted_count
    
    def get_metrics_count(self, session_id: str) -> int:
        """
        Get the total number of metric entries for a session.
        
        Args:
            session_id: Unique identifier for the training session
            
        Returns:
            Number of metric entries
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM metrics
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            return row[0] if row else 0
