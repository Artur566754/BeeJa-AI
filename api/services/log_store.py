"""
Log Store service for the Server Management API.

This module provides storage and retrieval of training session logs using SQLite.
All log entries include timestamps and severity levels.
"""

from datetime import datetime, timezone
from typing import List, Optional
import sqlite3

from api.database import Database, get_database
from api.models.data_models import LogEntry, LogLevel


class LogStore:
    """
    Manages storage and retrieval of training session logs.
    
    Provides methods to append logs in real-time and retrieve logs
    for specific sessions with filtering options.
    """
    
    def __init__(self, database: Optional[Database] = None):
        """
        Initialize the LogStore.
        
        Args:
            database: Database instance. If None, uses global database.
        """
        self.db = database or get_database()
    
    def append_log(
        self,
        session_id: str,
        message: str,
        level: str = "INFO"
    ) -> None:
        """
        Append a log entry for a training session.
        
        Args:
            session_id: Session identifier
            message: Log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
            
        Raises:
            ValueError: If level is not a valid LogLevel
        """
        # Validate log level
        try:
            log_level = LogLevel(level)
        except ValueError:
            raise ValueError(
                f"Invalid log level '{level}'. Must be one of: "
                f"{', '.join([l.value for l in LogLevel])}"
            )
        
        timestamp = datetime.now(timezone.utc)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO logs (session_id, timestamp, level, message)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, timestamp.isoformat(), log_level.value, message)
            )
    
    def get_logs(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[LogEntry]:
        """
        Get log entries for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of log entries to return (default: 100)
            
        Returns:
            List of LogEntry objects, ordered by timestamp (newest first)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, timestamp, level, message
                FROM logs
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit)
            )
            
            rows = cursor.fetchall()
            
            # Convert rows to LogEntry objects
            logs = []
            for row in rows:
                logs.append(LogEntry(
                    session_id=row['session_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    level=LogLevel(row['level']),
                    message=row['message']
                ))
            
            # Return in chronological order (oldest first)
            return list(reversed(logs))
    
    def get_logs_since(
        self,
        session_id: str,
        timestamp: datetime
    ) -> List[LogEntry]:
        """
        Get log entries for a session since a specific timestamp.
        
        Args:
            session_id: Session identifier
            timestamp: Only return logs after this timestamp
            
        Returns:
            List of LogEntry objects, ordered by timestamp (oldest first)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, timestamp, level, message
                FROM logs
                WHERE session_id = ? AND timestamp > ?
                ORDER BY timestamp ASC
                """,
                (session_id, timestamp.isoformat())
            )
            
            rows = cursor.fetchall()
            
            # Convert rows to LogEntry objects
            logs = []
            for row in rows:
                logs.append(LogEntry(
                    session_id=row['session_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    level=LogLevel(row['level']),
                    message=row['message']
                ))
            
            return logs
    
    def get_all_logs(self, session_id: str) -> List[LogEntry]:
        """
        Get all log entries for a session without limit.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of all LogEntry objects, ordered by timestamp (oldest first)
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, timestamp, level, message
                FROM logs
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,)
            )
            
            rows = cursor.fetchall()
            
            # Convert rows to LogEntry objects
            logs = []
            for row in rows:
                logs.append(LogEntry(
                    session_id=row['session_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    level=LogLevel(row['level']),
                    message=row['message']
                ))
            
            return logs
    
    def delete_logs(self, session_id: str) -> int:
        """
        Delete all log entries for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of log entries deleted
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM logs WHERE session_id = ?",
                (session_id,)
            )
            return cursor.rowcount
