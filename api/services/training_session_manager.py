"""
Training Session Manager for the Server Management API.

This module manages the lifecycle of training sessions including creation,
execution, queueing, and state management.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict
from pathlib import Path
import json
import uuid
import threading
import asyncio
from collections import deque

from api.database import Database, get_database
from api.models.data_models import (
    TrainingConfig, SessionStatus, SessionInfo, SessionState, Metrics
)
from api.services.training_executor import TrainingExecutor, get_training_executor
from api.services.metrics_store import MetricsStore
from api.services.log_store import LogStore
from api.config import APIConfig


class TrainingSessionManager:
    """
    Manages training session lifecycle, queueing, and execution.
    
    Coordinates between the training executor, metrics store, and log store
    to provide a complete training session management system.
    """
    
    def __init__(
        self,
        database: Optional[Database] = None,
        executor: Optional[TrainingExecutor] = None,
        metrics_store: Optional[MetricsStore] = None,
        log_store: Optional[LogStore] = None
    ):
        """
        Initialize the TrainingSessionManager.
        
        Args:
            database: Database instance
            executor: Training executor instance
            metrics_store: Metrics store instance
            log_store: Log store instance
        """
        self.db = database or get_database()
        self.executor = executor or get_training_executor()
        self.metrics_store = metrics_store or MetricsStore()
        self.log_store = log_store or LogStore(self.db)
        
        # Queue for pending sessions
        self._queue: deque = deque()
        self._queue_lock = threading.Lock()
        
        # Active sessions tracking
        self._active_sessions: Dict[str, threading.Thread] = {}
        self._active_lock = threading.Lock()
    
    def create_session(self, config: TrainingConfig) -> str:
        """
        Create a new training session.
        
        Args:
            config: Training configuration
            
        Returns:
            Session ID
        """
        # Generate unique session ID
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        # Create session in database
        created_at = datetime.now(timezone.utc)
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions 
                (session_id, config_json, state, current_epoch, total_epochs, 
                 start_time, end_time, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    json.dumps(config.model_dump()),
                    SessionState.QUEUED.value,
                    None,
                    config.epochs,
                    None,
                    None,
                    None,
                    created_at.isoformat()
                )
            )
        
        # Log session creation
        self.log_store.append_log(
            session_id,
            f"Session created with config: {config.model_architecture}, "
            f"dataset: {config.dataset_name}, epochs: {config.epochs}",
            "INFO"
        )
        
        # Add to queue or start immediately
        if self.can_start_new_session():
            self._start_session_async(session_id, config)
        else:
            with self._queue_lock:
                self._queue.append((session_id, config))
            self.log_store.append_log(
                session_id,
                "Session queued - waiting for available resources",
                "INFO"
            )
        
        return session_id
    
    def start_session(self, session_id: str) -> None:
        """
        Start a queued training session.
        
        Args:
            session_id: Session identifier
            
        Raises:
            ValueError: If session doesn't exist or is not queued
        """
        status = self.get_status(session_id)
        if status is None:
            raise ValueError(f"Session {session_id} not found")
        
        if status.state != SessionState.QUEUED:
            raise ValueError(f"Session {session_id} is not queued (state: {status.state})")
        
        # Get config from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT config_json FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Session {session_id} not found")
            
            config_dict = json.loads(row['config_json'])
            config = TrainingConfig(**config_dict)
        
        self._start_session_async(session_id, config)
    
    def _start_session_async(self, session_id: str, config: TrainingConfig) -> None:
        """Start a training session in a background thread."""
        # Update state to running
        self._update_session_state(session_id, SessionState.RUNNING)
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._run_training,
            args=(session_id, config),
            daemon=True
        )
        
        with self._active_lock:
            self._active_sessions[session_id] = training_thread
        
        training_thread.start()
    
    def _run_training(self, session_id: str, config: TrainingConfig) -> None:
        """Run training session (called in background thread)."""
        try:
            # Update start time
            start_time = datetime.now(timezone.utc)
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE sessions SET start_time = ? WHERE session_id = ?",
                    (start_time.isoformat(), session_id)
                )
            
            # Define callbacks
            def metrics_callback(metrics: Metrics):
                self.metrics_store.save_metrics(
                    session_id,
                    metrics.epoch,
                    metrics
                )
                # Update current epoch
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE sessions SET current_epoch = ? WHERE session_id = ?",
                        (metrics.epoch, session_id)
                    )
            
            def log_callback(message: str, level: str):
                self.log_store.append_log(session_id, message, level)
            
            # Execute training
            self.executor.execute_training(
                session_id,
                config,
                metrics_callback,
                log_callback
            )
            
            # Training completed successfully
            self._update_session_state(
                session_id,
                SessionState.COMPLETED,
                end_time=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            # Training failed
            error_msg = str(e)
            self._update_session_state(
                session_id,
                SessionState.FAILED,
                end_time=datetime.now(timezone.utc),
                error_message=error_msg
            )
            self.log_store.append_log(
                session_id,
                f"Training failed: {error_msg}",
                "ERROR"
            )
        
        finally:
            # Remove from active sessions
            with self._active_lock:
                if session_id in self._active_sessions:
                    del self._active_sessions[session_id]
            
            # Start next queued session if available
            self._start_next_queued_session()
    
    def stop_session(self, session_id: str) -> None:
        """
        Stop an active training session.
        
        Args:
            session_id: Session identifier
            
        Raises:
            ValueError: If session is not running
        """
        status = self.get_status(session_id)
        if status is None:
            raise ValueError(f"Session {session_id} not found")
        
        if status.state != SessionState.RUNNING:
            raise ValueError(f"Session {session_id} is not running (state: {status.state})")
        
        # Request stop
        stopped = self.executor.stop_training(session_id)
        
        if stopped:
            self._update_session_state(
                session_id,
                SessionState.STOPPED,
                end_time=datetime.now(timezone.utc)
            )
            self.log_store.append_log(
                session_id,
                "Training stopped by user request",
                "WARNING"
            )
    
    def get_status(self, session_id: str) -> Optional[SessionStatus]:
        """
        Get the status of a training session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionStatus object, or None if session doesn't exist
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, state, current_epoch, total_epochs,
                       start_time, end_time, error_message
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return SessionStatus(
                session_id=row['session_id'],
                state=SessionState(row['state']),
                current_epoch=row['current_epoch'],
                total_epochs=row['total_epochs'],
                start_time=datetime.fromisoformat(row['start_time']) if row['start_time'] else None,
                end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                error_message=row['error_message']
            )
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """
        Get all active (running) training sessions.
        
        Returns:
            List of SessionInfo objects for running sessions
        """
        return self._get_sessions_by_state(SessionState.RUNNING)
    
    def get_queued_sessions(self) -> List[SessionInfo]:
        """
        Get all queued training sessions.
        
        Returns:
            List of SessionInfo objects for queued sessions
        """
        return self._get_sessions_by_state(SessionState.QUEUED)
    
    def get_all_sessions(self) -> List[SessionInfo]:
        """
        Get all training sessions regardless of state.
        
        Returns:
            List of all sessions
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, config_json, state, current_epoch, total_epochs,
                       start_time, end_time, error_message, created_at
                FROM sessions
                ORDER BY created_at DESC
                """
            )
            
            rows = cursor.fetchall()
            sessions = []
            
            for row in rows:
                config = TrainingConfig(**json.loads(row['config_json']))
                status = SessionStatus(
                    session_id=row['session_id'],
                    state=SessionState(row['state']),
                    current_epoch=row['current_epoch'],
                    total_epochs=row['total_epochs'],
                    start_time=datetime.fromisoformat(row['start_time']) if row['start_time'] else None,
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                    error_message=row['error_message']
                )
                sessions.append(SessionInfo(
                    session_id=row['session_id'],
                    config=config,
                    status=status,
                    created_at=datetime.fromisoformat(row['created_at'])
                ))
            
            return sessions
    
    def cancel_queued_session(self, session_id: str) -> None:
        """
        Cancel a queued training session.
        
        Args:
            session_id: Session identifier
            
        Raises:
            ValueError: If session is not queued
        """
        status = self.get_status(session_id)
        if status is None:
            raise ValueError(f"Session {session_id} not found")
        
        if status.state != SessionState.QUEUED:
            raise ValueError(f"Session {session_id} is not queued (state: {status.state})")
        
        # Remove from queue
        with self._queue_lock:
            self._queue = deque([
                (sid, cfg) for sid, cfg in self._queue if sid != session_id
            ])
        
        # Update state to stopped
        self._update_session_state(
            session_id,
            SessionState.STOPPED,
            end_time=datetime.now(timezone.utc)
        )
        
        self.log_store.append_log(
            session_id,
            "Queued session cancelled by user",
            "WARNING"
        )
    
    def can_start_new_session(self) -> bool:
        """
        Check if a new training session can be started.
        
        Returns:
            True if resources are available, False otherwise
        """
        with self._active_lock:
            active_count = len(self._active_sessions)
        
        return active_count < APIConfig.MAX_CONCURRENT_SESSIONS
    
    def _get_sessions_by_state(self, state: SessionState) -> List[SessionInfo]:
        """Get all sessions with a specific state."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, config_json, state, current_epoch, total_epochs,
                       start_time, end_time, error_message, created_at
                FROM sessions
                WHERE state = ?
                ORDER BY created_at DESC
                """,
                (state.value,)
            )
            
            rows = cursor.fetchall()
            
            sessions = []
            for row in rows:
                config_dict = json.loads(row['config_json'])
                config = TrainingConfig(**config_dict)
                
                status = SessionStatus(
                    session_id=row['session_id'],
                    state=SessionState(row['state']),
                    current_epoch=row['current_epoch'],
                    total_epochs=row['total_epochs'],
                    start_time=datetime.fromisoformat(row['start_time']) if row['start_time'] else None,
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                    error_message=row['error_message']
                )
                
                sessions.append(SessionInfo(
                    session_id=row['session_id'],
                    config=config,
                    status=status,
                    created_at=datetime.fromisoformat(row['created_at'])
                ))
            
            return sessions
    
    def _update_session_state(
        self,
        session_id: str,
        state: SessionState,
        end_time: Optional[datetime] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Update the state of a session in the database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            if end_time:
                cursor.execute(
                    """
                    UPDATE sessions 
                    SET state = ?, end_time = ?, error_message = ?
                    WHERE session_id = ?
                    """,
                    (state.value, end_time.isoformat(), error_message, session_id)
                )
            else:
                cursor.execute(
                    "UPDATE sessions SET state = ? WHERE session_id = ?",
                    (state.value, session_id)
                )
    
    def _start_next_queued_session(self) -> None:
        """Start the next queued session if resources are available."""
        if not self.can_start_new_session():
            return
        
        with self._queue_lock:
            if not self._queue:
                return
            
            session_id, config = self._queue.popleft()
        
        self.log_store.append_log(
            session_id,
            "Starting queued session - resources now available",
            "INFO"
        )
        
        self._start_session_async(session_id, config)


# Global session manager instance
_manager_instance: Optional[TrainingSessionManager] = None


def get_session_manager() -> TrainingSessionManager:
    """
    Get the global training session manager instance.
    
    Returns:
        TrainingSessionManager instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = TrainingSessionManager()
    return _manager_instance
