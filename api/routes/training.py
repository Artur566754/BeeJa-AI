"""
Training API routes for the Server Management API.

This module implements all training-related endpoints including session management,
metrics retrieval, log access, and queue management.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional, List
from pydantic import BaseModel

from api.models.data_models import (
    TrainingConfig,
    SessionStatus,
    SessionInfo,
    Metrics,
    LogEntry
)
from api.services.training_session_manager import get_session_manager
from api.services.metrics_store import get_metrics_store
from api.services.log_store import get_log_store
from api.services.auth_manager import get_auth_manager


router = APIRouter(prefix="/api/v1/training", tags=["training"])


# Response models
class StartTrainingResponse(BaseModel):
    """Response for starting a training session."""
    session_id: str
    status: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: dict


# Authentication dependency
async def verify_api_key(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None)
):
    """
    Verify API key from request headers.
    
    Raises:
        HTTPException: If authentication fails
    """
    auth_manager = get_auth_manager()
    
    # Build headers dict
    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    if x_api_key:
        headers["X-API-Key"] = x_api_key
    
    # Get the endpoint (simplified for now)
    endpoint = "/api/v1/training/*"
    
    is_authenticated, error_message = auth_manager.authenticate_request(headers, endpoint)
    
    if not is_authenticated:
        raise HTTPException(status_code=401, detail=error_message)


@router.post("/start", response_model=StartTrainingResponse, dependencies=[Depends(verify_api_key)])
def start_training(config: TrainingConfig):
    """
    Start a new training session.
    
    Args:
        config: Training configuration
        
    Returns:
        Session ID and status
        
    Raises:
        HTTPException: If configuration is invalid or session creation fails
    """
    try:
        session_manager = get_session_manager()
        session_id = session_manager.create_session(config)
        
        # Try to start the session
        started = session_manager.start_session(session_id)
        
        if started:
            status = "running"
            message = "Training session created and started"
        else:
            status = "queued"
            message = "Training session created and queued"
        
        return StartTrainingResponse(
            session_id=session_id,
            status=status,
            message=message
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.post("/{session_id}/stop", dependencies=[Depends(verify_api_key)])
def stop_training(session_id: str):
    """
    Stop an active training session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If session not found or cannot be stopped
    """
    try:
        session_manager = get_session_manager()
        session_manager.stop_session(session_id)
        
        return {"message": f"Training session {session_id} stopped successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop training: {str(e)}")


@router.get("/{session_id}/status", response_model=SessionStatus, dependencies=[Depends(verify_api_key)])
def get_training_status(session_id: str):
    """
    Get the status of a training session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session status information
        
    Raises:
        HTTPException: If session not found
    """
    try:
        session_manager = get_session_manager()
        status = session_manager.get_status(session_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/{session_id}/metrics", response_model=Optional[Metrics], dependencies=[Depends(verify_api_key)])
def get_training_metrics(session_id: str):
    """
    Get the latest metrics for a training session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Latest metrics or None if no metrics available
        
    Raises:
        HTTPException: If session not found
    """
    try:
        # First verify session exists
        session_manager = get_session_manager()
        status = session_manager.get_status(session_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Get metrics
        metrics_store = get_metrics_store()
        metrics = metrics_store.get_latest_metrics(session_id)
        
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/{session_id}/history", response_model=List[Metrics], dependencies=[Depends(verify_api_key)])
def get_training_history(session_id: str):
    """
    Get the complete metrics history for a training session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of all metrics entries
        
    Raises:
        HTTPException: If session not found
    """
    try:
        # First verify session exists
        session_manager = get_session_manager()
        status = session_manager.get_status(session_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Get metrics history
        metrics_store = get_metrics_store()
        history = metrics_store.get_metrics_history(session_id)
        
        return history
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/{session_id}/logs", response_model=List[LogEntry], dependencies=[Depends(verify_api_key)])
def get_training_logs(session_id: str, limit: int = 100):
    """
    Get logs for a training session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of log entries to return (default: 100)
        
    Returns:
        List of log entries
        
    Raises:
        HTTPException: If session not found
    """
    try:
        # First verify session exists
        session_manager = get_session_manager()
        status = session_manager.get_status(session_id)
        
        if status is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Get logs
        log_store = get_log_store()
        logs = log_store.get_logs(session_id, limit=limit)
        
        return logs
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.get("/sessions", response_model=List[SessionInfo], dependencies=[Depends(verify_api_key)])
def list_sessions():
    """
    List all training sessions.
    
    Returns:
        List of all sessions with their status and configuration
    """
    try:
        session_manager = get_session_manager()
        sessions = session_manager.get_all_sessions()
        
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/queue", response_model=List[SessionInfo], dependencies=[Depends(verify_api_key)])
def get_queue_status():
    """
    Get the status of the training queue.
    
    Returns:
        List of queued sessions
    """
    try:
        session_manager = get_session_manager()
        queued_sessions = session_manager.get_queued_sessions()
        
        return queued_sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")


@router.delete("/queue/{session_id}", dependencies=[Depends(verify_api_key)])
def cancel_queued_session(session_id: str):
    """
    Cancel a queued training session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If session not found or not queued
    """
    try:
        session_manager = get_session_manager()
        session_manager.cancel_queued_session(session_id)
        
        return {"message": f"Queued session {session_id} cancelled successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel session: {str(e)}")
