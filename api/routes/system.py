"""
System API routes for the Server Management API.

This module implements system monitoring endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional

from api.models.data_models import SystemInfo
from api.services.system_monitor import get_system_monitor
from api.services.auth_manager import get_auth_manager


router = APIRouter(prefix="/api/v1/system", tags=["system"])


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
    
    # Get the endpoint
    endpoint = "/api/v1/system/*"
    
    is_authenticated, error_message = auth_manager.authenticate_request(headers, endpoint)
    
    if not is_authenticated:
        raise HTTPException(status_code=401, detail=error_message)


@router.get("/info", response_model=SystemInfo, dependencies=[Depends(verify_api_key)])
def get_system_info():
    """
    Get current system resource information.
    
    Returns:
        System resource information including CPU, memory, disk, and GPU usage
    """
    try:
        system_monitor = get_system_monitor()
        info = system_monitor.get_system_info()
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")
