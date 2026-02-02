"""
Model management API routes for the Server Management API.

This module implements endpoints for listing, downloading, uploading, and deleting models.
"""

from fastapi import APIRouter, HTTPException, Depends, Header, UploadFile, File
from fastapi.responses import FileResponse
from typing import Optional, List

from api.models.data_models import ModelInfo
from api.services.model_registry import get_model_registry
from api.services.auth_manager import get_auth_manager


router = APIRouter(prefix="/api/v1/models", tags=["models"])


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
    endpoint = "/api/v1/models/*"
    
    is_authenticated, error_message = auth_manager.authenticate_request(headers, endpoint)
    
    if not is_authenticated:
        raise HTTPException(status_code=401, detail=error_message)


@router.get("", response_model=List[ModelInfo], dependencies=[Depends(verify_api_key)])
def list_models():
    """
    List all models in the model registry.
    
    Returns:
        List of model information
    """
    try:
        model_registry = get_model_registry()
        models = model_registry.list_models()
        
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/{model_name}", dependencies=[Depends(verify_api_key)])
def download_model(model_name: str):
    """
    Download a model file.
    
    Args:
        model_name: Name of the model to download
        
    Returns:
        Model file
        
    Raises:
        HTTPException: If model not found
    """
    try:
        model_registry = get_model_registry()
        model_path = model_registry.get_model_path(model_name)
        
        if model_path is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return FileResponse(
            path=str(model_path),
            media_type="application/octet-stream",
            filename=model_path.name
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@router.post("", dependencies=[Depends(verify_api_key)])
async def upload_model(
    model_name: str,
    file: UploadFile = File(...)
):
    """
    Upload a model file.
    
    Args:
        model_name: Name for the model
        file: Model file to upload
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If upload fails or model already exists
    """
    try:
        model_registry = get_model_registry()
        
        # Check if model already exists
        if model_registry.model_exists(model_name):
            raise HTTPException(
                status_code=409,
                detail=f"Model '{model_name}' already exists"
            )
        
        # Read file data
        file_data = await file.read()
        
        # Save model
        model_registry.save_model(model_name, file_data)
        
        return {
            "message": f"Model '{model_name}' uploaded successfully",
            "name": model_name,
            "size_mb": len(file_data) / (1024 * 1024)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")


@router.delete("/{model_name}", dependencies=[Depends(verify_api_key)])
def delete_model(model_name: str):
    """
    Delete a model from the registry.
    
    Args:
        model_name: Name of the model to delete
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If model not found
    """
    try:
        model_registry = get_model_registry()
        
        success = model_registry.delete_model(model_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return {"message": f"Model '{model_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")
