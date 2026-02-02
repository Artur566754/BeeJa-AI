"""
Dataset management API routes for the Server Management API.

This module implements endpoints for listing, getting info, and uploading datasets.
"""

from fastapi import APIRouter, HTTPException, Depends, Header, UploadFile, File
from typing import Optional, List

from api.models.data_models import DatasetInfo
from api.services.dataset_registry import get_dataset_registry
from api.services.auth_manager import get_auth_manager


router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


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
    endpoint = "/api/v1/datasets/*"
    
    is_authenticated, error_message = auth_manager.authenticate_request(headers, endpoint)
    
    if not is_authenticated:
        raise HTTPException(status_code=401, detail=error_message)


@router.get("", response_model=List[DatasetInfo], dependencies=[Depends(verify_api_key)])
def list_datasets():
    """
    List all datasets in the dataset registry.
    
    Returns:
        List of dataset information
    """
    try:
        dataset_registry = get_dataset_registry()
        datasets = dataset_registry.list_datasets()
        
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/{dataset_name}", response_model=DatasetInfo, dependencies=[Depends(verify_api_key)])
def get_dataset_info(dataset_name: str):
    """
    Get information about a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dataset information
        
    Raises:
        HTTPException: If dataset not found
    """
    try:
        dataset_registry = get_dataset_registry()
        dataset_info = dataset_registry.get_dataset_info(dataset_name)
        
        if dataset_info is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
        
        return dataset_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")


@router.post("", dependencies=[Depends(verify_api_key)])
async def upload_dataset(
    dataset_name: str,
    file: UploadFile = File(...)
):
    """
    Upload a dataset file.
    
    Args:
        dataset_name: Name for the dataset
        file: Dataset file to upload
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If upload fails or dataset already exists
    """
    try:
        dataset_registry = get_dataset_registry()
        
        # Check if dataset already exists
        if dataset_registry.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=409,
                detail=f"Dataset '{dataset_name}' already exists"
            )
        
        # Read file data
        file_data = await file.read()
        
        # Save dataset
        dataset_registry.save_dataset(dataset_name, file_data)
        
        return {
            "message": f"Dataset '{dataset_name}' uploaded successfully",
            "name": dataset_name,
            "size_mb": len(file_data) / (1024 * 1024)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")
