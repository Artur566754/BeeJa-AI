"""
Data models for the Server Management API.

This module defines Pydantic models for all API data structures including
training configurations, session information, metrics, logs, and resource metadata.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

from api.config import APIConfig


class SessionState(str, Enum):
    """Enumeration of possible training session states."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class LogLevel(str, Enum):
    """Enumeration of log levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class TrainingConfig(BaseModel):
    """
    Configuration for a training session.
    
    Validates all training parameters including learning rate ranges,
    positive batch sizes, and valid optimizer/loss function choices.
    """
    model_architecture: str = Field(
        ...,
        description="Model architecture name",
        min_length=1
    )
    dataset_name: str = Field(
        ...,
        description="Dataset name to use for training",
        min_length=1
    )
    learning_rate: float = Field(
        ...,
        description="Learning rate for optimizer",
        gt=0.0
    )
    batch_size: int = Field(
        ...,
        description="Batch size for training",
        gt=0
    )
    epochs: int = Field(
        ...,
        description="Number of training epochs",
        gt=0
    )
    optimizer: str = Field(
        default=APIConfig.DEFAULT_OPTIMIZER,
        description="Optimizer algorithm"
    )
    loss_function: str = Field(
        default=APIConfig.DEFAULT_LOSS_FUNCTION,
        description="Loss function to use"
    )
    
    @field_validator('learning_rate')
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate is within acceptable range."""
        if v < APIConfig.MIN_LEARNING_RATE:
            raise ValueError(
                f"Learning rate must be at least {APIConfig.MIN_LEARNING_RATE}, got {v}"
            )
        if v > APIConfig.MAX_LEARNING_RATE:
            raise ValueError(
                f"Learning rate must be at most {APIConfig.MAX_LEARNING_RATE}, got {v}"
            )
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is within acceptable range."""
        if v < APIConfig.MIN_BATCH_SIZE:
            raise ValueError(
                f"Batch size must be at least {APIConfig.MIN_BATCH_SIZE}, got {v}"
            )
        if v > APIConfig.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size must be at most {APIConfig.MAX_BATCH_SIZE}, got {v}"
            )
        return v
    
    @field_validator('epochs')
    @classmethod
    def validate_epochs(cls, v: int) -> int:
        """Validate epochs is within acceptable range."""
        if v < APIConfig.MIN_EPOCHS:
            raise ValueError(
                f"Epochs must be at least {APIConfig.MIN_EPOCHS}, got {v}"
            )
        if v > APIConfig.MAX_EPOCHS:
            raise ValueError(
                f"Epochs must be at most {APIConfig.MAX_EPOCHS}, got {v}"
            )
        return v
    
    @field_validator('optimizer')
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        """Validate optimizer is a supported option."""
        if v not in APIConfig.VALID_OPTIMIZERS:
            raise ValueError(
                f"Optimizer must be one of {APIConfig.VALID_OPTIMIZERS}, got '{v}'"
            )
        return v
    
    @field_validator('loss_function')
    @classmethod
    def validate_loss_function(cls, v: str) -> str:
        """Validate loss function is a supported option."""
        if v not in APIConfig.VALID_LOSS_FUNCTIONS:
            raise ValueError(
                f"Loss function must be one of {APIConfig.VALID_LOSS_FUNCTIONS}, got '{v}'"
            )
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "model_architecture": "resnet50",
                "dataset_name": "cifar10",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "optimizer": "adam",
                "loss_function": "cross_entropy"
            }
        }
    }


class SessionStatus(BaseModel):
    """
    Status information for a training session.
    
    Tracks the current state, progress, and timing of a training session.
    """
    session_id: str = Field(..., description="Unique session identifier")
    state: SessionState = Field(..., description="Current session state")
    current_epoch: Optional[int] = Field(
        None,
        description="Current epoch number (if running)",
        ge=0
    )
    total_epochs: int = Field(..., description="Total number of epochs", gt=0)
    start_time: Optional[datetime] = Field(
        None,
        description="Session start timestamp"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="Session end timestamp"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if session failed"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "sess_abc123",
                "state": "running",
                "current_epoch": 5,
                "total_epochs": 10,
                "start_time": "2024-01-15T10:30:00Z",
                "end_time": None,
                "error_message": None
            }
        }
    }


class SessionInfo(BaseModel):
    """
    Complete information about a training session.
    
    Combines session status with the original configuration.
    """
    session_id: str = Field(..., description="Unique session identifier")
    config: TrainingConfig = Field(..., description="Training configuration")
    status: SessionStatus = Field(..., description="Current session status")
    created_at: datetime = Field(..., description="Session creation timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "sess_abc123",
                "config": {
                    "model_architecture": "resnet50",
                    "dataset_name": "cifar10",
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 10
                },
                "status": {
                    "session_id": "sess_abc123",
                    "state": "running",
                    "current_epoch": 5,
                    "total_epochs": 10,
                    "start_time": "2024-01-15T10:30:00Z"
                },
                "created_at": "2024-01-15T10:25:00Z"
            }
        }
    }


class Metrics(BaseModel):
    """
    Training metrics for a specific epoch.
    
    Contains loss and accuracy values for both training and validation.
    """
    session_id: str = Field(..., description="Session identifier")
    epoch: int = Field(..., description="Epoch number", ge=0)
    loss: float = Field(..., description="Training loss")
    accuracy: float = Field(..., description="Training accuracy", ge=0.0, le=1.0)
    val_loss: Optional[float] = Field(None, description="Validation loss")
    val_accuracy: Optional[float] = Field(
        None,
        description="Validation accuracy",
        ge=0.0,
        le=1.0
    )
    timestamp: datetime = Field(..., description="Metrics timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "sess_abc123",
                "epoch": 5,
                "loss": 0.234,
                "accuracy": 0.892,
                "val_loss": 0.267,
                "val_accuracy": 0.875,
                "timestamp": "2024-01-15T10:35:00Z"
            }
        }
    }


class LogEntry(BaseModel):
    """
    A single log entry for a training session.
    
    Contains timestamped log messages with severity levels.
    """
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(..., description="Log entry timestamp")
    level: LogLevel = Field(..., description="Log severity level")
    message: str = Field(..., description="Log message", min_length=1)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "sess_abc123",
                "timestamp": "2024-01-15T10:35:00Z",
                "level": "INFO",
                "message": "Starting epoch 5/10"
            }
        }
    }


class SystemInfo(BaseModel):
    """
    System resource information.
    
    Contains CPU, memory, disk, and GPU usage statistics.
    """
    cpu_usage_percent: float = Field(
        ...,
        description="CPU usage percentage",
        ge=0.0,
        le=100.0
    )
    memory_used_mb: float = Field(
        ...,
        description="Memory used in MB",
        ge=0.0
    )
    memory_total_mb: float = Field(
        ...,
        description="Total memory in MB",
        gt=0.0
    )
    memory_percent: float = Field(
        ...,
        description="Memory usage percentage",
        ge=0.0,
        le=100.0
    )
    disk_free_gb: float = Field(
        ...,
        description="Free disk space in GB",
        ge=0.0
    )
    disk_total_gb: float = Field(
        ...,
        description="Total disk space in GB",
        gt=0.0
    )
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_usage_percent: Optional[float] = Field(
        None,
        description="GPU usage percentage",
        ge=0.0,
        le=100.0
    )
    gpu_memory_used_mb: Optional[float] = Field(
        None,
        description="GPU memory used in MB",
        ge=0.0
    )
    gpu_memory_total_mb: Optional[float] = Field(
        None,
        description="Total GPU memory in MB",
        gt=0.0
    )
    timestamp: datetime = Field(..., description="Measurement timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu_usage_percent": 45.2,
                "memory_used_mb": 8192,
                "memory_total_mb": 16384,
                "memory_percent": 50.0,
                "disk_free_gb": 250.5,
                "disk_total_gb": 500.0,
                "gpu_available": True,
                "gpu_usage_percent": 78.5,
                "gpu_memory_used_mb": 6144,
                "gpu_memory_total_mb": 8192,
                "timestamp": "2024-01-15T10:35:00Z"
            }
        }
    }


class ModelInfo(BaseModel):
    """
    Metadata for a trained model.
    
    Contains model name, size, and creation information.
    """
    name: str = Field(..., description="Model name", min_length=1)
    size_mb: float = Field(..., description="Model file size in MB", ge=0.0)
    created_at: datetime = Field(..., description="Model creation timestamp")
    file_path: str = Field(..., description="Path to model file", min_length=1)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "resnet50_cifar10",
                "size_mb": 102.4,
                "created_at": "2024-01-15T11:00:00Z",
                "file_path": "models/resnet50_cifar10_20240115.pth"
            }
        }
    }


class DatasetInfo(BaseModel):
    """
    Metadata for a dataset.
    
    Contains dataset name, size, format, and sample information.
    """
    name: str = Field(..., description="Dataset name", min_length=1)
    size_mb: float = Field(..., description="Dataset file size in MB", ge=0.0)
    sample_count: int = Field(..., description="Number of samples", ge=0)
    format: str = Field(..., description="Dataset format (csv, images, numpy, etc.)")
    dimensions: Optional[str] = Field(
        None,
        description="Data dimensions (e.g., '28x28x1' for MNIST)"
    )
    created_at: datetime = Field(..., description="Dataset creation timestamp")
    file_path: str = Field(..., description="Path to dataset file", min_length=1)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "cifar10",
                "size_mb": 163.0,
                "sample_count": 60000,
                "format": "images",
                "dimensions": "32x32x3",
                "created_at": "2024-01-10T09:00:00Z",
                "file_path": "datasets/cifar10.tar.gz"
            }
        }
    }
