"""
Unit tests for API data models.

Tests specific examples and edge cases for data model validation.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from api.models import (
    TrainingConfig,
    SessionStatus,
    SessionState,
    SessionInfo,
    Metrics,
    LogEntry,
    LogLevel,
    SystemInfo,
    ModelInfo,
    DatasetInfo,
)
from api.config import APIConfig


class TestTrainingConfig:
    """Unit tests for TrainingConfig model."""
    
    def test_valid_config_with_all_fields(self):
        """Test creating a valid config with all fields specified."""
        config = TrainingConfig(
            model_architecture="resnet50",
            dataset_name="cifar10",
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            optimizer="adam",
            loss_function="cross_entropy"
        )
        
        assert config.model_architecture == "resnet50"
        assert config.dataset_name == "cifar10"
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
        assert config.epochs == 10
        assert config.optimizer == "adam"
        assert config.loss_function == "cross_entropy"
    
    def test_valid_config_with_defaults(self):
        """Test creating a valid config using default values."""
        config = TrainingConfig(
            model_architecture="vgg16",
            dataset_name="mnist",
            learning_rate=0.01,
            batch_size=64,
            epochs=5
        )
        
        assert config.optimizer == APIConfig.DEFAULT_OPTIMIZER
        assert config.loss_function == APIConfig.DEFAULT_LOSS_FUNCTION
    
    def test_learning_rate_too_low(self):
        """Test that learning rate below minimum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=0.00001,  # Below MIN_LEARNING_RATE (0.0001)
                batch_size=32,
                epochs=10
            )
        
        assert "learning" in str(exc_info.value).lower()
    
    def test_learning_rate_too_high(self):
        """Test that learning rate above maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=5.0,  # Above MAX_LEARNING_RATE (1.0)
                batch_size=32,
                epochs=10
            )
        
        assert "learning" in str(exc_info.value).lower()
    
    def test_negative_learning_rate(self):
        """Test that negative learning rate is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=-0.001,
                batch_size=32,
                epochs=10
            )
        
        assert "learning" in str(exc_info.value).lower()
    
    def test_zero_batch_size(self):
        """Test that zero batch size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=0.001,
                batch_size=0,
                epochs=10
            )
        
        assert "batch" in str(exc_info.value).lower()
    
    def test_negative_batch_size(self):
        """Test that negative batch size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=0.001,
                batch_size=-32,
                epochs=10
            )
        
        assert "batch" in str(exc_info.value).lower()
    
    def test_batch_size_too_large(self):
        """Test that batch size above maximum is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=0.001,
                batch_size=1000,  # Above MAX_BATCH_SIZE (256)
                epochs=10
            )
        
        assert "batch" in str(exc_info.value).lower()
    
    def test_zero_epochs(self):
        """Test that zero epochs is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=0.001,
                batch_size=32,
                epochs=0
            )
        
        assert "epoch" in str(exc_info.value).lower()
    
    def test_negative_epochs(self):
        """Test that negative epochs is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=0.001,
                batch_size=32,
                epochs=-10
            )
        
        assert "epoch" in str(exc_info.value).lower()
    
    def test_invalid_optimizer(self):
        """Test that invalid optimizer name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=0.001,
                batch_size=32,
                epochs=10,
                optimizer="invalid_optimizer"
            )
        
        assert "optimizer" in str(exc_info.value).lower()
    
    def test_invalid_loss_function(self):
        """Test that invalid loss function name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="cifar10",
                learning_rate=0.001,
                batch_size=32,
                epochs=10,
                loss_function="invalid_loss"
            )
        
        assert "loss" in str(exc_info.value).lower()
    
    def test_empty_model_architecture(self):
        """Test that empty model architecture is rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                model_architecture="",
                dataset_name="cifar10",
                learning_rate=0.001,
                batch_size=32,
                epochs=10
            )
    
    def test_empty_dataset_name(self):
        """Test that empty dataset name is rejected."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                model_architecture="resnet50",
                dataset_name="",
                learning_rate=0.001,
                batch_size=32,
                epochs=10
            )
    
    def test_boundary_values(self):
        """Test boundary values for all numeric fields."""
        # Minimum valid values
        config_min = TrainingConfig(
            model_architecture="model",
            dataset_name="dataset",
            learning_rate=APIConfig.MIN_LEARNING_RATE,
            batch_size=APIConfig.MIN_BATCH_SIZE,
            epochs=APIConfig.MIN_EPOCHS
        )
        assert config_min.learning_rate == APIConfig.MIN_LEARNING_RATE
        assert config_min.batch_size == APIConfig.MIN_BATCH_SIZE
        assert config_min.epochs == APIConfig.MIN_EPOCHS
        
        # Maximum valid values
        config_max = TrainingConfig(
            model_architecture="model",
            dataset_name="dataset",
            learning_rate=APIConfig.MAX_LEARNING_RATE,
            batch_size=APIConfig.MAX_BATCH_SIZE,
            epochs=APIConfig.MAX_EPOCHS
        )
        assert config_max.learning_rate == APIConfig.MAX_LEARNING_RATE
        assert config_max.batch_size == APIConfig.MAX_BATCH_SIZE
        assert config_max.epochs == APIConfig.MAX_EPOCHS


class TestSessionStatus:
    """Unit tests for SessionStatus model."""
    
    def test_valid_running_session(self):
        """Test creating a running session status."""
        status = SessionStatus(
            session_id="sess_123",
            state=SessionState.RUNNING,
            current_epoch=5,
            total_epochs=10,
            start_time=datetime.now()
        )
        
        assert status.session_id == "sess_123"
        assert status.state == SessionState.RUNNING
        assert status.current_epoch == 5
        assert status.total_epochs == 10
        assert status.start_time is not None
        assert status.end_time is None
        assert status.error_message is None
    
    def test_valid_failed_session(self):
        """Test creating a failed session status."""
        status = SessionStatus(
            session_id="sess_456",
            state=SessionState.FAILED,
            current_epoch=3,
            total_epochs=10,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error_message="Out of memory"
        )
        
        assert status.state == SessionState.FAILED
        assert status.error_message == "Out of memory"
    
    def test_valid_queued_session(self):
        """Test creating a queued session status."""
        status = SessionStatus(
            session_id="sess_789",
            state=SessionState.QUEUED,
            total_epochs=20
        )
        
        assert status.state == SessionState.QUEUED
        assert status.current_epoch is None
        assert status.start_time is None


class TestMetrics:
    """Unit tests for Metrics model."""
    
    def test_valid_metrics_with_validation(self):
        """Test creating metrics with validation data."""
        metrics = Metrics(
            session_id="sess_123",
            epoch=5,
            loss=0.234,
            accuracy=0.892,
            val_loss=0.267,
            val_accuracy=0.875,
            timestamp=datetime.now()
        )
        
        assert metrics.session_id == "sess_123"
        assert metrics.epoch == 5
        assert metrics.loss == 0.234
        assert metrics.accuracy == 0.892
        assert metrics.val_loss == 0.267
        assert metrics.val_accuracy == 0.875
    
    def test_valid_metrics_without_validation(self):
        """Test creating metrics without validation data."""
        metrics = Metrics(
            session_id="sess_123",
            epoch=5,
            loss=0.234,
            accuracy=0.892,
            timestamp=datetime.now()
        )
        
        assert metrics.val_loss is None
        assert metrics.val_accuracy is None
    
    def test_accuracy_out_of_range(self):
        """Test that accuracy outside [0, 1] is rejected."""
        with pytest.raises(ValidationError):
            Metrics(
                session_id="sess_123",
                epoch=5,
                loss=0.234,
                accuracy=1.5,  # Above 1.0
                timestamp=datetime.now()
            )


class TestLogEntry:
    """Unit tests for LogEntry model."""
    
    def test_valid_log_entry(self):
        """Test creating a valid log entry."""
        log = LogEntry(
            session_id="sess_123",
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Training started"
        )
        
        assert log.session_id == "sess_123"
        assert log.level == LogLevel.INFO
        assert log.message == "Training started"
    
    def test_all_log_levels(self):
        """Test all log levels are valid."""
        for level in LogLevel:
            log = LogEntry(
                session_id="sess_123",
                timestamp=datetime.now(),
                level=level,
                message="Test message"
            )
            assert log.level == level


class TestSystemInfo:
    """Unit tests for SystemInfo model."""
    
    def test_valid_system_info_with_gpu(self):
        """Test creating system info with GPU data."""
        info = SystemInfo(
            cpu_usage_percent=45.2,
            memory_used_mb=8192,
            memory_total_mb=16384,
            memory_percent=50.0,
            disk_free_gb=250.5,
            disk_total_gb=500.0,
            gpu_available=True,
            gpu_usage_percent=78.5,
            gpu_memory_used_mb=6144,
            gpu_memory_total_mb=8192,
            timestamp=datetime.now()
        )
        
        assert info.gpu_available is True
        assert info.gpu_usage_percent == 78.5
        assert info.gpu_memory_used_mb == 6144
    
    def test_valid_system_info_without_gpu(self):
        """Test creating system info without GPU data."""
        info = SystemInfo(
            cpu_usage_percent=45.2,
            memory_used_mb=8192,
            memory_total_mb=16384,
            memory_percent=50.0,
            disk_free_gb=250.5,
            disk_total_gb=500.0,
            gpu_available=False,
            timestamp=datetime.now()
        )
        
        assert info.gpu_available is False
        assert info.gpu_usage_percent is None
        assert info.gpu_memory_used_mb is None


class TestModelInfo:
    """Unit tests for ModelInfo model."""
    
    def test_valid_model_info(self):
        """Test creating valid model info."""
        info = ModelInfo(
            name="resnet50_cifar10",
            size_mb=102.4,
            created_at=datetime.now(),
            file_path="models/resnet50_cifar10.pth"
        )
        
        assert info.name == "resnet50_cifar10"
        assert info.size_mb == 102.4
        assert info.file_path == "models/resnet50_cifar10.pth"


class TestDatasetInfo:
    """Unit tests for DatasetInfo model."""
    
    def test_valid_dataset_info(self):
        """Test creating valid dataset info."""
        info = DatasetInfo(
            name="cifar10",
            size_mb=163.0,
            sample_count=60000,
            format="images",
            dimensions="32x32x3",
            created_at=datetime.now(),
            file_path="datasets/cifar10.tar.gz"
        )
        
        assert info.name == "cifar10"
        assert info.size_mb == 163.0
        assert info.sample_count == 60000
        assert info.format == "images"
        assert info.dimensions == "32x32x3"
    
    def test_dataset_info_without_dimensions(self):
        """Test creating dataset info without dimensions."""
        info = DatasetInfo(
            name="custom_dataset",
            size_mb=50.0,
            sample_count=10000,
            format="csv",
            created_at=datetime.now(),
            file_path="datasets/custom.csv"
        )
        
        assert info.dimensions is None
