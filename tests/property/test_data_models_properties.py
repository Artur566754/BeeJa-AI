"""
Property-based tests for API data models.

Tests validation logic for TrainingConfig and other data models using hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume
from pydantic import ValidationError
from datetime import datetime

from api.models import (
    TrainingConfig,
    SessionStatus,
    SessionState,
    Metrics,
    LogEntry,
    LogLevel,
    SystemInfo,
    ModelInfo,
    DatasetInfo,
)
from api.config import APIConfig


# Hypothesis strategies for generating test data

# Valid training configuration strategy
valid_training_config = st.fixed_dictionaries({
    "model_architecture": st.text(min_size=1, max_size=50),
    "dataset_name": st.text(min_size=1, max_size=50),
    "learning_rate": st.floats(
        min_value=APIConfig.MIN_LEARNING_RATE,
        max_value=APIConfig.MAX_LEARNING_RATE,
        allow_nan=False,
        allow_infinity=False
    ),
    "batch_size": st.integers(
        min_value=APIConfig.MIN_BATCH_SIZE,
        max_value=APIConfig.MAX_BATCH_SIZE
    ),
    "epochs": st.integers(
        min_value=APIConfig.MIN_EPOCHS,
        max_value=APIConfig.MAX_EPOCHS
    ),
    "optimizer": st.sampled_from(APIConfig.VALID_OPTIMIZERS),
    "loss_function": st.sampled_from(APIConfig.VALID_LOSS_FUNCTIONS),
})

# Invalid training configuration strategies
invalid_learning_rate_config = st.fixed_dictionaries({
    "model_architecture": st.text(min_size=1, max_size=50),
    "dataset_name": st.text(min_size=1, max_size=50),
    "learning_rate": st.one_of(
        st.floats(max_value=0.0, exclude_max=True),  # Negative or zero
        st.floats(min_value=APIConfig.MAX_LEARNING_RATE, exclude_min=True, max_value=10.0),  # Too high
        st.floats(max_value=APIConfig.MIN_LEARNING_RATE, exclude_max=True, min_value=0.0, exclude_min=True),  # Too low
    ),
    "batch_size": st.integers(
        min_value=APIConfig.MIN_BATCH_SIZE,
        max_value=APIConfig.MAX_BATCH_SIZE
    ),
    "epochs": st.integers(
        min_value=APIConfig.MIN_EPOCHS,
        max_value=APIConfig.MAX_EPOCHS
    ),
})

invalid_batch_size_config = st.fixed_dictionaries({
    "model_architecture": st.text(min_size=1, max_size=50),
    "dataset_name": st.text(min_size=1, max_size=50),
    "learning_rate": st.floats(
        min_value=APIConfig.MIN_LEARNING_RATE,
        max_value=APIConfig.MAX_LEARNING_RATE,
        allow_nan=False,
        allow_infinity=False
    ),
    "batch_size": st.one_of(
        st.integers(max_value=0),  # Zero or negative
        st.integers(min_value=APIConfig.MAX_BATCH_SIZE + 1, max_value=10000),  # Too high
    ),
    "epochs": st.integers(
        min_value=APIConfig.MIN_EPOCHS,
        max_value=APIConfig.MAX_EPOCHS
    ),
})

invalid_epochs_config = st.fixed_dictionaries({
    "model_architecture": st.text(min_size=1, max_size=50),
    "dataset_name": st.text(min_size=1, max_size=50),
    "learning_rate": st.floats(
        min_value=APIConfig.MIN_LEARNING_RATE,
        max_value=APIConfig.MAX_LEARNING_RATE,
        allow_nan=False,
        allow_infinity=False
    ),
    "batch_size": st.integers(
        min_value=APIConfig.MIN_BATCH_SIZE,
        max_value=APIConfig.MAX_BATCH_SIZE
    ),
    "epochs": st.one_of(
        st.integers(max_value=0),  # Zero or negative
        st.integers(min_value=APIConfig.MAX_EPOCHS + 1, max_value=100000),  # Too high
    ),
})

invalid_optimizer_config = st.fixed_dictionaries({
    "model_architecture": st.text(min_size=1, max_size=50),
    "dataset_name": st.text(min_size=1, max_size=50),
    "learning_rate": st.floats(
        min_value=APIConfig.MIN_LEARNING_RATE,
        max_value=APIConfig.MAX_LEARNING_RATE,
        allow_nan=False,
        allow_infinity=False
    ),
    "batch_size": st.integers(
        min_value=APIConfig.MIN_BATCH_SIZE,
        max_value=APIConfig.MAX_BATCH_SIZE
    ),
    "epochs": st.integers(
        min_value=APIConfig.MIN_EPOCHS,
        max_value=APIConfig.MAX_EPOCHS
    ),
    "optimizer": st.text(min_size=1, max_size=20).filter(
        lambda x: x not in APIConfig.VALID_OPTIMIZERS
    ),
})

invalid_loss_function_config = st.fixed_dictionaries({
    "model_architecture": st.text(min_size=1, max_size=50),
    "dataset_name": st.text(min_size=1, max_size=50),
    "learning_rate": st.floats(
        min_value=APIConfig.MIN_LEARNING_RATE,
        max_value=APIConfig.MAX_LEARNING_RATE,
        allow_nan=False,
        allow_infinity=False
    ),
    "batch_size": st.integers(
        min_value=APIConfig.MIN_BATCH_SIZE,
        max_value=APIConfig.MAX_BATCH_SIZE
    ),
    "epochs": st.integers(
        min_value=APIConfig.MIN_EPOCHS,
        max_value=APIConfig.MAX_EPOCHS
    ),
    "loss_function": st.text(min_size=1, max_size=20).filter(
        lambda x: x not in APIConfig.VALID_LOSS_FUNCTIONS
    ),
})


class TestTrainingConfigValidation:
    """
    Property tests for TrainingConfig validation.
    
    Feature: server-management-api, Property 19: Invalid configuration rejected with validation errors
    Validates: Requirements 6.4
    """
    
    @given(config_data=valid_training_config)
    def test_valid_config_accepted(self, config_data):
        """Valid training configurations should be accepted without errors."""
        # Valid configuration should not raise any exception
        config = TrainingConfig(**config_data)
        
        # Verify all fields are set correctly
        assert config.model_architecture == config_data["model_architecture"]
        assert config.dataset_name == config_data["dataset_name"]
        assert config.learning_rate == config_data["learning_rate"]
        assert config.batch_size == config_data["batch_size"]
        assert config.epochs == config_data["epochs"]
        assert config.optimizer == config_data["optimizer"]
        assert config.loss_function == config_data["loss_function"]
    
    @given(config_data=invalid_learning_rate_config)
    def test_invalid_learning_rate_rejected(self, config_data):
        """
        Invalid learning rates should be rejected with validation errors.
        
        Feature: server-management-api, Property 19: Invalid configuration rejected with validation errors
        Validates: Requirements 6.4
        """
        # Invalid learning rate should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**config_data)
        
        # Verify error message mentions learning rate
        error_str = str(exc_info.value).lower()
        assert "learning" in error_str or "learning_rate" in error_str
    
    @given(config_data=invalid_batch_size_config)
    def test_invalid_batch_size_rejected(self, config_data):
        """
        Invalid batch sizes should be rejected with validation errors.
        
        Feature: server-management-api, Property 19: Invalid configuration rejected with validation errors
        Validates: Requirements 6.4
        """
        # Invalid batch size should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**config_data)
        
        # Verify error message mentions batch size
        error_str = str(exc_info.value).lower()
        assert "batch" in error_str or "batch_size" in error_str
    
    @given(config_data=invalid_epochs_config)
    def test_invalid_epochs_rejected(self, config_data):
        """
        Invalid epoch counts should be rejected with validation errors.
        
        Feature: server-management-api, Property 19: Invalid configuration rejected with validation errors
        Validates: Requirements 6.4
        """
        # Invalid epochs should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**config_data)
        
        # Verify error message mentions epochs
        error_str = str(exc_info.value).lower()
        assert "epoch" in error_str
    
    @given(config_data=invalid_optimizer_config)
    def test_invalid_optimizer_rejected(self, config_data):
        """
        Invalid optimizer names should be rejected with validation errors.
        
        Feature: server-management-api, Property 19: Invalid configuration rejected with validation errors
        Validates: Requirements 6.4
        """
        # Invalid optimizer should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**config_data)
        
        # Verify error message mentions optimizer
        error_str = str(exc_info.value).lower()
        assert "optimizer" in error_str
    
    @given(config_data=invalid_loss_function_config)
    def test_invalid_loss_function_rejected(self, config_data):
        """
        Invalid loss function names should be rejected with validation errors.
        
        Feature: server-management-api, Property 19: Invalid configuration rejected with validation errors
        Validates: Requirements 6.4
        """
        # Invalid loss function should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**config_data)
        
        # Verify error message mentions loss function
        error_str = str(exc_info.value).lower()
        assert "loss" in error_str or "loss_function" in error_str


class TestTrainingConfigDefaults:
    """
    Property tests for TrainingConfig default values.
    
    Feature: server-management-api, Property 20: Optional parameters use defaults
    Validates: Requirements 6.5
    """
    
    @given(
        model_arch=st.text(min_size=1, max_size=50),
        dataset=st.text(min_size=1, max_size=50),
        lr=st.floats(
            min_value=APIConfig.MIN_LEARNING_RATE,
            max_value=APIConfig.MAX_LEARNING_RATE,
            allow_nan=False,
            allow_infinity=False
        ),
        batch_size=st.integers(
            min_value=APIConfig.MIN_BATCH_SIZE,
            max_value=APIConfig.MAX_BATCH_SIZE
        ),
        epochs=st.integers(
            min_value=APIConfig.MIN_EPOCHS,
            max_value=APIConfig.MAX_EPOCHS
        ),
    )
    def test_optional_parameters_use_defaults(
        self, model_arch, dataset, lr, batch_size, epochs
    ):
        """
        When optional parameters are omitted, default values should be used.
        
        Feature: server-management-api, Property 20: Optional parameters use defaults
        Validates: Requirements 6.5
        """
        # Create config without optional parameters
        config = TrainingConfig(
            model_architecture=model_arch,
            dataset_name=dataset,
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epochs
        )
        
        # Verify default values are used
        assert config.optimizer == APIConfig.DEFAULT_OPTIMIZER
        assert config.loss_function == APIConfig.DEFAULT_LOSS_FUNCTION
    
    @given(
        model_arch=st.text(min_size=1, max_size=50),
        dataset=st.text(min_size=1, max_size=50),
        lr=st.floats(
            min_value=APIConfig.MIN_LEARNING_RATE,
            max_value=APIConfig.MAX_LEARNING_RATE,
            allow_nan=False,
            allow_infinity=False
        ),
        batch_size=st.integers(
            min_value=APIConfig.MIN_BATCH_SIZE,
            max_value=APIConfig.MAX_BATCH_SIZE
        ),
        epochs=st.integers(
            min_value=APIConfig.MIN_EPOCHS,
            max_value=APIConfig.MAX_EPOCHS
        ),
        optimizer=st.sampled_from(APIConfig.VALID_OPTIMIZERS),
    )
    def test_explicit_optimizer_overrides_default(
        self, model_arch, dataset, lr, batch_size, epochs, optimizer
    ):
        """
        When optimizer is explicitly provided, it should override the default.
        
        Feature: server-management-api, Property 20: Optional parameters use defaults
        Validates: Requirements 6.5
        """
        # Create config with explicit optimizer
        config = TrainingConfig(
            model_architecture=model_arch,
            dataset_name=dataset,
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epochs,
            optimizer=optimizer
        )
        
        # Verify explicit value is used
        assert config.optimizer == optimizer
    
    @given(
        model_arch=st.text(min_size=1, max_size=50),
        dataset=st.text(min_size=1, max_size=50),
        lr=st.floats(
            min_value=APIConfig.MIN_LEARNING_RATE,
            max_value=APIConfig.MAX_LEARNING_RATE,
            allow_nan=False,
            allow_infinity=False
        ),
        batch_size=st.integers(
            min_value=APIConfig.MIN_BATCH_SIZE,
            max_value=APIConfig.MAX_BATCH_SIZE
        ),
        epochs=st.integers(
            min_value=APIConfig.MIN_EPOCHS,
            max_value=APIConfig.MAX_EPOCHS
        ),
        loss_function=st.sampled_from(APIConfig.VALID_LOSS_FUNCTIONS),
    )
    def test_explicit_loss_function_overrides_default(
        self, model_arch, dataset, lr, batch_size, epochs, loss_function
    ):
        """
        When loss function is explicitly provided, it should override the default.
        
        Feature: server-management-api, Property 20: Optional parameters use defaults
        Validates: Requirements 6.5
        """
        # Create config with explicit loss function
        config = TrainingConfig(
            model_architecture=model_arch,
            dataset_name=dataset,
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epochs,
            loss_function=loss_function
        )
        
        # Verify explicit value is used
        assert config.loss_function == loss_function


class TestOtherDataModels:
    """Basic property tests for other data models to ensure they validate correctly."""
    
    @given(
        session_id=st.text(min_size=1, max_size=50),
        epoch=st.integers(min_value=0, max_value=1000),
        loss=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        accuracy=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_metrics_model_valid(self, session_id, epoch, loss, accuracy):
        """Metrics model should accept valid data."""
        metrics = Metrics(
            session_id=session_id,
            epoch=epoch,
            loss=loss,
            accuracy=accuracy,
            timestamp=datetime.now()
        )
        
        assert metrics.session_id == session_id
        assert metrics.epoch == epoch
        assert metrics.loss == loss
        assert metrics.accuracy == accuracy
    
    @given(
        session_id=st.text(min_size=1, max_size=50),
        message=st.text(min_size=1, max_size=500),
        level=st.sampled_from(list(LogLevel)),
    )
    def test_log_entry_model_valid(self, session_id, message, level):
        """LogEntry model should accept valid data."""
        log_entry = LogEntry(
            session_id=session_id,
            timestamp=datetime.now(),
            level=level,
            message=message
        )
        
        assert log_entry.session_id == session_id
        assert log_entry.message == message
        assert log_entry.level == level
    
    @given(
        cpu=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        mem_used=st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        mem_total=st.floats(min_value=1.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        mem_pct=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        disk_free=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        disk_total=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    def test_system_info_model_valid(
        self, cpu, mem_used, mem_total, mem_pct, disk_free, disk_total
    ):
        """SystemInfo model should accept valid data."""
        system_info = SystemInfo(
            cpu_usage_percent=cpu,
            memory_used_mb=mem_used,
            memory_total_mb=mem_total,
            memory_percent=mem_pct,
            disk_free_gb=disk_free,
            disk_total_gb=disk_total,
            gpu_available=False,
            timestamp=datetime.now()
        )
        
        assert system_info.cpu_usage_percent == cpu
        assert system_info.memory_used_mb == mem_used
        assert system_info.gpu_available is False
