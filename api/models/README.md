# API Data Models

This directory contains Pydantic data models for the Server Management API.

## Overview

All data models use Pydantic for automatic validation, serialization, and documentation generation. The models enforce strict validation rules to ensure data integrity throughout the API.

## Models

### TrainingConfig

Configuration for a training session with comprehensive validation:

- **model_architecture**: Model architecture name (required, non-empty)
- **dataset_name**: Dataset name (required, non-empty)
- **learning_rate**: Learning rate (0.0001 to 1.0)
- **batch_size**: Batch size (1 to 256)
- **epochs**: Number of epochs (1 to 1000)
- **optimizer**: Optimizer algorithm (default: "adam", options: adam, sgd, rmsprop, adamw)
- **loss_function**: Loss function (default: "cross_entropy", options: cross_entropy, mse, mae, bce)

**Validation Rules:**
- Learning rate must be between MIN_LEARNING_RATE (0.0001) and MAX_LEARNING_RATE (1.0)
- Batch size must be between MIN_BATCH_SIZE (1) and MAX_BATCH_SIZE (256)
- Epochs must be between MIN_EPOCHS (1) and MAX_EPOCHS (1000)
- Optimizer must be one of the valid options
- Loss function must be one of the valid options
- Optional parameters use configured defaults when omitted

### SessionStatus

Status information for a training session:

- **session_id**: Unique session identifier
- **state**: Current state (queued, running, completed, failed, stopped)
- **current_epoch**: Current epoch number (optional, for running sessions)
- **total_epochs**: Total number of epochs
- **start_time**: Session start timestamp (optional)
- **end_time**: Session end timestamp (optional)
- **error_message**: Error message if failed (optional)

### SessionInfo

Complete information about a training session:

- **session_id**: Unique session identifier
- **config**: TrainingConfig object
- **status**: SessionStatus object
- **created_at**: Session creation timestamp

### Metrics

Training metrics for a specific epoch:

- **session_id**: Session identifier
- **epoch**: Epoch number (>= 0)
- **loss**: Training loss
- **accuracy**: Training accuracy (0.0 to 1.0)
- **val_loss**: Validation loss (optional)
- **val_accuracy**: Validation accuracy (optional, 0.0 to 1.0)
- **timestamp**: Metrics timestamp

### LogEntry

A single log entry for a training session:

- **session_id**: Session identifier
- **timestamp**: Log entry timestamp
- **level**: Log severity level (INFO, WARNING, ERROR, DEBUG)
- **message**: Log message (non-empty)

### SystemInfo

System resource information:

- **cpu_usage_percent**: CPU usage percentage (0.0 to 100.0)
- **memory_used_mb**: Memory used in MB (>= 0)
- **memory_total_mb**: Total memory in MB (> 0)
- **memory_percent**: Memory usage percentage (0.0 to 100.0)
- **disk_free_gb**: Free disk space in GB (>= 0)
- **disk_total_gb**: Total disk space in GB (> 0)
- **gpu_available**: Whether GPU is available
- **gpu_usage_percent**: GPU usage percentage (optional, 0.0 to 100.0)
- **gpu_memory_used_mb**: GPU memory used in MB (optional, >= 0)
- **gpu_memory_total_mb**: Total GPU memory in MB (optional, > 0)
- **timestamp**: Measurement timestamp

### ModelInfo

Metadata for a trained model:

- **name**: Model name (non-empty)
- **size_mb**: Model file size in MB (>= 0)
- **created_at**: Model creation timestamp
- **file_path**: Path to model file (non-empty)

### DatasetInfo

Metadata for a dataset:

- **name**: Dataset name (non-empty)
- **size_mb**: Dataset file size in MB (>= 0)
- **sample_count**: Number of samples (>= 0)
- **format**: Dataset format (e.g., "csv", "images", "numpy")
- **dimensions**: Data dimensions (optional, e.g., "32x32x3")
- **created_at**: Dataset creation timestamp
- **file_path**: Path to dataset file (non-empty)

## Enumerations

### SessionState

Possible training session states:
- `QUEUED`: Session is waiting to start
- `RUNNING`: Session is currently executing
- `COMPLETED`: Session finished successfully
- `FAILED`: Session encountered an error
- `STOPPED`: Session was manually stopped

### LogLevel

Log severity levels:
- `INFO`: Informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `DEBUG`: Debug messages

## Usage Examples

### Creating a Training Configuration

```python
from api.models import TrainingConfig

# With all parameters
config = TrainingConfig(
    model_architecture="resnet50",
    dataset_name="cifar10",
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
    optimizer="adam",
    loss_function="cross_entropy"
)

# With defaults (optimizer and loss_function will use configured defaults)
config = TrainingConfig(
    model_architecture="resnet50",
    dataset_name="cifar10",
    learning_rate=0.001,
    batch_size=32,
    epochs=10
)
```

### Creating Session Status

```python
from api.models import SessionStatus, SessionState
from datetime import datetime

status = SessionStatus(
    session_id="sess_abc123",
    state=SessionState.RUNNING,
    current_epoch=5,
    total_epochs=10,
    start_time=datetime.now()
)
```

### Creating Metrics

```python
from api.models import Metrics
from datetime import datetime

metrics = Metrics(
    session_id="sess_abc123",
    epoch=5,
    loss=0.234,
    accuracy=0.892,
    val_loss=0.267,
    val_accuracy=0.875,
    timestamp=datetime.now()
)
```

## Validation

All models use Pydantic validators to ensure data integrity:

1. **Range Validation**: Numeric fields are validated against configured min/max values
2. **Enum Validation**: String fields with limited options are validated against allowed values
3. **Required Fields**: Non-optional fields must be provided
4. **Type Validation**: All fields are type-checked automatically
5. **Custom Validators**: Complex validation logic is implemented using Pydantic validators

When validation fails, Pydantic raises a `ValidationError` with detailed information about which fields failed validation and why.

## Testing

The models are thoroughly tested with both unit tests and property-based tests:

- **Unit Tests**: `tests/unit/test_data_models.py` - Tests specific examples and edge cases
- **Property Tests**: `tests/property/test_data_models_properties.py` - Tests validation across wide range of inputs

Run tests with:
```bash
pytest tests/unit/test_data_models.py tests/property/test_data_models_properties.py -v
```

## Requirements Validation

These models satisfy the following requirements:

- **Requirement 6.1**: Accept configuration parameters (learning rate, batch size, epochs)
- **Requirement 6.2**: Accept model architecture selection
- **Requirement 6.3**: Accept dataset selection
- **Requirement 6.4**: Reject invalid configuration with validation errors
- **Requirement 6.5**: Use default values for optional parameters

The models implement:
- **Property 19**: Invalid configuration rejected with validation errors
- **Property 20**: Optional parameters use defaults
