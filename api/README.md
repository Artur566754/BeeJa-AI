# Server Management API

## Overview

This is the Server Management API for remotely managing and monitoring PyTorch-based AI model training. The API provides comprehensive control over training sessions, system monitoring, model management, and configuration.

## Project Structure

```
api/
├── __init__.py           # Package initialization
├── config.py             # Configuration management
├── database.py           # SQLite database management
├── models/               # Data models (Pydantic)
├── services/             # Business logic layer
├── routes/               # API endpoint handlers
└── utils/                # Utility functions

tests/
├── unit/                 # Unit tests
└── integration/          # Integration tests
```

## Core Infrastructure

### Configuration (`api/config.py`)

The `APIConfig` class manages all configuration settings:

- **API Server Settings**: Host, port
- **Authentication**: API keys (comma-separated in environment variable)
- **Database**: SQLite database path
- **File Storage**: Models and datasets directories
- **Training Defaults**: Optimizer, loss function, max concurrent sessions
- **Validation Ranges**: Learning rate, batch size, epochs limits
- **Valid Options**: Supported optimizers and loss functions

#### Environment Variables

- `API_HOST`: API server host (default: "0.0.0.0")
- `API_PORT`: API server port (default: 8000)
- `API_KEYS`: Comma-separated list of valid API keys
- `DATABASE_PATH`: Path to SQLite database (default: "api_data.db")
- `MODELS_DIR`: Directory for model storage (default: "models")
- `DATASETS_DIR`: Directory for dataset storage (default: "datasets")
- `MAX_CONCURRENT_SESSIONS`: Maximum concurrent training sessions (default: 1)

### Database (`api/database.py`)

The `Database` class provides SQLite database management with:

- Thread-local connection pooling
- Context manager for automatic commit/rollback
- Schema creation and management
- Six core tables:
  - `sessions`: Training session metadata
  - `metrics`: Training metrics per epoch
  - `logs`: Training logs
  - `models`: Model registry
  - `datasets`: Dataset registry
  - `auth_logs`: Authentication audit logs

#### Database Schema

**sessions table**:
- `session_id` (TEXT, PRIMARY KEY)
- `config_json` (TEXT, NOT NULL)
- `state` (TEXT, NOT NULL)
- `current_epoch` (INTEGER)
- `total_epochs` (INTEGER, NOT NULL)
- `start_time` (TIMESTAMP)
- `end_time` (TIMESTAMP)
- `error_message` (TEXT)
- `created_at` (TIMESTAMP, NOT NULL)

**metrics table**:
- `id` (INTEGER, PRIMARY KEY)
- `session_id` (TEXT, FOREIGN KEY)
- `epoch` (INTEGER, NOT NULL)
- `loss` (REAL, NOT NULL)
- `accuracy` (REAL, NOT NULL)
- `val_loss` (REAL)
- `val_accuracy` (REAL)
- `timestamp` (TIMESTAMP, NOT NULL)
- UNIQUE constraint on (session_id, epoch)
- Index on session_id

**logs table**:
- `id` (INTEGER, PRIMARY KEY)
- `session_id` (TEXT, FOREIGN KEY)
- `timestamp` (TIMESTAMP, NOT NULL)
- `level` (TEXT, NOT NULL)
- `message` (TEXT, NOT NULL)
- Indexes on session_id and timestamp

**models table**:
- `name` (TEXT, PRIMARY KEY)
- `size_mb` (REAL, NOT NULL)
- `file_path` (TEXT, NOT NULL)
- `created_at` (TIMESTAMP, NOT NULL)

**datasets table**:
- `name` (TEXT, PRIMARY KEY)
- `size_mb` (REAL, NOT NULL)
- `sample_count` (INTEGER, NOT NULL)
- `format` (TEXT, NOT NULL)
- `dimensions` (TEXT)
- `file_path` (TEXT, NOT NULL)
- `created_at` (TIMESTAMP, NOT NULL)

**auth_logs table**:
- `id` (INTEGER, PRIMARY KEY)
- `api_key_hash` (TEXT, NOT NULL)
- `success` (BOOLEAN, NOT NULL)
- `endpoint` (TEXT, NOT NULL)
- `timestamp` (TIMESTAMP, NOT NULL)
- Index on timestamp

## Dependencies

The API requires the following dependencies (see `requirements.txt`):

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **Pydantic**: Data validation using Python type annotations
- **psutil**: System and process monitoring
- **pynvml**: NVIDIA GPU monitoring
- **pytest**: Testing framework
- **hypothesis**: Property-based testing

## Testing

The infrastructure includes comprehensive unit tests:

- `tests/unit/test_database.py`: Database creation, tables, indexes, transactions
- `tests/unit/test_database_schema.py`: Schema validation, foreign keys, constraints
- `tests/unit/test_config.py`: Configuration defaults and validation

Run tests with:
```bash
pytest tests/unit/test_database.py tests/unit/test_config.py tests/unit/test_database_schema.py -v
```

## Next Steps

The following components need to be implemented:

1. **Data Models** (Task 2): Pydantic models for validation
2. **System Monitor** (Task 3): CPU, memory, GPU monitoring
3. **Metrics Store** (Task 4): Metrics storage and retrieval
4. **Log Store** (Task 5): Log management
5. **Model Registry** (Task 6): Model file management
6. **Dataset Registry** (Task 7): Dataset management
7. **Training Executor** (Task 8): Training pipeline integration
8. **Session Manager** (Task 9): Training session lifecycle
9. **Authentication** (Task 11): API key validation
10. **API Routes** (Tasks 12-13): REST endpoints
11. **Main Server** (Task 14): FastAPI application

## Requirements Validation

This infrastructure setup validates the following requirements:

- **Requirement 8.3**: API key-based authentication (configuration support)
- **Requirement 9.1**: JSON response format (FastAPI provides this)

All database tables and indexes are created according to the design specification, providing the foundation for the complete API implementation.
