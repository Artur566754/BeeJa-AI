# Implementation Plan: Server Management API

## Overview

This implementation plan breaks down the Server Management API into discrete coding tasks. The approach follows a bottom-up strategy: first establishing core infrastructure (database, models, utilities), then building individual components (training manager, metrics store, model registry), and finally wiring everything together with the API layer and authentication.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - Create directory structure: `api/`, `api/models/`, `api/services/`, `api/routes/`, `api/utils/`, `tests/unit/`, `tests/integration/`
  - Create `api/database.py` with SQLite connection management and table creation
  - Create database schema (sessions, metrics, logs, models, datasets, auth_logs tables)
  - Set up `requirements.txt` with dependencies: fastapi, uvicorn, sqlalchemy, psutil, pynvml, hypothesis, pytest
  - Create `api/config.py` for configuration management (API keys, paths, defaults)
  - _Requirements: 8.3, 9.1_

- [x] 2. Implement data models and validation
  - [x] 2.1 Create core data model classes
    - Write Pydantic models for TrainingConfig, SessionStatus, SessionInfo, Metrics, LogEntry, SystemInfo, ModelInfo, DatasetInfo
    - Implement validation logic for TrainingConfig (learning rate range, positive batch size, etc.)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 2.2 Write property test for configuration validation
    - **Property 19: Invalid configuration rejected with validation errors**
    - **Validates: Requirements 6.4**

  - [x] 2.3 Write property test for default values
    - **Property 20: Optional parameters use defaults**
    - **Validates: Requirements 6.5**

- [x] 3. Implement System Monitor component
  - [x] 3.1 Create `api/services/system_monitor.py`
    - Implement SystemMonitor class with get_system_info() method
    - Use psutil for CPU, memory, disk metrics
    - Use pynvml for GPU metrics (with graceful fallback if unavailable)
    - Implement background refresh task (every 5 seconds)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.2 Write unit test for system info completeness
    - **Property 11: System info contains required fields**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [x] 4. Implement Metrics Store component
  - [x] 4.1 Create `api/services/metrics_store.py`
    - Implement MetricsStore class with SQLite backend
    - Implement save_metrics(), get_latest_metrics(), get_metrics_history(), get_metrics_by_epoch()
    - Add database indexes for efficient queries
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 4.2 Write property test for metrics retrieval
    - **Property 7: Metrics request returns latest data**
    - **Validates: Requirements 2.2**

  - [x] 4.3 Write property test for metrics history
    - **Property 8: Metrics history returns all epochs**
    - **Validates: Requirements 2.3**

  - [x] 4.4 Write property test for metrics per epoch
    - **Property 9: Metrics recorded per epoch**
    - **Validates: Requirements 2.4**

- [x] 5. Implement Log Store component
  - [x] 5.1 Create `api/services/log_store.py`
    - Implement LogStore class with SQLite backend
    - Implement append_log(), get_logs(), get_logs_since()
    - Ensure all log entries include timestamps
    - _Requirements: 7.1, 7.3, 7.4, 7.5_

  - [x] 5.2 Write property test for log timestamps
    - **Property 25: All log entries have timestamps**
    - **Validates: Requirements 7.5**

  - [x] 5.3 Write property test for log persistence
    - **Property 24: Completed session persists logs**
    - **Validates: Requirements 7.4**

- [x] 6. Implement Model Registry component
  - [x] 6.1 Create `api/services/model_registry.py`
    - Implement ModelRegistry class with file system and SQLite backend
    - Implement list_models(), get_model_path(), save_model(), delete_model(), model_exists()
    - Store models in `models/` directory with metadata in database
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 6.2 Write property test for model listing
    - **Property 12: Model list returns all models**
    - **Validates: Requirements 4.1**

  - [x] 6.3 Write property test for model round-trip
    - **Property 13: Model download round-trip**
    - **Validates: Requirements 4.2, 4.3**

  - [x] 6.4 Write property test for model deletion
    - **Property 14: Model deletion removes all traces**
    - **Validates: Requirements 4.4**

- [x] 7. Implement Dataset Registry component
  - [x] 7.1 Create `api/services/dataset_registry.py`
    - Implement DatasetRegistry class with file system and SQLite backend
    - Implement list_datasets(), get_dataset_info(), save_dataset(), dataset_exists()
    - Extract metadata (format, dimensions, sample count) during upload
    - Store datasets in `datasets/` directory
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 7.2 Write property test for dataset listing
    - **Property 16: Dataset list returns complete metadata**
    - **Validates: Requirements 5.1**

  - [x] 7.3 Write property test for dataset metadata
    - **Property 17: Dataset info returns correct metadata**
    - **Validates: Requirements 5.2**

  - [x] 7.4 Write property test for dataset upload
    - **Property 18: Dataset upload stores with metadata**
    - **Validates: Requirements 5.3**

- [x] 8. Implement Training Pipeline Integration
  - [x] 8.1 Create `api/services/training_executor.py`
    - Implement TrainingExecutor class that wraps existing `src/training_pipeline.py`
    - Implement execute_training() with callbacks for metrics and logs
    - Implement stop_training() for graceful shutdown
    - Handle model saving on completion
    - Handle error logging on failure
    - _Requirements: 1.1, 1.2, 1.4, 1.5_

  - [x] 8.2 Write property test for training completion
    - **Property 4: Completed session saves model**
    - **Validates: Requirements 1.4**

  - [x] 8.3 Write property test for training failure
    - **Property 5: Failed session logs error**
    - **Validates: Requirements 1.5**

- [x] 9. Implement Training Session Manager
  - [x] 9.1 Create `api/services/training_session_manager.py`
    - Implement TrainingSessionManager class
    - Implement create_session(), start_session(), stop_session(), get_status()
    - Implement get_active_sessions(), get_queued_sessions(), cancel_queued_session()
    - Implement can_start_new_session() with resource checking
    - Implement queue management and automatic session starting
    - Use asyncio background tasks for training execution
    - Persist session state to database
    - _Requirements: 1.1, 1.2, 1.3, 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 9.2 Write property test for session creation
    - **Property 1: Valid training configuration creates session**
    - **Validates: Requirements 1.1, 6.1, 6.2, 6.3**

  - [x] 9.3 Write property test for session stopping
    - **Property 2: Stopping active session preserves state**
    - **Validates: Requirements 1.2**

  - [x] 9.4 Write property test for session status
    - **Property 3: Session status reflects actual state**
    - **Validates: Requirements 1.3**

  - [x] 9.5 Write property test for concurrent sessions
    - **Property 33: Concurrent sessions run when resources allow**
    - **Validates: Requirements 10.1**

  - [x] 9.6 Write property test for queueing
    - **Property 34: Requests queued when capacity full**
    - **Validates: Requirements 10.2**

  - [x] 9.7 Write property test for queue processing
    - **Property 36: Queue automatically processes on completion**
    - **Validates: Requirements 10.4**

- [x] 10. Checkpoint - Ensure core services work
  - Run all unit and property tests for services
  - Verify database operations work correctly
  - Verify file system operations work correctly
  - Ask the user if questions arise

- [x] 11. Implement Authentication component
  - [x] 11.1 Create `api/services/auth_manager.py`
    - Implement AuthenticationManager class
    - Implement validate_api_key(), get_api_key_from_header()
    - Implement log_auth_attempt() with database logging
    - Load API keys from environment variable or config file
    - _Requirements: 8.1, 8.2, 8.4, 8.5_

  - [x] 11.2 Write property test for authentication rejection
    - **Property 26: Unauthenticated requests rejected**
    - **Validates: Requirements 8.1**

  - [x] 11.3 Write property test for invalid API keys
    - **Property 28: Invalid API key returns auth error**
    - **Validates: Requirements 8.4**

  - [x] 11.4 Write property test for auth logging
    - **Property 29: Authentication attempts logged**
    - **Validates: Requirements 8.5**

- [x] 12. Implement API routes - Training endpoints
  - [x] 12.1 Create `api/routes/training.py`
    - Implement POST /api/v1/training/start endpoint
    - Implement POST /api/v1/training/{session_id}/stop endpoint
    - Implement GET /api/v1/training/{session_id}/status endpoint
    - Implement GET /api/v1/training/{session_id}/metrics endpoint
    - Implement GET /api/v1/training/{session_id}/history endpoint
    - Implement GET /api/v1/training/{session_id}/logs endpoint
    - Implement GET /api/v1/training/sessions endpoint
    - Implement GET /api/v1/training/queue endpoint
    - Implement DELETE /api/v1/training/queue/{session_id} endpoint
    - Add authentication dependency to all endpoints
    - Add error handling with consistent error response format
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 7.1, 7.2, 10.3, 10.5_

  - [x] 12.2 Write integration tests for training endpoints
    - Test complete training lifecycle (start → status → metrics → logs)
    - Test queue management workflow
    - Test error cases (non-existent session, invalid config)

- [x] 13. Implement API routes - System and resource endpoints
  - [x] 13.1 Create `api/routes/system.py`
    - Implement GET /api/v1/system/info endpoint
    - Add authentication dependency
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 13.2 Create `api/routes/models.py`
    - Implement GET /api/v1/models endpoint
    - Implement GET /api/v1/models/{model_name} endpoint (file download)
    - Implement POST /api/v1/models endpoint (file upload)
    - Implement DELETE /api/v1/models/{model_name} endpoint
    - Add authentication dependency to all endpoints
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 13.3 Create `api/routes/datasets.py`
    - Implement GET /api/v1/datasets endpoint
    - Implement GET /api/v1/datasets/{dataset_name} endpoint
    - Implement POST /api/v1/datasets endpoint (file upload)
    - Add authentication dependency to all endpoints
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 13.4 Write property test for non-existent resources
    - **Property 15: Non-existent resource returns 404**
    - **Validates: Requirements 4.5, 5.4**

- [x] 14. Implement main API server
  - [x] 14.1 Create `api/main.py`
    - Initialize FastAPI application
    - Register all route modules (training, system, models, datasets)
    - Add CORS middleware if needed
    - Add global exception handler for consistent error responses
    - Implement startup event to initialize database and services
    - Implement shutdown event to cleanup resources
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 14.2 Write property test for JSON responses
    - **Property 30: All responses are valid JSON**
    - **Validates: Requirements 9.1**

  - [x] 14.3 Write property test for HTTP status codes
    - **Property 31: HTTP status codes match operation result**
    - **Validates: Requirements 9.2, 9.3, 9.4**

  - [x] 14.4 Write property test for error structure
    - **Property 32: Error responses have consistent structure**
    - **Validates: Requirements 9.5**

- [x] 15. Create API startup script and documentation
  - [x] 15.1 Create `run_api.py` script
    - Parse command line arguments (port, host, API keys)
    - Initialize configuration
    - Start uvicorn server
    - _Requirements: All_

  - [x] 15.2 Create `API_DOCUMENTATION.md`
    - Document all endpoints with request/response examples
    - Document authentication setup
    - Document configuration options
    - Include example curl commands for each endpoint
    - _Requirements: All_

- [x] 16. Integration with Telegram bot
  - [x] 16.1 Update `telegram_bot/bot.py` to use the API
    - Add API client class for making HTTP requests
    - Update bot commands to call API endpoints
    - Add error handling for API failures
    - Add configuration for API URL and API key
    - _Requirements: All_

  - [x] 16.2 Write integration tests for bot-API interaction
    - Test bot commands trigger correct API calls
    - Test bot handles API responses correctly
    - Test bot handles API errors gracefully

- [x] 17. Final checkpoint - End-to-end testing
  - Start the API server
  - Run complete integration test suite
  - Test with actual Telegram bot commands
  - Verify all property tests pass (100 iterations each)
  - Verify all unit tests pass
  - Generate test coverage report
  - Ask the user if questions arise

## Notes

- All tasks are required for comprehensive implementation
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples and edge cases
- The implementation follows a bottom-up approach: infrastructure → services → API layer
- Database and file system operations are isolated in service classes for testability
- All API responses follow consistent JSON format with proper error handling
