# Server Management API Documentation

## Overview

The Server Management API is a RESTful HTTP API that provides comprehensive control over a PyTorch-based AI model training server. It enables remote management of training sessions, system monitoring, model management, and dataset management.

## Base URL

```
http://localhost:8000
```

## Authentication

All API endpoints (except `/health` and `/`) require authentication using an API key.

### Authentication Methods

The API supports two authentication methods:

1. **Bearer Token** (recommended):
   ```
   Authorization: Bearer YOUR_API_KEY
   ```

2. **X-API-Key Header**:
   ```
   X-API-Key: YOUR_API_KEY
   ```

### Setting Up API Keys

API keys can be configured in two ways:

1. **Environment Variable**:
   ```bash
   export API_KEYS="key1,key2,key3"
   ```

2. **Command-Line Argument**:
   ```bash
   python run_api.py --api-keys "key1,key2,key3"
   ```

## Starting the Server

### Basic Usage

```bash
python run_api.py
```

### With Custom Configuration

```bash
python run_api.py --host 0.0.0.0 --port 8000 --api-keys "your_secret_key"
```

### Development Mode (with auto-reload)

```bash
python run_api.py --reload
```

### Command-Line Options

- `--host HOST`: Host to bind to (default: 0.0.0.0)
- `--port PORT`: Port to bind to (default: 8000)
- `--api-keys KEYS`: Comma-separated list of API keys
- `--reload`: Enable auto-reload for development
- `--log-level LEVEL`: Log level (critical, error, warning, info, debug, trace)

## API Endpoints

### Health Check

#### GET /health

Check if the API server is running.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Server Management API",
  "version": "1.0.0"
}
```

---

### Training Management

#### POST /api/v1/training/start

Start a new training session.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_architecture": "resnet50",
    "dataset_name": "cifar10",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "adam",
    "loss_function": "cross_entropy"
  }'
```

**Response:**
```json
{
  "session_id": "sess_abc123def456",
  "status": "running",
  "message": "Training session created and started"
}
```

**Parameters:**
- `model_architecture` (required): Model architecture name
- `dataset_name` (required): Dataset name to use
- `learning_rate` (required): Learning rate (0.0001 - 1.0)
- `batch_size` (required): Batch size (1 - 256)
- `epochs` (required): Number of epochs (1 - 1000)
- `optimizer` (optional): Optimizer (adam, sgd, rmsprop, adamw) - default: adam
- `loss_function` (optional): Loss function (cross_entropy, mse, mae, bce) - default: cross_entropy

---

#### POST /api/v1/training/{session_id}/stop

Stop an active training session.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/training/sess_abc123/stop \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "message": "Training session sess_abc123 stopped successfully"
}
```

---

#### GET /api/v1/training/{session_id}/status

Get the status of a training session.

**Request:**
```bash
curl http://localhost:8000/api/v1/training/sess_abc123/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "state": "running",
  "current_epoch": 5,
  "total_epochs": 10,
  "start_time": "2024-01-15T10:30:00Z",
  "end_time": null,
  "error_message": null
}
```

**States:**
- `queued`: Session is waiting to start
- `running`: Session is currently training
- `completed`: Session finished successfully
- `failed`: Session failed with an error
- `stopped`: Session was manually stopped

---

#### GET /api/v1/training/{session_id}/metrics

Get the latest metrics for a training session.

**Request:**
```bash
curl http://localhost:8000/api/v1/training/sess_abc123/metrics \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "epoch": 5,
  "loss": 0.234,
  "accuracy": 0.892,
  "val_loss": 0.267,
  "val_accuracy": 0.875,
  "timestamp": "2024-01-15T10:35:00Z"
}
```

---

#### GET /api/v1/training/{session_id}/history

Get the complete metrics history for a training session.

**Request:**
```bash
curl http://localhost:8000/api/v1/training/sess_abc123/history \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
[
  {
    "session_id": "sess_abc123",
    "epoch": 1,
    "loss": 0.856,
    "accuracy": 0.654,
    "val_loss": 0.892,
    "val_accuracy": 0.632,
    "timestamp": "2024-01-15T10:31:00Z"
  },
  {
    "session_id": "sess_abc123",
    "epoch": 2,
    "loss": 0.543,
    "accuracy": 0.782,
    "val_loss": 0.589,
    "val_accuracy": 0.756,
    "timestamp": "2024-01-15T10:32:00Z"
  }
]
```

---

#### GET /api/v1/training/{session_id}/logs

Get logs for a training session.

**Request:**
```bash
curl http://localhost:8000/api/v1/training/sess_abc123/logs?limit=100 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
[
  {
    "session_id": "sess_abc123",
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "message": "Starting training session"
  },
  {
    "session_id": "sess_abc123",
    "timestamp": "2024-01-15T10:31:00Z",
    "level": "INFO",
    "message": "Epoch 1/10 completed"
  }
]
```

**Query Parameters:**
- `limit` (optional): Maximum number of log entries to return (default: 100)

---

#### GET /api/v1/training/sessions

List all training sessions.

**Request:**
```bash
curl http://localhost:8000/api/v1/training/sessions \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
[
  {
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
]
```

---

#### GET /api/v1/training/queue

Get the status of the training queue.

**Request:**
```bash
curl http://localhost:8000/api/v1/training/queue \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
[
  {
    "session_id": "sess_def456",
    "config": {
      "model_architecture": "vgg16",
      "dataset_name": "mnist",
      "learning_rate": 0.001,
      "batch_size": 64,
      "epochs": 5
    },
    "status": {
      "session_id": "sess_def456",
      "state": "queued",
      "current_epoch": null,
      "total_epochs": 5
    },
    "created_at": "2024-01-15T10:40:00Z"
  }
]
```

---

#### DELETE /api/v1/training/queue/{session_id}

Cancel a queued training session.

**Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/training/queue/sess_def456 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "message": "Queued session sess_def456 cancelled successfully"
}
```

---

### System Monitoring

#### GET /api/v1/system/info

Get current system resource information.

**Request:**
```bash
curl http://localhost:8000/api/v1/system/info \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "cpu_usage_percent": 45.2,
  "memory_used_mb": 8192,
  "memory_total_mb": 16384,
  "memory_percent": 50.0,
  "disk_free_gb": 250.5,
  "disk_total_gb": 500.0,
  "gpu_available": true,
  "gpu_usage_percent": 78.5,
  "gpu_memory_used_mb": 6144,
  "gpu_memory_total_mb": 8192,
  "timestamp": "2024-01-15T10:35:00Z"
}
```

---

### Model Management

#### GET /api/v1/models

List all models in the model registry.

**Request:**
```bash
curl http://localhost:8000/api/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
[
  {
    "name": "resnet50_cifar10",
    "size_mb": 102.4,
    "created_at": "2024-01-15T11:00:00Z",
    "file_path": "models/resnet50_cifar10_20240115.pth"
  }
]
```

---

#### GET /api/v1/models/{model_name}

Download a model file.

**Request:**
```bash
curl http://localhost:8000/api/v1/models/resnet50_cifar10 \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -o model.pth
```

**Response:** Binary file download

---

#### POST /api/v1/models

Upload a model file.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/models?model_name=my_model \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@model.pth"
```

**Response:**
```json
{
  "message": "Model 'my_model' uploaded successfully",
  "name": "my_model",
  "size_mb": 102.4
}
```

---

#### DELETE /api/v1/models/{model_name}

Delete a model from the registry.

**Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/models/my_model \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "message": "Model 'my_model' deleted successfully"
}
```

---

### Dataset Management

#### GET /api/v1/datasets

List all datasets in the dataset registry.

**Request:**
```bash
curl http://localhost:8000/api/v1/datasets \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
[
  {
    "name": "cifar10",
    "size_mb": 163.0,
    "sample_count": 60000,
    "format": "images",
    "dimensions": "32x32x3",
    "created_at": "2024-01-10T09:00:00Z",
    "file_path": "datasets/cifar10.tar.gz"
  }
]
```

---

#### GET /api/v1/datasets/{dataset_name}

Get information about a specific dataset.

**Request:**
```bash
curl http://localhost:8000/api/v1/datasets/cifar10 \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "name": "cifar10",
  "size_mb": 163.0,
  "sample_count": 60000,
  "format": "images",
  "dimensions": "32x32x3",
  "created_at": "2024-01-10T09:00:00Z",
  "file_path": "datasets/cifar10.tar.gz"
}
```

---

#### POST /api/v1/datasets

Upload a dataset file.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/datasets?dataset_name=my_dataset \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@dataset.tar.gz"
```

**Response:**
```json
{
  "message": "Dataset 'my_dataset' uploaded successfully",
  "name": "my_dataset",
  "size_mb": 163.0
}
```

---

## Error Responses

All error responses follow a consistent format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

### Example Error Response

```json
{
  "detail": "Session 'sess_invalid' not found"
}
```

---

## Configuration

### Environment Variables

- `API_HOST`: Host to bind to (default: 0.0.0.0)
- `API_PORT`: Port to bind to (default: 8000)
- `API_KEYS`: Comma-separated list of API keys (required)
- `DATABASE_PATH`: Path to SQLite database (default: api_data.db)
- `MODELS_DIR`: Directory for model files (default: models)
- `DATASETS_DIR`: Directory for dataset files (default: datasets)
- `MAX_CONCURRENT_SESSIONS`: Maximum concurrent training sessions (default: 1)

### Example Configuration

```bash
export API_KEYS="secret_key_1,secret_key_2"
export MAX_CONCURRENT_SESSIONS=2
export MODELS_DIR="/path/to/models"
export DATASETS_DIR="/path/to/datasets"

python run_api.py
```

---

## Interactive API Documentation

The API provides interactive documentation powered by Swagger UI:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- Browse all available endpoints
- View request/response schemas
- Test API calls directly from the browser
- See example requests and responses

---

## Examples

### Complete Training Workflow

```bash
# 1. Start a training session
SESSION_ID=$(curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_architecture": "resnet50",
    "dataset_name": "cifar10",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
  }' | jq -r '.session_id')

echo "Started session: $SESSION_ID"

# 2. Check status
curl http://localhost:8000/api/v1/training/$SESSION_ID/status \
  -H "Authorization: Bearer YOUR_API_KEY"

# 3. Get current metrics
curl http://localhost:8000/api/v1/training/$SESSION_ID/metrics \
  -H "Authorization: Bearer YOUR_API_KEY"

# 4. View logs
curl http://localhost:8000/api/v1/training/$SESSION_ID/logs \
  -H "Authorization: Bearer YOUR_API_KEY"

# 5. Stop training (if needed)
curl -X POST http://localhost:8000/api/v1/training/$SESSION_ID/stop \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Monitor System Resources

```bash
# Get system info
curl http://localhost:8000/api/v1/system/info \
  -H "Authorization: Bearer YOUR_API_KEY" | jq

# Watch system resources (updates every 5 seconds)
watch -n 5 'curl -s http://localhost:8000/api/v1/system/info \
  -H "Authorization: Bearer YOUR_API_KEY" | jq'
```

### Manage Models

```bash
# List all models
curl http://localhost:8000/api/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"

# Download a model
curl http://localhost:8000/api/v1/models/resnet50_cifar10 \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -o downloaded_model.pth

# Upload a model
curl -X POST http://localhost:8000/api/v1/models?model_name=new_model \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@model.pth"

# Delete a model
curl -X DELETE http://localhost:8000/api/v1/models/old_model \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Troubleshooting

### API Key Not Configured

**Error:**
```
Configuration error: No API keys configured. Set API_KEYS environment variable.
```

**Solution:**
```bash
python run_api.py --api-keys "your_secret_key"
```

### Port Already in Use

**Error:**
```
[ERROR] [Errno 48] Address already in use
```

**Solution:**
```bash
python run_api.py --port 8001
```

### GPU Not Available

If GPU monitoring is not available, the API will still work but `gpu_available` will be `false` in system info responses. This is normal if:
- No GPU is installed
- NVIDIA drivers are not installed
- `pynvml` package is not installed

---

## Support

For issues, questions, or contributions, please refer to the project repository.
