# Telegram Bot API Integration

## Overview

The Telegram bot has been updated to use the Server Management API for all training and management operations. This provides better separation of concerns, enables remote management, and allows multiple clients to interact with the training server.

## Changes Made

### 1. API Client Class

Added a comprehensive `APIClient` class that handles all communication with the Server Management API:

**Features:**
- Authentication using Bearer token (API key)
- Automatic error handling and user-friendly error messages
- Timeout handling (30 seconds default)
- Connection error detection
- Support for all API endpoints

**Methods:**
- `health_check()` - Check if API server is running
- `start_training(config)` - Start a new training session
- `stop_training(session_id)` - Stop an active training session
- `get_training_status(session_id)` - Get session status
- `get_training_metrics(session_id)` - Get current metrics
- `get_training_history(session_id)` - Get full metrics history
- `get_training_logs(session_id, limit)` - Get training logs
- `list_sessions()` - List all training sessions
- `get_queue_status()` - Get queued sessions
- `get_system_info()` - Get system resource information
- `list_models()` - List all models in registry
- `list_datasets()` - List all datasets in registry

### 2. Updated Bot Commands

All bot commands now use the API instead of direct function calls:

**🎓 Обучить модель (Train Model)**
- Checks for active sessions via API
- Creates training session with configurable epochs
- Monitors progress asynchronously
- Sends updates on epoch completion
- Notifies on completion/failure

**📊 Статус модели (Model Status)**
- Shows system resources (CPU, RAM, GPU, disk)
- Shows active and completed training sessions
- Shows models in registry with total size

**📁 Список датасетов (Dataset List)**
- Lists all datasets from API registry
- Shows size, sample count, and format for each dataset

### 3. Training Monitoring

Added asynchronous training monitoring that:
- Checks training status every 30 seconds
- Sends progress updates when epochs complete
- Shows loss and accuracy metrics
- Notifies on completion, failure, or manual stop
- Handles errors gracefully without stopping monitoring

### 4. Configuration

Added new environment variables in `.env`:

```bash
# Server Management API Configuration
API_URL=http://localhost:8000
API_KEY=your_api_key_here
```

### 5. Error Handling

Comprehensive error handling for:
- API server not running (connection errors)
- Invalid API key (authentication errors)
- Invalid parameters (validation errors)
- Resource not found (404 errors)
- Timeout errors
- Network errors

All errors are translated to user-friendly Russian messages.

### 6. Backward Compatibility

The bot maintains backward compatibility:
- Local chat interface still works if API is unavailable
- Graceful degradation when API_KEY is not configured
- Warning messages guide users to configure API

## Setup Instructions

### 1. Install Dependencies

```bash
cd telegram_bot
pip install -r requirements.txt
```

The updated `requirements.txt` includes `httpx>=0.24.0` for HTTP requests.

### 2. Configure API Access

Edit `telegram_bot/.env` and add:

```bash
API_URL=http://localhost:8000
API_KEY=your_secret_api_key
```

**Important:** The API_KEY must match one of the keys configured in the API server.

### 3. Start the API Server

Before running the bot, start the API server:

```bash
python run_api.py --api-keys "your_secret_api_key"
```

### 4. Start the Bot

```bash
cd telegram_bot
python bot.py
```

## Usage Examples

### Starting Training

1. Open Telegram and start the bot with `/start`
2. Click "🎓 Обучить модель"
3. Enter number of epochs (e.g., 10)
4. Bot will create a training session and start monitoring
5. You'll receive updates as training progresses

### Checking Status

1. Click "📊 Статус модели"
2. View system resources and training sessions
3. See active sessions, completed sessions, and models

### Viewing Datasets

1. Click "📁 Список датасетов"
2. View all available datasets with metadata

## API Integration Details

### Training Session Lifecycle

```
User Request → Bot → API Client → API Server → Training Pipeline
                ↓
         Monitor Task (async)
                ↓
         Status Updates → User
```

### Error Flow

```
API Error → APIClient._request() → Exception with Russian message → User
```

### Monitoring Flow

```
Start Training → Create Monitor Task
                      ↓
                Check Status (every 30s)
                      ↓
                Epoch Changed? → Send Update
                      ↓
                Completed/Failed? → Send Final Message & Stop
```

## Testing

### Test API Client

Run the test script to verify API client functionality:

```bash
python telegram_bot/test_api_client.py
```

### Test with Bot

1. Ensure API server is running
2. Start the bot
3. Try each command:
   - `/start` - Should show menu
   - "🎓 Обучить модель" - Should create training session
   - "📊 Статус модели" - Should show system info
   - "📁 Список датасетов" - Should list datasets

### Test Error Handling

1. Stop the API server
2. Try bot commands
3. Should see user-friendly error messages about API not being available

## Troubleshooting

### "API клиент не настроен"

**Problem:** API_KEY not set in `.env` file

**Solution:** Add `API_KEY=your_key` to `telegram_bot/.env`

### "Не удалось подключиться к API серверу"

**Problem:** API server is not running or wrong URL

**Solution:** 
1. Check API server is running: `curl http://localhost:8000/health`
2. Verify API_URL in `.env` matches server address

### "Ошибка аутентификации: неверный API ключ"

**Problem:** API_KEY doesn't match server configuration

**Solution:** Ensure API_KEY in bot's `.env` matches one of the keys in API server configuration

### Training doesn't start

**Problem:** No datasets available

**Solution:** Upload a dataset via API:
```bash
curl -X POST http://localhost:8000/api/v1/datasets?dataset_name=my_dataset \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@dataset.tar.gz"
```

## Architecture Benefits

### Before (Direct Integration)
- Bot directly called training functions
- Used subprocess to run training
- No visibility into training progress
- Single point of failure
- Hard to scale

### After (API Integration)
- Clean separation of concerns
- RESTful API for all operations
- Real-time progress monitoring
- Multiple clients can connect
- Easy to scale and deploy separately

## Future Enhancements

Possible improvements:
1. Add commands to stop training sessions
2. Add commands to view training logs
3. Add commands to download models
4. Add commands to upload datasets
5. Add inline buttons for session management
6. Add charts/graphs for metrics visualization
7. Add notifications for training completion (even if user didn't start it)

## Security Considerations

1. **API Key Storage**: API keys are stored in `.env` file (not committed to git)
2. **HTTPS**: For production, use HTTPS for API communication
3. **Admin Only**: Training commands are restricted to admin user
4. **Rate Limiting**: Consider adding rate limiting to prevent abuse
5. **Input Validation**: All inputs are validated by API server

## Performance

- API requests have 30-second timeout
- Monitoring checks status every 30 seconds
- Minimal overhead for bot operations
- Asynchronous operations don't block bot

## Compatibility

- **Python**: 3.8+
- **API Server**: Server Management API v1.0.0+
- **Dependencies**: See `requirements.txt`

## Support

For issues or questions:
1. Check API server logs: Look for errors in API server output
2. Check bot logs: Look for errors in bot output
3. Verify configuration: Ensure `.env` is properly configured
4. Test API directly: Use curl to test API endpoints
5. Check network: Ensure bot can reach API server

## References

- [API Documentation](../API_DOCUMENTATION.md)
- [Bot README](README.md)
- [Server Management API Design](.kiro/specs/server-management-api/design.md)
