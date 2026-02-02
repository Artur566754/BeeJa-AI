"""
Integration tests for bot-API interaction.

These tests verify that the Telegram bot correctly interacts with the Server Management API,
including making API calls, handling responses, and handling errors gracefully.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock, patch
import httpx
from telegram_bot.bot import APIClient


@pytest.fixture
def api_client():
    """Create an APIClient instance for testing."""
    return APIClient(base_url="http://localhost:8000", api_key="test_key_123")


@pytest.mark.asyncio
class TestAPIClientSuccessfulCalls:
    """Test successful API calls from the bot."""
    
    async def test_health_check_success(self, api_client):
        """Test successful health check API call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.health_check()
            
            assert result is True
    
    async def test_start_training_success(self, api_client):
        """Test successful training start API call."""
        config = {
            "model_architecture": "test_model",
            "dataset_name": "test_dataset",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_id": "sess_abc123",
            "status": "running",
            "message": "Training started"
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.start_training(config)
            
            assert result["session_id"] == "sess_abc123"
            assert result["status"] == "running"
    
    async def test_get_training_status_success(self, api_client):
        """Test successful training status retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_id": "sess_abc123",
            "state": "running",
            "current_epoch": 5,
            "total_epochs": 10
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.get_training_status("sess_abc123")
            
            assert result["session_id"] == "sess_abc123"
            assert result["state"] == "running"
            assert result["current_epoch"] == 5
    
    async def test_stop_training_success(self, api_client):
        """Test successful training stop API call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": "Training session stopped successfully"
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.stop_training("sess_abc123")
            
            assert "message" in result
    
    async def test_get_training_metrics_success(self, api_client):
        """Test successful metrics retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_id": "sess_abc123",
            "epoch": 5,
            "loss": 0.234,
            "accuracy": 0.892
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.get_training_metrics("sess_abc123")
            
            assert result["epoch"] == 5
            assert result["loss"] == 0.234
            assert result["accuracy"] == 0.892
    
    async def test_get_training_history_success(self, api_client):
        """Test successful training history retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"epoch": 1, "loss": 0.856, "accuracy": 0.654},
            {"epoch": 2, "loss": 0.543, "accuracy": 0.782}
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.get_training_history("sess_abc123")
            
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["epoch"] == 1
    
    async def test_get_training_logs_success(self, api_client):
        """Test successful training logs retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "message": "Training started"},
            {"timestamp": "2024-01-15T10:31:00Z", "level": "INFO", "message": "Epoch 1 completed"}
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.get_training_logs("sess_abc123", limit=50)
            
            assert isinstance(result, list)
            assert len(result) == 2
    
    async def test_list_sessions_success(self, api_client):
        """Test successful sessions list retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"session_id": "sess_abc123", "status": {"state": "running"}},
            {"session_id": "sess_def456", "status": {"state": "completed"}}
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.list_sessions()
            
            assert isinstance(result, list)
            assert len(result) == 2
    
    async def test_get_queue_status_success(self, api_client):
        """Test successful queue status retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"session_id": "sess_queued1", "status": {"state": "queued"}}
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.get_queue_status()
            
            assert isinstance(result, list)
    
    async def test_get_system_info_success(self, api_client):
        """Test successful system info retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cpu_usage_percent": 45.2,
            "memory_used_mb": 8192,
            "memory_total_mb": 16384,
            "gpu_available": True
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.get_system_info()
            
            assert result["cpu_usage_percent"] == 45.2
            assert result["gpu_available"] is True
    
    async def test_list_models_success(self, api_client):
        """Test successful models list retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "model1", "size_mb": 102.4},
            {"name": "model2", "size_mb": 256.8}
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.list_models()
            
            assert isinstance(result, list)
            assert len(result) == 2
    
    async def test_list_datasets_success(self, api_client):
        """Test successful datasets list retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "dataset1", "size_mb": 163.0, "sample_count": 60000},
            {"name": "dataset2", "size_mb": 89.5, "sample_count": 10000}
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            result = await api_client.list_datasets()
            
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["sample_count"] == 60000


@pytest.mark.asyncio
class TestAPIClientErrorHandling:
    """Test error handling in API client."""
    
    async def test_connection_error_handling(self, api_client):
        """Test bot handles connection errors gracefully with Russian error message."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            
            with pytest.raises(Exception) as exc_info:
                await api_client.start_training({})
            
            # Verify error message is in Russian
            assert "Не удалось подключиться к API серверу" in str(exc_info.value)
    
    async def test_timeout_error_handling(self, api_client):
        """Test bot handles timeout errors gracefully with Russian error message."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                side_effect=httpx.TimeoutException("Request timeout")
            )
            
            with pytest.raises(Exception) as exc_info:
                await api_client.get_training_status("sess_abc123")
            
            # Verify error message is in Russian
            assert "Превышено время ожидания" in str(exc_info.value)
    
    async def test_404_error_handling(self, api_client):
        """Test bot handles 404 errors gracefully with Russian error message."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Session not found"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            with pytest.raises(Exception) as exc_info:
                await api_client.get_training_status("nonexistent_session")
            
            # Verify error message is in Russian
            assert "Не найдено" in str(exc_info.value)
    
    async def test_401_auth_error_handling(self, api_client):
        """Test bot handles authentication errors gracefully with Russian error message."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Invalid API key"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            with pytest.raises(Exception) as exc_info:
                await api_client.start_training({})
            
            # Verify error message is in Russian
            assert "Ошибка аутентификации" in str(exc_info.value)
    
    async def test_400_validation_error_handling(self, api_client):
        """Test bot handles validation errors gracefully with Russian error message."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Invalid learning rate"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            with pytest.raises(Exception) as exc_info:
                await api_client.start_training({})
            
            # Verify error message is in Russian
            assert "Неверные параметры" in str(exc_info.value)
    
    async def test_500_server_error_handling(self, api_client):
        """Test bot handles server errors gracefully with Russian error message."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal server error"}
        mock_response.text = '{"detail": "Internal server error"}'
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            with pytest.raises(Exception) as exc_info:
                await api_client.get_system_info()
            
            # Verify error message is in Russian
            assert "Ошибка API" in str(exc_info.value)
    
    async def test_health_check_failure(self, api_client):
        """Test health check returns False on failure."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            
            result = await api_client.health_check()
            
            assert result is False
    
    async def test_generic_exception_handling(self, api_client):
        """Test bot handles generic exceptions gracefully."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                side_effect=Exception("Unexpected error")
            )
            
            with pytest.raises(Exception) as exc_info:
                await api_client.list_datasets()
            
            # Verify error message is in Russian
            assert "Ошибка связи с API" in str(exc_info.value)


@pytest.mark.asyncio
class TestAPIClientRequestFormat:
    """Test that API client sends requests in correct format."""
    
    async def test_authentication_header_included(self, api_client):
        """Test that authentication header is included in requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_request = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.request = mock_request
            
            await api_client.health_check()
            
            # Verify Authorization header was sent
            call_kwargs = mock_request.call_args[1]
            assert "headers" in call_kwargs
            assert "Authorization" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Authorization"] == "Bearer test_key_123"
    
    async def test_content_type_header_included(self, api_client):
        """Test that Content-Type header is included in requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"session_id": "sess_abc123"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_request = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.request = mock_request
            
            await api_client.start_training({})
            
            # Verify Content-Type header was sent
            call_kwargs = mock_request.call_args[1]
            assert "headers" in call_kwargs
            assert "Content-Type" in call_kwargs["headers"]
            assert call_kwargs["headers"]["Content-Type"] == "application/json"
    
    async def test_correct_endpoint_url_construction(self, api_client):
        """Test that endpoint URLs are constructed correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"state": "running"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_request = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.request = mock_request
            
            await api_client.get_training_status("sess_abc123")
            
            # Verify correct URL was called
            call_args = mock_request.call_args[1]
            assert call_args["url"] == "http://localhost:8000/api/v1/training/sess_abc123/status"
    
    async def test_timeout_configuration(self, api_client):
        """Test that timeout is configured correctly."""
        assert api_client.timeout == 30.0


@pytest.mark.asyncio
class TestBotCommandHandlers:
    """Test bot command handlers interact correctly with API."""
    
    async def test_training_command_creates_session(self):
        """Test that training command creates a training session via API."""
        # This test verifies the bot command handler calls the API correctly
        # In a real scenario, this would test the actual bot command handler
        
        api_client = APIClient(base_url="http://localhost:8000", api_key="test_key")
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "session_id": "sess_new123",
            "status": "running",
            "message": "Training started"
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            config = {
                "model_architecture": "custom_ai_model",
                "dataset_name": "test_dataset",
                "learning_rate": 0.0001,
                "batch_size": 16,
                "epochs": 10,
                "optimizer": "adam",
                "loss_function": "cross_entropy"
            }
            
            result = await api_client.start_training(config)
            
            assert result["session_id"] == "sess_new123"
            assert result["status"] == "running"
    
    async def test_status_command_retrieves_info(self):
        """Test that status command retrieves system and session info."""
        api_client = APIClient(base_url="http://localhost:8000", api_key="test_key")
        
        # Mock list_sessions response
        sessions_response = Mock()
        sessions_response.status_code = 200
        sessions_response.json.return_value = [
            {"session_id": "sess1", "status": {"state": "running"}},
            {"session_id": "sess2", "status": {"state": "completed"}}
        ]
        
        # Mock system_info response
        system_response = Mock()
        system_response.status_code = 200
        system_response.json.return_value = {
            "cpu_usage_percent": 45.2,
            "memory_percent": 50.0,
            "memory_used_mb": 8192,
            "memory_total_mb": 16384,
            "gpu_available": True,
            "gpu_usage_percent": 78.5,
            "disk_free_gb": 250.5,
            "disk_total_gb": 500.0
        }
        
        # Mock models response
        models_response = Mock()
        models_response.status_code = 200
        models_response.json.return_value = [
            {"name": "model1", "size_mb": 102.4}
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_request = AsyncMock()
            mock_request.side_effect = [sessions_response, system_response, models_response]
            mock_client.return_value.__aenter__.return_value.request = mock_request
            
            sessions = await api_client.list_sessions()
            system_info = await api_client.get_system_info()
            models = await api_client.list_models()
            
            assert len(sessions) == 2
            assert system_info["cpu_usage_percent"] == 45.2
            assert len(models) == 1
    
    async def test_dataset_list_command(self):
        """Test that dataset list command retrieves datasets."""
        api_client = APIClient(base_url="http://localhost:8000", api_key="test_key")
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "name": "dataset1",
                "size_mb": 163.0,
                "sample_count": 60000,
                "format": "csv"
            },
            {
                "name": "dataset2",
                "size_mb": 89.5,
                "sample_count": 10000,
                "format": "images"
            }
        ]
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            datasets = await api_client.list_datasets()
            
            assert len(datasets) == 2
            assert datasets[0]["name"] == "dataset1"
            assert datasets[0]["sample_count"] == 60000
            assert datasets[1]["format"] == "images"


@pytest.mark.asyncio
class TestBotErrorMessageFormatting:
    """Test that bot formats error messages correctly in Russian."""
    
    async def test_connection_error_message_in_russian(self, api_client):
        """Test connection error message is in Russian."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            
            try:
                await api_client.start_training({})
                assert False, "Should have raised exception"
            except Exception as e:
                error_msg = str(e)
                # Verify Russian error message
                assert "Не удалось подключиться к API серверу" in error_msg
                assert "Проверьте, что сервер запущен" in error_msg
    
    async def test_timeout_error_message_in_russian(self, api_client):
        """Test timeout error message is in Russian."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )
            
            try:
                await api_client.get_training_status("sess_abc123")
                assert False, "Should have raised exception"
            except Exception as e:
                error_msg = str(e)
                # Verify Russian error message
                assert "Превышено время ожидания ответа от API сервера" in error_msg
    
    async def test_not_found_error_message_in_russian(self, api_client):
        """Test 404 error message is in Russian."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Resource not found"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            try:
                await api_client.get_training_metrics("nonexistent")
                assert False, "Should have raised exception"
            except Exception as e:
                error_msg = str(e)
                # Verify Russian error message
                assert "Не найдено" in error_msg
    
    async def test_auth_error_message_in_russian(self, api_client):
        """Test authentication error message is in Russian."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Unauthorized"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            try:
                await api_client.list_models()
                assert False, "Should have raised exception"
            except Exception as e:
                error_msg = str(e)
                # Verify Russian error message
                assert "Ошибка аутентификации" in error_msg
                assert "неверный API ключ" in error_msg
    
    async def test_validation_error_message_in_russian(self, api_client):
        """Test validation error message is in Russian."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Invalid parameters"}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            try:
                await api_client.start_training({})
                assert False, "Should have raised exception"
            except Exception as e:
                error_msg = str(e)
                # Verify Russian error message
                assert "Неверные параметры" in error_msg
