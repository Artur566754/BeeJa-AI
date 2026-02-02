"""Telegram бот для управления AI моделью через Server Management API."""
import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import httpx

# Загружаем переменные окружения из .env файла
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Добавляем путь к AI модели (для локального чата, если API недоступен)
# Для Docker
sys.path.append('/app')
# Для локального запуска
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from src.model import CustomAIModel
    from src.dataset_loader import DatasetLoader
    from src.chat_interface import ChatInterface
    LOCAL_CHAT_AVAILABLE = True
except ImportError:
    LOCAL_CHAT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Local chat modules not available, will use API only")

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ID администратора
ADMIN_ID = int(os.getenv('ADMIN_ID', '843366380'))

# API Configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')
API_KEY = os.getenv('API_KEY', '')

# Состояния для ConversationHandler
WAITING_EPOCHS = 1

# Глобальная переменная для модели (локальный чат, если API недоступен)
ai_model = None
chat_interface = None


class APIClient:
    """Client for interacting with the Server Management API."""
    
    def __init__(self, base_url: str, api_key: str):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.timeout = 30.0
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments for httpx request
            
        Returns:
            Response JSON data
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    **kwargs
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    error_data = response.json()
                    raise Exception(f"Не найдено: {error_data.get('detail', 'Resource not found')}")
                elif response.status_code == 401:
                    raise Exception("Ошибка аутентификации: неверный API ключ")
                elif response.status_code == 400:
                    error_data = response.json()
                    raise Exception(f"Неверные параметры: {error_data.get('detail', 'Bad request')}")
                else:
                    error_data = response.json() if response.text else {}
                    raise Exception(f"Ошибка API ({response.status_code}): {error_data.get('detail', 'Unknown error')}")
                    
        except httpx.TimeoutException:
            raise Exception("Превышено время ожидания ответа от API сервера")
        except httpx.ConnectError:
            raise Exception("Не удалось подключиться к API серверу. Проверьте, что сервер запущен.")
        except Exception as e:
            if "Ошибка" in str(e) or "Не найдено" in str(e):
                raise
            raise Exception(f"Ошибка связи с API: {str(e)}")
    
    async def health_check(self) -> bool:
        """
        Check if API server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = await self._request('GET', '/health')
            return response.get('status') == 'healthy'
        except:
            return False
    
    async def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a new training session.
        
        Args:
            config: Training configuration
            
        Returns:
            Response with session_id and status
        """
        return await self._request('POST', '/api/v1/training/start', json=config)
    
    async def stop_training(self, session_id: str) -> Dict[str, Any]:
        """
        Stop a training session.
        
        Args:
            session_id: ID of the session to stop
            
        Returns:
            Response with success message
        """
        return await self._request('POST', f'/api/v1/training/{session_id}/stop')
    
    async def get_training_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get status of a training session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session status information
        """
        return await self._request('GET', f'/api/v1/training/{session_id}/status')
    
    async def get_training_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get current metrics for a training session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Current metrics (loss, accuracy, etc.)
        """
        return await self._request('GET', f'/api/v1/training/{session_id}/metrics')
    
    async def get_training_history(self, session_id: str) -> list:
        """
        Get metrics history for a training session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of metrics for all epochs
        """
        return await self._request('GET', f'/api/v1/training/{session_id}/history')
    
    async def get_training_logs(self, session_id: str, limit: int = 50) -> list:
        """
        Get logs for a training session.
        
        Args:
            session_id: ID of the session
            limit: Maximum number of log entries
            
        Returns:
            List of log entries
        """
        return await self._request('GET', f'/api/v1/training/{session_id}/logs', params={'limit': limit})
    
    async def list_sessions(self) -> list:
        """
        List all training sessions.
        
        Returns:
            List of all sessions
        """
        return await self._request('GET', '/api/v1/training/sessions')
    
    async def get_queue_status(self) -> list:
        """
        Get training queue status.
        
        Returns:
            List of queued sessions
        """
        return await self._request('GET', '/api/v1/training/queue')
    
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get system resource information.
        
        Returns:
            System info (CPU, memory, GPU, disk)
        """
        return await self._request('GET', '/api/v1/system/info')
    
    async def list_models(self) -> list:
        """
        List all models in the registry.
        
        Returns:
            List of models
        """
        return await self._request('GET', '/api/v1/models')
    
    async def list_datasets(self) -> list:
        """
        List all datasets in the registry.
        
        Returns:
            List of datasets
        """
        return await self._request('GET', '/api/v1/datasets')


# Initialize API client
api_client = APIClient(API_URL, API_KEY) if API_KEY else None


def is_admin(user_id: int) -> bool:
    """Проверка, является ли пользователь администратором."""
    return user_id == ADMIN_ID


def load_ai_model():
    """Загрузка AI модели для локального чата (fallback если API недоступен)."""
    global ai_model, chat_interface
    
    if not LOCAL_CHAT_AVAILABLE:
        logger.warning("Local chat modules not available")
        return False
    
    try:
        model_path = "/app/models/ai_model.pth"
        
        if os.path.exists(model_path):
            logger.info("Загрузка существующей модели...")
            import torch
            checkpoint = torch.load(model_path, weights_only=False)
            vocab_size = checkpoint.get('vocab_size', 100)
            
            ai_model = CustomAIModel(
                vocab_size=vocab_size,
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2
            )
            ai_model.load_weights(model_path)
            chat_interface = ChatInterface(ai_model)
            logger.info("✓ Модель загружена успешно")
            return True
        else:
            logger.warning("Модель не найдена. Требуется обучение.")
            return False
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    if is_admin(user_id):
        # Админская клавиатура
        keyboard = [
            [KeyboardButton("💬 Чат с AI"), KeyboardButton("🎓 Обучить модель")],
            [KeyboardButton("📊 Статус модели"), KeyboardButton("📁 Список датасетов")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            f"👋 Привет, {user_name}!\n\n"
            f"🔑 Вы вошли как администратор.\n\n"
            f"Доступные функции:\n"
            f"💬 Чат с AI - общение с моделью\n"
            f"🎓 Обучить модель - дообучение на новых данных\n"
            f"📊 Статус модели - информация о модели\n"
            f"📁 Список датасетов - просмотр файлов для обучения",
            reply_markup=reply_markup
        )
    else:
        # Обычная клавиатура для пользователей
        keyboard = [[KeyboardButton("💬 Чат с AI")]]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            f"👋 Привет, {user_name}!\n\n"
            f"Я AI бот, созданный Jamsaide.\n"
            f"Нажми '💬 Чат с AI' чтобы начать общение!",
            reply_markup=reply_markup
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений."""
    global chat_interface
    
    user_id = update.effective_user.id
    text = update.message.text
    
    # Обработка кнопок
    if text == "💬 Чат с AI":
        if chat_interface is None:
            await update.message.reply_text(
                "⚠️ Модель не загружена. Администратор должен сначала обучить модель."
            )
            return
        
        await update.message.reply_text(
            "💬 Режим чата активирован!\n"
            "Просто пиши мне сообщения, и я буду отвечать.\n\n"
            "Для выхода нажми кнопку меню."
        )
        return
    
    elif text == "🎓 Обучить модель":
        if not is_admin(user_id):
            await update.message.reply_text("⛔ Эта функция доступна только администратору.")
            return
        
        if not api_client:
            await update.message.reply_text("⚠️ API клиент не настроен. Установите API_KEY в .env файле.")
            return
        
        # Check if there are active sessions
        try:
            sessions = await api_client.list_sessions()
            active_sessions = [s for s in sessions if s.get('status', {}).get('state') in ['running', 'queued']]
            
            if active_sessions:
                await update.message.reply_text(
                    "⚠️ Обучение уже идет! Подождите завершения.\n"
                    f"Активных сессий: {len(active_sessions)}"
                )
                return
        except Exception as e:
            logger.error(f"Error checking sessions: {e}")
            await update.message.reply_text(f"❌ Ошибка проверки сессий: {str(e)}")
            return
        
        await update.message.reply_text(
            "🎓 Начинаем процесс обучения.\n\n"
            "Сколько эпох обучения использовать?\n"
            "Рекомендуется: 10-30 для дообучения, 50-100 для первого обучения.\n\n"
            "Введите число:"
        )
        context.user_data['waiting_for_epochs'] = True
        return
    
    elif text == "📊 Статус модели":
        if not is_admin(user_id):
            await update.message.reply_text("⛔ Эта функция доступна только администратору.")
            return
        
        if not api_client:
            await update.message.reply_text("⚠️ API клиент не настроен. Установите API_KEY в .env файле.")
            return
        
        try:
            # Get all sessions
            sessions = await api_client.list_sessions()
            
            # Get system info
            system_info = await api_client.get_system_info()
            
            # Get models
            models = await api_client.list_models()
            
            # Format status message
            status_parts = ["📊 Статус системы:\n"]
            
            # System resources
            status_parts.append(f"💻 CPU: {system_info.get('cpu_usage_percent', 0):.1f}%")
            status_parts.append(f"🧠 RAM: {system_info.get('memory_percent', 0):.1f}% ({system_info.get('memory_used_mb', 0):.0f}/{system_info.get('memory_total_mb', 0):.0f} MB)")
            
            if system_info.get('gpu_available'):
                status_parts.append(f"🎮 GPU: {system_info.get('gpu_usage_percent', 0):.1f}% ({system_info.get('gpu_memory_used_mb', 0):.0f}/{system_info.get('gpu_memory_total_mb', 0):.0f} MB)")
            
            status_parts.append(f"💾 Диск: {system_info.get('disk_free_gb', 0):.1f}/{system_info.get('disk_total_gb', 0):.1f} GB свободно")
            
            # Training sessions
            active_sessions = [s for s in sessions if s.get('status', {}).get('state') in ['running', 'queued']]
            completed_sessions = [s for s in sessions if s.get('status', {}).get('state') == 'completed']
            
            status_parts.append(f"\n🎓 Сессии обучения:")
            status_parts.append(f"  Активных: {len(active_sessions)}")
            status_parts.append(f"  Завершено: {len(completed_sessions)}")
            status_parts.append(f"  Всего: {len(sessions)}")
            
            # Models
            status_parts.append(f"\n📦 Моделей в реестре: {len(models)}")
            if models:
                total_size = sum(m.get('size_mb', 0) for m in models)
                status_parts.append(f"  Общий размер: {total_size:.1f} MB")
            
            await update.message.reply_text("\n".join(status_parts))
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            await update.message.reply_text(f"❌ Ошибка получения статуса: {str(e)}")
        
        return
    
    elif text == "📁 Список датасетов":
        if not is_admin(user_id):
            await update.message.reply_text("⛔ Эта функция доступна только администратору.")
            return
        
        if not api_client:
            await update.message.reply_text("⚠️ API клиент не настроен. Установите API_KEY в .env файле.")
            return
        
        try:
            datasets = await api_client.list_datasets()
            
            if datasets:
                dataset_list = []
                for ds in datasets:
                    name = ds.get('name', 'Unknown')
                    size = ds.get('size_mb', 0)
                    samples = ds.get('sample_count', 0)
                    format_type = ds.get('format', 'unknown')
                    dataset_list.append(f"📄 {name}\n   Размер: {size:.1f} MB, Образцов: {samples}, Формат: {format_type}")
                
                await update.message.reply_text(f"📁 Датасеты:\n\n" + "\n\n".join(dataset_list))
            else:
                await update.message.reply_text("📁 Датасеты не найдены")
                
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            await update.message.reply_text(f"❌ Ошибка получения списка датасетов: {str(e)}")
        
        return
    
    # Обработка ввода количества эпох
    if context.user_data.get('waiting_for_epochs'):
        try:
            epochs = int(text)
            if epochs < 1 or epochs > 200:
                await update.message.reply_text("⚠️ Введите число от 1 до 200")
                return
            
            context.user_data['waiting_for_epochs'] = False
            await start_training(update, context, epochs)
            return
        except ValueError:
            await update.message.reply_text("⚠️ Пожалуйста, введите число")
            return
    
    # Обычный чат с AI
    if chat_interface is None:
        await update.message.reply_text(
            "⚠️ Модель не загружена. Администратор должен сначала обучить модель."
        )
        return
    
    # Генерация ответа
    try:
        await update.message.reply_text("🤔 Думаю...")
        response = chat_interface.process_message(text)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Ошибка генерации ответа: {e}")
        await update.message.reply_text("❌ Ошибка при генерации ответа")


async def start_training(update: Update, context: ContextTypes.DEFAULT_TYPE, epochs: int):
    """Запуск обучения модели через API."""
    
    if not api_client:
        await update.message.reply_text("⚠️ API клиент не настроен. Установите API_KEY в .env файле.")
        return
    
    await update.message.reply_text(
        f"🎓 Начинаю обучение на {epochs} эпох...\n"
        f"⏳ Это может занять некоторое время.\n"
        f"Я буду отправлять обновления о прогрессе!"
    )
    
    try:
        # Get available datasets
        datasets = await api_client.list_datasets()
        
        if not datasets:
            await update.message.reply_text("❌ Датасеты не найдены. Загрузите датасет через API.")
            return
        
        # Use the first available dataset
        dataset_name = datasets[0].get('name', 'default')
        
        # Start training session
        config = {
            "model_architecture": "custom_ai_model",
            "dataset_name": dataset_name,
            "learning_rate": 0.0001,
            "batch_size": 16,
            "epochs": epochs,
            "optimizer": "adam",
            "loss_function": "cross_entropy"
        }
        
        response = await api_client.start_training(config)
        session_id = response.get('session_id')
        status = response.get('status', 'unknown')
        
        await update.message.reply_text(
            f"✅ Сессия обучения создана!\n"
            f"🆔 ID: {session_id}\n"
            f"📊 Статус: {status}\n\n"
            f"Используйте /status_{session_id} для проверки прогресса"
        )
        
        # Store session ID in context for monitoring
        if 'training_sessions' not in context.bot_data:
            context.bot_data['training_sessions'] = []
        context.bot_data['training_sessions'].append(session_id)
        
        # Start monitoring task
        asyncio.create_task(monitor_training(update, context, session_id))
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        await update.message.reply_text(f"❌ Ошибка запуска обучения: {str(e)}")


async def monitor_training(update: Update, context: ContextTypes.DEFAULT_TYPE, session_id: str):
    """Мониторинг прогресса обучения."""
    
    if not api_client:
        return
    
    last_epoch = 0
    check_interval = 30  # seconds
    
    try:
        while True:
            await asyncio.sleep(check_interval)
            
            try:
                # Get current status
                status = await api_client.get_training_status(session_id)
                state = status.get('state')
                current_epoch = status.get('current_epoch')
                total_epochs = status.get('total_epochs')
                
                # If training completed or failed, send final message
                if state == 'completed':
                    # Get final metrics
                    try:
                        metrics = await api_client.get_training_metrics(session_id)
                        await update.message.reply_text(
                            f"✅ Обучение завершено успешно!\n"
                            f"🆔 Сессия: {session_id}\n"
                            f"🎓 Эпох: {total_epochs}\n"
                            f"📉 Финальная потеря: {metrics.get('loss', 'N/A'):.4f}\n"
                            f"🎯 Точность: {metrics.get('accuracy', 'N/A'):.2%}\n"
                            f"🔄 Модель обновлена и готова к работе."
                        )
                    except:
                        await update.message.reply_text(
                            f"✅ Обучение завершено успешно!\n"
                            f"🆔 Сессия: {session_id}\n"
                            f"🎓 Эпох: {total_epochs}"
                        )
                    break
                    
                elif state == 'failed':
                    error_msg = status.get('error_message', 'Unknown error')
                    await update.message.reply_text(
                        f"❌ Обучение завершилось с ошибкой!\n"
                        f"🆔 Сессия: {session_id}\n"
                        f"⚠️ Ошибка: {error_msg}"
                    )
                    break
                    
                elif state == 'stopped':
                    await update.message.reply_text(
                        f"⏹️ Обучение остановлено\n"
                        f"🆔 Сессия: {session_id}\n"
                        f"🎓 Завершено эпох: {current_epoch}/{total_epochs}"
                    )
                    break
                
                # Send progress update if epoch changed
                if current_epoch and current_epoch > last_epoch:
                    last_epoch = current_epoch
                    
                    try:
                        metrics = await api_client.get_training_metrics(session_id)
                        await update.message.reply_text(
                            f"📊 Прогресс обучения\n"
                            f"🆔 Сессия: {session_id}\n"
                            f"🎓 Эпоха: {current_epoch}/{total_epochs}\n"
                            f"📉 Потеря: {metrics.get('loss', 'N/A'):.4f}\n"
                            f"🎯 Точность: {metrics.get('accuracy', 'N/A'):.2%}"
                        )
                    except:
                        await update.message.reply_text(
                            f"📊 Прогресс: эпоха {current_epoch}/{total_epochs}"
                        )
                
            except Exception as e:
                logger.error(f"Error monitoring training: {e}")
                # Continue monitoring despite errors
                
    except asyncio.CancelledError:
        logger.info(f"Monitoring cancelled for session {session_id}")
    except Exception as e:
        logger.error(f"Fatal error in monitoring: {e}")


def main():
    """Запуск бота."""
    # Check API configuration
    if not API_KEY:
        logger.warning("API_KEY not set! Bot will have limited functionality.")
        logger.warning("Set API_KEY in telegram_bot/.env file to enable training features.")
    else:
        logger.info(f"API client configured for {API_URL}")
    
    # Try to load local model for chat (fallback)
    if LOCAL_CHAT_AVAILABLE:
        load_ai_model()
    
    # Получаем токен из переменной окружения
    token = os.getenv('BOT_TOKEN')
    if not token:
        logger.error("BOT_TOKEN не установлен!")
        return
    
    # Создаем приложение
    application = Application.builder().token(token).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    logger.info("🤖 Бот запущен!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
