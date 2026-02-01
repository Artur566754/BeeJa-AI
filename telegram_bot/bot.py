"""Telegram бот для управления AI моделью."""
import os
import sys
import logging
import subprocess
import threading
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler

# Загружаем переменные окружения из .env файла
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Добавляем путь к AI модели
# Для Docker
sys.path.append('/app')
# Для локального запуска
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.model import CustomAIModel
from src.dataset_loader import DatasetLoader
from src.chat_interface import ChatInterface

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ID администратора
ADMIN_ID = 843366380

# Состояния для ConversationHandler
WAITING_EPOCHS = 1

# Глобальная переменная для модели
ai_model = None
chat_interface = None
training_in_progress = False


def is_admin(user_id: int) -> bool:
    """Проверка, является ли пользователь администратором."""
    return user_id == ADMIN_ID


def load_ai_model():
    """Загрузка AI модели."""
    global ai_model, chat_interface
    
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
    global chat_interface, training_in_progress
    
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
        
        if training_in_progress:
            await update.message.reply_text("⚠️ Обучение уже идет! Подождите завершения.")
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
        
        model_path = "/app/models/ai_model.pth"
        if os.path.exists(model_path):
            import torch
            checkpoint = torch.load(model_path, weights_only=False)
            vocab_size = checkpoint.get('vocab_size', 'N/A')
            
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            status = (
                f"📊 Статус модели:\n\n"
                f"✅ Модель обучена\n"
                f"📦 Размер словаря: {vocab_size}\n"
                f"💾 Размер файла: {file_size:.2f} MB\n"
                f"📍 Путь: {model_path}"
            )
        else:
            status = "❌ Модель не обучена"
        
        await update.message.reply_text(status)
        return
    
    elif text == "📁 Список датасетов":
        if not is_admin(user_id):
            await update.message.reply_text("⛔ Эта функция доступна только администратору.")
            return
        
        datasets_dir = "/app/datasets"
        if os.path.exists(datasets_dir):
            files = os.listdir(datasets_dir)
            if files:
                file_list = "\n".join([f"📄 {f}" for f in files])
                await update.message.reply_text(f"📁 Датасеты:\n\n{file_list}")
            else:
                await update.message.reply_text("📁 Папка datasets пуста")
        else:
            await update.message.reply_text("❌ Папка datasets не найдена")
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
    """Запуск обучения модели."""
    global training_in_progress
    
    training_in_progress = True
    
    await update.message.reply_text(
        f"🎓 Начинаю обучение на {epochs} эпох...\n"
        f"⏳ Это может занять некоторое время.\n"
        f"Я сообщу, когда закончу!"
    )
    
    def train():
        global ai_model, chat_interface, training_in_progress
        
        try:
            # Запускаем обучение
            cmd = [
                "python", "/app/main.py",
                "--train",
                "--epochs", str(epochs),
                "--lr", "0.0001",
                "--batch-size", "16"
            ]
            
            logger.info(f"Запуск команды: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Перезагружаем модель
                load_ai_model()
                
                # Отправляем сообщение об успехе
                context.application.create_task(
                    update.message.reply_text(
                        f"✅ Обучение завершено успешно!\n"
                        f"🎓 Эпох: {epochs}\n"
                        f"🔄 Модель обновлена и готова к работе."
                    )
                )
            else:
                context.application.create_task(
                    update.message.reply_text(
                        f"❌ Ошибка при обучении:\n{result.stderr[:500]}"
                    )
                )
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            context.application.create_task(
                update.message.reply_text(f"❌ Ошибка: {str(e)}")
            )
        finally:
            training_in_progress = False
    
    # Запускаем обучение в отдельном потоке
    thread = threading.Thread(target=train)
    thread.start()


def main():
    """Запуск бота."""
    # Загружаем модель при старте
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
