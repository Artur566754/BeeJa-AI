FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    python-telegram-bot==20.7 \
    torch>=2.0.0 \
    PyPDF2>=3.0.0 \
    python-docx>=0.8.11 \
    hypothesis>=6.0.0

# Копируем весь проект
COPY src/ /app/src/
COPY main.py /app/main.py
COPY telegram_bot/bot.py /app/bot.py

# Создаем необходимые директории
RUN mkdir -p /app/datasets /app/models /app/bot_data

# Запускаем бота
CMD ["python", "/app/bot.py"]
