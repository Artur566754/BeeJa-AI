"""Обучение улучшенной модели."""
import os
from src.model import CustomAIModel
from src.dataset_loader import DatasetLoader
from src.training_pipeline import TrainingPipeline

print("=" * 60)
print("Обучение улучшенной модели")
print("=" * 60)

# Создаем БОЛЬШУЮ модель
print("\n📊 Создание модели...")
print("   Параметры:")
print("   - Embedding: 256 (было 128)")
print("   - Hidden: 512 (было 256)")
print("   - Layers: 3 (было 2)")

model = CustomAIModel(
    vocab_size=100,
    embedding_dim=256,    # Увеличили в 2 раза
    hidden_dim=512,       # Увеличили в 2 раза
    num_layers=3          # Добавили слой
)

total_params = sum(p.numel() for p in model.parameters())
print(f"   ✓ Всего параметров: {total_params:,}")

# Загружаем данные
print("\n📚 Загрузка данных...")
loader = DatasetLoader("datasets")
text, errors = loader.load_all_datasets()
print(f"   ✓ Загружено {len(text):,} символов")

# Обучаем
print("\n🎓 Начинаю обучение...")
print("   Это займет больше времени, но результат будет лучше!")
print("   Параметры:")
print("   - Epochs: 100 (больше эпох)")
print("   - Learning Rate: 0.001")
print("   - Batch Size: 16 (меньше батч для лучшего обучения)")

pipeline = TrainingPipeline(model, loader)

try:
    pipeline.train(epochs=100, learning_rate=0.001, batch_size=16)
    
    # Сохраняем
    pipeline.model.save_weights("models/ai_model_better.pth")
    print("\n✓ Модель сохранена: models/ai_model_better.pth")
    
    # Тестируем
    print("\n🧪 Тестирование модели...")
    test_prompts = ["Jamsaide", "создатель", "BeeBoo", "привет"]
    
    for prompt in test_prompts:
        response = pipeline.model.generate(prompt, max_length=80, temperature=0.7)
        print(f"\n   Prompt: {prompt}")
        print(f"   Response: {response[:150]}...")
    
except Exception as e:
    print(f"\n❌ Ошибка: {e}")

print("\n" + "=" * 60)
print("Для использования этой модели, переименуй файл:")
print("  models/ai_model_better.pth → models/ai_model.pth")
print("=" * 60)
