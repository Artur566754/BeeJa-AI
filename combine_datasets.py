"""
Скрипт для объединения всех датасетов в один файл для обучения
"""
import os
from pathlib import Path

def combine_datasets():
    """Объединяет все .txt файлы из папки datasets в один файл"""
    datasets_dir = Path("datasets")
    output_file = datasets_dir / "all_combined.txt"
    
    # Список всех датасетов
    dataset_files = [
        "kniga_AI_Model.txt",
        "kniga_matematika.txt",
        "kniga_psihologiy.txt",
        "kniga_python.txt",
        "training_data.txt",
        "monolog.txt",
        "monolog2.txt",
        "hello_my_bro.txt",
        "grubo.txt",
        "initial_dataset.txt"
    ]
    
    print("🔄 Объединение датасетов...")
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in dataset_files:
            filepath = datasets_dir / filename
            
            if filepath.exists():
                print(f"  ✓ Добавляю {filename}...")
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    lines = len(content.split('\n'))
                    total_lines += lines
                    
                    # Добавляем содержимое файла
                    outfile.write(content)
                    # Добавляем разделитель между файлами
                    outfile.write("\n\n")
            else:
                print(f"  ⚠ Файл {filename} не найден, пропускаю...")
    
    print(f"\n✅ Готово! Создан файл: {output_file}")
    print(f"📊 Всего строк: {total_lines}")
    print(f"📁 Размер: {output_file.stat().st_size / 1024:.2f} KB")
    
    return str(output_file)

if __name__ == "__main__":
    combined_file = combine_datasets()
    print(f"\n🚀 Теперь запусти обучение командой:")
    print(f"python main.py --train --model-type transformer --epochs 500 --batch-size 64 --lr 0.0005")
