"""Configuration classes for the AI model."""
from dataclasses import dataclass, asdict
import json
from typing import Dict


@dataclass
class ModelConfig:
    """Конфигурация модели"""
    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    seq_length: int = 100
    
    def save(self, filepath: str) -> None:
        """Сохранение конфигурации в JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Загрузка конфигурации из JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 32
    temperature: float = 1.0
