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


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    vocab_size: int
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    hidden_dim: int = 1024  # FFN hidden dimension
    max_seq_length: int = 512
    dropout: float = 0.1
    use_memory_efficient: bool = False  # Use memory-efficient attention for long sequences
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Check ranges first
        if not (2 <= self.num_layers <= 12):
            raise ValueError(
                f"num_layers must be between 2 and 12, got {self.num_layers}"
            )
        if not (2 <= self.num_heads <= 16):
            raise ValueError(
                f"num_heads must be between 2 and 16, got {self.num_heads}"
            )
        if not (128 <= self.embedding_dim <= 1024):
            raise ValueError(
                f"embedding_dim must be between 128 and 1024, got {self.embedding_dim}"
            )
        if not (1000 <= self.vocab_size <= 50000):
            raise ValueError(
                f"vocab_size must be between 1000 and 50000, got {self.vocab_size}"
            )
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(
                f"dropout must be between 0.0 and 1.0, got {self.dropout}"
            )
        
        # Check compatibility
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
    
    @classmethod
    def small(cls, vocab_size: int) -> 'TransformerConfig':
        """Small model configuration (fast training)."""
        return cls(
            vocab_size=vocab_size,
            embedding_dim=128,
            num_heads=4,
            num_layers=4,
            hidden_dim=512,
            max_seq_length=256
        )
    
    @classmethod
    def medium(cls, vocab_size: int) -> 'TransformerConfig':
        """Medium model configuration (balanced)."""
        return cls(
            vocab_size=vocab_size,
            embedding_dim=256,
            num_heads=8,
            num_layers=6,
            hidden_dim=1024,
            max_seq_length=512
        )
    
    @classmethod
    def large(cls, vocab_size: int) -> 'TransformerConfig':
        """Large model configuration (best quality)."""
        return cls(
            vocab_size=vocab_size,
            embedding_dim=512,
            num_heads=16,
            num_layers=12,
            hidden_dim=2048,
            max_seq_length=1024
        )
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'TransformerConfig':
        """Load configuration from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TransformerTrainingConfig:
    """Configuration for Transformer training."""
    epochs: int = 50
    learning_rate: float = 0.0001
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Generation parameters
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
