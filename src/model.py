"""Custom AI Model with LSTM architecture."""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from src.vocabulary import Vocabulary


class CustomAIModel(nn.Module):
    """LSTM-based text generation model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int):
        """
        Args:
            vocab_size: Размер словаря (количество уникальных символов)
            embedding_dim: Размерность эмбеддингов (например, 128)
            hidden_dim: Размерность скрытого слоя LSTM (например, 256)
            num_layers: Количество слоев LSTM (например, 2)
        """
        super(CustomAIModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.vocabulary: Optional[Vocabulary] = None
    
    def forward(self, input_sequence: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass через сеть
        
        Args:
            input_sequence: Входная последовательность символов [batch_size, seq_len]
            hidden_state: Скрытое состояние LSTM (h, c)
            
        Returns:
            output: Предсказания для следующего символа [batch_size, seq_len, vocab_size]
            hidden_state: Обновленное скрытое состояние (h, c)
        """
        # Embedding
        embedded = self.embedding(input_sequence)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM forward pass
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(embedded)
        else:
            lstm_out, hidden_state = self.lstm(embedded, hidden_state)
        
        # Output layer
        output = self.fc(lstm_out)  # [batch_size, seq_len, vocab_size]
        
        return output, hidden_state
    
    def generate(self, seed_text: str, max_length: int, temperature: float = 1.0) -> str:
        """
        Генерация текста на основе начальной строки
        
        Args:
            seed_text: Начальный текст для генерации
            max_length: Максимальная длина генерируемого текста
            temperature: Параметр для контроля случайности (0.5-1.5)
            
        Returns:
            Сгенерированный текст
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not set. Call set_vocabulary() first.")
        
        self.eval()
        
        with torch.no_grad():
            # Encode seed text
            input_indices = self.vocabulary.encode(seed_text)
            if not input_indices:
                input_indices = [0]  # Start with first character if seed is empty
            
            generated = list(input_indices)
            hidden = None
            
            # Generate character by character
            for _ in range(max_length):
                # Prepare input
                input_tensor = torch.tensor([[generated[-1]]], dtype=torch.long)
                
                # Forward pass
                output, hidden = self.forward(input_tensor, hidden)
                
                # Apply temperature
                logits = output[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=0)
                
                # Sample next character
                next_idx = torch.multinomial(probs, 1).item()
                generated.append(next_idx)
            
            # Decode to text
            generated_text = self.vocabulary.decode(generated)
            return generated_text
    
    def set_vocabulary(self, vocabulary: Vocabulary) -> None:
        """Set the vocabulary for encoding/decoding"""
        self.vocabulary = vocabulary
    
    def save_weights(self, filepath: str) -> None:
        """
        Сохранение весов модели
        
        Args:
            filepath: Путь для сохранения весов
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not set. Cannot save model without vocabulary.")
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'vocabulary': {
                'char_to_idx': self.vocabulary.char_to_idx,
                'idx_to_char': self.vocabulary.idx_to_char,
                'vocab_size': self.vocabulary.vocab_size
            }
        }
        
        torch.save(checkpoint, filepath)
    
    def load_weights(self, filepath: str) -> None:
        """
        Загрузка весов модели
        
        Args:
            filepath: Путь к файлу с весами
        """
        checkpoint = torch.load(filepath, weights_only=False)
        
        # Load model weights
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load vocabulary
        vocab = Vocabulary()
        vocab.char_to_idx = checkpoint['vocabulary']['char_to_idx']
        vocab.idx_to_char = {int(k): v for k, v in checkpoint['vocabulary']['idx_to_char'].items()}
        vocab.vocab_size = checkpoint['vocabulary']['vocab_size']
        
        self.vocabulary = vocab
