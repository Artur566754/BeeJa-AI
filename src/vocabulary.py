"""Vocabulary class for character encoding/decoding."""
import json
from typing import Dict, List


class Vocabulary:
    """Словарь для маппинга символов"""
    
    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
    
    def build_from_text(self, text: str) -> None:
        """Построение словаря из текста"""
        unique_chars = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(unique_chars)
    
    def encode(self, text: str) -> List[int]:
        """Кодирование текста в индексы"""
        return [self.char_to_idx.get(char, 0) for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Декодирование индексов в текст"""
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])
    
    def save(self, filepath: str) -> None:
        """Сохранение словаря"""
        data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Загрузка словаря"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls()
        vocab.char_to_idx = data['char_to_idx']
        vocab.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        vocab.vocab_size = data['vocab_size']
        return vocab
