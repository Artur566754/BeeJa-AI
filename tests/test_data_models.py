"""Unit tests for data models."""
import os
import tempfile
import pytest
from src.config import ModelConfig, TrainingConfig
from src.vocabulary import Vocabulary


class TestModelConfig:
    """Tests for ModelConfig"""
    
    def test_save_load_round_trip(self):
        """Test ModelConfig save/load round-trip"""
        config = ModelConfig(
            vocab_size=100,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=3,
            seq_length=50
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            config.save(filepath)
            loaded_config = ModelConfig.load(filepath)
            
            assert loaded_config.vocab_size == config.vocab_size
            assert loaded_config.embedding_dim == config.embedding_dim
            assert loaded_config.hidden_dim == config.hidden_dim
            assert loaded_config.num_layers == config.num_layers
            assert loaded_config.seq_length == config.seq_length
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestVocabulary:
    """Tests for Vocabulary"""
    
    def test_encoding_decoding(self):
        """Test Vocabulary encoding/decoding"""
        vocab = Vocabulary()
        text = "Hello, World!"
        
        vocab.build_from_text(text)
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        
        assert decoded == text
        assert vocab.vocab_size == len(set(text))
    
    def test_save_load_round_trip(self):
        """Test Vocabulary save/load round-trip"""
        vocab = Vocabulary()
        text = "Test vocabulary with various characters: 123, АБВ!"
        vocab.build_from_text(text)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            vocab.save(filepath)
            loaded_vocab = Vocabulary.load(filepath)
            
            assert loaded_vocab.vocab_size == vocab.vocab_size
            assert loaded_vocab.char_to_idx == vocab.char_to_idx
            
            # Test encoding with loaded vocab
            encoded = loaded_vocab.encode(text)
            decoded = loaded_vocab.decode(encoded)
            assert decoded == text
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestInitialDataset:
    """Tests for initial dataset"""
    
    def test_initial_dataset_content(self):
        """Test initial dataset contains creator information"""
        dataset_path = "datasets/initial_dataset.txt"
        
        assert os.path.exists(dataset_path), "Initial dataset file should exist"
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Requirements 5.2, 5.3, 5.4
        assert "Jamsaide" in content or "Артур" in content, "Should contain creator name"
        assert "BeeBoo" in content or "Никита" in content, "Should contain helper name"
        assert "Araxium" in content or "Лев" in content, "Should contain helper name"
        assert "люблю" in content or "благодарен" in content, "Should express affection"
