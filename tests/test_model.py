"""Unit tests for CustomAIModel."""
import os
import tempfile
import torch
import pytest
from src.model import CustomAIModel
from src.vocabulary import Vocabulary


class TestCustomAIModel:
    """Tests for CustomAIModel"""
    
    def test_model_initialization(self):
        """Test model initialization with various configurations"""
        model = CustomAIModel(
            vocab_size=50,
            embedding_dim=64,
            hidden_dim=128,
            num_layers=2
        )
        
        assert model.vocab_size == 50
        assert model.embedding_dim == 64
        assert model.hidden_dim == 128
        assert model.num_layers == 2
        assert isinstance(model.embedding, torch.nn.Embedding)
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert isinstance(model.fc, torch.nn.Linear)
    
    def test_forward_pass(self):
        """Test forward pass with sample input"""
        model = CustomAIModel(
            vocab_size=50,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Create sample input [batch_size=2, seq_len=10]
        input_tensor = torch.randint(0, 50, (2, 10))
        
        # Forward pass
        output, hidden = model.forward(input_tensor)
        
        # Check output shape
        assert output.shape == (2, 10, 50)  # [batch_size, seq_len, vocab_size]
        assert len(hidden) == 2  # (h, c)
        assert hidden[0].shape == (1, 2, 64)  # [num_layers, batch_size, hidden_dim]
    
    def test_generation_with_seed(self):
        """Test generation with seed text"""
        model = CustomAIModel(
            vocab_size=50,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Create and set vocabulary
        vocab = Vocabulary()
        vocab.build_from_text("Hello, World! This is a test.")
        model.set_vocabulary(vocab)
        
        # Generate text
        generated = model.generate("Hello", max_length=20, temperature=1.0)
        
        # Check that generation produces output
        assert len(generated) > len("Hello")
        assert generated.startswith("Hello")
    
    def test_save_and_load_weights(self):
        """Test saving and loading model weights"""
        model = CustomAIModel(
            vocab_size=30,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        # Create and set vocabulary
        vocab = Vocabulary()
        vocab.build_from_text("Test vocabulary for model")
        model.set_vocabulary(vocab)
        
        # Save weights
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            filepath = f.name
        
        try:
            model.save_weights(filepath)
            
            # Create new model and load weights
            new_model = CustomAIModel(
                vocab_size=30,
                embedding_dim=32,
                hidden_dim=64,
                num_layers=1
            )
            new_model.load_weights(filepath)
            
            # Check that vocabulary was loaded
            assert new_model.vocabulary is not None
            assert new_model.vocabulary.vocab_size == vocab.vocab_size
            
            # Check that weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_generation_without_vocabulary_raises_error(self):
        """Test that generation without vocabulary raises error"""
        model = CustomAIModel(
            vocab_size=50,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        with pytest.raises(ValueError, match="Vocabulary not set"):
            model.generate("test", max_length=10)
    
    def test_save_without_vocabulary_raises_error(self):
        """Test that saving without vocabulary raises error"""
        model = CustomAIModel(
            vocab_size=50,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            filepath = f.name
        
        try:
            with pytest.raises(ValueError, match="Vocabulary not set"):
                model.save_weights(filepath)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
