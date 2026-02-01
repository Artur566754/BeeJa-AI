"""Property-based tests for CustomAIModel."""
import os
import tempfile
import torch
from hypothesis import given, strategies as st, settings
from src.model import CustomAIModel
from src.vocabulary import Vocabulary


class TestCustomAIModelProperties:
    """Property-based tests for CustomAIModel
    
    Feature: custom-ai-model
    """
    
    @given(st.text(min_size=10, max_size=200).filter(lambda x: x.strip()))
    @settings(max_examples=100)
    def test_text_generation_capability(self, text: str):
        """
        Property 18: Text Generation Capability
        For any text input, the model should generate non-empty output
        
        Validates: Requirements 6.2, 6.3
        """
        # Create vocabulary from text
        vocab = Vocabulary()
        vocab.build_from_text(text)
        
        # Create model
        model = CustomAIModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        model.set_vocabulary(vocab)
        
        # Take first few characters as seed
        seed = text[:min(5, len(text))]
        
        # Generate text
        generated = model.generate(seed, max_length=20, temperature=1.0)
        
        # Check that output is non-empty and starts with seed
        assert len(generated) > 0, "Generated text should be non-empty"
        assert generated.startswith(seed), "Generated text should start with seed"
    
    @given(st.text(min_size=10, max_size=100).filter(lambda x: x.strip()))
    @settings(max_examples=100)
    def test_weight_persistence_round_trip(self, text: str):
        """
        Property 19: Weight Persistence Round-Trip
        For any model weights, saving and loading should preserve them
        
        Validates: Requirements 6.4, 6.5
        """
        # Create vocabulary
        vocab = Vocabulary()
        vocab.build_from_text(text)
        
        # Create model
        model = CustomAIModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1
        )
        model.set_vocabulary(vocab)
        
        # Save weights
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            filepath = f.name
        
        try:
            model.save_weights(filepath)
            
            # Create new model and load weights
            new_model = CustomAIModel(
                vocab_size=vocab.vocab_size,
                embedding_dim=32,
                hidden_dim=64,
                num_layers=1
            )
            new_model.load_weights(filepath)
            
            # Check that all parameters match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2, atol=1e-6), "Weights should be preserved after save/load"
            
            # Check vocabulary is preserved
            assert new_model.vocabulary.vocab_size == vocab.vocab_size
            assert new_model.vocabulary.char_to_idx == vocab.char_to_idx
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
