"""Property-based tests for Vocabulary."""
import os
import tempfile
from hypothesis import given, strategies as st
from src.vocabulary import Vocabulary


class TestVocabularyProperties:
    """Property-based tests for Vocabulary
    
    Feature: custom-ai-model, Property 19: Weight Persistence Round-Trip
    Validates: Requirements 6.4, 6.5
    """
    
    @given(st.text(min_size=1, max_size=1000))
    def test_encoding_decoding_preserves_text(self, text: str):
        """
        Property 19: Weight Persistence Round-Trip
        For any vocabulary built from random text, encoding then decoding 
        should preserve the original text
        
        Validates: Requirements 6.4, 6.5
        """
        vocab = Vocabulary()
        vocab.build_from_text(text)
        
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        
        assert decoded == text, f"Decoding should preserve original text"
    
    @given(st.text(min_size=1, max_size=500))
    def test_save_load_preserves_vocabulary(self, text: str):
        """
        Property 19: Weight Persistence Round-Trip
        For any vocabulary, saving and loading should preserve encoding/decoding
        
        Validates: Requirements 6.4, 6.5
        """
        vocab = Vocabulary()
        vocab.build_from_text(text)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            vocab.save(filepath)
            loaded_vocab = Vocabulary.load(filepath)
            
            # Test that loaded vocab produces same encoding
            original_encoded = vocab.encode(text)
            loaded_encoded = loaded_vocab.encode(text)
            
            assert original_encoded == loaded_encoded
            assert loaded_vocab.decode(loaded_encoded) == text
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
