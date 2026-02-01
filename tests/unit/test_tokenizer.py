"""Unit tests for BPE Tokenizer edge cases."""
import pytest
import tempfile
import os
from src.tokenizer import BPETokenizer


class TestBPETokenizerEdgeCases:
    """Unit tests for BPE Tokenizer edge cases."""
    
    def test_invalid_vocab_size_too_small(self):
        """Test that vocab_size below minimum raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be between 1000 and 50000"):
            BPETokenizer(vocab_size=500)
    
    def test_invalid_vocab_size_too_large(self):
        """Test that vocab_size above maximum raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be between 1000 and 50000"):
            BPETokenizer(vocab_size=60000)
    
    def test_valid_vocab_size_boundaries(self):
        """Test that boundary vocab sizes are accepted."""
        # Minimum boundary
        tokenizer1 = BPETokenizer(vocab_size=1000)
        assert tokenizer1.vocab_size == 1000
        
        # Maximum boundary
        tokenizer2 = BPETokenizer(vocab_size=50000)
        assert tokenizer2.vocab_size == 50000
    
    def test_build_from_empty_text(self):
        """Test that building from empty text raises ValueError."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        with pytest.raises(ValueError, match="Training text must contain at least 10 characters"):
            tokenizer.build_from_text("")
    
    def test_build_from_very_short_text(self):
        """Test that building from very short text raises ValueError."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        with pytest.raises(ValueError, match="Training text must contain at least 10 characters"):
            tokenizer.build_from_text("short")
    
    def test_encode_before_training(self):
        """Test that encoding before training raises RuntimeError."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        with pytest.raises(RuntimeError, match="Tokenizer not trained"):
            tokenizer.encode("test text")
    
    def test_decode_before_training(self):
        """Test that decoding before training raises RuntimeError."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        with pytest.raises(RuntimeError, match="Tokenizer not trained"):
            tokenizer.decode([1, 2, 3])
    
    def test_save_before_training(self):
        """Test that saving before training raises RuntimeError."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            with pytest.raises(RuntimeError, match="Cannot save untrained tokenizer"):
                tokenizer.save(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_empty_input_handling(self):
        """Test that empty inputs are handled correctly."""
        tokenizer = BPETokenizer(vocab_size=2000)
        training_text = "This is a sample text for training the tokenizer. " * 10
        tokenizer.build_from_text(training_text)
        
        # Empty string encoding
        assert tokenizer.encode("") == []
        
        # Empty list decoding
        assert tokenizer.decode([]) == ""
    
    def test_very_long_sequence(self):
        """Test encoding and decoding very long sequences."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        # Create long training text
        training_text = "The quick brown fox jumps over the lazy dog. " * 100
        tokenizer.build_from_text(training_text)
        
        # Create very long test text
        long_text = "Testing with a very long sequence of text. " * 50
        
        # Should handle without errors
        token_ids = tokenizer.encode(long_text)
        assert len(token_ids) > 0
        
        decoded = tokenizer.decode(token_ids)
        assert len(decoded) > 0
    
    def test_special_characters(self):
        """Test handling of special characters."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        training_text = "Hello! How are you? I'm fine, thanks. #hashtag @mention $100 50% test@email.com"
        training_text = training_text * 10  # Make it longer
        tokenizer.build_from_text(training_text)
        
        test_text = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/"
        
        # Should encode without errors
        token_ids = tokenizer.encode(test_text)
        assert len(token_ids) > 0
        
        # Should decode without errors
        decoded = tokenizer.decode(token_ids)
        assert len(decoded) > 0
    
    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        # Training text with some unicode
        training_text = "Hello world! Testing unicode: café, naïve, résumé. " * 10
        tokenizer.build_from_text(training_text)
        
        # Test with unicode
        test_text = "Unicode test: café résumé"
        
        # Should handle without crashing
        token_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(token_ids)
        
        # Basic sanity check
        assert len(token_ids) > 0
        assert len(decoded) > 0
    
    def test_character_level_fallback(self):
        """Test character-level fallback for unknown tokens."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        # Train on English text
        training_text = "This is English text for training. " * 20
        tokenizer.build_from_text(training_text)
        
        # Test with text containing unknown characters
        test_text = "English with some новые символы"
        
        # Should encode using character-level fallback for unknown chars
        token_ids = tokenizer.encode(test_text)
        assert len(token_ids) > 0
        
        # All token IDs should be valid
        vocab_size = tokenizer.get_vocab_size()
        for tid in token_ids:
            assert 0 <= tid < vocab_size
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises FileNotFoundError."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
            tokenizer.load("/nonexistent/path/tokenizer.json")
    
    def test_load_corrupted_file(self):
        """Test loading corrupted file raises ValueError."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        # Create corrupted JSON file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("{ corrupted json content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Corrupted tokenizer file"):
                tokenizer.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_load_invalid_format(self):
        """Test loading file with missing fields raises ValueError."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        # Create file with missing fields
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write('{"vocab_size": 2000}')  # Missing other required fields
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid tokenizer file: missing field"):
                tokenizer.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_special_tokens_in_vocabulary(self):
        """Test that special tokens are always in vocabulary."""
        tokenizer = BPETokenizer(vocab_size=2000)
        training_text = "Sample training text for the tokenizer. " * 10
        tokenizer.build_from_text(training_text)
        
        # Check special tokens exist
        assert BPETokenizer.PAD_TOKEN in tokenizer.vocab
        assert BPETokenizer.UNK_TOKEN in tokenizer.vocab
        assert BPETokenizer.BOS_TOKEN in tokenizer.vocab
        assert BPETokenizer.EOS_TOKEN in tokenizer.vocab
        
        # Check they have expected indices
        assert tokenizer.vocab[BPETokenizer.PAD_TOKEN] == 0
        assert tokenizer.vocab[BPETokenizer.UNK_TOKEN] == 1
        assert tokenizer.vocab[BPETokenizer.BOS_TOKEN] == 2
        assert tokenizer.vocab[BPETokenizer.EOS_TOKEN] == 3
    
    def test_is_trained_flag(self):
        """Test that is_trained flag works correctly."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        # Should not be trained initially
        assert not tokenizer.is_trained()
        
        # Should be trained after building
        training_text = "Training text for the tokenizer. " * 10
        tokenizer.build_from_text(training_text)
        assert tokenizer.is_trained()
    
    def test_get_vocab_size(self):
        """Test that get_vocab_size returns actual vocabulary size."""
        tokenizer = BPETokenizer(vocab_size=5000)
        training_text = "Sample text for vocabulary building. " * 20
        tokenizer.build_from_text(training_text)
        
        actual_size = tokenizer.get_vocab_size()
        
        # Actual size should be positive and reasonable
        assert actual_size > 0
        assert actual_size <= 5000  # Should not exceed target
        
        # Should match length of vocab dict
        assert actual_size == len(tokenizer.vocab)
    
    def test_decode_invalid_indices(self):
        """Test decoding with invalid indices."""
        tokenizer = BPETokenizer(vocab_size=2000)
        training_text = "Training text for testing. " * 10
        tokenizer.build_from_text(training_text)
        
        vocab_size = tokenizer.get_vocab_size()
        
        # Decode with some invalid indices (should skip them)
        invalid_ids = [0, 1, vocab_size + 100, vocab_size + 200]
        
        # Should not crash, just skip invalid indices
        decoded = tokenizer.decode(invalid_ids)
        
        # Should return some result (may be empty or partial)
        assert isinstance(decoded, str)
    
    def test_whitespace_handling(self):
        """Test handling of various whitespace characters."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        training_text = "Text with spaces\ttabs\nand newlines. " * 10
        tokenizer.build_from_text(training_text)
        
        test_text = "Multiple   spaces\t\ttabs\n\nnewlines"
        
        # Should encode and decode
        token_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(token_ids)
        
        # Whitespace may be normalized, but should not crash
        assert len(token_ids) > 0
        assert len(decoded) > 0
    
    def test_numbers_and_punctuation(self):
        """Test handling of numbers and punctuation."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        training_text = "Numbers 123 456 789 and punctuation!!! ... ??? " * 10
        tokenizer.build_from_text(training_text)
        
        test_text = "Test 123 with numbers and punctuation!!!"
        
        token_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(token_ids)
        
        assert len(token_ids) > 0
        assert len(decoded) > 0
    
    def test_case_sensitivity(self):
        """Test that tokenizer handles case (converts to lowercase)."""
        tokenizer = BPETokenizer(vocab_size=2000)
        
        training_text = "Mixed Case Text WITH different CASES. " * 10
        tokenizer.build_from_text(training_text)
        
        # Test with different cases
        text1 = "Hello World"
        text2 = "hello world"
        
        ids1 = tokenizer.encode(text1)
        ids2 = tokenizer.encode(text2)
        
        # Should produce same encoding (case-insensitive)
        assert ids1 == ids2
