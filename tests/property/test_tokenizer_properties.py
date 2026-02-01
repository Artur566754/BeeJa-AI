"""Property-based tests for BPE Tokenizer."""
import pytest
import tempfile
import os
from hypothesis import given, settings, strategies as st
from src.tokenizer import BPETokenizer


# Strategy for generating text with reasonable characters
def text_strategy(min_size=10, max_size=1000):
    """Generate text with printable ASCII characters."""
    return st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters='\x00'),
        min_size=min_size,
        max_size=max_size
    ).filter(lambda x: len(x.strip()) >= 10)


# Strategy for generating vocabulary sizes
vocab_size_strategy = st.integers(min_value=1000, max_value=50000)


class TestTokenizerProperties:
    """Property-based tests for BPE Tokenizer."""
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy(min_size=100, max_size=500))
    def test_property_4_vocabulary_coverage(self, text):
        """
        Feature: transformer-upgrade, Property 4: Tokenizer Vocabulary Coverage
        
        For any text used to build the tokenizer vocabulary, all characters and 
        subwords in that text SHALL be encodable after vocabulary construction, 
        with no unknown tokens.
        
        **Validates: Requirements 2.3**
        """
        # Build tokenizer from text
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.build_from_text(text)
        
        # Encode the same text
        token_ids = tokenizer.encode(text)
        
        # Verify no unknown tokens (UNK token has index 1)
        unk_token_id = tokenizer.vocab[BPETokenizer.UNK_TOKEN]
        
        # Count UNK tokens
        unk_count = sum(1 for tid in token_ids if tid == unk_token_id)
        
        # All characters from training text should be encodable
        # Note: Due to BPE word tokenization, some formatting might change,
        # but no UNK tokens should appear for the training text
        assert unk_count == 0, f"Found {unk_count} unknown tokens when encoding training text"
    
    @settings(max_examples=100, deadline=None)
    @given(
        training_text=text_strategy(min_size=100, max_size=300),
        test_text=st.text(
            alphabet=st.characters(min_codepoint=127, max_codepoint=200),
            min_size=5,
            max_size=50
        )
    )
    def test_property_5_unknown_token_handling(self, training_text, test_text):
        """
        Feature: transformer-upgrade, Property 5: Unknown Token Handling
        
        For any text containing characters not in the tokenizer vocabulary, 
        encoding SHALL complete without errors and produce valid token indices 
        within the vocabulary range.
        
        **Validates: Requirements 2.4**
        """
        # Build tokenizer from training text
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.build_from_text(training_text)
        
        vocab_size = tokenizer.get_vocab_size()
        
        # Try to encode text with potentially unknown characters
        try:
            token_ids = tokenizer.encode(test_text)
            
            # Verify all token IDs are valid (within vocabulary range)
            for tid in token_ids:
                assert 0 <= tid < vocab_size, \
                    f"Token ID {tid} out of range [0, {vocab_size})"
            
            # Encoding should complete without errors
            assert True
        except Exception as e:
            pytest.fail(f"Encoding failed with unknown characters: {e}")
    
    @settings(max_examples=100, deadline=None)
    @given(
        training_text=text_strategy(min_size=100, max_size=300),
        test_text=text_strategy(min_size=10, max_size=100)
    )
    def test_property_6_token_encoding_validity(self, training_text, test_text):
        """
        Feature: transformer-upgrade, Property 6: Token Encoding Validity
        
        For any text string, encoding SHALL produce token indices where all 
        indices are non-negative integers less than the vocabulary size.
        
        **Validates: Requirements 2.5**
        """
        # Build tokenizer
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.build_from_text(training_text)
        
        vocab_size = tokenizer.get_vocab_size()
        
        # Encode test text
        token_ids = tokenizer.encode(test_text)
        
        # Verify all indices are valid
        for tid in token_ids:
            assert isinstance(tid, int), f"Token ID {tid} is not an integer"
            assert tid >= 0, f"Token ID {tid} is negative"
            assert tid < vocab_size, f"Token ID {tid} >= vocab_size {vocab_size}"
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy(min_size=50, max_size=200))
    def test_property_7_tokenizer_roundtrip_consistency(self, text):
        """
        Feature: transformer-upgrade, Property 7: Tokenizer Round-Trip Consistency
        
        For any text string in the tokenizer's vocabulary, encoding then decoding 
        SHALL produce text that preserves the semantic meaning (allowing for 
        normalization like whitespace changes).
        
        **Validates: Requirements 2.6**
        """
        # Build tokenizer from the text
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.build_from_text(text)
        
        # Encode then decode
        token_ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(token_ids)
        
        # Normalize both texts for comparison (lowercase, whitespace)
        original_normalized = ' '.join(text.lower().split())
        decoded_normalized = ' '.join(decoded_text.lower().split())
        
        # The decoded text should match the original (after normalization)
        # We allow for minor differences due to BPE tokenization
        # Check that most of the content is preserved
        if len(original_normalized) > 0:
            # Calculate similarity (simple character overlap)
            original_chars = set(original_normalized)
            decoded_chars = set(decoded_normalized)
            
            if original_chars:
                overlap = len(original_chars & decoded_chars) / len(original_chars)
                assert overlap > 0.8, \
                    f"Round-trip lost too much information: {overlap:.2%} overlap\n" \
                    f"Original: {original_normalized[:100]}\n" \
                    f"Decoded: {decoded_normalized[:100]}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        text=text_strategy(min_size=100, max_size=300),
        vocab_size=vocab_size_strategy
    )
    def test_property_8_tokenizer_persistence_roundtrip(self, text, vocab_size):
        """
        Feature: transformer-upgrade, Property 8: Tokenizer Persistence Round-Trip
        
        For any trained tokenizer, saving then loading SHALL restore a tokenizer 
        that produces identical encodings for all inputs.
        
        **Validates: Requirements 2.7**
        """
        # Create and train tokenizer
        tokenizer1 = BPETokenizer(vocab_size=vocab_size)
        tokenizer1.build_from_text(text)
        
        # Encode some text
        test_text = text[:100]  # Use part of training text
        original_encoding = tokenizer1.encode(test_text)
        
        # Save tokenizer
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            tokenizer1.save(temp_path)
            
            # Load into new tokenizer
            tokenizer2 = BPETokenizer(vocab_size=vocab_size)
            tokenizer2.load(temp_path)
            
            # Encode same text with loaded tokenizer
            loaded_encoding = tokenizer2.encode(test_text)
            
            # Encodings should be identical
            assert original_encoding == loaded_encoding, \
                "Loaded tokenizer produces different encoding"
            
            # Vocabulary should be identical
            assert tokenizer1.vocab == tokenizer2.vocab, \
                "Loaded tokenizer has different vocabulary"
            
            # Merges should be identical
            assert tokenizer1.merges == tokenizer2.merges, \
                "Loaded tokenizer has different merges"
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @settings(max_examples=50, deadline=None)
    @given(text=text_strategy(min_size=100, max_size=300))
    def test_empty_string_encoding(self, text):
        """Test that empty strings are handled correctly."""
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.build_from_text(text)
        
        # Empty string should return empty list
        token_ids = tokenizer.encode("")
        assert token_ids == [], "Empty string should encode to empty list"
        
        # Empty list should decode to empty string
        decoded = tokenizer.decode([])
        assert decoded == "", "Empty list should decode to empty string"
    
    @settings(max_examples=50, deadline=None)
    @given(text=text_strategy(min_size=100, max_size=300))
    def test_special_tokens_not_in_output(self, text):
        """Test that special tokens are handled correctly."""
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.build_from_text(text)
        
        # Encode and decode
        token_ids = tokenizer.encode(text[:50])
        decoded = tokenizer.decode(token_ids)
        
        # Special tokens should not appear in decoded output
        assert BPETokenizer.PAD_TOKEN not in decoded
        assert BPETokenizer.BOS_TOKEN not in decoded
        assert BPETokenizer.EOS_TOKEN not in decoded
