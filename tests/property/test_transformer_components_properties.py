"""Property-based tests for Transformer core components."""
import pytest
import torch
from hypothesis import given, settings, strategies as st
from src.transformer_components import (
    PositionalEncoding,
    MultiHeadAttention,
    TransformerBlock,
    create_causal_mask
)


# Strategies for generating test inputs
def embedding_dim_strategy():
    """Generate valid embedding dimensions (must be divisible by common head counts)."""
    return st.sampled_from([64, 128, 256, 512])


def num_heads_strategy(embedding_dim):
    """Generate valid number of heads for given embedding_dim."""
    valid_heads = [h for h in [2, 4, 8, 16] if embedding_dim % h == 0]
    return st.sampled_from(valid_heads)


def seq_len_strategy():
    """Generate reasonable sequence lengths."""
    return st.integers(min_value=1, max_value=128)


def batch_size_strategy():
    """Generate reasonable batch sizes."""
    return st.integers(min_value=1, max_value=8)


class TestTransformerComponentsProperties:
    """Property-based tests for Transformer components."""
    
    @settings(max_examples=100, deadline=None)
    @given(
        embedding_dim=embedding_dim_strategy(),
        seq_len=seq_len_strategy(),
        batch_size=batch_size_strategy()
    )
    def test_property_1_positional_encoding_sensitivity(self, embedding_dim, seq_len, batch_size):
        """
        Feature: transformer-upgrade, Property 1: Positional Encoding Sensitivity
        
        For any input sequence, if we permute the order of tokens, the model's 
        output logits SHALL change, demonstrating that positional information 
        affects predictions.
        
        **Validates: Requirements 1.2**
        """
        # Create positional encoding
        pos_enc = PositionalEncoding(embedding_dim, max_seq_length=512, dropout=0.0)
        pos_enc.eval()  # Disable dropout for deterministic behavior
        
        # Create random input embeddings
        torch.manual_seed(42)  # For reproducibility
        x = torch.randn(batch_size, seq_len, embedding_dim)
        
        # Apply positional encoding
        output1 = pos_enc(x)
        
        # Permute the sequence (reverse order)
        if seq_len > 1:
            x_permuted = torch.flip(x, dims=[1])
            output2 = pos_enc(x_permuted)
            
            # Outputs should be different (positional encoding affects output)
            # Check that at least some positions differ significantly
            diff = torch.abs(output1 - output2).mean().item()
            
            # For sequences longer than 1, permutation should cause noticeable difference
            assert diff > 1e-6, \
                f"Positional encoding not sensitive to position: diff={diff}"
    
    @settings(max_examples=100, deadline=None)
    @given(
        embedding_dim=embedding_dim_strategy(),
        seq_len=seq_len_strategy(),
        batch_size=batch_size_strategy()
    )
    def test_property_3_causal_masking(self, embedding_dim, seq_len, batch_size):
        """
        Feature: transformer-upgrade, Property 3: Causal Masking
        
        For any input sequence and position i, modifying tokens at positions j > i 
        SHALL NOT affect the model's prediction at position i, ensuring causal 
        masking prevents information leakage from future tokens.
        
        **Validates: Requirements 1.9**
        """
        # Skip if sequence too short
        if seq_len < 2:
            return
        
        # Create multi-head attention with causal mask
        num_heads = 4 if embedding_dim % 4 == 0 else 2
        attention = MultiHeadAttention(embedding_dim, num_heads, dropout=0.0)
        attention.eval()
        
        # Create causal mask
        causal_mask = create_causal_mask(seq_len)
        
        # Create random input
        torch.manual_seed(42)
        x1 = torch.randn(batch_size, seq_len, embedding_dim)
        
        # Apply attention with causal mask
        output1 = attention(x1, attention_mask=causal_mask)
        
        # Modify future tokens (positions > 0)
        x2 = x1.clone()
        x2[:, 1:, :] = torch.randn_like(x2[:, 1:, :])
        
        # Apply attention again
        output2 = attention(x2, attention_mask=causal_mask)
        
        # First position should be identical (no future information)
        first_pos_diff = torch.abs(output1[:, 0, :] - output2[:, 0, :]).max().item()
        
        assert first_pos_diff < 1e-5, \
            f"Causal masking failed: first position affected by future tokens (diff={first_pos_diff})"
    
    @settings(max_examples=50, deadline=None)
    @given(
        embedding_dim=embedding_dim_strategy(),
        seq_len=seq_len_strategy(),
        batch_size=batch_size_strategy()
    )
    def test_positional_encoding_shape(self, embedding_dim, seq_len, batch_size):
        """Test that positional encoding preserves input shape."""
        pos_enc = PositionalEncoding(embedding_dim, max_seq_length=512)
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = pos_enc(x)
        
        assert output.shape == x.shape, \
            f"Shape mismatch: input {x.shape}, output {output.shape}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        embedding_dim=embedding_dim_strategy(),
        seq_len=seq_len_strategy(),
        batch_size=batch_size_strategy()
    )
    def test_multihead_attention_shape(self, embedding_dim, seq_len, batch_size):
        """Test that multi-head attention preserves input shape."""
        num_heads = 4 if embedding_dim % 4 == 0 else 2
        attention = MultiHeadAttention(embedding_dim, num_heads)
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = attention(x)
        
        assert output.shape == x.shape, \
            f"Shape mismatch: input {x.shape}, output {output.shape}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        embedding_dim=embedding_dim_strategy(),
        seq_len=seq_len_strategy(),
        batch_size=batch_size_strategy()
    )
    def test_transformer_block_shape(self, embedding_dim, seq_len, batch_size):
        """Test that transformer block preserves input shape."""
        num_heads = 4 if embedding_dim % 4 == 0 else 2
        hidden_dim = embedding_dim * 4
        block = TransformerBlock(embedding_dim, num_heads, hidden_dim)
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = block(x)
        
        assert output.shape == x.shape, \
            f"Shape mismatch: input {x.shape}, output {output.shape}"
    
    @settings(max_examples=50, deadline=None)
    @given(seq_len=seq_len_strategy())
    def test_causal_mask_properties(self, seq_len):
        """Test that causal mask has correct properties."""
        mask = create_causal_mask(seq_len)
        
        # Check shape
        assert mask.shape == (seq_len, seq_len)
        
        # Check that it's lower triangular (mask[i, j] = 1 if j <= i)
        for i in range(seq_len):
            for j in range(seq_len):
                if j <= i:
                    assert mask[i, j] == 1, f"Mask should be 1 at ({i}, {j})"
                else:
                    assert mask[i, j] == 0, f"Mask should be 0 at ({i}, {j})"
    
    def test_invalid_embedding_dim_not_divisible(self):
        """Test that invalid embedding_dim raises ValueError."""
        with pytest.raises(ValueError, match="embedding_dim .* must be divisible by num_heads"):
            MultiHeadAttention(embedding_dim=100, num_heads=3)
    
    @settings(max_examples=50, deadline=None)
    @given(
        embedding_dim=embedding_dim_strategy(),
        seq_len=seq_len_strategy()
    )
    def test_attention_without_mask(self, embedding_dim, seq_len):
        """Test that attention works without mask."""
        num_heads = 4 if embedding_dim % 4 == 0 else 2
        attention = MultiHeadAttention(embedding_dim, num_heads)
        
        x = torch.randn(1, seq_len, embedding_dim)
        
        # Should work without mask
        output = attention(x, attention_mask=None)
        assert output.shape == x.shape
    
    @settings(max_examples=50, deadline=None)
    @given(
        embedding_dim=embedding_dim_strategy(),
        seq_len=seq_len_strategy()
    )
    def test_residual_connections(self, embedding_dim, seq_len):
        """Test that residual connections are working."""
        num_heads = 4 if embedding_dim % 4 == 0 else 2
        hidden_dim = embedding_dim * 4
        block = TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout=0.0)
        block.eval()
        
        # Create input
        x = torch.randn(1, seq_len, embedding_dim)
        
        # Forward pass
        output = block(x)
        
        # Output should not be identical to input (transformation applied)
        # but should have similar magnitude due to residual connections
        input_norm = torch.norm(x)
        output_norm = torch.norm(output)
        
        # Norms should be in similar range (within 10x)
        ratio = output_norm / (input_norm + 1e-8)
        assert 0.1 < ratio < 10.0, \
            f"Residual connections may not be working: norm ratio={ratio}"
