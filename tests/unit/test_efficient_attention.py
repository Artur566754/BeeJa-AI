"""Unit tests for efficient attention computation."""
import pytest
import torch
from src.transformer_components import MultiHeadAttention, create_causal_mask
from src.transformer_model import TransformerModel
from src.config import TransformerConfig


class TestEfficientAttention:
    """Test efficient attention computation."""
    
    def test_standard_attention_short_sequence(self):
        """Test standard attention works for short sequences."""
        attention = MultiHeadAttention(
            embedding_dim=128,
            num_heads=4,
            dropout=0.1,
            use_memory_efficient=False
        )
        
        # Short sequence (< 512)
        x = torch.randn(2, 100, 128)
        mask = create_causal_mask(100)
        
        output = attention(x, mask)
        
        assert output.shape == (2, 100, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_memory_efficient_attention_long_sequence(self):
        """Test memory-efficient attention for long sequences (>512 tokens)."""
        attention = MultiHeadAttention(
            embedding_dim=128,
            num_heads=4,
            dropout=0.1,
            use_memory_efficient=True
        )
        attention.eval()  # Set to eval mode to disable dropout for consistency
        
        # Long sequence (> 512)
        x = torch.randn(1, 600, 128)
        mask = create_causal_mask(600)
        
        output = attention(x, mask)
        
        assert output.shape == (1, 600, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_memory_efficient_vs_standard_short_sequence(self):
        """Test that memory-efficient and standard attention produce similar results for short sequences."""
        embedding_dim = 128
        num_heads = 4
        seq_len = 100
        
        # Create two attention modules with same initialization
        torch.manual_seed(42)
        attention_standard = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.0,  # No dropout for comparison
            use_memory_efficient=False
        )
        
        torch.manual_seed(42)
        attention_efficient = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=0.0,
            use_memory_efficient=True
        )
        
        # Copy weights to ensure they're identical
        attention_efficient.load_state_dict(attention_standard.state_dict())
        
        # Set to eval mode
        attention_standard.eval()
        attention_efficient.eval()
        
        # Same input
        torch.manual_seed(123)
        x = torch.randn(2, seq_len, embedding_dim)
        mask = create_causal_mask(seq_len)
        
        # Forward pass
        with torch.no_grad():
            output_standard = attention_standard(x, mask)
            output_efficient = attention_efficient(x, mask)
        
        # Should produce similar results (allowing for numerical differences)
        assert torch.allclose(output_standard, output_efficient, rtol=1e-4, atol=1e-5)
    
    def test_attention_with_no_mask(self):
        """Test attention works without mask."""
        attention = MultiHeadAttention(
            embedding_dim=128,
            num_heads=4,
            dropout=0.1,
            use_memory_efficient=False
        )
        
        x = torch.randn(2, 50, 128)
        
        output = attention(x, attention_mask=None)
        
        assert output.shape == (2, 50, 128)
        assert not torch.isnan(output).any()
    
    def test_transformer_model_with_memory_efficient_attention(self):
        """Test full Transformer model with memory-efficient attention."""
        config = TransformerConfig(
            vocab_size=1000,
            embedding_dim=128,
            num_heads=4,
            num_layers=2,
            hidden_dim=512,
            max_seq_length=1024,
            dropout=0.1,
            use_memory_efficient=True
        )
        
        model = TransformerModel(config)
        model.eval()
        
        # Long sequence
        input_ids = torch.randint(0, 1000, (1, 600))
        
        with torch.no_grad():
            logits = model(input_ids)
        
        assert logits.shape == (1, 600, 1000)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_attention_output_shape_consistency(self):
        """Test that attention output shape is consistent across different sequence lengths."""
        attention = MultiHeadAttention(
            embedding_dim=256,
            num_heads=8,
            dropout=0.1,
            use_memory_efficient=True
        )
        
        for seq_len in [50, 100, 256, 512, 600, 1000]:
            x = torch.randn(2, seq_len, 256)
            mask = create_causal_mask(seq_len)
            
            output = attention(x, mask)
            
            assert output.shape == (2, seq_len, 256), f"Failed for seq_len={seq_len}"
    
    def test_chunked_attention_boundary_conditions(self):
        """Test memory-efficient attention at chunk boundaries."""
        attention = MultiHeadAttention(
            embedding_dim=128,
            num_heads=4,
            dropout=0.0,
            use_memory_efficient=True
        )
        attention.eval()
        
        # Test at exact chunk boundary (512)
        x = torch.randn(1, 512, 128)
        mask = create_causal_mask(512)
        
        with torch.no_grad():
            output = attention(x, mask)
        
        assert output.shape == (1, 512, 128)
        assert not torch.isnan(output).any()
        
        # Test just over chunk boundary (513)
        x = torch.randn(1, 513, 128)
        mask = create_causal_mask(513)
        
        with torch.no_grad():
            output = attention(x, mask)
        
        assert output.shape == (1, 513, 128)
        assert not torch.isnan(output).any()
    
    def test_attention_with_batch_size_variations(self):
        """Test attention works with different batch sizes."""
        attention = MultiHeadAttention(
            embedding_dim=128,
            num_heads=4,
            dropout=0.1,
            use_memory_efficient=True
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 600, 128)
            mask = create_causal_mask(600)
            
            output = attention(x, mask)
            
            assert output.shape == (batch_size, 600, 128), f"Failed for batch_size={batch_size}"
