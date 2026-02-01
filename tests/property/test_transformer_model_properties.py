"""Property-based tests for TransformerModel configuration and persistence."""
import pytest
import torch
import tempfile
import os
from hypothesis import given, settings, strategies as st
from src.config import TransformerConfig
from src.transformer_model import TransformerModel
from src.tokenizer import BPETokenizer


# Feature: transformer-upgrade, Property 2: Configuration Validation
# Validates: Requirements 1.3, 1.4, 1.5, 2.2, 7.6
@settings(max_examples=100, deadline=None)
@given(
    vocab_size=st.integers(min_value=1000, max_value=50000),
    num_heads=st.integers(min_value=2, max_value=16),
    num_layers=st.integers(min_value=2, max_value=12),
    dropout=st.floats(min_value=0.0, max_value=1.0),
    head_dim_multiplier=st.integers(min_value=8, max_value=64)
)
def test_property_2_valid_configuration_accepted(
    vocab_size, num_heads, num_layers, dropout, head_dim_multiplier
):
    """
    Property 2: Configuration Validation
    
    For any model configuration within valid ranges, the system SHALL accept it.
    Valid ranges:
    - layers: 2-12
    - heads: 2-16
    - dimensions: 128-1024
    - vocab: 1000-50000
    - dropout: 0.0-1.0
    """
    # Generate embedding_dim that's divisible by num_heads and in valid range
    embedding_dim = num_heads * head_dim_multiplier
    
    # Clamp to valid range while maintaining divisibility
    if embedding_dim < 128:
        # Round up to nearest multiple of num_heads that's >= 128
        embedding_dim = ((128 + num_heads - 1) // num_heads) * num_heads
    if embedding_dim > 1024:
        # Round down to nearest multiple of num_heads that's <= 1024
        embedding_dim = (1024 // num_heads) * num_heads
    
    config = TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=embedding_dim * 4,
        max_seq_length=512,
        dropout=dropout
    )
    
    # Should not raise any exception
    config.validate()
    
    # Should be able to create model with valid config
    model = TransformerModel(config)
    assert model is not None
    assert model.vocab_size == vocab_size
    assert model.embedding_dim == embedding_dim
    assert model.num_heads == num_heads
    assert model.num_layers == num_layers


# Feature: transformer-upgrade, Property 2: Configuration Validation (Invalid Cases)
# Validates: Requirements 1.3, 1.4, 1.5, 2.2, 7.6
@settings(max_examples=100, deadline=None)
@given(
    num_layers=st.integers(min_value=-10, max_value=1) | st.integers(min_value=13, max_value=100)
)
def test_property_2_invalid_num_layers_rejected(num_layers):
    """Property 2: Configuration with invalid num_layers SHALL be rejected."""
    config = TransformerConfig(
        vocab_size=5000,
        embedding_dim=256,
        num_heads=8,
        num_layers=num_layers,
        hidden_dim=1024,
        max_seq_length=512,
        dropout=0.1
    )
    
    with pytest.raises(ValueError, match="num_layers must be between 2 and 12"):
        config.validate()


@settings(max_examples=100, deadline=None)
@given(
    num_heads=st.integers(min_value=-10, max_value=1) | st.integers(min_value=17, max_value=100)
)
def test_property_2_invalid_num_heads_rejected(num_heads):
    """Property 2: Configuration with invalid num_heads SHALL be rejected."""
    config = TransformerConfig(
        vocab_size=5000,
        embedding_dim=256,
        num_heads=num_heads,
        num_layers=6,
        hidden_dim=1024,
        max_seq_length=512,
        dropout=0.1
    )
    
    with pytest.raises(ValueError, match="num_heads must be between 2 and 16"):
        config.validate()


@settings(max_examples=100, deadline=None)
@given(
    embedding_dim=st.integers(min_value=-100, max_value=127) | st.integers(min_value=1025, max_value=5000)
)
def test_property_2_invalid_embedding_dim_rejected(embedding_dim):
    """Property 2: Configuration with invalid embedding_dim SHALL be rejected."""
    config = TransformerConfig(
        vocab_size=5000,
        embedding_dim=embedding_dim,
        num_heads=8,
        num_layers=6,
        hidden_dim=1024,
        max_seq_length=512,
        dropout=0.1
    )
    
    with pytest.raises(ValueError, match="embedding_dim must be between 128 and 1024"):
        config.validate()


@settings(max_examples=100, deadline=None)
@given(
    vocab_size=st.integers(min_value=-1000, max_value=999) | st.integers(min_value=50001, max_value=100000)
)
def test_property_2_invalid_vocab_size_rejected(vocab_size):
    """Property 2: Configuration with invalid vocab_size SHALL be rejected."""
    config = TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=1024,
        max_seq_length=512,
        dropout=0.1
    )
    
    with pytest.raises(ValueError, match="vocab_size must be between 1000 and 50000"):
        config.validate()


@settings(max_examples=100, deadline=None)
@given(
    dropout=st.floats(min_value=-1.0, max_value=-0.01) | st.floats(min_value=1.01, max_value=10.0)
)
def test_property_2_invalid_dropout_rejected(dropout):
    """Property 2: Configuration with invalid dropout SHALL be rejected."""
    config = TransformerConfig(
        vocab_size=5000,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=1024,
        max_seq_length=512,
        dropout=dropout
    )
    
    with pytest.raises(ValueError, match="dropout must be between 0.0 and 1.0"):
        config.validate()


@settings(max_examples=100, deadline=None)
@given(
    embedding_dim=st.integers(min_value=128, max_value=1024),
    num_heads=st.integers(min_value=2, max_value=16)
)
def test_property_2_incompatible_embedding_heads_rejected(embedding_dim, num_heads):
    """Property 2: Configuration where embedding_dim not divisible by num_heads SHALL be rejected."""
    # Ensure they're incompatible
    if embedding_dim % num_heads == 0:
        embedding_dim += 1
        if embedding_dim > 1024:
            embedding_dim -= 2
    
    config = TransformerConfig(
        vocab_size=5000,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=6,
        hidden_dim=1024,
        max_seq_length=512,
        dropout=0.1
    )
    
    with pytest.raises(ValueError, match="embedding_dim .* must be divisible by num_heads"):
        config.validate()


def test_property_2_preset_configurations():
    """Property 2: Preset configurations (small, medium, large) SHALL be valid."""
    vocab_size = 5000
    
    # Test small configuration
    small_config = TransformerConfig.small(vocab_size)
    small_config.validate()
    model = TransformerModel(small_config)
    assert model is not None
    
    # Test medium configuration
    medium_config = TransformerConfig.medium(vocab_size)
    medium_config.validate()
    model = TransformerModel(medium_config)
    assert model is not None
    
    # Test large configuration
    large_config = TransformerConfig.large(vocab_size)
    large_config.validate()
    model = TransformerModel(large_config)
    assert model is not None



# Feature: transformer-upgrade, Property 20: Checkpoint Persistence Round-Trip
# Validates: Requirements 4.5, 4.6, 5.1, 5.2
@settings(max_examples=100, deadline=None)
@given(
    vocab_size=st.integers(min_value=1000, max_value=5000),
    num_heads=st.sampled_from([2, 4, 8]),
    num_layers=st.integers(min_value=2, max_value=6),
    seq_len=st.integers(min_value=5, max_value=50),
    batch_size=st.integers(min_value=1, max_value=4)
)
def test_property_20_checkpoint_round_trip(vocab_size, num_heads, num_layers, seq_len, batch_size):
    """
    Property 20: Checkpoint Persistence Round-Trip
    
    For any trained model, saving a checkpoint then loading it SHALL restore
    a model that produces identical outputs for all inputs.
    """
    # Create a valid embedding_dim
    embedding_dim = num_heads * 32
    if embedding_dim < 128:
        embedding_dim = 128
    if embedding_dim > 512:
        embedding_dim = 512
    
    # Create config
    config = TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=embedding_dim * 2,
        max_seq_length=512,
        dropout=0.1
    )
    
    # Create model and tokenizer
    model1 = TransformerModel(config)
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.build_from_text("hello world test sample text for tokenizer")
    model1.set_tokenizer(tokenizer)
    
    # Create random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Get output from original model
    model1.eval()
    with torch.no_grad():
        output1 = model1(input_ids)
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        checkpoint_path = f.name
    
    try:
        model1.save_checkpoint(checkpoint_path)
        
        # Create new model with same config
        model2 = TransformerModel(config)
        
        # Load checkpoint
        model2.load_checkpoint(checkpoint_path)
        
        # Get output from loaded model
        model2.eval()
        with torch.no_grad():
            output2 = model2(input_ids)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-6), \
            "Loaded model produces different outputs than original model"
        
        # Tokenizer should also be restored
        assert model2.tokenizer is not None
        assert model2.tokenizer.vocab_size == vocab_size
        
    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


# Feature: transformer-upgrade, Property 21: Checkpoint Compatibility Validation
# Validates: Requirements 5.3
@settings(max_examples=50, deadline=None)
@given(
    vocab_size1=st.integers(min_value=1000, max_value=3000),
    vocab_size2=st.integers(min_value=3001, max_value=5000),
    embedding_dim1=st.sampled_from([128, 256]),
    embedding_dim2=st.sampled_from([384, 512])
)
def test_property_21_incompatible_checkpoint_rejected(vocab_size1, vocab_size2, embedding_dim1, embedding_dim2):
    """
    Property 21: Checkpoint Compatibility Validation
    
    For any checkpoint file with incompatible configuration, loading SHALL raise
    an appropriate error rather than silently failing.
    """
    # Create first model
    config1 = TransformerConfig(
        vocab_size=vocab_size1,
        embedding_dim=embedding_dim1,
        num_heads=4,
        num_layers=4,
        hidden_dim=embedding_dim1 * 2,
        max_seq_length=256,
        dropout=0.1
    )
    
    model1 = TransformerModel(config1)
    tokenizer = BPETokenizer(vocab_size=vocab_size1)
    tokenizer.build_from_text("test text for tokenizer")
    model1.set_tokenizer(tokenizer)
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        checkpoint_path = f.name
    
    try:
        model1.save_checkpoint(checkpoint_path)
        
        # Create second model with incompatible config
        config2 = TransformerConfig(
            vocab_size=vocab_size2,  # Different vocab size
            embedding_dim=embedding_dim2,  # Different embedding dim
            num_heads=4,
            num_layers=4,
            hidden_dim=embedding_dim2 * 2,
            max_seq_length=256,
            dropout=0.1
        )
        
        model2 = TransformerModel(config2)
        
        # Loading should raise ValueError
        with pytest.raises(ValueError, match="mismatch|incompatible|Invalid"):
            model2.load_checkpoint(checkpoint_path)
            
    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


# Feature: transformer-upgrade, Property 22: PyTorch Serialization Compatibility
# Validates: Requirements 5.5
@settings(max_examples=50, deadline=None)
@given(
    vocab_size=st.integers(min_value=1000, max_value=3000),
    num_heads=st.sampled_from([2, 4, 8])
)
def test_property_22_pytorch_serialization_format(vocab_size, num_heads):
    """
    Property 22: PyTorch Serialization Compatibility
    
    For any saved checkpoint, the file SHALL be loadable using torch.load()
    and SHALL contain the expected keys.
    """
    embedding_dim = num_heads * 32
    if embedding_dim < 128:
        embedding_dim = 128
    
    config = TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=4,
        hidden_dim=embedding_dim * 2,
        max_seq_length=256,
        dropout=0.1
    )
    
    model = TransformerModel(config)
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.build_from_text("test text for building tokenizer vocabulary")
    model.set_tokenizer(tokenizer)
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        checkpoint_path = f.name
    
    try:
        model.save_checkpoint(checkpoint_path, epoch=10, step=100, loss=0.5)
        
        # Load using torch.load
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify expected keys exist
        assert 'model_state_dict' in checkpoint
        assert 'config' in checkpoint
        assert 'tokenizer' in checkpoint
        assert 'training_info' in checkpoint
        
        # Verify config structure
        assert 'vocab_size' in checkpoint['config']
        assert 'embedding_dim' in checkpoint['config']
        assert 'num_heads' in checkpoint['config']
        assert 'num_layers' in checkpoint['config']
        
        # Verify tokenizer structure
        assert 'type' in checkpoint['tokenizer']
        assert 'vocab' in checkpoint['tokenizer']
        
        # Verify training info
        assert checkpoint['training_info']['epoch'] == 10
        assert checkpoint['training_info']['step'] == 100
        assert checkpoint['training_info']['loss'] == 0.5
        
    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def test_property_22_missing_checkpoint_file():
    """Property 22: Loading non-existent checkpoint SHALL raise FileNotFoundError."""
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    
    with pytest.raises(FileNotFoundError):
        model.load_checkpoint("nonexistent_checkpoint.pt")
