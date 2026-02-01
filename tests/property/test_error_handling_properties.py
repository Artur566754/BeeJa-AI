"""Property-based tests for error handling."""
import pytest
from hypothesis import given, settings, strategies as st
import torch
import os
import tempfile
from src.transformer_model import TransformerModel
from src.config import TransformerConfig
from src.tokenizer import BPETokenizer
from src.chat_interface import ChatInterface


# Feature: transformer-upgrade, Property 23: Model Loading Error Handling
# **Validates: Requirements 6.5**


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_23_missing_checkpoint_error_handling(seed):
    """
    Property 23: Model Loading Error Handling
    
    For any error during model loading (missing file), the system SHALL handle 
    the error gracefully without crashing and SHALL provide a meaningful error message.
    
    **Validates: Requirements 6.5**
    """
    torch.manual_seed(seed)
    
    # Create model
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    
    # Try to load non-existent checkpoint
    non_existent_path = f"non_existent_checkpoint_{seed}.pth"
    
    # Should raise FileNotFoundError with meaningful message
    with pytest.raises(FileNotFoundError):
        model.load_checkpoint(non_existent_path)


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_23_corrupted_checkpoint_error_handling(seed):
    """
    Test that corrupted checkpoint files are handled gracefully.
    
    **Validates: Requirements 6.5**
    """
    torch.manual_seed(seed)
    
    # Create model
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    
    # Create a corrupted checkpoint file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pth', delete=False) as f:
        f.write("This is not a valid checkpoint file")
        corrupted_path = f.name
    
    try:
        # Try to load corrupted checkpoint
        # Should raise an error (RuntimeError or similar)
        with pytest.raises(Exception) as exc_info:
            model.load_checkpoint(corrupted_path)
        
        # Error message should be meaningful (not empty)
        assert len(str(exc_info.value)) > 0, "Error message should not be empty"
        
    finally:
        # Clean up
        if os.path.exists(corrupted_path):
            os.remove(corrupted_path)


@settings(max_examples=50, deadline=None)
@given(
    vocab_size1=st.integers(min_value=500, max_value=1000),
    vocab_size2=st.integers(min_value=1001, max_value=1500),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_23_incompatible_checkpoint_error_handling(vocab_size1, vocab_size2, seed):
    """
    Test that incompatible checkpoint configurations are detected.
    
    **Validates: Requirements 6.5**
    """
    torch.manual_seed(seed)
    
    # Create model with vocab_size1
    config1 = TransformerConfig.small(vocab_size=vocab_size1)
    model1 = TransformerModel(config1)
    
    # Create tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size1)
    tokenizer.build_from_text("hello world test " * 20)
    model1.set_tokenizer(tokenizer)
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        checkpoint_path = f.name
    
    try:
        model1.save_checkpoint(checkpoint_path)
        
        # Create model with different vocab_size
        config2 = TransformerConfig.small(vocab_size=vocab_size2)
        model2 = TransformerModel(config2)
        
        # Try to load incompatible checkpoint
        # Should raise an error about incompatibility
        with pytest.raises(Exception) as exc_info:
            model2.load_checkpoint(checkpoint_path)
        
        # Error should mention incompatibility or mismatch
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ['incompatible', 'mismatch', 'size', 'shape']), \
            f"Error message should indicate incompatibility: {error_msg}"
        
    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


@settings(max_examples=50, deadline=None)
@given(
    user_input=st.text(min_size=1, max_size=100),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_chat_interface_handles_generation_errors_gracefully(user_input, seed):
    """
    Test that ChatInterface handles generation errors without crashing.
    
    **Validates: Requirements 6.5**
    """
    torch.manual_seed(seed)
    
    # Create model and tokenizer
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.build_from_text("hello world test " * 20)
    model.set_tokenizer(tokenizer)
    
    # Create chat interface
    chat = ChatInterface(model, use_context=True)
    
    # Process message should not crash, even with unusual input
    try:
        response = chat.process_message(user_input)
        
        # Response should be a string
        assert isinstance(response, str), "Response should be a string"
        
        # Response should not be empty (either generated text or error message)
        assert len(response) > 0, "Response should not be empty"
        
    except Exception as e:
        # If an exception occurs, it should be caught and handled
        # The test should not fail here
        pytest.fail(f"ChatInterface should handle errors gracefully, but raised: {e}")


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10000)
)
def test_model_without_tokenizer_error_handling(seed):
    """
    Test that attempting to generate without a tokenizer is handled.
    
    **Validates: Requirements 6.5**
    """
    torch.manual_seed(seed)
    
    # Create model without setting tokenizer
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    
    # Try to generate without tokenizer
    # Should raise an error with meaningful message
    with pytest.raises(Exception) as exc_info:
        model.generate(seed_text="hello", max_length=10)
    
    # Error message should mention tokenizer
    error_msg = str(exc_info.value).lower()
    assert 'tokenizer' in error_msg, \
        f"Error message should mention tokenizer: {error_msg}"


@settings(max_examples=30, deadline=None)
@given(
    max_length=st.integers(min_value=-10, max_value=0),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_invalid_generation_parameters_error_handling(max_length, seed):
    """
    Test that invalid generation parameters are handled.
    
    **Validates: Requirements 6.5**
    """
    torch.manual_seed(seed)
    
    # Create model and tokenizer
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.build_from_text("hello world test " * 20)
    model.set_tokenizer(tokenizer)
    
    # Try to generate with invalid max_length
    # Should raise ValueError or handle gracefully
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        model.generate(seed_text="hello", max_length=max_length)
    
    # Error message should be meaningful
    assert len(str(exc_info.value)) > 0, "Error message should not be empty"


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10000)
)
def test_empty_seed_text_error_handling(seed):
    """
    Test that empty seed text is handled gracefully.
    
    **Validates: Requirements 6.5**
    """
    torch.manual_seed(seed)
    
    # Create model and tokenizer
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.build_from_text("hello world test " * 20)
    model.set_tokenizer(tokenizer)
    
    # Try to generate with empty seed text
    # Should either handle gracefully or raise meaningful error
    try:
        response = model.generate(seed_text="", max_length=10)
        # If it succeeds, response should be a string
        assert isinstance(response, str)
    except Exception as e:
        # If it raises an error, message should be meaningful
        assert len(str(e)) > 0, "Error message should not be empty"
