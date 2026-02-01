"""Property-based tests for TrainingPipeline with Transformer support."""
import pytest
import torch
import tempfile
import os
from hypothesis import given, settings, strategies as st
from src.config import TransformerConfig
from src.transformer_model import TransformerModel
from src.training_pipeline import TrainingPipeline
from src.dataset_loader import DatasetLoader


def create_test_model_and_pipeline(vocab_size=1000):
    """Helper to create a small model and training pipeline for testing."""
    config = TransformerConfig(
        vocab_size=vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=256,
        max_seq_length=128,
        dropout=0.1
    )
    
    model = TransformerModel(config)
    
    # Create a temporary dataset directory
    temp_dir = tempfile.mkdtemp()
    dataset_loader = DatasetLoader(temp_dir)
    
    pipeline = TrainingPipeline(model, dataset_loader)
    
    return model, pipeline, temp_dir


# Feature: transformer-upgrade, Property 17: Batch Shape Correctness
# Validates: Requirements 4.2
@settings(max_examples=100, deadline=None)
@given(
    seq_length=st.integers(min_value=10, max_value=50),
    batch_size=st.integers(min_value=1, max_value=8)
)
def test_property_17_batch_shape_correctness(seq_length, batch_size):
    """
    Property 17: Batch Shape Correctness
    
    For any prepared training batch, the input tensor SHALL have shape
    [batch_size, seq_len] and the target tensor SHALL have shape [batch_size, seq_len],
    with all values being valid token indices.
    """
    model, pipeline, temp_dir = create_test_model_and_pipeline()
    
    try:
        # Create training text
        text = "hello world test sample text for training " * 20
        
        # Create tokenizer
        pipeline.tokenizer = pipeline.create_tokenizer(text)
        
        # Prepare data
        input_data, target_data = pipeline.prepare_data(text, seq_length)
        
        # Check shapes
        assert input_data.dim() == 2, "Input should be 2D tensor"
        assert target_data.dim() == 2, "Target should be 2D tensor"
        
        assert input_data.size(1) == seq_length, f"Input seq_len should be {seq_length}"
        assert target_data.size(1) == seq_length, f"Target seq_len should be {seq_length}"
        
        # Check that batch sizes match
        assert input_data.size(0) == target_data.size(0), "Batch sizes should match"
        
        # Check that all values are valid token indices
        assert torch.all(input_data >= 0), "All input indices should be non-negative"
        assert torch.all(input_data < model.vocab_size), f"All input indices should be < {model.vocab_size}"
        assert torch.all(target_data >= 0), "All target indices should be non-negative"
        assert torch.all(target_data < model.vocab_size), f"All target indices should be < {model.vocab_size}"
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# Feature: transformer-upgrade, Property 18: Loss Non-Negativity
# Validates: Requirements 4.3
@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_18_loss_non_negativity(seed):
    """
    Property 18: Loss Non-Negativity
    
    For any training batch, the computed cross-entropy loss SHALL be
    non-negative and finite (not NaN or Inf).
    """
    torch.manual_seed(seed)
    
    model, pipeline, temp_dir = create_test_model_and_pipeline()
    
    try:
        # Create training text
        text = "hello world test sample text for training " * 10
        
        # Create tokenizer
        pipeline.tokenizer = pipeline.create_tokenizer(text)
        model.set_tokenizer(pipeline.tokenizer)
        
        # Prepare data
        input_data, target_data = pipeline.prepare_data(text, seq_length=20)
        
        # Get a batch
        batch_input = input_data[:4]
        batch_target = target_data[:4]
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch_input)
            
            # Reshape for loss
            output = output.reshape(-1, model.vocab_size)
            target = batch_target.reshape(-1)
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(output, target)
            
            # Verify loss properties
            assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"
            assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
            assert not torch.isnan(loss), "Loss should not be NaN"
            assert not torch.isinf(loss), "Loss should not be Inf"
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# Feature: transformer-upgrade, Property 19: Gradient Clipping
# Validates: Requirements 4.4
@settings(max_examples=50, deadline=None)
@given(
    max_grad_norm=st.floats(min_value=0.1, max_value=5.0)
)
def test_property_19_gradient_clipping(max_grad_norm):
    """
    Property 19: Gradient Clipping
    
    For any training step with gradient clipping norm N, after clipping,
    the global gradient norm SHALL be at most N.
    """
    model, pipeline, temp_dir = create_test_model_and_pipeline()
    
    try:
        # Create training text
        text = "hello world test sample text for training " * 10
        
        # Create tokenizer
        pipeline.tokenizer = pipeline.create_tokenizer(text)
        model.set_tokenizer(pipeline.tokenizer)
        
        # Prepare data
        input_data, target_data = pipeline.prepare_data(text, seq_length=20)
        
        # Get a batch
        batch_input = input_data[:4]
        batch_target = target_data[:4]
        
        # Forward pass
        model.train()
        output = model(batch_input)
        
        # Compute loss
        output = output.reshape(-1, model.vocab_size)
        target = batch_target.reshape(-1)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # Compute global gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Verify gradient norm is at most max_grad_norm (with tolerance for numerical errors)
        assert total_norm <= max_grad_norm + 1e-3, \
            f"Gradient norm {total_norm} should be <= {max_grad_norm}"
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


# Feature: transformer-upgrade, Property 24: Gradient Accumulation Equivalence
# Validates: Requirements 8.3
@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=1000),
    batch_size=st.integers(min_value=2, max_value=4),
    accumulation_steps=st.integers(min_value=2, max_value=4)
)
def test_property_24_gradient_accumulation_equivalence(seed, batch_size, accumulation_steps):
    """
    Property 24: Gradient Accumulation Equivalence
    
    For any training data, training with batch size B and gradient accumulation
    steps G SHALL produce equivalent weight updates to training with batch size
    B*G and no gradient accumulation (within numerical precision).
    """
    torch.manual_seed(seed)
    
    # Create two identical models
    config = TransformerConfig(
        vocab_size=1000,
        embedding_dim=128,  # Must be >= 128
        num_heads=2,
        num_layers=2,
        hidden_dim=256,
        max_seq_length=64,
        dropout=0.0  # Disable dropout for deterministic behavior
    )
    
    model1 = TransformerModel(config)
    model2 = TransformerModel(config)
    
    # Copy weights to ensure they start identical
    model2.load_state_dict(model1.state_dict())
    
    # Create optimizers
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    
    # Create training data
    text = "hello world test sample text for training gradient accumulation " * 10
    
    temp_dir1 = tempfile.mkdtemp()
    temp_dir2 = tempfile.mkdtemp()
    
    try:
        dataset_loader1 = DatasetLoader(temp_dir1)
        dataset_loader2 = DatasetLoader(temp_dir2)
        
        pipeline1 = TrainingPipeline(model1, dataset_loader1)
        pipeline2 = TrainingPipeline(model2, dataset_loader2)
        
        pipeline1.tokenizer = pipeline1.create_tokenizer(text)
        pipeline2.tokenizer = pipeline2.create_tokenizer(text)
        
        model1.set_tokenizer(pipeline1.tokenizer)
        model2.set_tokenizer(pipeline2.tokenizer)
        
        input_data, target_data = pipeline1.prepare_data(text, seq_length=20)
        
        # Ensure we have enough data
        total_samples = batch_size * accumulation_steps
        if len(input_data) < total_samples:
            return  # Skip if not enough data
        
        # Model 1: Use gradient accumulation with small batches
        model1.train()
        optimizer1.zero_grad()
        
        for step in range(accumulation_steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            
            batch_input = input_data[start_idx:end_idx]
            batch_target = target_data[start_idx:end_idx]
            
            output = model1(batch_input)
            output = output.reshape(-1, model1.vocab_size)
            target = batch_target.reshape(-1)
            
            loss = torch.nn.functional.cross_entropy(output, target)
            loss = loss / accumulation_steps  # Scale loss
            loss.backward()
        
        optimizer1.step()
        
        # Model 2: Use large batch without accumulation
        model2.train()
        optimizer2.zero_grad()
        
        large_batch_input = input_data[:total_samples]
        large_batch_target = target_data[:total_samples]
        
        output = model2(large_batch_input)
        output = output.reshape(-1, model2.vocab_size)
        target = large_batch_target.reshape(-1)
        
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer2.step()
        
        # Compare weights (should be very similar, allowing for numerical precision)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, rtol=1e-4, atol=1e-5), \
                "Gradient accumulation should produce equivalent updates"
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


def test_training_pipeline_model_type_detection():
    """Test that pipeline correctly detects model type."""
    # Test with Transformer
    config = TransformerConfig.small(vocab_size=1000)
    transformer_model = TransformerModel(config)
    temp_dir = tempfile.mkdtemp()
    
    try:
        dataset_loader = DatasetLoader(temp_dir)
        pipeline = TrainingPipeline(transformer_model, dataset_loader)
        
        assert pipeline.is_transformer == True, "Should detect Transformer model"
    
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_tokenizer_creation():
    """Test that appropriate tokenizer is created based on model type."""
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    temp_dir = tempfile.mkdtemp()
    
    try:
        dataset_loader = DatasetLoader(temp_dir)
        pipeline = TrainingPipeline(model, dataset_loader)
        
        text = "hello world test sample text for tokenizer"
        tokenizer = pipeline.create_tokenizer(text)
        
        # Should create BPETokenizer for Transformer
        from src.tokenizer import BPETokenizer
        assert isinstance(tokenizer, BPETokenizer), "Should create BPETokenizer for Transformer"
    
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
