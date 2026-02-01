"""Unit tests for mixed precision training support."""
import pytest
import torch
from src.training_pipeline import TrainingPipeline, AMP_AVAILABLE
from src.transformer_model import TransformerModel
from src.config import TransformerConfig
from src.dataset_loader import DatasetLoader
from src.tokenizer import BPETokenizer


class TestMixedPrecision:
    """Test mixed precision training support."""
    
    def test_amp_availability_detection(self):
        """Test that AMP availability is correctly detected."""
        # AMP_AVAILABLE should be True for PyTorch 1.6+
        assert isinstance(AMP_AVAILABLE, bool)
    
    def test_enable_mixed_precision_on_cpu(self):
        """Test that mixed precision is disabled on CPU."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        model.to_cpu()
        
        dataset_loader = DatasetLoader("datasets")
        pipeline = TrainingPipeline(model, dataset_loader)
        pipeline.set_device('cpu')
        
        # Try to enable mixed precision on CPU
        pipeline.enable_mixed_precision(True)
        
        # Should be disabled on CPU
        assert pipeline.use_amp is False
        assert pipeline.scaler is None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_enable_mixed_precision_on_gpu(self):
        """Test that mixed precision can be enabled on GPU."""
        if not AMP_AVAILABLE:
            pytest.skip("AMP not available")
        
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        dataset_loader = DatasetLoader("datasets")
        pipeline = TrainingPipeline(model, dataset_loader)
        pipeline.set_device('cuda')
        
        # Enable mixed precision on GPU
        pipeline.enable_mixed_precision(True)
        
        # Should be enabled on GPU
        assert pipeline.use_amp is True
        assert pipeline.scaler is not None
    
    def test_disable_mixed_precision(self):
        """Test disabling mixed precision."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        dataset_loader = DatasetLoader("datasets")
        pipeline = TrainingPipeline(model, dataset_loader)
        
        # Disable mixed precision
        pipeline.enable_mixed_precision(False)
        
        assert pipeline.use_amp is False
        assert pipeline.scaler is None
    
    @pytest.mark.skipif(not torch.cuda.is_available() or not AMP_AVAILABLE, 
                        reason="CUDA or AMP not available")
    def test_mixed_precision_forward_pass(self):
        """Test forward pass with mixed precision."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        model.to_gpu(0)
        
        # Create input on GPU
        input_ids = torch.randint(0, 1000, (2, 50), device='cuda:0')
        
        # Enable mixed precision
        from torch.cuda.amp import autocast
        
        with autocast():
            logits = model(input_ids)
        
        assert logits.shape == (2, 50, 1000)
        assert logits.device.type == 'cuda'
        # In mixed precision, intermediate computations may use float16
        # but output is typically float32
    
    def test_training_with_mixed_precision_parameter(self):
        """Test that training accepts use_mixed_precision parameter."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        # Set tokenizer
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.build_from_text("hello world test sample text for training")
        model.set_tokenizer(tokenizer)
        
        dataset_loader = DatasetLoader("datasets")
        pipeline = TrainingPipeline(model, dataset_loader)
        
        # Create a small test dataset
        import os
        os.makedirs("datasets", exist_ok=True)
        with open("datasets/test_mixed_precision.txt", "w") as f:
            f.write("hello world " * 100)
        
        try:
            # Train with mixed precision parameter (will be disabled on CPU)
            pipeline.train(
                epochs=1,
                learning_rate=0.001,
                batch_size=2,
                use_mixed_precision=True
            )
            
            # Training should complete without errors
            assert True
        finally:
            # Clean up
            if os.path.exists("datasets/test_mixed_precision.txt"):
                os.remove("datasets/test_mixed_precision.txt")
    
    @pytest.mark.skipif(not torch.cuda.is_available() or not AMP_AVAILABLE,
                        reason="CUDA or AMP not available")
    def test_gradient_scaling_with_amp(self):
        """Test that gradient scaling works with AMP."""
        from torch.cuda.amp import GradScaler
        
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        model.to_gpu(0)
        
        dataset_loader = DatasetLoader("datasets")
        pipeline = TrainingPipeline(model, dataset_loader)
        pipeline.set_device('cuda')
        pipeline.enable_mixed_precision(True)
        
        # Verify scaler is created
        assert isinstance(pipeline.scaler, GradScaler)
    
    def test_mixed_precision_state_persistence(self):
        """Test that mixed precision state can be toggled."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        dataset_loader = DatasetLoader("datasets")
        pipeline = TrainingPipeline(model, dataset_loader)
        
        # Initially disabled
        assert pipeline.use_amp is False
        
        # Enable (will fail on CPU but state should update)
        pipeline.enable_mixed_precision(True)
        
        # Disable
        pipeline.enable_mixed_precision(False)
        assert pipeline.use_amp is False
        assert pipeline.scaler is None
    
    def test_training_without_mixed_precision(self):
        """Test that training works without mixed precision (baseline)."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        # Set tokenizer
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.build_from_text("hello world test sample text for training")
        model.set_tokenizer(tokenizer)
        
        dataset_loader = DatasetLoader("datasets")
        pipeline = TrainingPipeline(model, dataset_loader)
        
        # Create a small test dataset
        import os
        os.makedirs("datasets", exist_ok=True)
        with open("datasets/test_no_mixed_precision.txt", "w") as f:
            f.write("hello world " * 100)
        
        try:
            # Train without mixed precision
            pipeline.train(
                epochs=1,
                learning_rate=0.001,
                batch_size=2,
                use_mixed_precision=False
            )
            
            # Training should complete without errors
            assert True
        finally:
            # Clean up
            if os.path.exists("datasets/test_no_mixed_precision.txt"):
                os.remove("datasets/test_no_mixed_precision.txt")
