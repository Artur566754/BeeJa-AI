"""Unit tests for device management (CPU/GPU support)."""
import pytest
import torch
from src.transformer_model import TransformerModel
from src.config import TransformerConfig
from src.tokenizer import BPETokenizer


class TestDeviceManagement:
    """Test CPU and GPU device management."""
    
    def test_default_device_is_cpu(self):
        """Test that model defaults to CPU device."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        assert model.get_device().type == 'cpu'
    
    def test_to_cpu(self):
        """Test moving model to CPU."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        model.to_cpu()
        
        assert model.get_device().type == 'cpu'
        # Verify parameters are on CPU
        for param in model.parameters():
            assert param.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_gpu(self):
        """Test moving model to GPU."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        model.to_gpu(0)
        
        assert model.get_device().type == 'cuda'
        # Verify parameters are on GPU
        for param in model.parameters():
            assert param.device.type == 'cuda'
    
    def test_to_gpu_without_cuda_raises_error(self):
        """Test that requesting GPU without CUDA raises error."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test error case")
        
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            model.to_gpu(0)
    
    def test_to_device_with_string(self):
        """Test moving model using device string."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        model.to_device('cpu')
        assert model.get_device().type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_with_cuda_string(self):
        """Test moving model to GPU using device string."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        model.to_device('cuda:0')
        assert model.get_device().type == 'cuda'
    
    def test_auto_device_selects_cpu_when_no_cuda(self):
        """Test auto_device selects CPU when CUDA not available."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test CPU fallback")
        
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        model.auto_device()
        assert model.get_device().type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_auto_device_selects_gpu_when_available(self):
        """Test auto_device selects GPU when CUDA available."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        model.auto_device()
        assert model.get_device().type == 'cuda'
    
    def test_forward_pass_on_cpu(self):
        """Test forward pass works on CPU."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        model.to_cpu()
        
        # Create input on CPU
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Forward pass
        logits = model(input_ids)
        
        assert logits.shape == (2, 10, 1000)
        assert logits.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass_on_gpu(self):
        """Test forward pass works on GPU."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        model.to_gpu(0)
        
        # Create input on GPU
        input_ids = torch.randint(0, 1000, (2, 10), device='cuda:0')
        
        # Forward pass
        logits = model(input_ids)
        
        assert logits.shape == (2, 10, 1000)
        assert logits.device.type == 'cuda'
    
    def test_device_persistence_after_checkpoint_save_load(self):
        """Test device setting persists after save/load."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        # Set tokenizer
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.build_from_text("hello world test")
        model.set_tokenizer(tokenizer)
        
        # Save checkpoint
        model.save_checkpoint('test_device_checkpoint.pth')
        
        # Create new model and load
        model2 = TransformerModel(config)
        model2.load_checkpoint('test_device_checkpoint.pth')
        
        # Device should be CPU (default)
        assert model2.get_device().type == 'cpu'
        
        # Clean up
        import os
        os.remove('test_device_checkpoint.pth')
    
    def test_method_chaining(self):
        """Test that device methods support chaining."""
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        # Chain multiple device operations
        result = model.to_cpu().to_cpu()
        
        assert result is model
        assert model.get_device().type == 'cpu'
