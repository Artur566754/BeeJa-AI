"""Integration tests for LSTM checkpoint migration and backward compatibility."""
import pytest
import torch
import os
import tempfile
from src.model import CustomAIModel
from src.transformer_model import TransformerModel
from src.config import TransformerConfig
from src.vocabulary import Vocabulary
from src.tokenizer import BPETokenizer
from src.training_pipeline import TrainingPipeline
from src.dataset_loader import DatasetLoader


class TestLSTMCheckpointLoading:
    """Test loading LSTM checkpoints for backward compatibility."""
    
    def test_load_lstm_checkpoint(self):
        """
        Test loading LSTM checkpoints.
        
        Validates: Requirements 5.4
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save LSTM model
            vocab = Vocabulary()
            vocab.build_from_text("hello world test sample " * 50)
            
            lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            lstm_model.set_vocabulary(vocab)
            
            # Save LSTM checkpoint
            checkpoint_path = os.path.join(temp_dir, "lstm_model.pth")
            lstm_model.save_weights(checkpoint_path)
            
            # Load LSTM checkpoint into new model
            lstm_model2 = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            lstm_model2.load_weights(checkpoint_path)
            
            # Verify loaded model works
            output = lstm_model2.generate("hello", max_length=20)
            assert isinstance(output, str) and len(output) > 0
    
    def test_lstm_checkpoint_format(self):
        """
        Test that LSTM checkpoints have the expected format.
        
        Validates: Requirements 5.4
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create LSTM model
            vocab = Vocabulary()
            vocab.build_from_text("hello world test " * 50)
            
            lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            lstm_model.set_vocabulary(vocab)
            
            # Save checkpoint
            checkpoint_path = os.path.join(temp_dir, "lstm_model.pth")
            lstm_model.save_weights(checkpoint_path)
            
            # Load checkpoint and verify format
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            # Check expected keys
            assert 'model_state_dict' in checkpoint, "Should have model_state_dict"
            assert 'vocab_size' in checkpoint, "Should have vocab_size"
            assert 'vocabulary' in checkpoint, "Should have vocabulary"
            
            # Verify vocabulary structure
            vocab_data = checkpoint['vocabulary']
            assert 'char_to_idx' in vocab_data, "Vocabulary should have char_to_idx"
            assert 'idx_to_char' in vocab_data, "Vocabulary should have idx_to_char"


class TestBackwardCompatibility:
    """Test backward compatibility between LSTM and Transformer."""
    
    def test_training_pipeline_supports_both_models(self):
        """
        Test that TrainingPipeline supports both LSTM and Transformer models.
        
        Validates: Requirements 5.4, 6.2
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            os.makedirs(dataset_dir)
            
            # Create training data
            training_text = "hello world test " * 100
            with open(os.path.join(dataset_dir, "train.txt"), "w") as f:
                f.write(training_text)
            
            loader = DatasetLoader(dataset_dir)
            
            # Test with LSTM
            vocab = Vocabulary()
            vocab.build_from_text(training_text)
            lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            
            lstm_pipeline = TrainingPipeline(lstm_model, loader)
            lstm_pipeline.train(epochs=1, learning_rate=0.001, batch_size=4)
            
            # Verify LSTM model trained
            assert lstm_model.vocabulary is not None
            
            # Test with Transformer
            config = TransformerConfig.small(vocab_size=1000)
            transformer_model = TransformerModel(config)
            
            transformer_pipeline = TrainingPipeline(transformer_model, loader)
            transformer_pipeline.train(epochs=1, learning_rate=0.001, batch_size=4)
            
            # Verify Transformer model trained
            assert transformer_model.tokenizer is not None
    
    def test_both_models_can_generate(self):
        """
        Test that both LSTM and Transformer can generate text.
        
        Validates: Requirements 5.4, 6.2
        """
        # Create LSTM model
        vocab = Vocabulary()
        vocab.build_from_text("hello world test " * 50)
        lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
        lstm_model.set_vocabulary(vocab)
        
        # Create Transformer model
        config = TransformerConfig.small(vocab_size=1000)
        transformer_model = TransformerModel(config)
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.build_from_text("hello world test " * 50)
        transformer_model.set_tokenizer(tokenizer)
        
        # Both should generate
        lstm_output = lstm_model.generate("hello", max_length=20)
        transformer_output = transformer_model.generate("hello", max_length=20)
        
        assert isinstance(lstm_output, str) and len(lstm_output) > 0
        assert isinstance(transformer_output, str) and len(transformer_output) > 0
    
    def test_checkpoint_formats_are_distinct(self):
        """
        Test that LSTM and Transformer checkpoints have distinct formats.
        
        Validates: Requirements 5.4, 5.5
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save LSTM checkpoint
            vocab = Vocabulary()
            vocab.build_from_text("hello world test " * 50)
            lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            lstm_model.set_vocabulary(vocab)
            
            lstm_path = os.path.join(temp_dir, "lstm.pth")
            lstm_model.save_weights(lstm_path)
            
            # Create and save Transformer checkpoint
            config = TransformerConfig.small(vocab_size=1000)
            transformer_model = TransformerModel(config)
            tokenizer = BPETokenizer(vocab_size=1000)
            tokenizer.build_from_text("hello world test " * 50)
            transformer_model.set_tokenizer(tokenizer)
            
            transformer_path = os.path.join(temp_dir, "transformer.pth")
            transformer_model.save_checkpoint(transformer_path)
            
            # Load both checkpoints
            lstm_checkpoint = torch.load(lstm_path, weights_only=False)
            transformer_checkpoint = torch.load(transformer_path, weights_only=False)
            
            # LSTM checkpoint should have 'vocabulary'
            assert 'vocabulary' in lstm_checkpoint
            
            # Transformer checkpoint should have 'config' and 'tokenizer'
            assert 'config' in transformer_checkpoint
            assert 'tokenizer' in transformer_checkpoint
            
            # They should have different structures
            assert 'vocabulary' not in transformer_checkpoint
            assert 'config' not in lstm_checkpoint


class TestMigrationScenarios:
    """Test various migration scenarios from LSTM to Transformer."""
    
    def test_cannot_load_lstm_checkpoint_into_transformer(self):
        """
        Test that loading LSTM checkpoint into Transformer raises appropriate error.
        
        Validates: Requirements 5.3, 5.4
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save LSTM checkpoint
            vocab = Vocabulary()
            vocab.build_from_text("hello world test " * 50)
            lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            lstm_model.set_vocabulary(vocab)
            
            lstm_path = os.path.join(temp_dir, "lstm.pth")
            lstm_model.save_weights(lstm_path)
            
            # Try to load into Transformer
            config = TransformerConfig.small(vocab_size=1000)
            transformer_model = TransformerModel(config)
            
            # Should raise an error
            with pytest.raises(Exception):
                transformer_model.load_checkpoint(lstm_path)
    
    def test_cannot_load_transformer_checkpoint_into_lstm(self):
        """
        Test that loading Transformer checkpoint into LSTM raises appropriate error.
        
        Validates: Requirements 5.3, 5.4
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save Transformer checkpoint
            config = TransformerConfig.small(vocab_size=1000)
            transformer_model = TransformerModel(config)
            tokenizer = BPETokenizer(vocab_size=1000)
            tokenizer.build_from_text("hello world test " * 50)
            transformer_model.set_tokenizer(tokenizer)
            
            transformer_path = os.path.join(temp_dir, "transformer.pth")
            transformer_model.save_checkpoint(transformer_path)
            
            # Try to load into LSTM
            vocab = Vocabulary()
            vocab.build_from_text("hello world test " * 50)
            lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            
            # Should raise an error
            with pytest.raises(Exception):
                lstm_model.load_weights(transformer_path)
    
    def test_separate_checkpoint_paths_for_different_models(self):
        """
        Test that LSTM and Transformer use separate checkpoint paths.
        
        Validates: Requirements 5.4
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save both models to different paths
            vocab = Vocabulary()
            vocab.build_from_text("hello world test " * 50)
            lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            lstm_model.set_vocabulary(vocab)
            
            lstm_path = os.path.join(temp_dir, "lstm_model.pth")
            lstm_model.save_weights(lstm_path)
            
            config = TransformerConfig.small(vocab_size=1000)
            transformer_model = TransformerModel(config)
            tokenizer = BPETokenizer(vocab_size=1000)
            tokenizer.build_from_text("hello world test " * 50)
            transformer_model.set_tokenizer(tokenizer)
            
            transformer_path = os.path.join(temp_dir, "transformer_model.pth")
            transformer_model.save_checkpoint(transformer_path)
            
            # Both files should exist
            assert os.path.exists(lstm_path)
            assert os.path.exists(transformer_path)
            
            # They should be different files
            assert lstm_path != transformer_path
            
            # Load each into appropriate model
            lstm_model2 = CustomAIModel(vocab_size=len(vocab.char_to_idx))
            lstm_model2.load_weights(lstm_path)
            
            config2 = TransformerConfig.small(vocab_size=1000)
            transformer_model2 = TransformerModel(config2)
            transformer_model2.load_checkpoint(transformer_path)
            
            # Both should work
            assert lstm_model2.vocabulary is not None
            assert transformer_model2.tokenizer is not None


class TestCoexistence:
    """Test that LSTM and Transformer can coexist in the same system."""
    
    def test_both_models_in_same_session(self):
        """
        Test that both LSTM and Transformer can be used in the same session.
        
        Validates: Requirements 5.4, 6.2
        """
        # Create LSTM model
        vocab = Vocabulary()
        vocab.build_from_text("hello world test " * 50)
        lstm_model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
        lstm_model.set_vocabulary(vocab)
        
        # Create Transformer model
        config = TransformerConfig.small(vocab_size=1000)
        transformer_model = TransformerModel(config)
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.build_from_text("hello world test " * 50)
        transformer_model.set_tokenizer(tokenizer)
        
        # Use both models
        lstm_output = lstm_model.generate("hello", max_length=20)
        transformer_output = transformer_model.generate("hello", max_length=20)
        
        # Both should work independently
        assert isinstance(lstm_output, str) and len(lstm_output) > 0
        assert isinstance(transformer_output, str) and len(transformer_output) > 0
        
        # They should not interfere with each other
        lstm_output2 = lstm_model.generate("world", max_length=20)
        transformer_output2 = transformer_model.generate("world", max_length=20)
        
        assert isinstance(lstm_output2, str) and len(lstm_output2) > 0
        assert isinstance(transformer_output2, str) and len(transformer_output2) > 0
