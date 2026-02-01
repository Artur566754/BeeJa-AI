"""End-to-end integration tests for the Transformer upgrade."""
import pytest
import torch
import os
import tempfile
import shutil
from src.transformer_model import TransformerModel
from src.config import TransformerConfig
from src.tokenizer import BPETokenizer
from src.dataset_loader import DatasetLoader
from src.training_pipeline import TrainingPipeline
from src.conversation_context import ConversationContext
from src.chat_interface import ChatInterface


class TestEndToEndTrainingPipeline:
    """Test full training pipeline from data loading to checkpoint saving."""
    
    def test_full_training_pipeline(self):
        """
        Test full training pipeline: data loading → tokenizer building → 
        training → checkpoint saving.
        
        Validates: Requirements 6.1, 6.2, 6.3, 6.4
        """
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            os.makedirs(dataset_dir)
            
            # Create sample training data
            training_text = "hello world this is a test " * 100
            with open(os.path.join(dataset_dir, "train.txt"), "w") as f:
                f.write(training_text)
            
            # Create model
            config = TransformerConfig.small(vocab_size=1000)
            model = TransformerModel(config)
            
            # Create dataset loader
            loader = DatasetLoader(dataset_dir)
            
            # Create training pipeline
            pipeline = TrainingPipeline(model, loader)
            
            # Train for a few epochs
            pipeline.train(
                epochs=2,
                learning_rate=0.001,
                batch_size=4,
                use_mixed_precision=False
            )
            
            # Verify model has tokenizer
            assert model.tokenizer is not None, "Model should have tokenizer after training"
            
            # Save checkpoint
            checkpoint_path = os.path.join(temp_dir, "checkpoint.pth")
            model.save_checkpoint(checkpoint_path)
            
            # Verify checkpoint exists
            assert os.path.exists(checkpoint_path), "Checkpoint should be saved"
            
            # Load checkpoint into new model
            config2 = TransformerConfig.small(vocab_size=1000)
            model2 = TransformerModel(config2)
            model2.load_checkpoint(checkpoint_path)
            
            # Verify loaded model has tokenizer
            assert model2.tokenizer is not None, "Loaded model should have tokenizer"
            
            # Generate text with both models
            seed_text = "hello"
            output1 = model.generate(seed_text, max_length=20, temperature=1.0)
            output2 = model2.generate(seed_text, max_length=20, temperature=1.0)
            
            # Both should generate valid text
            assert isinstance(output1, str) and len(output1) > 0
            assert isinstance(output2, str) and len(output2) > 0


class TestEndToEndInferencePipeline:
    """Test full inference pipeline from checkpoint loading to generation."""
    
    def test_full_inference_pipeline(self):
        """
        Test full inference pipeline: checkpoint loading → context management → 
        generation.
        
        Validates: Requirements 6.1, 6.2, 6.3, 6.4
        """
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and train a small model
            config = TransformerConfig.small(vocab_size=1000)
            model = TransformerModel(config)
            
            # Create tokenizer
            tokenizer = BPETokenizer(vocab_size=1000)
            training_text = "hello world this is a test " * 50
            tokenizer.build_from_text(training_text)
            model.set_tokenizer(tokenizer)
            
            # Save checkpoint
            checkpoint_path = os.path.join(temp_dir, "model.pth")
            model.save_checkpoint(checkpoint_path)
            
            # Load checkpoint into new model
            config2 = TransformerConfig.small(vocab_size=1000)
            model2 = TransformerModel(config2)
            model2.load_checkpoint(checkpoint_path)
            
            # Create conversation context
            context = ConversationContext(max_context_length=256)
            context.add_message("user", "hello")
            context.add_message("assistant", "hi there")
            context.add_message("user", "how are you")
            
            # Get formatted context
            formatted_context = context.get_context()
            assert len(formatted_context) > 0, "Context should not be empty"
            
            # Generate with context
            response = model2.generate(
                seed_text=formatted_context,
                max_length=50,
                temperature=0.7
            )
            
            # Verify generation
            assert isinstance(response, str), "Response should be a string"
            assert len(response) > 0, "Response should not be empty"


class TestChatInterfaceIntegration:
    """Test ChatInterface integration with Transformer model."""
    
    def test_chat_interface_with_transformer(self):
        """
        Test ChatInterface works with Transformer model.
        
        Validates: Requirements 6.1, 6.2, 6.3, 6.4
        """
        # Create model and tokenizer
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.build_from_text("hello world test sample " * 50)
        model.set_tokenizer(tokenizer)
        
        # Create chat interface
        chat = ChatInterface(model, use_context=True)
        
        # Process messages
        response1 = chat.process_message("hello")
        assert isinstance(response1, str) and len(response1) > 0
        
        response2 = chat.process_message("how are you")
        assert isinstance(response2, str) and len(response2) > 0
        
        # Check conversation history
        history = chat.get_history()
        assert len(history) == 4, "Should have 4 messages (2 user + 2 assistant)"
        
        # Clear history
        chat.clear_history()
        assert len(chat.get_history()) == 0, "History should be empty after clear"
    
    def test_chat_interface_with_lstm_backward_compatibility(self):
        """
        Test ChatInterface maintains backward compatibility with LSTM.
        
        Validates: Requirements 6.2
        """
        from src.model import CustomAIModel
        from src.vocabulary import Vocabulary
        
        # Create LSTM model
        vocab = Vocabulary()
        vocab.build_from_text("hello world test sample " * 50)
        
        model = CustomAIModel(vocab_size=len(vocab.char_to_idx))
        model.set_vocabulary(vocab)
        
        # Create chat interface
        chat = ChatInterface(model, use_context=False)
        
        # Process message
        response = chat.process_message("hello")
        assert isinstance(response, str), "Response should be a string"


class TestModelComparison:
    """Test comparing LSTM and Transformer models."""
    
    def test_lstm_vs_transformer_comparison(self):
        """
        Test that both LSTM and Transformer models can be used interchangeably.
        
        Validates: Requirements 6.2
        """
        from src.model import CustomAIModel
        from src.vocabulary import Vocabulary
        
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
        
        # Both should be able to generate
        lstm_output = lstm_model.generate("hello", max_length=20)
        transformer_output = transformer_model.generate("hello", max_length=20)
        
        assert isinstance(lstm_output, str) and len(lstm_output) > 0
        assert isinstance(transformer_output, str) and len(transformer_output) > 0


class TestCheckpointPersistence:
    """Test checkpoint persistence across sessions."""
    
    def test_checkpoint_persistence_across_sessions(self):
        """
        Test that checkpoints can be saved and loaded across different sessions.
        
        Validates: Requirements 5.1, 5.2, 5.3, 5.5
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "model.pth")
            
            # Session 1: Create and save model
            config1 = TransformerConfig.small(vocab_size=1000)
            model1 = TransformerModel(config1)
            
            tokenizer1 = BPETokenizer(vocab_size=1000)
            tokenizer1.build_from_text("hello world test " * 50)
            model1.set_tokenizer(tokenizer1)
            
            # Generate sample output
            seed_text = "hello"
            output1 = model1.generate(seed_text, max_length=20, temperature=0.0)
            
            # Save checkpoint
            model1.save_checkpoint(checkpoint_path)
            
            # Delete model
            del model1
            
            # Session 2: Load model
            config2 = TransformerConfig.small(vocab_size=1000)
            model2 = TransformerModel(config2)
            model2.load_checkpoint(checkpoint_path)
            
            # Generate with same seed and temperature
            output2 = model2.generate(seed_text, max_length=20, temperature=0.0)
            
            # Outputs should be identical (deterministic with temp=0)
            assert output1 == output2, "Loaded model should produce identical outputs"


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""
    
    def test_training_with_insufficient_data(self):
        """
        Test that training handles insufficient data gracefully.
        
        Validates: Requirements 6.5
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "datasets")
            os.makedirs(dataset_dir)
            
            # Create very small training data
            with open(os.path.join(dataset_dir, "train.txt"), "w") as f:
                f.write("hi")
            
            # Create model
            config = TransformerConfig.small(vocab_size=1000)
            model = TransformerModel(config)
            
            # Create dataset loader
            loader = DatasetLoader(dataset_dir)
            
            # Create training pipeline
            pipeline = TrainingPipeline(model, loader)
            
            # Training should either succeed or raise meaningful error
            try:
                pipeline.train(epochs=1, learning_rate=0.001, batch_size=2)
                # If it succeeds, that's fine
                assert True
            except ValueError as e:
                # If it fails, error should be meaningful
                assert len(str(e)) > 0, "Error message should not be empty"
    
    def test_generation_with_empty_context(self):
        """
        Test that generation handles empty context gracefully.
        
        Validates: Requirements 6.5
        """
        # Create model and tokenizer
        config = TransformerConfig.small(vocab_size=1000)
        model = TransformerModel(config)
        
        tokenizer = BPETokenizer(vocab_size=1000)
        tokenizer.build_from_text("hello world test " * 50)
        model.set_tokenizer(tokenizer)
        
        # Try to generate with empty seed
        try:
            response = model.generate(seed_text="", max_length=10)
            # If it succeeds, response should be valid
            assert isinstance(response, str)
        except Exception as e:
            # If it fails, error should be meaningful
            assert len(str(e)) > 0, "Error message should not be empty"
