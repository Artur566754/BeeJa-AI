"""Unit tests for TrainingPipeline."""
import os
import tempfile
import shutil
import torch
import pytest
from src.model import CustomAIModel
from src.dataset_loader import DatasetLoader
from src.training_pipeline import TrainingPipeline


class TestTrainingPipeline:
    """Tests for TrainingPipeline"""
    
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory with sample data"""
        temp_dir = tempfile.mkdtemp()
        
        # Create sample dataset
        sample_text = """
        Это тестовый датасет для обучения модели.
        Модель должна научиться генерировать текст на основе этих данных.
        Чем больше данных, тем лучше будет модель.
        Обучение - это важный процесс для любой нейронной сети.
        """
        
        with open(os.path.join(temp_dir, "sample.txt"), 'w', encoding='utf-8') as f:
            f.write(sample_text * 10)  # Repeat to have enough data
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_vocabulary_creation(self, temp_dataset_dir):
        """Test vocabulary creation from text"""
        model = CustomAIModel(vocab_size=100, embedding_dim=32, hidden_dim=64, num_layers=1)
        loader = DatasetLoader(temp_dataset_dir)
        pipeline = TrainingPipeline(model, loader)
        
        text = "Hello, World! This is a test."
        vocab = pipeline.create_vocabulary(text)
        
        assert vocab.vocab_size > 0
        assert len(vocab.char_to_idx) == vocab.vocab_size
    
    def test_data_preparation(self, temp_dataset_dir):
        """Test data preparation with sample text"""
        model = CustomAIModel(vocab_size=100, embedding_dim=32, hidden_dim=64, num_layers=1)
        loader = DatasetLoader(temp_dataset_dir)
        pipeline = TrainingPipeline(model, loader)
        
        text = "This is a test text for data preparation." * 5
        pipeline.vocabulary = pipeline.create_vocabulary(text)
        
        seq_length = 20
        input_data, target_data = pipeline.prepare_data(text, seq_length)
        
        assert input_data.shape[1] == seq_length
        assert target_data.shape[1] == seq_length
        assert input_data.shape[0] == target_data.shape[0]
    
    def test_training_loop_small_dataset(self, temp_dataset_dir):
        """Test training loop with small dataset"""
        model = CustomAIModel(vocab_size=100, embedding_dim=32, hidden_dim=64, num_layers=1)
        loader = DatasetLoader(temp_dataset_dir)
        pipeline = TrainingPipeline(model, loader)
        
        # Train for just 2 epochs
        pipeline.train(epochs=2, learning_rate=0.001, batch_size=16)
        
        # Check that pipeline's model has vocabulary set
        assert pipeline.model.vocabulary is not None
        assert pipeline.model.vocabulary.vocab_size > 0
    
    def test_backup_and_restoration(self, temp_dataset_dir):
        """Test weight backup and restoration"""
        model = CustomAIModel(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        loader = DatasetLoader(temp_dataset_dir)
        pipeline = TrainingPipeline(model, loader)
        
        # Create vocabulary and set it
        text = "Test text for backup"
        vocab = pipeline.create_vocabulary(text)
        model.set_vocabulary(vocab)
        
        # Save backup
        backup_path = "models/test_backup.pth"
        os.makedirs("models", exist_ok=True)
        
        try:
            pipeline.save_backup(backup_path)
            assert os.path.exists(backup_path)
            
            # Modify model weights
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(1.0)
            
            # Restore backup
            pipeline.restore_backup(backup_path)
            
            # Cleanup
            pipeline.cleanup_backup(backup_path)
            assert not os.path.exists(backup_path)
        finally:
            if os.path.exists(backup_path):
                os.remove(backup_path)
    
    def test_insufficient_data_raises_error(self, temp_dataset_dir):
        """Test that insufficient data raises error"""
        # Create empty dataset
        empty_dir = tempfile.mkdtemp()
        
        try:
            with open(os.path.join(empty_dir, "tiny.txt"), 'w', encoding='utf-8') as f:
                f.write("Hi")  # Too short
            
            model = CustomAIModel(vocab_size=100, embedding_dim=32, hidden_dim=64, num_layers=1)
            loader = DatasetLoader(empty_dir)
            pipeline = TrainingPipeline(model, loader)
            
            with pytest.raises(ValueError, match="Insufficient training data"):
                pipeline.train(epochs=1, learning_rate=0.001, batch_size=16)
        finally:
            shutil.rmtree(empty_dir)
