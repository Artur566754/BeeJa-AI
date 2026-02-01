"""Training pipeline for the AI model."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Tuple, Optional
from src.model import CustomAIModel
from src.dataset_loader import DatasetLoader
from src.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Пайплайн для обучения модели"""
    
    def __init__(self, model: CustomAIModel, dataset_loader: DatasetLoader):
        """
        Args:
            model: Модель для обучения
            dataset_loader: Загрузчик датасетов
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.vocabulary: Optional[Vocabulary] = None
    
    def create_vocabulary(self, text: str) -> Vocabulary:
        """
        Создание словаря символов
        
        Args:
            text: Текст для построения словаря
            
        Returns:
            Маппинг символ -> индекс
        """
        vocab = Vocabulary()
        vocab.build_from_text(text)
        return vocab
    
    def prepare_data(self, text: str, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Подготовка данных для обучения
        
        Args:
            text: Исходный текст
            seq_length: Длина последовательности для обучения
            
        Returns:
            input_sequences: Входные последовательности
            target_sequences: Целевые последовательности (следующий символ)
        """
        if self.vocabulary is None:
            raise ValueError("Vocabulary not created. Call create_vocabulary() first.")
        
        # Encode text
        encoded = self.vocabulary.encode(text)
        
        if len(encoded) < seq_length + 1:
            raise ValueError(f"Text too short. Need at least {seq_length + 1} characters.")
        
        # Create sequences
        input_seqs = []
        target_seqs = []
        
        for i in range(len(encoded) - seq_length):
            input_seq = encoded[i:i + seq_length]
            target_seq = encoded[i + 1:i + seq_length + 1]
            
            input_seqs.append(input_seq)
            target_seqs.append(target_seq)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_seqs, dtype=torch.long)
        target_tensor = torch.tensor(target_seqs, dtype=torch.long)
        
        return input_tensor, target_tensor
    
    def train(self, epochs: int, learning_rate: float, batch_size: int) -> None:
        """
        Обучение модели
        
        Args:
            epochs: Количество эпох обучения
            learning_rate: Скорость обучения
            batch_size: Размер батча
        """
        backup_path = "models/backup_weights.pth"
        
        # Save backup before training
        if self.model.vocabulary is not None:
            try:
                self.save_backup(backup_path)
            except Exception as e:
                logger.warning(f"Could not save backup: {e}")
        
        try:
            # Load all datasets
            text, errors = self.dataset_loader.load_all_datasets()
            
            if not text or len(text) < 100:
                raise ValueError("Insufficient training data. Need at least 100 characters.")
            
            if errors:
                logger.warning(f"Encountered {len(errors)} errors while loading datasets")
            
            # Create or update vocabulary
            self.vocabulary = self.create_vocabulary(text)
            
            # Update model vocabulary and vocab_size if needed
            if self.model.vocab_size != self.vocabulary.vocab_size:
                logger.info(f"Updating model vocab_size from {self.model.vocab_size} to {self.vocabulary.vocab_size}")
                # Recreate model with new vocab size
                from src.model import CustomAIModel
                new_model = CustomAIModel(
                    vocab_size=self.vocabulary.vocab_size,
                    embedding_dim=self.model.embedding_dim,
                    hidden_dim=self.model.hidden_dim,
                    num_layers=self.model.num_layers
                )
                self.model = new_model
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            self.model.set_vocabulary(self.vocabulary)
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            
            # Prepare data
            seq_length = 100
            input_data, target_data = self.prepare_data(text, seq_length)
            
            logger.info(f"Training on {len(input_data)} sequences")
            
            # Training loop
            self.model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                # Create batches
                for i in range(0, len(input_data), batch_size):
                    batch_input = input_data[i:i + batch_size]
                    batch_target = target_data[i:i + batch_size]
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    output, _ = self.model.forward(batch_input)
                    
                    # Reshape for loss calculation
                    output = output.reshape(-1, self.model.vocab_size)
                    target = batch_target.reshape(-1)
                    
                    # Calculate loss
                    loss = self.criterion(output, target)
                    
                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"Training diverged at epoch {epoch}, batch {num_batches}")
                        raise RuntimeError("Training diverged: loss is NaN or Inf")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    # Update weights
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            
            logger.info("Training completed successfully")
            
            # Clean up backup after successful training
            self.cleanup_backup(backup_path)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Restore from backup on failure
            if os.path.exists(backup_path):
                try:
                    self.restore_backup(backup_path)
                    logger.info("Restored model from backup after training failure")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {restore_error}")
            raise
    
    def save_backup(self, backup_path: str) -> None:
        """Save model weights backup"""
        if self.model.vocabulary is not None:
            self.model.save_weights(backup_path)
            logger.info(f"Backup saved to {backup_path}")
    
    def restore_backup(self, backup_path: str) -> None:
        """Restore model weights from backup"""
        if os.path.exists(backup_path):
            self.model.load_weights(backup_path)
            logger.info(f"Restored from backup: {backup_path}")
    
    def cleanup_backup(self, backup_path: str) -> None:
        """Remove backup file"""
        if os.path.exists(backup_path):
            os.remove(backup_path)
            logger.info(f"Removed backup: {backup_path}")
    
    def auto_train(self, watch_interval: int = 5) -> None:
        """
        Автоматическое обучение при обнаружении новых файлов
        
        Args:
            watch_interval: Интервал проверки в секундах
        """
        import time
        
        logger.info(f"Starting auto-training mode. Watching {self.dataset_loader.dataset_dir}")
        
        # Track known files
        known_files = set(self.dataset_loader.get_supported_files())
        logger.info(f"Initial files: {len(known_files)}")
        
        while True:
            time.sleep(watch_interval)
            
            # Check for new files
            current_files = set(self.dataset_loader.get_supported_files())
            new_files = current_files - known_files
            
            if new_files:
                logger.info(f"Detected {len(new_files)} new file(s): {new_files}")
                
                try:
                    # Trigger training
                    self.train(epochs=50, learning_rate=0.001, batch_size=32)
                    
                    # Update known files
                    known_files = current_files
                    logger.info("Auto-training completed successfully")
                except Exception as e:
                    logger.error(f"Auto-training failed: {e}")
                    # Continue watching despite error
