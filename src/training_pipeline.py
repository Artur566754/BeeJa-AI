"""Training pipeline for the AI model."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Tuple, Optional, Union
from tqdm import tqdm
from src.model import CustomAIModel
from src.dataset_loader import DatasetLoader
from src.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

# Check if AMP is available (PyTorch 1.6+)
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    logger.warning("Automatic Mixed Precision (AMP) not available. Using full precision training.")


class TrainingPipeline:
    """Пайплайн для обучения модели"""
    
    def __init__(self, model: Union[CustomAIModel, 'TransformerModel'], dataset_loader: DatasetLoader):
        """
        Args:
            model: Модель для обучения (LSTM или Transformer)
            dataset_loader: Загрузчик датасетов
        """
        self.model = model
        self.dataset_loader = dataset_loader
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.vocabulary: Optional[Vocabulary] = None
        self.tokenizer: Optional['BPETokenizer'] = None
        
        # Detect model type
        self.is_transformer = self._is_transformer_model(model)
        
        # Device management
        self.device = self._get_device()
        
        # Mixed precision training
        self.use_amp = False
        self.scaler = None
    
    def _is_transformer_model(self, model) -> bool:
        """Check if model is a Transformer model."""
        return hasattr(model, 'transformer_blocks')
    
    def _get_device(self) -> torch.device:
        """Get the device for training (GPU if available, else CPU)."""
        if self.is_transformer and hasattr(self.model, 'get_device'):
            return self.model.get_device()
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_device(self, device: str) -> None:
        """
        Set the device for training.
        
        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.device = torch.device(device)
        if self.is_transformer and hasattr(self.model, 'to_device'):
            self.model.to_device(device)
        else:
            self.model.to(self.device)
        logger.info(f"Training device set to: {self.device}")
    
    def enable_mixed_precision(self, enable: bool = True) -> None:
        """
        Enable or disable mixed precision training (AMP).
        
        Args:
            enable: Whether to enable mixed precision training
            
        Note:
            Mixed precision training is only available on CUDA devices with AMP support.
        """
        if enable:
            if not AMP_AVAILABLE:
                logger.warning("AMP not available. Mixed precision training disabled.")
                self.use_amp = False
                return
            
            if not torch.cuda.is_available() or self.device.type != 'cuda':
                logger.warning("Mixed precision training requires CUDA. Disabled.")
                self.use_amp = False
                return
            
            self.use_amp = True
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.use_amp = False
            self.scaler = None
            logger.info("Mixed precision training disabled")
    
    def create_tokenizer(self, text: str):
        """
        Create appropriate tokenizer based on model type.
        
        Args:
            text: Text to build tokenizer from
            
        Returns:
            Tokenizer (Vocabulary for LSTM, BPETokenizer for Transformer)
        """
        if self.is_transformer:
            from src.tokenizer import BPETokenizer
            tokenizer = BPETokenizer(vocab_size=self.model.vocab_size)
            tokenizer.build_from_text(text)
            return tokenizer
        else:
            vocab = Vocabulary()
            vocab.build_from_text(text)
            return vocab
    
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
        if self.is_transformer:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not created. Call create_tokenizer() first.")
            encoded = self.tokenizer.encode(text)
        else:
            if self.vocabulary is None:
                raise ValueError("Vocabulary not created. Call create_vocabulary() first.")
            encoded = self.vocabulary.encode(text)
        
        if len(encoded) < seq_length + 1:
            raise ValueError(f"Text too short. Need at least {seq_length + 1} tokens.")
        
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
    
    def train(self, epochs: int, learning_rate: float, batch_size: int, 
              gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0,
              use_mixed_precision: bool = False) -> None:
        """
        Обучение модели
        
        Args:
            epochs: Количество эпох обучения
            learning_rate: Скорость обучения
            batch_size: Размер батча
            gradient_accumulation_steps: Steps for gradient accumulation (Transformer only)
            max_grad_norm: Maximum gradient norm for clipping
            use_mixed_precision: Use automatic mixed precision training (GPU only)
        """
        # Enable mixed precision if requested
        if use_mixed_precision:
            self.enable_mixed_precision(True)
        
        backup_path = "models/backup_weights.pth"
        
        # Save backup before training
        has_tokenizer_or_vocab = (
            (self.is_transformer and self.model.tokenizer is not None) or
            (not self.is_transformer and self.model.vocabulary is not None)
        )
        
        if has_tokenizer_or_vocab:
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
            
            # Create or update vocabulary/tokenizer
            if self.is_transformer:
                self.tokenizer = self.create_tokenizer(text)
                self.model.set_tokenizer(self.tokenizer)
                seq_length = min(self.model.max_seq_length // 2, 100)  # Use shorter sequences for Transformer
            else:
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
                seq_length = 100
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            
            # Prepare data
            input_data, target_data = self.prepare_data(text, seq_length)
            
            num_sequences = len(input_data)
            num_batches_per_epoch = (num_sequences + batch_size - 1) // batch_size
            
            logger.info(f"Training on {num_sequences} sequences")
            logger.info(f"Batches per epoch: {num_batches_per_epoch}")
            logger.info(f"Total epochs: {epochs}")
            print(f"\n{'='*60}")
            print(f"Starting training: {epochs} epochs, {num_sequences} sequences")
            print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
            print(f"{'='*60}\n")
            
            # Training loop
            self.model.train()
            
            # Progress bar for epochs
            epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch", position=0)
            
            for epoch in epoch_pbar:
                total_loss = 0
                num_batches = 0
                accumulation_counter = 0
                
                # Progress bar for batches within epoch
                batch_pbar = tqdm(
                    range(0, len(input_data), batch_size),
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    unit="batch",
                    leave=False,
                    position=1
                )
                
                # Create batches
                for i in batch_pbar:
                    batch_input = input_data[i:i + batch_size].to(self.device)
                    batch_target = target_data[i:i + batch_size].to(self.device)
                    
                    # Use mixed precision if enabled
                    if self.use_amp and AMP_AVAILABLE:
                        with autocast():
                            # Forward pass (different for LSTM vs Transformer)
                            if self.is_transformer:
                                output = self.model.forward(batch_input)  # Transformer returns logits directly
                            else:
                                output, _ = self.model.forward(batch_input)  # LSTM returns output and hidden state
                            
                            # Reshape for loss calculation
                            output = output.reshape(-1, self.model.vocab_size)
                            target = batch_target.reshape(-1)
                            
                            # Calculate loss
                            loss = self.criterion(output, target)
                            
                            # Scale loss for gradient accumulation
                            if gradient_accumulation_steps > 1:
                                loss = loss / gradient_accumulation_steps
                    else:
                        # Standard precision forward pass
                        if self.is_transformer:
                            output = self.model.forward(batch_input)
                        else:
                            output, _ = self.model.forward(batch_input)
                        
                        # Reshape for loss calculation
                        output = output.reshape(-1, self.model.vocab_size)
                        target = batch_target.reshape(-1)
                        
                        # Calculate loss
                        loss = self.criterion(output, target)
                        
                        # Scale loss for gradient accumulation
                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps
                    
                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"Training diverged at epoch {epoch}, batch {num_batches}")
                        raise RuntimeError("Training diverged: loss is NaN or Inf")
                    
                    # Backward pass
                    if self.use_amp and AMP_AVAILABLE:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    accumulation_counter += 1
                    
                    # Update weights after accumulation steps
                    if accumulation_counter >= gradient_accumulation_steps:
                        if self.use_amp and AMP_AVAILABLE:
                            # Unscale gradients before clipping
                            self.scaler.unscale_(self.optimizer)
                            
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                            
                            # Update weights with scaled gradients
                            self.scaler.step(self.optimizer)
                            
                            # Update scaler
                            self.scaler.update()
                        else:
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                            
                            # Update weights
                            self.optimizer.step()
                        
                        # Zero gradients
                        self.optimizer.zero_grad()
                        
                        accumulation_counter = 0
                    
                    total_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
                    num_batches += 1
                    
                    # Update batch progress bar with current loss
                    batch_pbar.set_postfix({
                        'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'avg_loss': f'{total_loss/num_batches:.4f}'
                    })
                
                # Handle remaining gradients if any
                if accumulation_counter > 0:
                    if self.use_amp and AMP_AVAILABLE:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                
                # Compute perplexity
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                
                # Update epoch progress bar
                epoch_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'perplexity': f'{perplexity:.2f}',
                    'batches': num_batches
                })
                
                # Log every epoch
                logger.info(f"Epoch {epoch + 1}/{epochs} completed - Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            print(f"\n{'='*60}")
            print(f"Training completed successfully!")
            print(f"Final average loss: {avg_loss:.4f}")
            print(f"Final perplexity: {perplexity:.2f}")
            print(f"{'='*60}\n")
            
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
        if self.is_transformer:
            if self.model.tokenizer is not None:
                self.model.save_checkpoint(backup_path)
                logger.info(f"Backup saved to {backup_path}")
        else:
            if self.model.vocabulary is not None:
                self.model.save_weights(backup_path)
                logger.info(f"Backup saved to {backup_path}")
    
    def restore_backup(self, backup_path: str) -> None:
        """Restore model weights from backup"""
        if os.path.exists(backup_path):
            if self.is_transformer:
                self.model.load_checkpoint(backup_path)
            else:
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
