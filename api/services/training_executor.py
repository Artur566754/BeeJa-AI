"""
Training Executor service for the Server Management API.

This module wraps the existing PyTorch training pipeline and provides
callbacks for metrics and logging integration.
"""

from datetime import datetime, timezone
from typing import Callable, Optional, Dict, Any
from pathlib import Path
import threading
import logging

from api.models.data_models import TrainingConfig, Metrics
from api.config import APIConfig


logger = logging.getLogger(__name__)


class TrainingExecutor:
    """
    Executes training sessions with callbacks for metrics and logs.
    
    Wraps the existing training pipeline and provides integration points
    for the API to monitor progress and collect metrics.
    """
    
    def __init__(self):
        """Initialize the TrainingExecutor."""
        self._active_sessions: Dict[str, threading.Thread] = {}
        self._stop_flags: Dict[str, threading.Event] = {}
    
    def execute_training(
        self,
        session_id: str,
        config: TrainingConfig,
        metrics_callback: Callable[[Metrics], None],
        log_callback: Callable[[str, str], None]
    ) -> None:
        """
        Execute a training session with callbacks.
        
        This is a simplified implementation that simulates training.
        In a real implementation, this would integrate with the actual
        PyTorch training pipeline (src/training_pipeline.py).
        
        Args:
            session_id: Unique session identifier
            config: Training configuration
            metrics_callback: Callback function for metrics updates
                             Takes Metrics object as parameter
            log_callback: Callback function for log messages
                         Takes (message, level) as parameters
        
        Raises:
            Exception: If training fails
        """
        try:
            log_callback("Starting training session", "INFO")
            log_callback(
                f"Configuration: {config.model_architecture}, "
                f"dataset: {config.dataset_name}, "
                f"epochs: {config.epochs}, "
                f"lr: {config.learning_rate}, "
                f"batch_size: {config.batch_size}",
                "INFO"
            )
            
            # Create stop flag for this session
            stop_flag = threading.Event()
            self._stop_flags[session_id] = stop_flag
            
            # Simulate training epochs
            for epoch in range(config.epochs):
                # Check if stop was requested
                if stop_flag.is_set():
                    log_callback(f"Training stopped at epoch {epoch + 1}", "WARNING")
                    break
                
                log_callback(f"Starting epoch {epoch + 1}/{config.epochs}", "INFO")
                
                # Simulate training metrics
                # In real implementation, these would come from actual training
                loss = 1.0 / (epoch + 1)  # Decreasing loss
                accuracy = min(0.95, 0.5 + (epoch * 0.05))  # Increasing accuracy
                val_loss = loss * 1.1  # Slightly higher validation loss
                val_accuracy = accuracy * 0.95  # Slightly lower validation accuracy
                
                # Create metrics object
                metrics = Metrics(
                    session_id=session_id,
                    epoch=epoch + 1,
                    loss=loss,
                    accuracy=accuracy,
                    val_loss=val_loss,
                    val_accuracy=val_accuracy,
                    timestamp=datetime.now(timezone.utc)
                )
                
                # Call metrics callback
                metrics_callback(metrics)
                
                log_callback(
                    f"Epoch {epoch + 1} completed - "
                    f"loss: {loss:.4f}, accuracy: {accuracy:.4f}",
                    "INFO"
                )
                
                # Simulate epoch duration
                import time
                time.sleep(0.1)  # Short sleep for testing
            
            # Training completed successfully
            if not stop_flag.is_set():
                log_callback("Training completed successfully", "INFO")
                
                # Save model
                model_name = f"{config.model_architecture}_{config.dataset_name}"
                model_path = self._save_model(session_id, model_name)
                log_callback(f"Model saved to {model_path}", "INFO")
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            log_callback(error_msg, "ERROR")
            logger.exception(f"Training session {session_id} failed")
            raise
        
        finally:
            # Cleanup
            if session_id in self._stop_flags:
                del self._stop_flags[session_id]
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
    
    def stop_training(self, session_id: str) -> bool:
        """
        Request graceful stop of a training session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if stop was requested, False if session not found
        """
        if session_id in self._stop_flags:
            self._stop_flags[session_id].set()
            logger.info(f"Stop requested for training session {session_id}")
            return True
        return False
    
    def _save_model(self, session_id: str, model_name: str) -> Path:
        """
        Save the trained model.
        
        In a real implementation, this would save the actual PyTorch model.
        For now, it creates a placeholder file.
        
        Args:
            session_id: Session identifier
            model_name: Name for the model
            
        Returns:
            Path to saved model file
        """
        models_dir = APIConfig.MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pth"
        model_path = models_dir / filename
        
        # In real implementation, would save actual model:
        # torch.save(model.state_dict(), model_path)
        
        # For now, create a placeholder
        model_path.write_bytes(b"placeholder model data")
        
        return model_path
    
    def is_session_active(self, session_id: str) -> bool:
        """
        Check if a training session is currently active.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session is active, False otherwise
        """
        return session_id in self._active_sessions


# Global training executor instance
_executor_instance: Optional[TrainingExecutor] = None


def get_training_executor() -> TrainingExecutor:
    """
    Get the global training executor instance.
    
    Returns:
        TrainingExecutor instance
    """
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = TrainingExecutor()
    return _executor_instance
