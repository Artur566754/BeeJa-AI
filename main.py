"""Main application entry point for Custom AI Model."""
import os
import argparse
import logging
from src.model import CustomAIModel
from src.dataset_loader import DatasetLoader
from src.training_pipeline import TrainingPipeline
from src.chat_interface import ChatInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_system():
    """
    Initialize the system: create directories, load or create model
    
    Returns:
        Tuple of (model, loader, pipeline)
    """
    # Create dataset directory if not exists
    dataset_dir = "datasets"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        logger.info(f"Created dataset directory: {dataset_dir}")
    
    # Create models directory if not exists
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")
    
    # Initialize dataset loader
    loader = DatasetLoader(dataset_dir)
    
    # Check if model exists
    model_path = "models/ai_model.pth"
    
    if os.path.exists(model_path):
        logger.info("Loading existing model...")
        # Load checkpoint to get correct vocab_size
        import torch
        checkpoint = torch.load(model_path, weights_only=False)
        vocab_size = checkpoint.get('vocab_size', 100)
        
        # Create model with correct vocab_size
        model = CustomAIModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
        try:
            model.load_weights(model_path)
            logger.info(f"Model loaded successfully (vocab_size={vocab_size})")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Creating new model.")
            model = CustomAIModel(vocab_size=100, embedding_dim=128, hidden_dim=256, num_layers=2)
    else:
        logger.info("Creating new model...")
        model = CustomAIModel(vocab_size=100, embedding_dim=128, hidden_dim=256, num_layers=2)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(model, loader)
    
    return model, loader, pipeline


def train_model(pipeline, epochs=50, learning_rate=0.001, batch_size=32):
    """
    Train the model
    
    Args:
        pipeline: TrainingPipeline instance
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
    """
    logger.info("Starting training...")
    
    try:
        pipeline.train(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
        
        # Save model after training
        model_path = "models/ai_model.pth"
        pipeline.model.save_weights(model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def start_chat(model):
    """
    Start chat interface
    
    Args:
        model: CustomAIModel instance
    """
    if model.vocabulary is None:
        logger.error("Model has no vocabulary. Please train the model first.")
        print("\nError: Model not trained yet. Please run with --train flag first.")
        return
    
    chat = ChatInterface(model)
    chat.start_chat()


def auto_train_mode(pipeline):
    """
    Start automatic training mode
    
    Args:
        pipeline: TrainingPipeline instance
    """
    logger.info("Starting automatic training mode...")
    print("\nAutomatic training mode activated.")
    print("The model will automatically retrain when new files are added to the datasets/ directory.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        pipeline.auto_train(watch_interval=5)
    except KeyboardInterrupt:
        logger.info("Automatic training stopped by user")
        print("\nAutomatic training stopped.")


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Custom AI Model - Chat and Training System")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--chat', action='store_true', help='Start chat interface (default)')
    parser.add_argument('--auto', action='store_true', help='Enable automatic training on new files')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    # Initialize system
    logger.info("Initializing Custom AI Model system...")
    model, loader, pipeline = initialize_system()
    
    # Determine mode
    if args.auto:
        auto_train_mode(pipeline)
    elif args.train:
        train_model(pipeline, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)
    else:
        # Default to chat mode
        start_chat(pipeline.model)


if __name__ == "__main__":
    main()
