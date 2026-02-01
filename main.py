"""Main application entry point for Custom AI Model."""
import os
import argparse
import logging
from src.model import CustomAIModel
from src.transformer_model import TransformerModel
from src.config import TransformerConfig, TransformerTrainingConfig
from src.dataset_loader import DatasetLoader
from src.training_pipeline import TrainingPipeline
from src.chat_interface import ChatInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_system(model_type='lstm', model_size='small'):
    """
    Initialize the system: create directories, load or create model
    
    Args:
        model_type: 'lstm' or 'transformer'
        model_size: 'small', 'medium', or 'large' (for Transformer)
    
    Returns:
        Tuple of (model, loader, pipeline)
        
    Validates: Requirements 4.1
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
    
    # Determine model path based on type
    if model_type == 'transformer':
        model_path = f"models/transformer_{model_size}.pth"
    else:
        model_path = "models/ai_model.pth"
    
    # Load or create model based on type
    if model_type == 'transformer':
        model = initialize_transformer(model_path, model_size)
    else:
        model = initialize_lstm(model_path)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(model, loader)
    
    return model, loader, pipeline


def initialize_lstm(model_path):
    """Initialize LSTM model (backward compatible)."""
    if os.path.exists(model_path):
        logger.info("Loading existing LSTM model...")
        # Load checkpoint to get correct vocab_size
        import torch
        checkpoint = torch.load(model_path, weights_only=False)
        vocab_size = checkpoint.get('vocab_size', 100)
        
        # Create model with correct vocab_size
        model = CustomAIModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
        try:
            model.load_weights(model_path)
            logger.info(f"LSTM model loaded successfully (vocab_size={vocab_size})")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Creating new model.")
            model = CustomAIModel(vocab_size=100, embedding_dim=128, hidden_dim=256, num_layers=2)
    else:
        logger.info("Creating new LSTM model...")
        model = CustomAIModel(vocab_size=100, embedding_dim=128, hidden_dim=256, num_layers=2)
    
    return model


def initialize_transformer(model_path, model_size='small'):
    """Initialize Transformer model."""
    if os.path.exists(model_path):
        logger.info(f"Loading existing Transformer model ({model_size})...")
        try:
            # Load checkpoint to get config
            import torch
            checkpoint = torch.load(model_path, weights_only=False)
            config_dict = checkpoint.get('config', {})
            config = TransformerConfig(**config_dict)
            
            # Create model
            model = TransformerModel(config)
            model.load_checkpoint(model_path)
            logger.info(f"Transformer model loaded successfully (vocab_size={config.vocab_size})")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Creating new model.")
            # Create new model with default vocab_size
            if model_size == 'small':
                config = TransformerConfig.small(vocab_size=5000)
            elif model_size == 'medium':
                config = TransformerConfig.medium(vocab_size=5000)
            elif model_size == 'large':
                config = TransformerConfig.large(vocab_size=5000)
            else:
                config = TransformerConfig.small(vocab_size=5000)
            model = TransformerModel(config)
    else:
        logger.info(f"Creating new Transformer model ({model_size})...")
        # Create new model
        if model_size == 'small':
            config = TransformerConfig.small(vocab_size=5000)
        elif model_size == 'medium':
            config = TransformerConfig.medium(vocab_size=5000)
        elif model_size == 'large':
            config = TransformerConfig.large(vocab_size=5000)
        else:
            config = TransformerConfig.small(vocab_size=5000)
        model = TransformerModel(config)
    
    return model


def train_model(pipeline, epochs=50, learning_rate=0.001, batch_size=32, 
                model_type='lstm', model_size='small', use_mixed_precision=False):
    """
    Train the model
    
    Args:
        pipeline: TrainingPipeline instance
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        model_type: 'lstm' or 'transformer'
        model_size: 'small', 'medium', or 'large'
        use_mixed_precision: Whether to use mixed precision training
    """
    logger.info(f"Starting training ({model_type} model)...")
    
    try:
        # Train with appropriate parameters
        if isinstance(pipeline.model, TransformerModel):
            # Transformer training with additional parameters
            pipeline.train(
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                use_mixed_precision=use_mixed_precision
            )
            # Save model after training
            model_path = f"models/transformer_{model_size}.pth"
            pipeline.model.save_checkpoint(model_path)
        else:
            # LSTM training (backward compatible)
            pipeline.train(
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
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
        model: CustomAIModel or TransformerModel instance
    """
    # Check if model is ready
    if isinstance(model, CustomAIModel):
        if model.vocabulary is None:
            logger.error("LSTM model has no vocabulary. Please train the model first.")
            print("\nError: Model not trained yet. Please run with --train flag first.")
            return
    elif isinstance(model, TransformerModel):
        if model.tokenizer is None:
            logger.error("Transformer model has no tokenizer. Please train the model first.")
            print("\nError: Model not trained yet. Please run with --train flag first.")
            return
    
    # Create chat interface with context support for Transformer
    use_context = isinstance(model, TransformerModel)
    chat = ChatInterface(model, use_context=use_context)
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
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'transformer'],
                        help='Model type: lstm or transformer (default: lstm)')
    parser.add_argument('--model-size', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='Model size for Transformer: small, medium, or large (default: small)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--mixed-precision', action='store_true', 
                        help='Use mixed precision training (GPU only)')
    
    args = parser.parse_args()
    
    # Adjust learning rate for Transformer
    if args.model_type == 'transformer' and args.lr == 0.001:
        args.lr = 0.0001  # Lower default for Transformer
    
    # Initialize system
    logger.info(f"Initializing Custom AI Model system ({args.model_type})...")
    model, loader, pipeline = initialize_system(
        model_type=args.model_type,
        model_size=args.model_size
    )
    
    # Determine mode
    if args.auto:
        auto_train_mode(pipeline)
    elif args.train:
        train_model(
            pipeline,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            model_type=args.model_type,
            model_size=args.model_size,
            use_mixed_precision=args.mixed_precision
        )
    else:
        # Default to chat mode
        start_chat(pipeline.model)


if __name__ == "__main__":
    main()
