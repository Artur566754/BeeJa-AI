"""Transformer decoder model for text generation."""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os
from src.config import TransformerConfig
from src.transformer_components import (
    PositionalEncoding,
    TransformerBlock,
    create_causal_mask
)


class TransformerModel(nn.Module):
    """Transformer decoder model for autoregressive text generation."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize Transformer model with configuration.
        
        Args:
            config: TransformerConfig with model hyperparameters
        """
        super().__init__()
        
        # Validate configuration
        config.validate()
        
        self.config = config
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.max_seq_length = config.max_seq_length
        self.dropout = config.dropout
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            config.embedding_dim,
            config.max_seq_length,
            config.dropout
        )
        
        # Stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.embedding_dim,
                config.num_heads,
                config.hidden_dim,
                config.dropout,
                config.use_memory_efficient
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)
        
        # Tokenizer (to be set externally)
        self.tokenizer = None
        
        # Device management
        self._device = torch.device('cpu')
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Optional mask [batch_size, seq_len] or [seq_len, seq_len]
            
        Returns:
            logits: Predictions [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        
        # Check sequence length
        if seq_len > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_length}"
            )
        
        # Token embedding
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_len, device=input_ids.device)
        
        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def set_tokenizer(self, tokenizer):
        """
        Set the tokenizer for encoding/decoding.
        
        Args:
            tokenizer: BPETokenizer or Vocabulary instance
        """
        self.tokenizer = tokenizer
    
    def get_config(self) -> TransformerConfig:
        """
        Get model configuration.
        
        Returns:
            config: TransformerConfig instance
        """
        return self.config
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.
        
        Returns:
            num_params: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self) -> torch.device:
        """
        Get the device the model is on.
        
        Returns:
            device: torch.device (cpu or cuda)
        """
        return self._device
    
    def to_device(self, device: str) -> 'TransformerModel':
        """
        Move model to specified device (CPU or GPU).
        
        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', etc.)
            
        Returns:
            self: Model instance for chaining
            
        Raises:
            RuntimeError: If CUDA is requested but not available
        """
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        
        device_obj = torch.device(device)
        self.to(device_obj)
        self._device = device_obj
        return self
    
    def to_cpu(self) -> 'TransformerModel':
        """
        Move model to CPU.
        
        Returns:
            self: Model instance for chaining
        """
        return self.to_device('cpu')
    
    def to_gpu(self, gpu_id: int = 0) -> 'TransformerModel':
        """
        Move model to GPU.
        
        Args:
            gpu_id: GPU device ID (default: 0)
            
        Returns:
            self: Model instance for chaining
            
        Raises:
            RuntimeError: If CUDA is not available
        """
        return self.to_device(f'cuda:{gpu_id}')
    
    def auto_device(self) -> 'TransformerModel':
        """
        Automatically select best available device (GPU if available, else CPU).
        
        Returns:
            self: Model instance for chaining
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.to_device(device)
    
    def generate(
        self,
        prompt: str,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        strategy: str = 'nucleus',
        conversation_context: Optional['ConversationContext'] = None
    ) -> str:
        """
        Generate text using various sampling strategies.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate (excluding prompt)
            temperature: Temperature for sampling (higher = more random)
            top_k: Number of top tokens for top-k sampling
            top_p: Cumulative probability threshold for nucleus sampling
            repetition_penalty: Penalty for repeating tokens (> 1.0 discourages repetition)
            strategy: Sampling strategy ('greedy', 'top_k', 'nucleus')
            conversation_context: Optional conversation context for contextual generation
            
        Returns:
            generated_text: Generated text string
            
        Raises:
            ValueError: If tokenizer is not set or invalid parameters
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before generation")
        
        # If conversation context is provided, use it to format the prompt
        if conversation_context is not None:
            prompt = conversation_context.get_context_for_generation(prompt)
        
        # Import here to avoid circular dependency
        from src.text_generator import TextGenerator
        
        generator = TextGenerator(self, self.tokenizer)
        
        # Use appropriate generation strategy
        if strategy == 'greedy':
            return generator.generate_greedy(prompt, max_length)
        elif strategy == 'top_k':
            if top_k is None:
                top_k = 50
            return generator.generate_top_k(prompt, max_length, k=top_k, temperature=temperature)
        elif strategy == 'nucleus':
            if top_p is None:
                top_p = 0.9
            return generator.generate_nucleus(prompt, max_length, p=top_p, temperature=temperature)
        else:
            # Default: use repetition penalty with specified strategy
            return generator.generate_with_repetition_penalty(
                prompt, max_length, temperature, top_k, top_p, repetition_penalty
            )
    
    def save_checkpoint(
        self,
        filepath: str,
        optimizer_state: Optional[Dict] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        loss: Optional[float] = None
    ) -> None:
        """
        Save model checkpoint with config and tokenizer.
        
        Args:
            filepath: Path to save checkpoint
            optimizer_state: Optional optimizer state dict
            epoch: Optional current epoch number
            step: Optional current step number
            loss: Optional current loss value
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be set before saving checkpoint")
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.config.vocab_size,
                'embedding_dim': self.config.embedding_dim,
                'num_heads': self.config.num_heads,
                'num_layers': self.config.num_layers,
                'hidden_dim': self.config.hidden_dim,
                'max_seq_length': self.config.max_seq_length,
                'dropout': self.config.dropout
            },
            'tokenizer': self._serialize_tokenizer(),
            'training_info': {
                'epoch': epoch,
                'step': step,
                'loss': loss,
                'optimizer_state': optimizer_state
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = False) -> Optional[Dict]:
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to return optimizer state
            
        Returns:
            optimizer_state: Optimizer state dict if load_optimizer=True, else None
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is incompatible with model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Validate checkpoint format
            required_keys = ['model_state_dict', 'config', 'tokenizer']
            for key in required_keys:
                if key not in checkpoint:
                    raise ValueError(f"Invalid checkpoint: missing key '{key}'")
            
            # Validate configuration compatibility
            saved_config = checkpoint['config']
            if saved_config['vocab_size'] != self.config.vocab_size:
                raise ValueError(
                    f"Vocabulary size mismatch: model has {self.config.vocab_size}, "
                    f"checkpoint has {saved_config['vocab_size']}"
                )
            if saved_config['embedding_dim'] != self.config.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: model has {self.config.embedding_dim}, "
                    f"checkpoint has {saved_config['embedding_dim']}"
                )
            if saved_config['num_layers'] != self.config.num_layers:
                raise ValueError(
                    f"Number of layers mismatch: model has {self.config.num_layers}, "
                    f"checkpoint has {saved_config['num_layers']}"
                )
            
            # Load model weights
            self.load_state_dict(checkpoint['model_state_dict'])
            
            # Load tokenizer
            self._deserialize_tokenizer(checkpoint['tokenizer'])
            
            # Return optimizer state if requested
            if load_optimizer and 'training_info' in checkpoint:
                return checkpoint['training_info'].get('optimizer_state')
            
            return None
            
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {str(e)}")
    
    def _serialize_tokenizer(self) -> Dict[str, Any]:
        """
        Serialize tokenizer for checkpoint.
        
        Returns:
            tokenizer_dict: Serialized tokenizer data
        """
        # Check tokenizer type
        if hasattr(self.tokenizer, 'merges'):
            # BPETokenizer
            return {
                'type': 'bpe',
                'vocab': self.tokenizer.vocab,
                'merges': self.tokenizer.merges,
                'vocab_size': self.tokenizer.vocab_size
            }
        elif hasattr(self.tokenizer, 'char_to_idx'):
            # Character-level Vocabulary
            return {
                'type': 'character',
                'char_to_idx': self.tokenizer.char_to_idx,
                'idx_to_char': self.tokenizer.idx_to_char,
                'vocab_size': self.tokenizer.vocab_size
            }
        else:
            raise ValueError(f"Unknown tokenizer type: {type(self.tokenizer)}")
    
    def _deserialize_tokenizer(self, tokenizer_dict: Dict[str, Any]) -> None:
        """
        Deserialize tokenizer from checkpoint.
        
        Args:
            tokenizer_dict: Serialized tokenizer data
        """
        tokenizer_type = tokenizer_dict.get('type')
        
        if tokenizer_type == 'bpe':
            # Import here to avoid circular dependency
            from src.tokenizer import BPETokenizer
            tokenizer = BPETokenizer(vocab_size=tokenizer_dict['vocab_size'])
            tokenizer.vocab = tokenizer_dict['vocab']
            tokenizer.merges = tokenizer_dict['merges']
            # Rebuild reverse vocab
            tokenizer.idx_to_token = {idx: token for token, idx in tokenizer.vocab.items()}
            self.tokenizer = tokenizer
        elif tokenizer_type == 'character':
            # Import here to avoid circular dependency
            from src.vocabulary import Vocabulary
            tokenizer = Vocabulary()
            tokenizer.char_to_idx = tokenizer_dict['char_to_idx']
            tokenizer.idx_to_char = tokenizer_dict['idx_to_char']
            tokenizer.vocab_size = tokenizer_dict['vocab_size']
            self.tokenizer = tokenizer
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
