"""Core Transformer components: PositionalEncoding, MultiHeadAttention, TransformerBlock."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, embedding_dim: int, max_seq_length: int, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            embedding_dim: Dimension of embeddings
            max_seq_length: Maximum sequence length
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term for sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        if embedding_dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # Handle odd embedding_dim
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, embedding_dim]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Token embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            output: Embeddings with position info [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        
        # Apply dropout
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1, use_memory_efficient: bool = False):
        """
        Initialize multi-head attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate for regularization
            use_memory_efficient: Use memory-efficient attention for long sequences
            
        Raises:
            ValueError: If embedding_dim is not divisible by num_heads
        """
        super().__init__()
        
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.use_memory_efficient = use_memory_efficient
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout
        
        # Check if PyTorch has scaled_dot_product_attention (PyTorch 2.0+)
        self.has_sdpa = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-head self-attention.
        
        Args:
            x: Input [batch_size, seq_len, embedding_dim]
            attention_mask: Causal mask [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            
        Returns:
            output: Attention output [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Use memory-efficient attention for long sequences (>512 tokens)
        if self.use_memory_efficient and seq_len > 512:
            return self._memory_efficient_attention(x, attention_mask)
        
        # Use PyTorch's optimized SDPA if available
        if self.has_sdpa and self.training:
            return self._sdpa_attention(x, attention_mask)
        
        # Standard attention computation
        return self._standard_attention(x, attention_mask)
    
    def _standard_attention(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention computation."""
        batch_size, seq_len, _ = x.size()
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch_size, seq_len, embedding_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: [batch_size, num_heads, seq_len, seq_len]
        
        # Apply attention mask (causal masking for autoregressive generation)
        if attention_mask is not None:
            # Expand mask for batch and heads if needed
            if attention_mask.dim() == 2:
                # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1)
            
            # Apply mask (set masked positions to large negative value)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back to [batch_size, seq_len, embedding_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output
    
    def _sdpa_attention(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Use PyTorch's optimized scaled_dot_product_attention (PyTorch 2.0+)."""
        batch_size, seq_len, _ = x.size()
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Prepare attention mask for SDPA
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attn_mask = attention_mask.unsqueeze(1)
            # Convert boolean mask to float mask
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf'))
        
        # Use PyTorch's optimized SDPA
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False  # We handle causality with our mask
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output
    
    def _memory_efficient_attention(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory-efficient attention for long sequences using chunking.
        Processes attention in chunks to reduce memory usage.
        """
        batch_size, seq_len, _ = x.size()
        chunk_size = 512  # Process in chunks of 512 tokens
        
        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Initialize output
        attn_output = torch.zeros_like(Q)
        
        # Process in chunks
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            Q_chunk = Q[:, :, i:end_i, :]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / self.scale
            
            # Apply attention mask
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask_chunk = attention_mask[i:end_i, :].unsqueeze(0).unsqueeze(0)
                elif attention_mask.dim() == 3:
                    mask_chunk = attention_mask[:, i:end_i, :].unsqueeze(1)
                else:
                    mask_chunk = attention_mask[:, :, i:end_i, :]
                
                scores = scores.masked_fill(mask_chunk == 0, float('-inf'))
            
            # Apply softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output[:, :, i:end_i, :] = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embedding_dim)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Single Transformer decoder block with self-attention and feed-forward layers."""
    
    def __init__(self, embedding_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1, use_memory_efficient: bool = False):
        """
        Initialize a Transformer block.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension of feed-forward network
            dropout: Dropout rate for regularization
            use_memory_efficient: Use memory-efficient attention for long sequences
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Multi-head self-attention
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout, use_memory_efficient)
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward network (2-layer MLP)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),  # Using GELU activation like GPT
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the block.
        
        Args:
            x: Input tensor [batch_size, seq_len, embedding_dim]
            attention_mask: Causal mask [seq_len, seq_len]
            
        Returns:
            output: Transformed tensor [batch_size, seq_len, embedding_dim]
        """
        # Pre-norm architecture with residual connections
        
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask)
        x = residual + x  # Residual connection
        
        # Feed-forward block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x  # Residual connection
        
        return x


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for autoregressive generation.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        mask: Causal mask [seq_len, seq_len] where mask[i, j] = 1 if j <= i, else 0
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask
