"""Text generation strategies for Transformer model."""
import torch
import torch.nn.functional as F
from typing import List, Optional
import math


class TextGenerator:
    """Implements various text generation strategies."""
    
    def __init__(self, model, tokenizer):
        """
        Initialize text generator.
        
        Args:
            model: TransformerModel instance
            tokenizer: BPETokenizer or Vocabulary instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate_greedy(
        self,
        prompt: str,
        max_length: int
    ) -> str:
        """
        Generate text using greedy decoding.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate
            
        Returns:
            generated_text: Generated text string
        """
        self.model.eval()
        
        # Encode prompt
        if not prompt:
            # Use a default start token if prompt is empty
            input_ids = [0]  # Assuming 0 is a valid token
        else:
            input_ids = self.tokenizer.encode(prompt)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input tensor
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                
                # Get model predictions
                logits = self.model(input_tensor)  # [1, seq_len, vocab_size]
                
                # Get logits for the last token
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # Select token with highest probability (greedy)
                next_token_id = torch.argmax(next_token_logits).item()
                
                # Add to generated sequence
                generated_ids.append(next_token_id)
                
                # Check for end of sequence (if tokenizer has EOS token)
                if hasattr(self.tokenizer, 'eos_token_id') and next_token_id == self.tokenizer.eos_token_id:
                    break
        
        # Decode generated sequence
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text
    
    def generate_top_k(
        self,
        prompt: str,
        max_length: int,
        k: int = 50,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text using top-k sampling.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate
            k: Number of top tokens to consider
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            generated_text: Generated text string
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        
        self.model.eval()
        
        # Encode prompt
        if not prompt:
            input_ids = [0]
        else:
            input_ids = self.tokenizer.encode(prompt)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input tensor
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                
                # Get model predictions
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Get top-k tokens
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(k, len(next_token_logits)))
                
                # Convert to probabilities
                top_k_probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from top-k
                sampled_index = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = top_k_indices[sampled_index].item()
                
                # Add to generated sequence
                generated_ids.append(next_token_id)
                
                # Check for EOS
                if hasattr(self.tokenizer, 'eos_token_id') and next_token_id == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text
    
    def generate_nucleus(
        self,
        prompt: str,
        max_length: int,
        p: float = 0.9,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text using nucleus (top-p) sampling.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate
            p: Cumulative probability threshold (0 < p <= 1)
            temperature: Temperature for sampling
            
        Returns:
            generated_text: Generated text string
        """
        if not (0.0 < p <= 1.0):
            raise ValueError(f"p must be in (0, 1], got {p}")
        
        self.model.eval()
        
        # Encode prompt
        if not prompt:
            input_ids = [0]
        else:
            input_ids = self.tokenizer.encode(prompt)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input tensor
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                
                # Get model predictions
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Find the minimal set of tokens whose cumulative probability exceeds p
                # Keep at least one token
                nucleus_mask = cumulative_probs <= p
                # Ensure at least the first token is included
                nucleus_mask[0] = True
                
                # Filter tokens
                nucleus_probs = sorted_probs[nucleus_mask]
                nucleus_indices = sorted_indices[nucleus_mask]
                
                # Renormalize probabilities
                nucleus_probs = nucleus_probs / nucleus_probs.sum()
                
                # Sample from nucleus
                sampled_index = torch.multinomial(nucleus_probs, num_samples=1)
                next_token_id = nucleus_indices[sampled_index].item()
                
                # Add to generated sequence
                generated_ids.append(next_token_id)
                
                # Check for EOS
                if hasattr(self.tokenizer, 'eos_token_id') and next_token_id == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text
    
    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        penalty: float = 1.2
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits.
        
        Args:
            logits: Logits tensor [vocab_size]
            generated_ids: List of already generated token IDs
            penalty: Repetition penalty factor (> 1.0 discourages repetition)
            
        Returns:
            penalized_logits: Logits with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        penalized_logits = logits.clone()
        
        # Apply penalty to tokens that have already been generated
        for token_id in set(generated_ids):
            if token_id < len(penalized_logits):
                # If logit is positive, divide by penalty
                # If logit is negative, multiply by penalty
                if penalized_logits[token_id] > 0:
                    penalized_logits[token_id] /= penalty
                else:
                    penalized_logits[token_id] *= penalty
        
        return penalized_logits
    
    def generate_with_repetition_penalty(
        self,
        prompt: str,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Generate text with repetition penalty.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_k: Optional top-k sampling parameter
            top_p: Optional nucleus sampling parameter
            repetition_penalty: Repetition penalty factor
            
        Returns:
            generated_text: Generated text string
        """
        self.model.eval()
        
        # Encode prompt
        if not prompt:
            input_ids = [0]
        else:
            input_ids = self.tokenizer.encode(prompt)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input tensor
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
                
                # Get model predictions
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :]
                
                # Apply repetition penalty
                next_token_logits = self.apply_repetition_penalty(
                    next_token_logits,
                    generated_ids,
                    repetition_penalty
                )
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply sampling strategy
                if top_k is not None and top_k > 0:
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
                    top_k_probs = F.softmax(top_k_logits, dim=-1)
                    sampled_index = torch.multinomial(top_k_probs, num_samples=1)
                    next_token_id = top_k_indices[sampled_index].item()
                elif top_p is not None and 0.0 < top_p <= 1.0:
                    # Nucleus sampling
                    probs = F.softmax(next_token_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    nucleus_mask = cumulative_probs <= top_p
                    nucleus_mask[0] = True
                    nucleus_probs = sorted_probs[nucleus_mask]
                    nucleus_indices = sorted_indices[nucleus_mask]
                    nucleus_probs = nucleus_probs / nucleus_probs.sum()
                    sampled_index = torch.multinomial(nucleus_probs, num_samples=1)
                    next_token_id = nucleus_indices[sampled_index].item()
                else:
                    # Greedy
                    next_token_id = torch.argmax(next_token_logits).item()
                
                # Add to generated sequence
                generated_ids.append(next_token_id)
                
                # Check for EOS
                if hasattr(self.tokenizer, 'eos_token_id') and next_token_id == self.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text
