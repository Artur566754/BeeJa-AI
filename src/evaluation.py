"""Model evaluation utilities for Transformer and LSTM models."""
import time
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union, Optional
import math
from src.transformer_model import TransformerModel
from src.model import CustomAIModel
from src.tokenizer import BPETokenizer
from src.vocabulary import Vocabulary


class ModelEvaluator:
    """Utilities for evaluating model performance."""
    
    def __init__(
        self,
        model: Union[TransformerModel, CustomAIModel],
        tokenizer: Union[BPETokenizer, Vocabulary]
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: The model to evaluate (Transformer or LSTM)
            tokenizer: The tokenizer (BPE or Vocabulary)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_model_device()
    
    def _get_model_device(self) -> torch.device:
        """Get the device the model is on."""
        if isinstance(self.model, TransformerModel):
            return self.model.get_device()
        else:
            # For LSTM model, check first parameter
            return next(self.model.parameters()).device
    
    def compute_perplexity(
        self,
        text: str,
        batch_size: int = 16,
        seq_length: int = 128
    ) -> float:
        """
        Compute perplexity on the given text.
        
        Perplexity measures how well the model predicts the text.
        Lower perplexity indicates better performance.
        
        Args:
            text: Text to evaluate on
            batch_size: Batch size for evaluation
            seq_length: Sequence length for batching
            
        Returns:
            perplexity: The perplexity score (positive finite number)
            
        Validates: Requirements 10.1
        """
        self.model.eval()
        
        # Encode text
        if isinstance(self.tokenizer, BPETokenizer):
            token_ids = self.tokenizer.encode(text)
        else:
            token_ids = [self.tokenizer.char_to_idx.get(c, 0) for c in text]
        
        if len(token_ids) < 2:
            raise ValueError("Text too short for perplexity computation")
        
        # Prepare sequences
        total_loss = 0.0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            # Process in chunks
            for i in range(0, len(token_ids) - 1, seq_length):
                # Get chunk
                chunk_end = min(i + seq_length, len(token_ids) - 1)
                input_chunk = token_ids[i:chunk_end]
                target_chunk = token_ids[i+1:chunk_end+1]
                
                if len(input_chunk) == 0:
                    continue
                
                # Convert to tensors
                input_tensor = torch.tensor([input_chunk], dtype=torch.long, device=self.device)
                target_tensor = torch.tensor(target_chunk, dtype=torch.long, device=self.device)
                
                # Forward pass
                if isinstance(self.model, TransformerModel):
                    logits = self.model(input_tensor)
                    logits = logits[0]  # Remove batch dimension
                else:
                    # LSTM model
                    hidden = self.model.init_hidden(1)
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.to(self.device) for h in hidden)
                    else:
                        hidden = hidden.to(self.device)
                    
                    logits_list = []
                    for token_id in input_chunk:
                        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                        output, hidden = self.model(token_tensor, hidden)
                        logits_list.append(output[0, 0])
                    logits = torch.stack(logits_list)
                
                # Compute loss
                loss = criterion(logits, target_tensor)
                total_loss += loss.item()
                total_tokens += len(target_chunk)
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        # Ensure perplexity is valid
        if not math.isfinite(perplexity) or perplexity <= 0:
            raise ValueError(f"Invalid perplexity computed: {perplexity}")
        
        return perplexity
    
    def generate_samples(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Generate sample outputs for qualitative evaluation.
        
        Args:
            prompts: List of prompts to generate from
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            samples: List of generated texts
            
        Validates: Requirements 10.2
        """
        self.model.eval()
        samples = []
        
        for prompt in prompts:
            if isinstance(self.model, TransformerModel):
                # Transformer generation
                generated = self.model.generate(
                    seed_text=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    **kwargs
                )
            else:
                # LSTM generation
                generated = self.model.generate_text(
                    seed_text=prompt,
                    length=max_length,
                    temperature=temperature
                )
            
            samples.append(generated)
        
        return samples
    
    def measure_inference_time(
        self,
        text: str,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Measure inference time for the model.
        
        Args:
            text: Input text for inference
            num_runs: Number of runs to average over
            
        Returns:
            timing_stats: Dictionary with timing statistics
            
        Validates: Requirements 10.4
        """
        self.model.eval()
        
        # Encode text
        if isinstance(self.tokenizer, BPETokenizer):
            token_ids = self.tokenizer.encode(text)
        else:
            token_ids = [self.tokenizer.char_to_idx.get(c, 0) for c in text]
        
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        
        # Warmup
        with torch.no_grad():
            if isinstance(self.model, TransformerModel):
                _ = self.model(input_tensor)
            else:
                hidden = self.model.init_hidden(1)
                if isinstance(hidden, tuple):
                    hidden = tuple(h.to(self.device) for h in hidden)
                else:
                    hidden = hidden.to(self.device)
                for token_id in token_ids:
                    token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                    _, hidden = self.model(token_tensor, hidden)
        
        # Measure time
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                if isinstance(self.model, TransformerModel):
                    _ = self.model(input_tensor)
                else:
                    hidden = self.model.init_hidden(1)
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.to(self.device) for h in hidden)
                    else:
                        hidden = hidden.to(self.device)
                    for token_id in token_ids:
                        token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                        _, hidden = self.model(token_tensor, hidden)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'total_time': sum(times),
            'tokens_per_second': len(token_ids) * num_runs / sum(times)
        }
    
    def get_model_size(self) -> Dict[str, Any]:
        """
        Report model size and memory usage.
        
        Returns:
            size_info: Dictionary with model size information
            
        Validates: Requirements 10.5
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate memory usage (in MB)
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 2)
        buffer_memory = sum(b.numel() * b.element_size() for b in self.model.buffers()) / (1024 ** 2)
        total_memory = param_memory + buffer_memory
        
        # Model type
        model_type = "Transformer" if isinstance(self.model, TransformerModel) else "LSTM"
        
        return {
            'model_type': model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_memory_mb': param_memory,
            'buffer_memory_mb': buffer_memory,
            'total_memory_mb': total_memory,
            'device': str(self.device)
        }
    
    def evaluate_full(
        self,
        validation_text: str,
        sample_prompts: List[str],
        batch_size: int = 16,
        seq_length: int = 128
    ) -> Dict[str, Any]:
        """
        Perform full evaluation with all metrics.
        
        Args:
            validation_text: Text for perplexity computation
            sample_prompts: Prompts for sample generation
            batch_size: Batch size for perplexity
            seq_length: Sequence length for perplexity
            
        Returns:
            results: Dictionary with all evaluation results
        """
        results = {}
        
        # Perplexity
        try:
            results['perplexity'] = self.compute_perplexity(
                validation_text,
                batch_size=batch_size,
                seq_length=seq_length
            )
        except Exception as e:
            results['perplexity'] = f"Error: {str(e)}"
        
        # Sample generation
        try:
            results['samples'] = self.generate_samples(sample_prompts)
        except Exception as e:
            results['samples'] = f"Error: {str(e)}"
        
        # Inference time
        try:
            results['inference_time'] = self.measure_inference_time(validation_text[:100])
        except Exception as e:
            results['inference_time'] = f"Error: {str(e)}"
        
        # Model size
        try:
            results['model_size'] = self.get_model_size()
        except Exception as e:
            results['model_size'] = f"Error: {str(e)}"
        
        return results


class ModelComparator:
    """Compare two models (e.g., LSTM vs Transformer)."""
    
    def __init__(
        self,
        model1: Union[TransformerModel, CustomAIModel],
        tokenizer1: Union[BPETokenizer, Vocabulary],
        model2: Union[TransformerModel, CustomAIModel],
        tokenizer2: Union[BPETokenizer, Vocabulary],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ):
        """
        Initialize model comparator.
        
        Args:
            model1: First model
            tokenizer1: First model's tokenizer
            model2: Second model
            tokenizer2: Second model's tokenizer
            model1_name: Name for first model
            model2_name: Name for second model
        """
        self.evaluator1 = ModelEvaluator(model1, tokenizer1)
        self.evaluator2 = ModelEvaluator(model2, tokenizer2)
        self.model1_name = model1_name
        self.model2_name = model2_name
    
    def compare_generation_quality(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, List[str]]:
        """
        Compare generation quality between models.
        
        Args:
            prompts: List of prompts to generate from
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            comparison: Dictionary with samples from both models
            
        Validates: Requirements 10.3
        """
        samples1 = self.evaluator1.generate_samples(prompts, max_length, temperature)
        samples2 = self.evaluator2.generate_samples(prompts, max_length, temperature)
        
        return {
            self.model1_name: samples1,
            self.model2_name: samples2,
            'prompts': prompts
        }
    
    def compare_training_metrics(
        self,
        validation_text: str,
        batch_size: int = 16,
        seq_length: int = 128
    ) -> Dict[str, Any]:
        """
        Compare training metrics (perplexity) between models.
        
        Args:
            validation_text: Text for evaluation
            batch_size: Batch size
            seq_length: Sequence length
            
        Returns:
            comparison: Dictionary with metrics from both models
            
        Validates: Requirements 10.3
        """
        try:
            perplexity1 = self.evaluator1.compute_perplexity(
                validation_text, batch_size, seq_length
            )
        except Exception as e:
            perplexity1 = f"Error: {str(e)}"
        
        try:
            perplexity2 = self.evaluator2.compute_perplexity(
                validation_text, batch_size, seq_length
            )
        except Exception as e:
            perplexity2 = f"Error: {str(e)}"
        
        return {
            self.model1_name: {'perplexity': perplexity1},
            self.model2_name: {'perplexity': perplexity2}
        }
    
    def compare_inference_speed(
        self,
        text: str,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Compare inference speed between models.
        
        Args:
            text: Input text for inference
            num_runs: Number of runs to average over
            
        Returns:
            comparison: Dictionary with timing stats from both models
            
        Validates: Requirements 10.3
        """
        timing1 = self.evaluator1.measure_inference_time(text, num_runs)
        timing2 = self.evaluator2.measure_inference_time(text, num_runs)
        
        return {
            self.model1_name: timing1,
            self.model2_name: timing2,
            'speedup': timing1['mean_time'] / timing2['mean_time'] if timing2['mean_time'] > 0 else float('inf')
        }
    
    def compare_full(
        self,
        validation_text: str,
        sample_prompts: List[str],
        batch_size: int = 16,
        seq_length: int = 128
    ) -> Dict[str, Any]:
        """
        Perform full comparison with all metrics.
        
        Args:
            validation_text: Text for perplexity computation
            sample_prompts: Prompts for sample generation
            batch_size: Batch size
            seq_length: Sequence length
            
        Returns:
            comparison: Dictionary with all comparison results
        """
        return {
            'generation_quality': self.compare_generation_quality(sample_prompts),
            'training_metrics': self.compare_training_metrics(validation_text, batch_size, seq_length),
            'inference_speed': self.compare_inference_speed(validation_text[:100]),
            'model_sizes': {
                self.model1_name: self.evaluator1.get_model_size(),
                self.model2_name: self.evaluator2.get_model_size()
            }
        }
