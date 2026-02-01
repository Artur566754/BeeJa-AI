"""BPE Tokenizer for subword tokenization."""
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import logging

logger = logging.getLogger(__name__)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer for subword tokenization."""
    
    # Special tokens
    UNK_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    def __init__(self, vocab_size: int = 5000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (minimum 1000, maximum 50000)
        """
        if not (1000 <= vocab_size <= 50000):
            raise ValueError(f"vocab_size must be between 1000 and 50000, got {vocab_size}")
        
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.char_vocab: Set[str] = set()
        self._is_trained = False
    
    def build_from_text(self, text: str) -> None:
        """
        Build BPE vocabulary from training text.
        
        Args:
            text: Training text to build vocabulary from
            
        Raises:
            ValueError: If text is empty or too short
        """
        if not text or len(text) < 10:
            raise ValueError("Training text must contain at least 10 characters")
        
        logger.info(f"Building BPE vocabulary with target size {self.vocab_size}")
        
        # Normalize text to lowercase for consistency
        text = text.lower()
        
        # Initialize with special tokens
        self.vocab = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        
        # Get all unique characters
        self.char_vocab = set(text)
        
        # Add individual characters to vocabulary
        for char in sorted(self.char_vocab):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # Add end-of-word marker
        if '</w>' not in self.vocab:
            self.vocab['</w>'] = len(self.vocab)
        
        # Tokenize text into words (split on whitespace and punctuation)
        words = self._tokenize_words(text)
        
        # Convert words to character sequences with end-of-word marker
        word_freqs = Counter(words)
        vocab_words = {}
        for word, freq in word_freqs.items():
            # Add space between characters and </w> at end
            vocab_words[' '.join(list(word)) + ' </w>'] = freq
        
        # Learn BPE merges
        self.merges = []
        current_vocab_size = len(self.vocab)
        
        while current_vocab_size < self.vocab_size:
            # Count all adjacent pairs
            pairs = self._count_pairs(vocab_words)
            
            if not pairs:
                logger.warning(f"No more pairs to merge. Final vocab size: {current_vocab_size}")
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the best pair in all words
            vocab_words = self._merge_pair(best_pair, vocab_words)
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                current_vocab_size += 1
            
            # Record the merge
            self.merges.append(best_pair)
            
            if len(self.merges) % 100 == 0:
                logger.debug(f"Learned {len(self.merges)} merges, vocab size: {current_vocab_size}")
        
        # Build reverse vocabulary
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self._is_trained = True
        
        logger.info(f"BPE vocabulary built: {len(self.vocab)} tokens, {len(self.merges)} merges")
    
    def _tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Split on whitespace and keep punctuation as separate tokens
        pattern = r'\w+|[^\w\s]'
        words = re.findall(pattern, text.lower())
        return words
    
    def _count_pairs(self, vocab_words: Dict[str, int]) -> Counter:
        """
        Count frequency of adjacent character pairs.
        
        Args:
            vocab_words: Dictionary of word representations to frequencies
            
        Returns:
            Counter of pair frequencies
        """
        pairs = Counter()
        
        for word, freq in vocab_words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], vocab_words: Dict[str, int]) -> Dict[str, int]:
        """
        Merge a pair in all words.
        
        Args:
            pair: Pair of symbols to merge
            vocab_words: Dictionary of word representations to frequencies
            
        Returns:
            Updated vocab_words with merged pair
        """
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in vocab_words.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        
        return new_vocab
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of token indices
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer not trained. Call build_from_text() first.")
        
        if not text:
            return []
        
        # Normalize to lowercase for consistency
        text = text.lower()
        
        # Tokenize into words
        words = self._tokenize_words(text)
        
        token_ids = []
        for word in words:
            # Apply BPE to each word
            word_tokens = self._encode_word(word)
            token_ids.extend(word_tokens)
        
        return token_ids
    
    def _encode_word(self, word: str) -> List[int]:
        """
        Encode a single word using BPE.
        
        Args:
            word: Word to encode
            
        Returns:
            List of token indices for the word
        """
        # Start with character-level representation
        symbols = list(word) + ['</w>']
        
        # Apply merges in order
        for merge in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == merge[0] and symbols[i + 1] == merge[1]:
                    # Merge the pair
                    symbols = symbols[:i] + [''.join(merge)] + symbols[i + 2:]
                else:
                    i += 1
        
        # Convert symbols to indices
        token_ids = []
        for symbol in symbols:
            if symbol in self.vocab:
                token_ids.append(self.vocab[symbol])
            else:
                # Symbol not in vocab - break down into characters
                # This handles merged tokens that weren't added to vocab
                if symbol == '</w>':
                    # End-of-word marker should be in vocab
                    if '</w>' in self.vocab:
                        token_ids.append(self.vocab['</w>'])
                    # If not, skip it (it's just a marker)
                else:
                    # Break down multi-character symbol into individual chars
                    for char in symbol:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
                        else:
                            # This character wasn't in training data
                            token_ids.append(self.vocab[self.UNK_TOKEN])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token indices back to text.
        
        Args:
            token_ids: List of token indices
            
        Returns:
            Decoded text string
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer not trained. Call build_from_text() first.")
        
        if not token_ids:
            return ""
        
        # Convert indices to tokens
        tokens = []
        for idx in token_ids:
            if idx in self.reverse_vocab:
                token = self.reverse_vocab[idx]
                # Skip special tokens in output
                if token not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
                    tokens.append(token)
            else:
                # Invalid index - skip or use UNK
                logger.warning(f"Invalid token index: {idx}")
        
        # Join tokens and clean up
        text = ''.join(tokens)
        
        # Remove end-of-word markers and restore spaces
        text = text.replace('</w>', ' ')
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def save(self, filepath: str) -> None:
        """
        Save tokenizer vocabulary and merges to file.
        
        Args:
            filepath: Path to save tokenizer state
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained tokenizer")
        
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': self.merges,
            'char_vocab': list(self.char_vocab),
            'is_trained': self._is_trained
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load tokenizer vocabulary and merges from file.
        
        Args:
            filepath: Path to load tokenizer state from
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupted or invalid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted tokenizer file: {e}")
        
        # Validate required fields
        required_fields = ['vocab_size', 'vocab', 'merges', 'char_vocab', 'is_trained']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Invalid tokenizer file: missing field '{field}'")
        
        # Load state
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.merges = [tuple(merge) for merge in data['merges']]
        self.char_vocab = set(data['char_vocab'])
        self._is_trained = data['is_trained']
        
        # Build reverse vocabulary
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        logger.info(f"Tokenizer loaded from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get the actual vocabulary size."""
        return len(self.vocab)
    
    def is_trained(self) -> bool:
        """Check if tokenizer has been trained."""
        return self._is_trained
