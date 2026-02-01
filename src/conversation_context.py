"""Conversation context management for maintaining chat history."""
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str


class ConversationContext:
    """Manages conversation history for contextual responses."""
    
    def __init__(self, max_context_length: int = 512):
        """
        Initialize conversation context manager.
        
        Args:
            max_context_length: Maximum number of tokens in context
        """
        self.max_context_length = max_context_length
        self.messages: List[Message] = []
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        """
        Set the tokenizer for token counting.
        
        Args:
            tokenizer: BPETokenizer or Vocabulary instance
        """
        self.tokenizer = tokenizer
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message text
            
        Raises:
            ValueError: If role is not 'user' or 'assistant'
        """
        if role not in ['user', 'assistant']:
            raise ValueError(f"Role must be 'user' or 'assistant', got '{role}'")
        
        message = Message(role=role, content=content)
        self.messages.append(message)
    
    def get_context(self, format_template: Optional[str] = None) -> str:
        """
        Get formatted conversation context.
        
        Args:
            format_template: Optional template for formatting messages.
                           Default: "{role}: {content}\n"
        
        Returns:
            context: Formatted conversation history
        """
        if not self.messages:
            return ""
        
        if format_template is None:
            format_template = "{role}: {content}\n"
        
        context_parts = []
        for message in self.messages:
            formatted = format_template.format(role=message.role, content=message.content)
            context_parts.append(formatted)
        
        return "".join(context_parts)
    
    def get_token_count(self, text: str) -> int:
        """
        Get token count for text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            token_count: Number of tokens
        """
        if self.tokenizer is None:
            # Fallback: estimate based on characters
            return len(text) // 4
        
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def truncate_to_length(self, max_tokens: int) -> None:
        """
        Truncate context to fit within token limit.
        Removes oldest messages first to preserve recent context.
        
        Args:
            max_tokens: Maximum number of tokens to keep
        """
        if not self.messages:
            return
        
        # Calculate current token count
        current_context = self.get_context()
        current_tokens = self.get_token_count(current_context)
        
        # Remove oldest messages until we're under the limit
        while current_tokens > max_tokens and len(self.messages) > 1:
            # Remove the oldest message
            self.messages.pop(0)
            
            # Recalculate token count
            current_context = self.get_context()
            current_tokens = self.get_token_count(current_context)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
    
    def get_message_count(self) -> int:
        """
        Get number of messages in context.
        
        Returns:
            count: Number of messages
        """
        return len(self.messages)
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """
        Get the last n messages from context.
        
        Args:
            n: Number of messages to retrieve
            
        Returns:
            messages: List of last n messages
        """
        if n <= 0:
            return []
        return self.messages[-n:]
    
    def get_context_for_generation(self, user_prompt: str) -> str:
        """
        Get context formatted for text generation.
        Includes conversation history and the new user prompt.
        
        Args:
            user_prompt: New user input
            
        Returns:
            context: Formatted context for generation
        """
        # Get existing context
        context = self.get_context()
        
        # Add new user prompt
        context += f"user: {user_prompt}\nassistant: "
        
        # Truncate if needed
        token_count = self.get_token_count(context)
        if token_count > self.max_context_length:
            # Truncate conversation history to make room for prompt
            prompt_tokens = self.get_token_count(f"user: {user_prompt}\nassistant: ")
            available_tokens = self.max_context_length - prompt_tokens
            
            if available_tokens > 0:
                self.truncate_to_length(available_tokens)
                context = self.get_context()
                context += f"user: {user_prompt}\nassistant: "
            else:
                # If prompt itself is too long, just use the prompt
                context = f"user: {user_prompt}\nassistant: "
        
        return context
