"""Chat interface for interacting with the AI model."""
from typing import List, Union, Optional
from src.model import CustomAIModel
from src.transformer_model import TransformerModel
from src.conversation_context import ConversationContext


class ChatInterface:
    """CLI интерфейс для общения с моделью
    
    Supports both LSTM (CustomAIModel) and Transformer models.
    Validates: Requirements 6.1, 6.2
    """
    
    def __init__(
        self,
        model: Union[CustomAIModel, TransformerModel],
        use_context: bool = True,
        max_context_length: int = 512
    ):
        """
        Args:
            model: Модель для генерации ответов (LSTM or Transformer)
            use_context: Whether to use conversation context (Transformer only)
            max_context_length: Maximum context length in tokens
            
        Validates: Requirements 6.1, 6.2
        """
        self.model = model
        self.conversation_history: List[tuple] = []
        
        # Detect model type
        self.is_transformer = isinstance(model, TransformerModel)
        
        # Initialize conversation context for Transformer
        self.conversation_context: Optional[ConversationContext] = None
        if self.is_transformer and use_context:
            self.conversation_context = ConversationContext(
                max_context_length=max_context_length
            )
    
    def process_message(self, user_input: str) -> str:
        """
        Обработка сообщения пользователя
        
        Args:
            user_input: Сообщение от пользователя
            
        Returns:
            Ответ модели
            
        Validates: Requirements 6.1, 6.2, 6.5
        """
        try:
            # Add user message to context (Transformer only)
            if self.conversation_context is not None:
                self.conversation_context.add_message("user", user_input)
            
            # Generate response based on model type
            if self.is_transformer:
                response = self._generate_transformer_response(user_input)
            else:
                response = self._generate_lstm_response(user_input)
            
            # Add to conversation history
            self.conversation_history.append(("user", user_input))
            self.conversation_history.append(("ai", response))
            
            # Add assistant response to context (Transformer only)
            if self.conversation_context is not None:
                self.conversation_context.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            # Handle errors gracefully (Requirement 6.5)
            error_message = f"Error generating response: {str(e)}"
            print(f"\n{error_message}")
            return "I apologize, but I encountered an error. Please try again."
    
    def _generate_transformer_response(self, user_input: str) -> str:
        """
        Generate response using Transformer model.
        
        Args:
            user_input: User message
            
        Returns:
            Generated response
        """
        # Use conversation context if available
        if self.conversation_context is not None:
            context = self.conversation_context.get_context()
            # Generate with context
            response = self.model.generate(
                seed_text=context,
                max_length=150,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2
            )
            # Extract only the new response (after the context)
            if response.startswith(context):
                response = response[len(context):].strip()
        else:
            # Generate without context
            response = self.model.generate(
                seed_text=user_input,
                max_length=150,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2
            )
            # Extract only the generated part
            if response.startswith(user_input):
                response = response[len(user_input):].strip()
        
        return response
    
    def _generate_lstm_response(self, user_input: str) -> str:
        """
        Generate response using LSTM model (backward compatible).
        
        Args:
            user_input: User message
            
        Returns:
            Generated response
        """
        # Use the full user input as seed for better context
        seed_text = user_input if len(user_input) <= 50 else user_input[:50]
        
        response = self.model.generate(
            seed_text=seed_text,
            max_length=150,
            temperature=0.5
        )
        
        # Extract only the generated part (remove seed)
        if response.startswith(seed_text):
            response = response[len(seed_text):]
        
        # Clean up response
        response = response.strip()
        
        return response
    
    def display_message(self, sender: str, message: str) -> None:
        """
        Отображение сообщения в чате
        
        Args:
            sender: Отправитель ("user" или "ai")
            message: Текст сообщения
        """
        if sender == "user":
            print(f"\nYou: {message}")
        elif sender == "ai":
            print(f"AI: {message}")
        else:
            print(f"{sender}: {message}")
    
    def start_chat(self) -> None:
        """
        Запуск чат-сессии
        """
        model_type = "Transformer" if self.is_transformer else "LSTM"
        context_status = "enabled" if self.conversation_context is not None else "disabled"
        
        print("=" * 60)
        print(f"Custom AI Model - Chat Interface ({model_type})")
        print(f"Conversation Context: {context_status}")
        print("=" * 60)
        print("Commands:")
        print("  /exit    - Exit the chat")
        print("  /retrain - Trigger model retraining")
        print("  /clear   - Clear conversation history")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input == "/exit":
                    print("\nGoodbye!")
                    break
                elif user_input == "/retrain":
                    print("\nRetraining command received. Please use main.py with --train flag.")
                    continue
                elif user_input == "/clear":
                    self.conversation_history.clear()
                    if self.conversation_context is not None:
                        self.conversation_context.clear()
                    print("\nConversation history cleared.")
                    continue
                
                # Process message and get response
                response = self.process_message(user_input)
                
                # Display response
                self.display_message("ai", response)
                
            except KeyboardInterrupt:
                print("\n\nChat interrupted. Goodbye!")
                break
            except Exception as e:
                # Handle errors gracefully (Requirement 6.5)
                print(f"\nError: {e}")
                print("Please try again.")
    
    def get_history(self) -> List[tuple]:
        """
        Get conversation history
        
        Returns:
            List of (sender, message) tuples
        """
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
