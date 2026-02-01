"""Chat interface for interacting with the AI model."""
from typing import List
from src.model import CustomAIModel


class ChatInterface:
    """CLI интерфейс для общения с моделью"""
    
    def __init__(self, model: CustomAIModel):
        """
        Args:
            model: Модель для генерации ответов
        """
        self.model = model
        self.conversation_history: List[tuple] = []
    
    def process_message(self, user_input: str) -> str:
        """
        Обработка сообщения пользователя
        
        Args:
            user_input: Сообщение от пользователя
            
        Returns:
            Ответ модели
        """
        # Generate response using the model
        # Use the full user input as seed for better context
        seed_text = user_input if len(user_input) <= 50 else user_input[:50]
        
        response = self.model.generate(
            seed_text=seed_text,
            max_length=150,  # Увеличили длину
            temperature=0.5  # Снизили temperature для более предсказуемого текста
        )
        
        # Extract only the generated part (remove seed)
        if response.startswith(seed_text):
            response = response[len(seed_text):]
        
        # Clean up response - take only first sentence or reasonable chunk
        response = response.strip()
        
        # Add to conversation history
        self.conversation_history.append(("user", user_input))
        self.conversation_history.append(("ai", response))
        
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
        print("=" * 60)
        print("Custom AI Model - Chat Interface")
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
