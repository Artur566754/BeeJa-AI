"""Property-based tests for ChatInterface."""
from hypothesis import given, strategies as st, settings
from src.model import CustomAIModel
from src.vocabulary import Vocabulary
from src.chat_interface import ChatInterface


class TestChatInterfaceProperties:
    """Property-based tests for ChatInterface
    
    Feature: custom-ai-model
    """
    
    @given(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    @settings(max_examples=100)
    def test_message_routing(self, user_message: str):
        """
        Property 1: Message Routing
        For any user message, the model should receive it for processing
        
        Validates: Requirements 1.1
        """
        # Create model with vocabulary
        model = CustomAIModel(vocab_size=100, embedding_dim=32, hidden_dim=64, num_layers=1)
        vocab = Vocabulary()
        vocab.build_from_text("Test vocabulary for chat: " + user_message)
        model.set_vocabulary(vocab)
        
        # Create chat interface
        chat = ChatInterface(model)
        
        # Process message
        response = chat.process_message(user_message)
        
        # Check that message was processed (response generated)
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should be non-empty"
        
        # Check that message is in history
        history = chat.get_history()
        user_messages = [msg for sender, msg in history if sender == "user"]
        assert user_message in user_messages, "User message should be in history"
    
    @given(st.lists(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()), min_size=1, max_size=10))
    @settings(max_examples=100, deadline=1000)
    def test_conversation_history_preservation(self, messages: list):
        """
        Property 3: Conversation History Preservation
        For any sequence of messages, all should be maintained in history
        
        Validates: Requirements 1.3
        """
        # Create model with vocabulary
        model = CustomAIModel(vocab_size=100, embedding_dim=32, hidden_dim=64, num_layers=1)
        vocab = Vocabulary()
        vocab.build_from_text("Test vocabulary: " + " ".join(messages))
        model.set_vocabulary(vocab)
        
        # Create chat interface
        chat = ChatInterface(model)
        
        # Send all messages
        for msg in messages:
            chat.process_message(msg)
        
        # Check history
        history = chat.get_history()
        user_messages = [msg for sender, msg in history if sender == "user"]
        
        # All user messages should be in history in order
        assert len(user_messages) == len(messages), "All messages should be in history"
        assert user_messages == messages, "Messages should be in correct order"
