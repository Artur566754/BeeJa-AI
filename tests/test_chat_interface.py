"""Unit tests for ChatInterface."""
import pytest
from src.model import CustomAIModel
from src.vocabulary import Vocabulary
from src.chat_interface import ChatInterface


class TestChatInterface:
    """Tests for ChatInterface"""
    
    @pytest.fixture
    def model_with_vocab(self):
        """Create a model with vocabulary"""
        model = CustomAIModel(vocab_size=50, embedding_dim=32, hidden_dim=64, num_layers=1)
        
        # Create and set vocabulary
        vocab = Vocabulary()
        vocab.build_from_text("Hello, World! This is a test for the chat interface.")
        model.set_vocabulary(vocab)
        
        return model
    
    def test_message_processing(self, model_with_vocab):
        """Test message processing"""
        chat = ChatInterface(model_with_vocab)
        
        user_message = "Hello"
        response = chat.process_message(user_message)
        
        # Check that response is generated
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_conversation_history(self, model_with_vocab):
        """Test conversation history"""
        chat = ChatInterface(model_with_vocab)
        
        # Send multiple messages
        chat.process_message("Hello")
        chat.process_message("How are you?")
        
        history = chat.get_history()
        
        # Check history contains all messages
        assert len(history) == 4  # 2 user messages + 2 AI responses
        assert history[0][0] == "user"
        assert history[1][0] == "ai"
        assert history[2][0] == "user"
        assert history[3][0] == "ai"
    
    def test_clear_history(self, model_with_vocab):
        """Test clearing conversation history"""
        chat = ChatInterface(model_with_vocab)
        
        # Send messages
        chat.process_message("Hello")
        chat.process_message("Test")
        
        # Clear history
        chat.clear_history()
        
        history = chat.get_history()
        assert len(history) == 0
    
    def test_display_message_user(self, model_with_vocab, capsys):
        """Test display formatting for user messages"""
        chat = ChatInterface(model_with_vocab)
        
        chat.display_message("user", "Test message")
        
        captured = capsys.readouterr()
        assert "You: Test message" in captured.out
    
    def test_display_message_ai(self, model_with_vocab, capsys):
        """Test display formatting for AI messages"""
        chat = ChatInterface(model_with_vocab)
        
        chat.display_message("ai", "AI response")
        
        captured = capsys.readouterr()
        assert "AI: AI response" in captured.out
    
    def test_history_preservation(self, model_with_vocab):
        """Test that history is preserved across multiple messages"""
        chat = ChatInterface(model_with_vocab)
        
        messages = ["First", "Second", "Third"]
        
        for msg in messages:
            chat.process_message(msg)
        
        history = chat.get_history()
        
        # Check all user messages are in history
        user_messages = [h[1] for h in history if h[0] == "user"]
        assert len(user_messages) == len(messages)
        for msg in messages:
            assert msg in user_messages
