"""Property-based tests for ConversationContext."""
import pytest
from hypothesis import given, settings, strategies as st
from src.conversation_context import ConversationContext, Message
from src.tokenizer import BPETokenizer
from src.config import TransformerConfig
from src.transformer_model import TransformerModel


def create_test_tokenizer(vocab_size=1000):
    """Helper to create a tokenizer for testing."""
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.build_from_text("hello world test sample text for building vocabulary")
    return tokenizer


# Feature: transformer-upgrade, Property 15: Context Influence on Generation
# Validates: Requirements 3.7
@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_15_context_influence_on_generation(seed):
    """
    Property 15: Context Influence on Generation
    
    For any prompt, generating with conversation context SHALL produce different
    outputs than generating without context (for non-deterministic sampling).
    """
    import torch
    torch.manual_seed(seed)
    
    # Create model and tokenizer
    config = TransformerConfig.small(vocab_size=1000)
    model = TransformerModel(config)
    tokenizer = create_test_tokenizer()
    model.set_tokenizer(tokenizer)
    model.eval()
    
    # Create conversation context
    context = ConversationContext(max_context_length=256)
    context.set_tokenizer(tokenizer)
    context.add_message("user", "Hello, how are you?")
    context.add_message("assistant", "I'm doing well, thank you!")
    
    prompt = "What's the weather like?"
    
    # Generate without context
    torch.manual_seed(seed)
    text_without_context = model.generate(
        prompt, max_length=10, strategy='nucleus', top_p=0.9, temperature=1.0
    )
    
    # Generate with context
    torch.manual_seed(seed)
    text_with_context = model.generate(
        prompt, max_length=10, strategy='nucleus', top_p=0.9, temperature=1.0,
        conversation_context=context
    )
    
    # The prompts are different (one has context, one doesn't), so outputs should differ
    # Note: With same seed but different prompts, outputs will be different
    # We're testing that the context is actually used
    assert text_without_context != text_with_context or len(text_with_context) > len(text_without_context), \
        "Context should influence generation"


# Feature: transformer-upgrade, Property 25: Conversation History Maintenance
# Validates: Requirements 9.1, 9.2
@settings(max_examples=100, deadline=None)
@given(
    num_messages=st.integers(min_value=1, max_value=20)
)
def test_property_25_conversation_history_maintenance(num_messages):
    """
    Property 25: Conversation History Maintenance
    
    For any sequence of messages in a conversation, the conversation context SHALL
    contain all messages up to the context window limit, in chronological order.
    """
    context = ConversationContext(max_context_length=1000)
    
    # Add messages alternating between user and assistant
    messages = []
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}"
        context.add_message(role, content)
        messages.append((role, content))
    
    # Verify all messages are stored
    assert context.get_message_count() == num_messages
    
    # Verify messages are in chronological order
    stored_messages = context.messages
    for i, (expected_role, expected_content) in enumerate(messages):
        assert stored_messages[i].role == expected_role
        assert stored_messages[i].content == expected_content
    
    # Verify formatted context contains all messages
    formatted_context = context.get_context()
    for role, content in messages:
        assert content in formatted_context


# Feature: transformer-upgrade, Property 26: Context Truncation Preserves Recent Messages
# Validates: Requirements 9.3, 9.4
@settings(max_examples=100, deadline=None)
@given(
    num_messages=st.integers(min_value=5, max_value=20),
    max_tokens=st.integers(min_value=10, max_value=100)
)
def test_property_26_context_truncation_preserves_recent(num_messages, max_tokens):
    """
    Property 26: Context Truncation Preserves Recent Messages
    
    For any conversation history exceeding the context window, after truncation,
    the most recent messages SHALL be preserved and the oldest messages SHALL be removed.
    """
    context = ConversationContext(max_context_length=1000)
    tokenizer = create_test_tokenizer()
    context.set_tokenizer(tokenizer)
    
    # Add messages
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"This is message number {i} with some content"
        context.add_message(role, content)
    
    # Store the last few messages before truncation
    messages_before = context.messages.copy()
    
    # Truncate
    context.truncate_to_length(max_tokens)
    
    # Verify that remaining messages are from the end of the original list
    messages_after = context.messages
    
    if len(messages_after) > 0:
        # The first message after truncation should appear in the original list
        first_after = messages_after[0]
        found = False
        for msg in messages_before:
            if msg.role == first_after.role and msg.content == first_after.content:
                found = True
                break
        assert found, "Truncated messages should come from original list"
        
        # The last message should be the same as before truncation
        assert messages_after[-1].content == messages_before[-1].content, \
            "Most recent message should be preserved"


def test_conversation_context_clear():
    """Test that clear() removes all messages."""
    context = ConversationContext()
    
    context.add_message("user", "Hello")
    context.add_message("assistant", "Hi there")
    
    assert context.get_message_count() == 2
    
    context.clear()
    
    assert context.get_message_count() == 0
    assert context.get_context() == ""


def test_conversation_context_invalid_role():
    """Test that invalid roles raise ValueError."""
    context = ConversationContext()
    
    with pytest.raises(ValueError, match="Role must be"):
        context.add_message("invalid_role", "Some content")


def test_conversation_context_get_last_n_messages():
    """Test getting last n messages."""
    context = ConversationContext()
    
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        context.add_message(role, f"Message {i}")
    
    # Get last 3 messages
    last_3 = context.get_last_n_messages(3)
    assert len(last_3) == 3
    assert last_3[0].content == "Message 2"
    assert last_3[1].content == "Message 3"
    assert last_3[2].content == "Message 4"
    
    # Get more messages than exist
    all_messages = context.get_last_n_messages(10)
    assert len(all_messages) == 5
    
    # Get 0 messages
    no_messages = context.get_last_n_messages(0)
    assert len(no_messages) == 0


def test_conversation_context_formatting():
    """Test custom formatting templates."""
    context = ConversationContext()
    
    context.add_message("user", "Hello")
    context.add_message("assistant", "Hi")
    
    # Default formatting
    default_format = context.get_context()
    assert "user: Hello" in default_format
    assert "assistant: Hi" in default_format
    
    # Custom formatting
    custom_format = context.get_context(format_template="[{role}] {content} | ")
    assert "[user] Hello |" in custom_format
    assert "[assistant] Hi |" in custom_format


def test_conversation_context_for_generation():
    """Test getting context formatted for generation."""
    context = ConversationContext(max_context_length=1000)
    tokenizer = create_test_tokenizer()
    context.set_tokenizer(tokenizer)
    
    context.add_message("user", "Hello")
    context.add_message("assistant", "Hi there")
    
    # Get context for new user prompt
    generation_context = context.get_context_for_generation("How are you?")
    
    # Should contain history and new prompt
    assert "user: Hello" in generation_context
    assert "assistant: Hi there" in generation_context
    assert "user: How are you?" in generation_context
    assert generation_context.endswith("assistant: ")


def test_conversation_context_token_counting():
    """Test token counting with and without tokenizer."""
    context = ConversationContext()
    
    # Without tokenizer (fallback to character-based estimation)
    text = "Hello world"
    count_without = context.get_token_count(text)
    assert count_without > 0
    
    # With tokenizer
    tokenizer = create_test_tokenizer()
    context.set_tokenizer(tokenizer)
    count_with = context.get_token_count(text)
    assert count_with > 0


def test_empty_conversation_context():
    """Test behavior with empty context."""
    context = ConversationContext()
    
    assert context.get_message_count() == 0
    assert context.get_context() == ""
    assert len(context.get_last_n_messages(5)) == 0
    
    # Truncation on empty context should not crash
    context.truncate_to_length(10)
    assert context.get_message_count() == 0
