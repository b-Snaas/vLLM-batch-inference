from transformers import AutoTokenizer

MAX_INPUT_LENGTH = 4096
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b-FP8")

def truncate_messages(messages: list, max_length: int) -> list:
    """Truncates messages to a maximum token length, optimizing for performance."""

    # Efficiently calculate total tokens using a generator expression
    total_tokens = sum(len(tokenizer.encode(msg.content)) for msg in messages)

    # Truncate if necessary
    if total_tokens > max_length:
        excess_tokens = total_tokens - max_length
        
        # Get the tokenized content of the last message
        last_message_content = messages[-1].content
        last_message_tokens = tokenizer.encode(last_message_content)
        
        # Truncate the tokens and decode back to string
        truncated_tokens = last_message_tokens[:-excess_tokens]
        messages[-1].content = tokenizer.decode(truncated_tokens)

    return messages
