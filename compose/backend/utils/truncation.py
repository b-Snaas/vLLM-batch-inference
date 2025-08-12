from transformers import AutoTokenizer

MAX_INPUT_LENGTH = 4096
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b-FP8")

def truncate_messages(messages: list, max_length: int) -> list:
    """Truncates messages to a maximum token length."""
    
    # Calculate total tokens
    total_tokens = 0
    for message in messages:
        total_tokens += len(tokenizer.encode(message.content))

    # Truncate if necessary
    if total_tokens > max_length:
        # Truncate the last message
        last_message = messages[-1]
        
        # Calculate the excess tokens
        excess_tokens = total_tokens - max_length
        
        # Truncate the content
        truncated_content = tokenizer.decode(tokenizer.encode(last_message.content)[:-excess_tokens])
        
        # Update the message
        last_message.content = truncated_content

    return messages
