from transformers import AutoTokenizer

MAX_INPUT_LENGTH = 4096
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b-FP8")

def truncate_messages(messages: list, max_length: int) -> list:
    """Truncates messages to a maximum token length, optimizing for performance."""

    total_tokens = sum(len(tokenizer.encode(msg.content)) for msg in messages)

    if total_tokens > max_length:
        excess_tokens = total_tokens - max_length
        
        last_message_content = messages[-1].content
        last_message_tokens = tokenizer.encode(last_message_content)
        
        truncated_tokens = last_message_tokens[:-excess_tokens]
        messages[-1].content = tokenizer.decode(truncated_tokens)

    return messages
