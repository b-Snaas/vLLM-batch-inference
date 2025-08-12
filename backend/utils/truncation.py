from transformers import AutoTokenizer

MAX_INPUT_LENGTH = 4096
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b-FP8")

def _get_message_content(message) -> str:
    """Safely extract content from either a dict-style or object-style message."""
    if hasattr(message, "content"):
        return getattr(message, "content") or ""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return ""

def _set_message_content(message, new_content: str) -> None:
    """Safely set content on either a dict-style or object-style message."""
    if hasattr(message, "content"):
        setattr(message, "content", new_content)
    elif isinstance(message, dict):
        message["content"] = new_content

def truncate_messages(messages: list, max_length: int) -> list:
    """Truncates messages to a maximum token length, optimizing for performance.

    Works with both dict messages ({"role": str, "content": str}) and
    object messages that expose a `.content` attribute (e.g., Pydantic models).
    Only the last message is truncated to fit the limit.
    """

    total_tokens = 0
    for msg in messages:
        content = _get_message_content(msg)
        total_tokens += len(tokenizer.encode(content))

    if total_tokens > max_length and messages:
        excess_tokens = total_tokens - max_length

        last_content = _get_message_content(messages[-1])
        last_tokens = tokenizer.encode(last_content)

        if excess_tokens >= len(last_tokens):
            truncated_tokens = []
        else:
            truncated_tokens = last_tokens[:-excess_tokens]

        _set_message_content(messages[-1], tokenizer.decode(truncated_tokens))

    return messages
