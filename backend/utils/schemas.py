from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.5
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class FileObject(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str

class BatchRequestCounts(BaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0

class Batch(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[Any] = None
    input_file_id: str
    completion_window: str
    status: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: BatchRequestCounts = Field(default_factory=BatchRequestCounts)
    usage: Optional[Dict[str, int]] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    metadata: Optional[Dict[str, str]] = None

class BatchCreate(BaseModel):
    input_file_id: str
    endpoint: str
    completion_window: str
    metadata: Optional[Dict[str, str]] = None