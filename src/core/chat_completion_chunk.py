from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ChoiceDelta(BaseModel):
    """Delta content in a choice."""
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    refusal: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class Choice(BaseModel):
    """Single choice in a chat completion chunk."""
    delta: ChoiceDelta
    finish_reason: Optional[str] = None
    index: int
    logprobs: Optional[Any] = None

class ChatCompletionChunk(BaseModel):
    """Chat completion chunk response matching OpenAI SDK structure."""
    id: str = Field(description="Unique identifier for this chat completion")
    choices: List[Choice] = Field(description="List of completion choices")
    created: int = Field(description="Unix timestamp of when this chunk was created")
    model: str = Field(description="Model used for completion")
    object: str = Field(
        default="chat.completion.chunk",
        description="Object type"
    )
    service_tier: Optional[str] = Field(
        default="default",
        description="Service tier for the completion"
    )
    system_fingerprint: Optional[str] = Field(
        description="System fingerprint for the completion"
    )
    usage: Optional[Dict[str, int]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chatcmpl-Bf2it41yaW3M3NuypDi9mVqBB8Gw9",
                "choices": [
                    {
                        "delta": {
                            "content": " today",
                            "function_call": None,
                            "refusal": None,
                            "role": None,
                            "tool_calls": None
                        },
                        "finish_reason": None,
                        "index": 0,
                        "logprobs": None
                    }
                ],
                "created": 1749121615,
                "model": "gpt-4.1-2025-04-14",
                "object": "chat.completion.chunk",
                "service_tier": "default",
                "system_fingerprint": "fp_51e1070cf2",
                "usage": None
            }
        }