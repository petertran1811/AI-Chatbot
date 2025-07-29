from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import time

def generate_thread_id(user_id: str) -> str:
    """
    Generates a unique thread ID.
    In a real-world scenario, this could be replaced with a more robust ID generation method.
    """
    return f"thread_{user_id}_{int(time.time())}"

class Thread(BaseModel):
    id: str = Field(
        description="The identifier, which can be referenced in API endpoints."
    )
    object: str = Field(
        default="thread",
        description="The object type, which is always 'thread'."
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp (in seconds) for when the thread was created."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to the thread. Keys are strings with max length 64 chars. Values are strings with max length 512 chars."
    )
    # tool_resources: Optional[Dict] = Field(
    #     default=None,
    #     description="A set of resources that are made available to the assistant's tools in this thread."
    # )

class CreateThreadsRequest(BaseModel):
    messages: Optional[List[dict]] = Field(
        default=None,
        examples=[],
        description="A list of messages to start the thread with."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to the thread. Keys are strings with max length 64 chars. Values are strings with max length 512 chars.",
    )
    # tool_resources: Optional[Dict] = Field(
    #     default=None,
    #     examples=[{}],
    #     description="A set of resources that are made available to the assistant's tools in this thread."
    # )
    