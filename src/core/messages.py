from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal, Union
import time

def generate_message_id(thread_id: str) -> str:
    """
    Generate a unique message ID.
    """
    return f"msg_{thread_id}_{int(time.time())}"

class TextContentProperties(BaseModel):
    value: str = Field(
        example="Hello, how are you?",
        description="The data that makes up the text."
        )
    annotations: Optional[List] = Field(
        default=None,
        example=[],
        description="Optional annotations for file citations or file paths"
    )
    
class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: TextContentProperties = Field(description="The text content that is part of a message")
    
class TextContentBase(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field(description="The text content that is part of a message")

class ImageURLProperties(BaseModel):
    url: str = Field(
        example="https://example.com/image.png",
        description="The URL of the image content that is part of a message."
    )
    detail: Optional[str] = Field(
        default=None,
        description="Specifies the detail level of the image. low uses fewer tokens, you can opt in to high resolution using high. Default value is auto"
    )

class ImageURL(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURLProperties = Field(
        description="The image URL content that is part of a message."
    )
    
class MessageObject(BaseModel):
    id: str = Field(
        description="The identifier, which can be referenced in API endpoints."
    )
    object: str = Field(
        default="thread.message",
        description="The object type, which is always 'thread.message'."
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp (in seconds) for when the message was created."
    )
    thread_id: str = Field(
        description="The thread ID that this message belongs to."
    )
    role: str = Field(
        description="The entity that produced the message. One of 'user' or 'assistant'."
    )
    content: List[TextContent | ImageURL] = Field(
        description="The content of the message in array of text and/or images."
    )
    assistant_id: Optional[str] = Field(
        default=None,
        description="If applicable, the ID of the assistant that authored this message."
    )
    run_id: Optional[str] = Field(
        default=None,
        description="The ID of the run associated with the creation of this message."
    )
    completed_at: Optional[int] = Field(
        default=None, 
        description="The Unix timestamp (in seconds) for when the message was completed."
    )
    incomplete_at: Optional[int] = Field(
        default=None,
        description="The Unix timestamp (in seconds) for when the message was marked as incomplete."
    )
    incomplete_details: Optional[Dict] = Field(
        default=None,
        description="On an incomplete message, details about why the message is incomplete."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object. Keys are strings with max length 64 chars. Values are strings with max length 512 chars."
    )
    attachments: Optional[List] = Field(
        default=None,
        description="A list of files attached to this message, and the tools they were added to."
    )
    status: Optional[str] = Field(
        default="completed",
        description="The status of this message, which can be either 'in_progress', 'incomplete', or 'completed'."
    )
    
class CreateMessageRequest(BaseModel):
    content: str | list[TextContentBase | ImageURL] = Field(
        description="The content of the message, which can be text or other types."
    )
    role: Literal["user", "assistant"] = Field(
        description="The role of the entity that is creating the message. One of 'user' or 'assistant'."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        example={},
        description="Set of 16 key-value pairs that can be attached to an object. Keys are strings with max length 64 chars. Values are strings with max length 512 chars."
    )

class DeltaContentProperties(BaseModel):
    index: int = Field(
        description="The index of the content in the message."
    )
    type: str = Field(
        description="The type of content, which is always 'text'."
    )
    text: TextContentProperties = Field(
        description="The text content that is part of the delta."
    )   


class DeltaContent(BaseModel):
    content: List[DeltaContentProperties] = Field(
        description="The content of the delta in array of text and/or images."
    )
    
class MessageDelta(BaseModel):
    id: str = Field(
        description="The identifier, which can be referenced in API endpoints."
    )
    object: str = Field(
        default="thread.message.delta",
        description="The object type, which is always 'thread.message.delta'."
    )
    delta: DeltaContent = Field(
        description="The delta content of the message."
    )