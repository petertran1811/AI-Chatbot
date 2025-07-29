import time
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from core.files import FileCounts
    
class VectorStore(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="The identifier, which can be referenced in API endpoints."
    )
    object: Optional[str] = Field(
        default="vector_store",
        description="The object type, which is always 'vector_store'."
    )
    created_at: Optional[int] = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp (in seconds) for when the vector store was created."
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the vector store."
    )
    status: Optional[str] = Field(
        default=None,
        description="The status of the vector store, which can be either 'expired', 'in_progress', or 'completed'."
    )
    usage_bytes: Optional[int] = Field(
        default=None,
        description="The total number of bytes used by the files in the vector store."
    )
    file_counts: Optional[FileCounts] = Field(
        default=None,
        description="Counts of files in various states (e.g., in_progress, completed, failed, etc.)."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters."
    )
    expires_after: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The expiration policy for a vector store."
    )
    chunking_strategy: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The chunking strategy used to chunk the file(s). If not set, will use the 'auto' strategy. Only applicable if file_ids is non-empty."
    )
    file_ids: Optional[List[str]] = Field(
        default=None,
        description="A list of File IDs that the vector store should use. Useful for tools like file_search that can access files."
    )
    last_used_at: Optional[int] = Field(
        default=None,
        description="The Unix timestamp (in seconds) for when the vector store was last used. This field is optional and can be used to track the last access time of the vector store."
    )

class CreateVectorStoreRequest(BaseModel):
    name: Optional[str] = Field(
        default=None,
        description="The name of the vector store."
    )
    chunking_strategy: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The chunking strategy used to chunk the file(s). If not set, will use the 'auto' strategy. Only applicable if file_ids is non-empty."
    )
    expires_after: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The expiration policy for a vector store."
    )
    file_ids: Optional[List[str]] = Field(
        default=None,
        description="A list of File IDs that the vector store should use. Useful for tools like file_search that can access files."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object. Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters."
    )