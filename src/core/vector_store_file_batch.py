from typing import Optional, Dict, List, Union, Literal, Any, BinaryIO

from pydantic import BaseModel, Field

from core.files import FileCounts

class VectorStoreFileBatch(BaseModel):
    id: str = Field(description="The identifier, which can be referenced in API endpoints.")
    object: str = Field(
        default="vector_store_file_batch",
        description="The object type, which is always 'vector_store_file_batch'"
    )
    created_at: int = Field(
        description="The Unix timestamp (in seconds) for when the vector store files batch was created"
    )
    status: str = Field(
        description="The status of the vector store files batch"
    )
    file_counts: FileCounts = Field(description="Statistics about the files in the batch", default_factory=FileCounts)
    vector_store_id: str = Field(description="The ID of the vector store that this batch is attached to")



class ChunkingStrategy(BaseModel):
    strategy: Literal["auto"] = Field(
        default="auto",
        description="The strategy to use for chunking files. Currently only 'auto' is supported."
    )


class CreateVectorStoreFileBatchRequest(BaseModel):
    file_ids: List[str] = Field(
        description="A list of File IDs that the vector store should use. Useful for tools like file_search that can access files."
    )
    attributes: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys are strings with maximum length of 64 characters. Values are strings with maximum length of 512 characters, booleans, or numbers."
    )
    chunking_strategy: Optional[ChunkingStrategy] = Field(
        default=None,
        description="The chunking strategy used to chunk the file(s). If not set, will use the 'auto' strategy."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "vector_store_id": "vs_abc123",
                "file_ids": ["file-abc123", "file-abc456"],
                "attributes": {"purpose": "document_indexing"},
                "chunking_strategy": {"strategy": "auto"}
            }
        }

class RetrieveVectorStoreFileBatchRequest(BaseModel):
    vector_store_id: str = Field(
        description="The ID of the vector store that the file batch belongs to"
    )
    batch_id: str = Field(
        description="The ID of the file batch being retrieved"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "vector_store_id": "vs_abc123",
                "batch_id": "vsfb_abc123"
            }
        }