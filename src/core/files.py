from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union, Literal, Any, BinaryIO
import time

class FileCounts(BaseModel):
    in_progress: int = Field(default=0, description="Number of files that are currently being processed")
    completed: int = Field(default=0, description="Number of files that have been successfully processed")
    failed: int = Field(default=0, description="Number of files that failed to process")
    cancelled: int = Field(default=0, description="Number of files that were cancelled during processing")
    total: int = Field(description="Total number of files in the batch", default=0)
    
class FileResponse(BaseModel):
    id: str = Field(
        description="The file identifier, which can be referenced in API endpoints."
    )
    object: str = Field(
        default="file",
        description="The object type, which is always 'file'."
    )
    bytes: int = Field(
        description="The size of the file, in bytes."
    )
    created_at: int = Field(
        description="The Unix timestamp (in seconds) for when the file was created."
    )
    expires_at: Optional[int] = Field(
        default=None,
        description="The Unix timestamp (in seconds) for when the file will expire."
    )
    filename: str = Field(
        description="The name of the file."
    )
    purpose: str = Field(
        description="The intended purpose of the file. Supported values are 'assistants', 'assistants_output', 'batch', 'batch_output', 'fine-tune', 'fine-tune-results', and 'vision'."
    )
