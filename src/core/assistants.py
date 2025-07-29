from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union, Literal, Any
import time

class Tool(BaseModel):
    type: str = Field(
        description="The type of tool"
    )

class CodeInterpreterTool(Tool):
    type: str = Field(
        default="code_interpreter",
        description="Code interpreter tool"
    )

class FunctionTool(Tool):
    type: str = Field(
        default="function",
        description="Function tool"
    )
    function: Dict = Field(
        description="The function definition"
    )

class FileSearchTool(Tool):
    type: str = Field(
        default="file_search",
        description="File search tool"
    )

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = Field(
        default="text",
        description="The format in which the model returns a response"
    )
    schema: Optional[Dict] = Field(
        default=None,
        description="Schema for JSON responses, when type is json_schema"
    )

class Assistant(BaseModel):
    id: str = Field(
        description="The identifier, which can be referenced in API endpoints."
    )
    object: str = Field(
        default="assistant",
        description="The object type, which is always 'assistant'."
    )
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp (in seconds) for when the assistant was created."
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the assistant. The maximum length is 256 characters."
    )
    description: Optional[str] = Field(
        default=None,
        description="The description of the assistant. The maximum length is 512 characters."
    )
    model: str = Field(
        description="ID of the model to use. You can use the List models API to see all of your available models."
    )
    instructions: Optional[str] = Field(
        default=None,
        description="The system instructions that the assistant uses. The maximum length is 256,000 characters."
    )
    tools: List[Union[CodeInterpreterTool, FileSearchTool, FunctionTool]] = Field(
        default_factory=list,
        description="A list of tools enabled on the assistant. There can be a maximum of 128 tools per assistant."
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format."
    )
    response_format: Optional[Literal["text"]] = Field(
        default="text",
        description="Specifies the format that the model must output."
    )

class CreateAssistantRequest(BaseModel):
    model: str = Field(
        description="ID of the model to use. You can use the List models API to see all available models."
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the assistant. The maximum length is 256 characters."
    )
    description: Optional[str] = Field(
        default=None,
        description="The description of the assistant. The maximum length is 512 characters."
    )
    instructions: Optional[str] = Field(
        default=None,
        description="The system instructions that the assistant uses. The maximum length is 256,000 characters."
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="A list of tools enabled on the assistant. There can be a maximum of 128 tools per assistant.",
        examples=[[{"type": "code_interpreter"}]]
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to the assistant. Keys are strings with maximum length of 64 characters. Values are strings with maximum length of 512 characters."
    )
    response_format: Optional[Dict[str, str]] = Field(
        default=None,
        description="Specifies the format that the model must output. Compatible with GPT-4o, GPT-4 Turbo, and all GPT-3.5 Turbo models since gpt-3.5-turbo-1106."
    )
    temperature: Optional[float] = Field(
        default=1.0,
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."
    )
    top_p: Optional[float] = Field(
        default=1.0,
        description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered."
    )
    tool_resources: Optional[Dict] = Field(
        default=None,
        description="A set of resources that are used by the assistant's tools. The resources are specific to the type of tool."
    )
    reasoning_effort: Optional[str] = Field(
        default="medium",
        description="Constrains effort on reasoning for reasoning models. Currently supported values are 'low', 'medium', and 'high'. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response. For o-series models only."
    )