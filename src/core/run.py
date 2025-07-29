from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union, Any
import time

def generate_run_id(thread_id: str) -> str:
    """
    Generate a unique run ID.
    """
    return f"run_{thread_id}_{int(time.time())}"

class TruncationStrategy(BaseModel):
    type: str = Field(
        default="auto",
        description="The truncation strategy to use for the thread. Options: 'auto' or 'last_messages'"
    )
    last_messages: Optional[int] = Field(
        default=None,
        description="The number of most recent messages from the thread when constructing the context for the run"
    )

class ToolCallFunction(BaseModel):
    name: str = Field(description="The name of the function")
    arguments: str = Field(description="The arguments that the model expects you to pass to the function")

class ToolCall(BaseModel):
    id: str = Field(description="The ID of the tool call. Must be referenced when submitting tool outputs")
    type: str = Field(default="function", description="The type of tool call the output is required for")
    function: ToolCallFunction = Field(description="The function definition")

class RequiredActionSubmitToolOutputs(BaseModel):
    tool_calls: List[ToolCall] = Field(description="A list of the relevant tool calls")
    type: str = Field(default="submit_tool_outputs", description="Always 'submit_tool_outputs'")

class Usage(BaseModel):
    completion_tokens: int = Field(description="Number of completion tokens used over the course of the run")
    prompt_tokens: int = Field(description="Number of prompt tokens used over the course of the run")
    total_tokens: int = Field(description="Total number of tokens used (prompt + completion)")

class LastError(BaseModel):
    code: str = Field(description="Error code: 'server_error', 'rate_limit_exceeded', or 'invalid_prompt'")
    message: str = Field(description="A human-readable description of the error")

class IncompleteDetails(BaseModel):
    reason: str = Field(description="The reason why the run is incomplete")

class RunObject(BaseModel):
    id: str = Field(description="The identifier, which can be referenced in API endpoints")
    object: str = Field(default="thread.run", description="The object type, which is always 'thread.run'")
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp (in seconds) for when the run was created"
    )
    assistant_id: str = Field(description="The ID of the assistant used for execution of this run")
    thread_id: str = Field(description="The ID of the thread that was executed on as a part of this run")
    status: str = Field(
        description="The status of the run: 'queued', 'in_progress', 'requires_action', 'cancelling', "
                    "'cancelled', 'failed', 'completed', 'incomplete', or 'expired'"
    )
    
    # Optional fields
    completed_at: Optional[int] = Field(None, description="The Unix timestamp (in seconds) for when the run was completed")
    cancelled_at: Optional[int] = Field(None, description="The Unix timestamp (in seconds) for when the run was cancelled")
    failed_at: Optional[int] = Field(None, description="The Unix timestamp (in seconds) for when the run failed")
    expires_at: Optional[int] = Field(None, description="The Unix timestamp (in seconds) for when the run will expire")
    started_at: Optional[int] = Field(None, description="The Unix timestamp (in seconds) for when the run was started")
    
    model: Optional[str] = Field(None, description="The model that the assistant used for this run")
    instructions: Optional[str] = Field(None, description="The instructions that the assistant used for this run")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="The list of tools that the assistant used for this run")
    
    # Run parameters
    temperature: Optional[float] = Field(None, description="The sampling temperature used for this run. Default is 1")
    top_p: Optional[float] = Field(None, description="The nucleus sampling value used for this run. Default is 1")
    truncation_strategy: Optional[TruncationStrategy] = Field(
        None, 
        description="Controls for how a thread will be truncated prior to the run"
    )
    
    # Response and format options
    response_format: Optional[Dict[str, Any]] = Field(None, description="Specifies the format that the model must output")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description="Controls which (if any) tool is called by the model"
    )
    
    # Usage statistics
    usage: Optional[Usage] = Field(None, description="Usage statistics related to the run")
    
    # Error handling
    last_error: Optional[LastError] = Field(None, description="The last error associated with this run")
    
    # For incomplete runs
    incomplete_details: Optional[IncompleteDetails] = Field(None, description="Details on why the run is incomplete")
    
    # For runs requiring action
    required_action: Optional[RequiredActionSubmitToolOutputs] = Field(
        None, 
        description="Details on the action required to continue the run"
    )
    
    # Tool-related options
    parallel_tool_calls: bool = Field(
        default=False,
        description="Whether to enable parallel function calling during tool use"
    )
    
    # Token limits
    max_completion_tokens: Optional[int] = Field(
        None, 
        description="The maximum number of completion tokens specified to have been used over the course of the run"
    )
    max_prompt_tokens: Optional[int] = Field(
        None,
        description="The maximum number of prompt tokens specified to have been used over the course of the run"
    )
    
    metadata: Optional[Dict[str, str]] = Field(
        None,
        description="Set of 16 key-value pairs that can be attached to an object for storing additional information"
    )

# ...existing code...
class CreateRunRequest(BaseModel):
    assistant_id: str = Field(description="The ID of the assistant to use for execution of this run")
    
    # Optional fields
    model: Optional[str] = Field(
        None, 
        description="The ID of the model to use for this run. If provided, overrides the model associated with the assistant"
    )
    instructions: Optional[str] = Field(
        None,
        description="Overrides the instructions of the assistant for this run"
    )
    additional_instructions: Optional[str] = Field(
        None,
        description="Appends additional instructions at the end of the instructions for the run"
    )
    additional_messages: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Adds additional messages to the thread before creating the run"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="The tools the assistant can use for this run. This is useful for modifying behavior on a per-run basis"
    )
    metadata: Optional[Dict[str, str]] = Field(
        None, 
        description="Set of 16 key-value pairs that can be attached to the run"
    )
    
    # Run parameters
    temperature: Optional[float] = Field(
        None, 
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 make output more random, while lower values like 0.2 make it more focused and deterministic"
    )
    top_p: Optional[float] = Field(
        None, 
        description="Nucleus sampling parameter. An alternative to sampling with temperature, where the model considers results of tokens with top_p probability mass"
    )
    stream: Optional[bool] = Field(
        None,
        description="If true, returns a stream of events that happen during the Run as server-sent events"
    )
    
    # Format options
    response_format: Optional[Dict[str, Any]] = Field(
        None, 
        description="Specifies the format that the model must output"
    )
    
    # Tool options
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description="Controls which (if any) tool is called by the model"
    )
    
    # Token limits
    max_completion_tokens: Optional[int] = Field(
        None,
        description="The maximum number of completion tokens that may be used over the course of the run"
    )
    max_prompt_tokens: Optional[int] = Field(
        None,
        description="The maximum number of prompt tokens that may be used over the course of the run"
    )
    
    # Thread truncation
    truncation_strategy: Optional[TruncationStrategy] = Field(
        None,
        description="Controls how a thread will be truncated prior to the run"
    )
    
    # Function calling settings
    parallel_tool_calls: Optional[bool] = Field(
        None,
        description="Whether to enable parallel function calling during tool use"
    )
    
    # For o-series models only
    reasoning_effort: Optional[str] = Field(
        None,
        description="Constrains effort on reasoning for reasoning models. Values: 'low', 'medium', or 'high'"
    )