import os
import json
import time
import uuid
import logging
import asyncio
import threading
import requests
from typing import Optional, Annotated

from fastapi import FastAPI, Depends, HTTPException, status, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.background import BackgroundTasks

from langgraph.types import StateSnapshot
from langchain_core.messages import HumanMessage, AIMessage

from db.manager import (
    get_all_data,
    update_data_by_name,
    delete_data_by_name
)

from runner.engine import get_async_app, get_sync_app
from runner.run import run_async, run_sync
from core.threads import (
    CreateThreadsRequest,
    Thread,
)
from core.messages import (
    generate_message_id,
    MessageObject,
    CreateMessageRequest,
    TextContent,
    TextContentBase,
    ImageURL,
    ImageURLProperties,
    TextContentProperties,
    MessageDelta,
    DeltaContent,
    DeltaContentProperties
)
from core.run import (
    RunObject,
    CreateRunRequest,
)
from core.assistants import (
    Assistant,
    CreateAssistantRequest,
)
from core.chat_completion import ChatCompletion, ChatCompletionChoice, ChatCompletionUsage, CreateChatCompletionRequest
from core.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta
from core.vector_store import (
    VectorStore,
    CreateVectorStoreRequest,
)
from core.vector_store_file_batch import (
    VectorStoreFileBatch,
    CreateVectorStoreFileBatchRequest,
    RetrieveVectorStoreFileBatchRequest
)
from core.files import (
    FileResponse,
    FileCounts
)
from models.llms import count_tokens, MODEL
from models.embedding import (
    VLLMEmbedding,
    create_vector_store,
    get_list_indices,
    VECTOR_STORE
)
from utils.utils import convert_message_to_langgraph_message
from constants import FILE_STORAGE_PATH
from configs.character_config import (
    save_new_character,
    update_character,
    delete_character,
    CHARACTER_SELECTOR_LIST
)


from log_config import MillisecondFormatter

os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
# logger = logging.getLogger("uvicorn")
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

logger = logging.getLogger(__name__)

for handler in logger.handlers:
    handler.setFormatter(MillisecondFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S.%f"
    ))
    
checkFilePath = "../HealthCheckFile.txt"
if not os.path.exists(checkFilePath):
    with open(checkFilePath, 'w') as file:
        file.write("200, OK")
    logger.info("Health Check file is created")
    
app = FastAPI()

def check_health_vllm():
    try:
        while True:
            resp = requests.get("http://localhost:8000/health", timeout=5)
            time.sleep(30)
    except:
        logger.error("VLLM is not running or not reachable.")
        if os.path.exists(checkFilePath):
            os.remove(checkFilePath)
    

monitor_thread = threading.Thread(target=check_health_vllm)
monitor_thread.setDaemon(True)
monitor_thread.start()

BEARER_TOKEN = os.getenv("API_KEY", "API_KEY")

bearer_scheme = HTTPBearer()

async def verify_static_token(
    auth: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
):
    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    is_correct_scheme = auth.scheme.lower() == "bearer"
    is_correct_token = auth.credentials == BEARER_TOKEN

    if not (is_correct_scheme and is_correct_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"token_verified": True, "token_value": auth.credentials}

def update_state_to_graph(
    user_id: str | None = None,
    character_id: str | None = None,
    thread_id: str | None = None,
    messages: str | list | None = None,
    cancelled: bool | None = None,
    as_node: str | None = None,
    # tool_resources: dict | None = None
) -> None:
    """
    Update the thread information to the graph.
    """
    with get_sync_app() as app:
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        values = {}
        if character_id is not None:
            values["character_id"] = character_id
        if user_id is not None:
            values["user_id"] = user_id
        if messages is not None and isinstance(messages, dict) and len(messages) > 0:
            values["messages"] = messages
        if messages is not None and (isinstance(messages, HumanMessage) or isinstance(messages, AIMessage)):
            values["messages"] = messages
        if cancelled is not None:
            values["cancelled"] = cancelled
        app.update_state(
            config,
            values=values,
            as_node=as_node
        )
        logger.debug(f"Updated state with config: {config}, values: {values}")
    return None


def get_state_from_graph(
    thread_id: str
) -> StateSnapshot:
    """
    Get the thread information from the graph.
    """
    with get_sync_app() as app:
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        state = app.get_state(config)
        logger.debug(f"Retrieved state with config: {config}, state: {state}")
    return state

@app.post("/assistants", dependencies=[Depends(verify_static_token)])
def create_assistant(payload: CreateAssistantRequest) -> Assistant:
    logger.info("[POST] /assistants")
    logger.debug(f"Creating assistant with payload: {payload}")
    assistant_id = payload.name

    assistant = Assistant(
        id=assistant_id,
        model=payload.model,
        name=payload.name,
        description=payload.description,
        instructions=payload.instructions,
        metadata=payload.metadata,
    )

    save_new_character(
        name=assistant_id,
        instruction=payload.instructions
    )
    
    logger.info(f"Successfully created assistant with ID: {assistant_id}")
    return assistant

@app.get("/assistants", dependencies=[Depends(verify_static_token)])
def list_assistants(
    limit: int = 20,
    order: str = "desc",
    after: Optional[str] = None,
    before: Optional[str] = None
):
    logger.info("[GET] /assistants")
    logger.info(
        f"Listing assistants with limit: {limit}, order: {order}, after: {after}, before: {before}")
    assistants = []
    data = get_all_data()
    for character in data:
        assistants.append(Assistant(
            id=character["name"],
            object="assistant",
            created_at=int(time.time()),
            name=character["name"],
            model="deepseek",
            instructions=character["instruction"],
            metadata={},
        ))

    limit = max(1, min(100, limit))
    assistants = assistants[:limit]
    logger.info(f"Number of assistants found: {len(assistants)}")
    return {
        "object": "list",
        "data": assistants,
    }

@app.get("/assistants/{assistant_id}", dependencies=[Depends(verify_static_token)])
def retrieve_assistant(assistant_id: str) -> Assistant:
    logger.info("[GET] /assistants/{assistant_id}")
    logger.info(f"Retrieving assistant with ID: {assistant_id}")
    data = get_all_data()
    assistants = []
    
    for character in data:
        if character["name"] == assistant_id:
            assistants.append(Assistant(
                id=character["name"],
                object="assistant",
                created_at=int(time.time()),
                name=character["name"],
                model="deepseek",
                instructions=character["instruction"],
                metadata={},
            ))
    
    if not assistants:
        return JSONResponse(status_code=404, content={'error': {'message': f"No assistant found with id '{assistant_id}'.", 'type': 'invalid_request_error', 'param': None, 'code': None}})
    
    logger.info(f"Retrieved assistant: {assistants[0]}")
    return assistants[0]

@app.delete("/assistants/{assistant_id}", dependencies=[Depends(verify_static_token)])
def delete_assistant(assistant_id: str) -> JSONResponse:
    logger.info("[DELETE] /assistants/{assistant_id}")
    logger.info(f"Deleting assistant with ID: {assistant_id}")
    data = get_all_data()
    
    for character in data:
        if character["name"] == assistant_id:
            delete_character(name=assistant_id)
            logger.info(f"Successfully deleted assistant with ID: {assistant_id}")
            return JSONResponse(status_code=200, content={"id": assistant_id, "object": "assistant.deleted", "deleted": True})
    return JSONResponse(status_code=404, content={'error': {'message': f"No assistant found with id '{assistant_id}'.", 'type': 'invalid_request_error', 'param': None, 'code': None}})

@app.post("/assistants/{assistant_id}", dependencies=[Depends(verify_static_token)])
def modify_assistant(assistant_id: str, payload: CreateAssistantRequest) -> Assistant:
    logger.info("[POST] /assistants/{assistant_id}")
    logger.debug(f"Modifying assistant with ID: {assistant_id} and payload: {payload}")
    data = get_all_data()
        
    for character in data:
        if character["name"] == assistant_id:
            update_character(
                name=assistant_id,
                instruction=payload.instructions,
            )
            logger.info(f"Successfully modified assistant with ID: {assistant_id}")
            return Assistant(
                id=assistant_id,
                object="assistant",
                created_at=int(time.time()),
                name=payload.name,
                model=payload.model,
                instructions=payload.instructions,
                metadata={},
            )

    return JSONResponse(status_code=404, content={'error': {'message': f"No assistant found with id '{assistant_id}'.", 'type': 'invalid_request_error', 'param': None, 'code': None}})

@app.post("/threads", dependencies=[Depends(verify_static_token)])
def create_thread(payload: CreateThreadsRequest) -> Thread:
    """
        Create a new thread for the conversation.
    """
    logger.info("[POST] /threads")
    logger.debug(f"Creating thread with payload: {payload}")
    thread = Thread(
        id=str(uuid.uuid4()),
        object="thread",
        created_at=int(time.time()),
        metadata=payload.metadata,
        # tool_resources=payload.tool_resources
    )

    update_state_to_graph(
        thread_id=thread.id,
        messages=payload.messages,
        # tool_resources=payload.tool_resources
    )
    logger.info(f"Successfully created thread with ID: {thread.id}")
    return thread

@app.get("/threads/{thread_id}", dependencies=[Depends(verify_static_token)])
def retrieve_thread(thread_id: str) -> Thread:
    """
        Retrieve a thread by its ID.
    """
    logger.info("[GET] /threads/{thread_id}")
    logger.info(f"Retrieving thread with ID: {thread_id}")
    state: StateSnapshot = get_state_from_graph(thread_id=thread_id)

    if not state:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    thread_id = state.config["configurable"]["thread_id"]
    if not thread_id:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    thread = Thread(
        id=thread_id,
        object="thread",
        created_at=int(time.time()),
    )
    logger.info(f"Successfully retrieved thread: {thread.id}")
    return thread

@app.post("/threads/{thread_id}/messages", dependencies=[Depends(verify_static_token)])
def create_message(thread_id: str, payload: CreateMessageRequest) -> MessageObject:
    """
        Create a new message in the thread.
    """
    logger.info("[POST] /threads/{thread_id}/messages")
    logger.debug(
        f"Creating message in thread {thread_id} with payload: {payload}")
    state: StateSnapshot = get_state_from_graph(thread_id=thread_id)

    if not state:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    if isinstance(payload.content, str):
        content = [
            TextContent(
                type="text",
                text=TextContentProperties(
                    value=payload.content,
                    annotations=[]
                )
            )
        ]
    elif isinstance(payload.content, list):
        content = []
        for item in payload.content:
            if item.type == "text":
                content.append(
                    TextContent(
                        type=item.type,
                        text=TextContentProperties(
                            value=item.text,
                            annotations=[]
                        )
                    )
                )
            elif item.type == "image_url":
                content.append(
                    ImageURL(
                        type=item.type,
                        image_url=ImageURLProperties(
                            url=item.image_url.url,
                            detail=item.image_url.detail
                        )
                    )
                )
        
    message = MessageObject(
        id=generate_message_id(thread_id=thread_id),
        object="thread.message",
        created_at=int(time.time()),
        thread_id=thread_id,
        role=payload.role,
        content=content,
        metadata=payload.metadata,
        attachments=[],
    )

    update_state_to_graph(
        thread_id=thread_id,
        messages=convert_message_to_langgraph_message(message.dict()),
    )
    logger.info(f"Successfully created message: {message.id} in thread {thread_id}")
    logger.info(f"Content of the message: {payload.content}")
    total_tokens = 0
    for content_item in message.content:
        if content_item.type == "text":
            total_tokens += count_tokens(content_item.text.value)
        elif content_item.type == "image_url":
            continue
    logger.info(f"Tokens: {total_tokens}")
    return message

@app.get("/threads/{thread_id}/runs", dependencies=[Depends(verify_static_token)])
def list_runs(
    thread_id: str,
    limit: int = 20,
    order: str = "desc",
    after: Optional[str] = None,
    before: Optional[str] = None
) -> dict:
    """
    List runs belonging to a thread.
    """
    logger.info("[GET] /threads/{thread_id}/runs")
    logger.info(
        f"Listing runs in thread {thread_id} with limit: {limit}, order: {order}, after: {after}, before: {before}")

    state: StateSnapshot = get_state_from_graph(thread_id=thread_id)

    if not state:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    # Get runs from state
    runs = [
        RunObject(
        id=state.metadata['run_id'],
        object="thread.run",
        created_at=int(time.time()),
        assistant_id="default",
        thread_id=thread_id,
        status="in_progress",
        model="deepseek",
        instructions="default"
    )

    ]
    # Apply limit and order
    limit = max(1, min(100, limit))
    runs = sorted(runs, key=lambda x: x.created_at, reverse=(order == "desc"))
    runs = runs[:limit]

    # Format response according to OpenAI API format
    response = {
        "object": "list",
        "data": runs,
        "first_id": runs[0].id if runs else None,
        "last_id": runs[-1].id if runs else None,
        "has_more": len(runs) >= limit
    }

    logger.info(f"Number of runs found: {len(runs)} in thread {thread_id}")
    return response

CANCELLED_RUNS = set()
@app.post("/threads/{thread_id}/runs/{run_id}/cancel", dependencies=[Depends(verify_static_token)])
async def cancel_run(thread_id: str, run_id: str) -> RunObject:
    """
    Cancel an in-progress run.
    """
    logger.info("[POST] /threads/{thread_id}/runs/{run_id}/cancel")
    logger.info(f"Cancelling run with ID: {run_id} in thread {thread_id}")

    state: StateSnapshot = get_state_from_graph(thread_id=thread_id)
    if not state:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    update_state_to_graph(
        thread_id=thread_id,
        cancelled=True,
        as_node="conversation"
    )

    
    # Create a cancelled run object
    run = RunObject(
        id=run_id,
        object="thread.run",
        created_at=int(time.time()),
        assistant_id="default",
        thread_id=thread_id,
        status="completed",
        model="deepseek",
        instructions="default"
    )

    logger.info(f"Cancelling: {run_id} in thread {thread_id}")
    return run

@app.get("/threads/{thread_id}/suggestions", dependencies=[Depends(verify_static_token)])
async def get_suggestion(thread_id: str) -> dict:
    """
    Get suggestions for the user based on the conversation in the thread.
    """
    logger.info("[GET] /threads/{thread_id}/suggestions")
    logger.info(f"Retrieving suggestions for thread {thread_id}")

    state: StateSnapshot = get_state_from_graph(thread_id=thread_id)
    
    if not state:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    # Assuming we have a function to generate suggestions based on the state
    suggestions = state.values.get("suggestions", [])
    
    if not suggestions:
        return JSONResponse(status_code=404, content={"message": "No suggestions available"})

    logger.info(f"Suggestions retrieved: {suggestions}")
    return {"suggestions": suggestions}


@app.post("/threads/{thread_id}/runs", dependencies=[Depends(verify_static_token)])
async def create_run(thread_id: str, payload: CreateRunRequest) -> RunObject:
    """
        Create a new run in the thread.
    """
    start_time = time.time()
    logger.info("[POST] /threads/{thread_id}/runs")
    logger.debug(f"Creating run in thread {thread_id} with payload: {payload}")
    if payload.assistant_id not in CHARACTER_SELECTOR_LIST:
        logger.error(f"Assistant with ID {payload.assistant_id} not found in {CHARACTER_SELECTOR_LIST}")
        return JSONResponse(status_code=404, content={"message": "Assistant not found"})

    state: StateSnapshot = get_state_from_graph(thread_id=thread_id)

    user_input = state.values["messages"][-1]
    character_id = payload.assistant_id
    
    suggestions = False
    if payload.metadata is not None:
        suggestions = True if "suggestions" in payload.metadata and payload.metadata["suggestions"] == "true" else False
    
    if not state:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    run = RunObject(
        id=str(uuid.uuid4()),
        object="thread.run",
        created_at=int(time.time()),
        assistant_id=payload.assistant_id,
        thread_id=thread_id,
        status="completed",
        model=payload.model,
        instructions=payload.instructions,
        tools=payload.tools,
        temperature=payload.temperature,
        top_p=payload.top_p,
        truncation_strategy=payload.truncation_strategy,
        response_format=payload.response_format,
        tool_choice=payload.tool_choice,
    )
    
    if payload.stream:
        message_id = f"msg_{thread_id}_{int(time.time())}"
        message_created = MessageObject(
            id=message_id,
            object="thread.message",
            created_at=int(time.time()),
            thread_id=thread_id,
            assistant_id=payload.assistant_id,
            run_id=run.id,
            status="in_progress",
            role="assistant",
            content=[],
            metadata={},
            attachments=[],
        )

        async def stream_run():
            async with get_async_app() as app:
                i_cnt = 0
                full_response = ""
                async for chunk, state in run_async(app, user_input, character_id, thread_id, run.id, suggestions):
                    if state.values.get("cancelled", False) == True:
                        yield f"event: thread.run.cancelled\ndata: {json.dumps(run.dict())}\n\n"
                        yield f"event: done\ndata: [CANCELLED]\n\n"
                        logger.warning(f"Run {run.id} in {thread_id} cancelled during stream.")
                        update_state_to_graph(
                            thread_id=thread_id,
                            cancelled=False,
                            as_node="conversation"
                        )
                        return
            
                    response = MessageDelta(
                        id=message_id,
                        object="thread.message.delta",
                        delta=DeltaContent(
                            content=[
                                DeltaContentProperties(
                                    index=0,
                                    type="text",
                                    text=TextContentProperties(
                                        value=chunk,
                                        annotations=[]
                                    )
                                )
                            ]
                        )
                    )
                    full_response += chunk
                    
                    if i_cnt == 0:
                        logger.info(f"Time to first token: {time.time() - start_time:.2f} seconds. Token: {chunk}")
                        yield f"event: thread.message.created\ndata: {json.dumps(message_created.dict())}\n\n"
                        yield f"event: thread.message.in_progress\ndata: {json.dumps(message_created.dict())}\n\n"
                        ttft = time.time() - start_time
                    yield f"event: thread.message.delta\ndata: {json.dumps(response.dict())}\n\n"

                    i_cnt += 1
                yield f"event: thread.run.completed\ndata: {json.dumps(run.dict())}\n\n"
                yield f"event: done\ndata: [DONE]\n\n"
                response_time = time.time() - start_time
            logger.info(f"Streaming run for thread {thread_id} with run ID {run.id}.")
            logger.info(f"Model Response:\n{full_response}.")
            output_stats = {
                'Tokens output': f"{count_tokens(full_response)} tokens",
                'Time to first token': f"{ttft:.2f} seconds",
                'Total response time': f"{response_time:.2f} seconds",
                'Throughput': f"{count_tokens(full_response) / response_time if response_time > 0 else 0:.2f} tokens/second"
            }
            logger.info(f"Statistics: \n{output_stats}")
        return StreamingResponse(stream_run(), media_type="text/event-stream")
    else:
        with get_sync_app() as app:
            model_out, state = run_sync(
                app, user_input, character_id, thread_id, run.id, suggestions)
            response_time = time.time() - start_time
        logger.info(f"Run completed for thread {thread_id} with run ID {run.id}.")
        logger.info(f"Model Response:\n{model_out}.")
        output_stats = {
            'Tokens output': f"{count_tokens(model_out)} tokens",
            'Total response time': f"{response_time:.2f} seconds",
            'Throughput': f"{count_tokens(model_out) / response_time if response_time > 0 else 0:.2f} tokens/second"
        }
        logger.info(f"Statistics: \n{output_stats}")
    return run

@app.get("/threads/{thread_id}/messages", dependencies=[Depends(verify_static_token)])
def list_messages(
    thread_id: str,
    limit: int = 20,
    order: str = "desc",
    after: str = None,
    before: str = None,
    run_id: str = None
) -> dict:
    logger.info("[GET] /threads/{thread_id}/messages")
    logger.info(
        f"Listing messages in thread {thread_id} with limit: {limit}, order: {order}, after: {after}, before: {before}, run_id: {run_id}")
    state: StateSnapshot = get_state_from_graph(thread_id=thread_id)
    if not state:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    # Get messages from state
    messages = []
    if "messages" in state.values and isinstance(state.values["messages"], list):
        raw_messages = state.values["messages"]
        # print(f"raw_messages: {raw_messages}")
        # Convert langgraph messages to MessageObject format
        for i, msg in enumerate(raw_messages):
            if isinstance(msg, HumanMessage):
                role = "user"
                msg_run_id = None
                
                if isinstance(msg.content, str):
                    content = [
                        TextContent(
                            type="text",
                            text=TextContentProperties(
                                value=msg.content,
                                annotations=[]
                            )
                        )
                    ]
                elif isinstance(msg.content, list):
                    content = []
                    for content_item in msg.content:
                        if content_item["type"] == "text":
                            content.append(
                                TextContent(
                                    type=content_item["type"],
                                    text=TextContentProperties(
                                        value=content_item["text"],
                                        annotations=[]
                                    )
                                )
                            )
                        elif content_item["type"] == "image_url":
                            content.append(
                                ImageURL(
                                    type=content_item["type"],
                                    image_url=ImageURLProperties(
                                        url=content_item["image_url"]["url"],
                                        detail=None
                                    )
                                )
                            )
                    # content = [
                    #     TextContent(
                    #         type=msg.content[0]["type"],
                    #         text=TextContentProperties(
                    #             value=msg.content[0]["text"],
                    #             annotations=[]
                    #         )
                    #     )
                    # ]
                    # if len(msg.content) > 1:
                    #     content.append(
                    #         ImageURL(
                    #             type=msg.content[1]["type"],
                    #             image_url=ImageURLProperties(
                    #                 url=msg.content[1]["image_url"]["url"],
                    #                 detail=None
                    #             )
                    #         )
                    #     )
                        
                message = MessageObject(
                    id=f"msg_{thread_id}_{i}_{int(time.time())}",
                    object="thread.message",
                    created_at=int(time.time()),
                    thread_id=thread_id,
                    assistant_id="default_assistant",
                    role=role,
                    content=content,
                    run_id=msg_run_id,
                    attachments=[],
                    metadata={},
                )
                
            elif isinstance(msg, AIMessage):
                role = "assistant"
                msg_run_id = msg.id

                message = MessageObject(
                    id=f"msg_{thread_id}_{i}_{int(time.time())}",
                    object="thread.message",
                    created_at=int(time.time()),
                    thread_id=thread_id,
                    assistant_id="default_assistant",
                    role=role,
                    content=[
                        TextContent(
                            type="text",
                            text=TextContentProperties(
                                value=msg.content,
                                annotations=[]
                            )
                        ),
                    ],
                    run_id=msg_run_id,
                    attachments=[],
                    metadata={},
                )
                
            
            messages.append(message)

    # Apply filtering by run_id if specified
    if run_id:
        messages = [msg for msg in messages if msg.run_id == run_id]
    # Apply limit
    messages.reverse() if order == "desc" else messages
    messages = messages[:limit]

    # Format response according to OpenAI API format
    response = {
        "object": "list",
        "data": messages,
        "first_id": messages[0].id if messages else None,
        "last_id": messages[-1].id if messages else None,
        "has_more": True
    }
    logger.info(f"Number of messages found: {len(messages)} in thread {thread_id}")
    return response

@app.get("/threads/{thread_id}/runs/{run_id}", dependencies=[Depends(verify_static_token)])
async def retrieve_run(
    thread_id: str,
    run_id: str
):
    """
        Retrieve a run by its ID.
    """
    logger.info("[GET] /threads/{thread_id}/runs/{run_id}")
    logger.info(f"Retrieving run with ID: {run_id} in thread {thread_id}")
    state: StateSnapshot = get_state_from_graph(thread_id=thread_id)
    
    if not state:
        return JSONResponse(status_code=404, content={"message": "Thread not found"})

    # Check if the run_id exists in the state
    if not run_id:
        return JSONResponse(status_code=404, content={"message": "Run not found"})

    status = "queued"
    full_run_id = "run--" + run_id + "-0"
    for msg in state.values["messages"]:
        if msg.id == full_run_id:
            status = "completed"
            
    run = RunObject(
        id=run_id,
        object="thread.run",
        created_at=int(time.time()),
        assistant_id="",
        thread_id=thread_id,
        status=status,
    )
    logger.info(f"Successfully retrieved run: {run.id} in thread {thread_id} with status: {status}")
    return run

@app.post("/chat/completions", response_model=ChatCompletion)
async def create_chat_completion(
    request: CreateChatCompletionRequest,
) -> ChatCompletion:
    """
    Create a chat completion.
    """
    logger.info("[POST] /chat/completions")
    logger.info(f"Creating chat completion with request: {request}")
    
    is_image = False
    messages = []
    for message in request.messages:
        if message["role"] == "user":
            content = []
            for content_item in message["content"]:
                if isinstance(content_item, str):
                    content.append(
                        TextContent(
                            type="text",
                            text=TextContentProperties(
                                value=content_item,
                                annotations=[]
                            )
                        )
                    )
                elif content_item["type"] == "text":
                    content.append(
                        TextContent(
                            type="text",
                            text=TextContentProperties(
                                value=content_item["text"],
                                annotations=[]
                            )
                        )
                    )
                elif content_item["type"] == "image_url":
                    is_image = True
                    content.append(
                        ImageURL(
                            type="image_url",
                            image_url=ImageURLProperties(
                                url=content_item["image_url"]["url"],
                                detail=None
                            )
                        )
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported content type: {content_item['type']}"
                    )
            messages.append(
                MessageObject(
                    id=generate_message_id(thread_id="chatcmpl"),
                    object="thread.message",
                    created_at=int(time.time()),
                    thread_id="default_thread",
                    assistant_id="default_assistant",
                    role=message["role"],
                    content=content,
                    metadata=message.get("metadata", {}),
                    attachments=message.get("attachments", []),
                )
            )
        elif message["role"] in ["assistant", "system"]:
            messages.append(
                MessageObject(
                    id=generate_message_id(thread_id="chatcmpl"),
                    object="thread.message",
                    created_at=int(time.time()),
                    thread_id="default_thread",
                    assistant_id="default_assistant",
                    role=message["role"],
                    content=[
                        TextContent(
                            type="text",
                            text=TextContentProperties(
                                value=message["content"],
                                annotations=[]
                            )
                        )
                    ],
                    metadata=message.get("metadata", {}),
                    attachments=message.get("attachments", []),
                )
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role in messages. Must be 'user', 'assistant' or 'system'."
            )
    messages = [convert_message_to_langgraph_message(msg.dict()) for msg in messages]
    
    model = MODEL
    
    if request.stream is False:
        try:
            response: AIMessage = model.invoke(
                input=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                stream=False
            )
        except Exception as e:
            logger.error(f"Error invoking model: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model invocation failed: {str(e)}"
            )
        
        return ChatCompletion(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": response.content
                    },
                    refusal=None,
                    annotations=[],
                    logprobs=None,
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=0,
                completion_tokens=count_tokens(response.content),
                total_tokens=count_tokens(response.content)
            )
        )
    else:
        async def event_generator():
            try:
                index = 0
                async for chunk in model.astream(
                    input=messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ):
                    # logger.info(f"Received chunk: {chunk}")
                    logger.debug(f"Streaming chunk: {chunk}")
                    # yield SSE format
                    chunk_data = ChatCompletionChunk(
                        id="chatcmpl-" + str(uuid.uuid4()),
                        choices=[
                            Choice(
                                delta=ChoiceDelta(
                                    content=chunk.content,
                                    function_call=None,
                                    refusal=None,
                                    role="assistant" if index == 0 else None,
                                    tool_calls=None
                                ),
                                finish_reason=None,
                                index=0,
                                logprobs=None
                            )
                        ],
                        created=int(time.time()),
                        model=request.model,
                        object="chat.completion.chunk",
                        service_tier="default",
                        system_fingerprint="fp_51e1070cf2",
                        usage={}
                    )
                    index += 1
                    # logger.info(f"Yielding chunk: {chunk_data}")
                    yield f"data: {chunk_data.model_dump_json()}\n\n"
                    
                finish_chunk = ChatCompletionChunk(
                    id="chatcmpl-" + str(uuid.uuid4()),
                    choices=[
                        Choice(
                            delta=ChoiceDelta(
                                content=None,
                                function_call=None,
                                refusal=None,
                                role=None,
                                tool_calls=None
                            ),
                            finish_reason="stop",
                            index=0,
                            logprobs=None
                        )
                    ],
                    created=int(time.time()),
                    model=request.model,
                    object="chat.completion.chunk",
                    service_tier="default",
                    system_fingerprint="fp_51e1070cf2",
                    usage={}
                )
                yield f"finish: {finish_chunk.model_dump_json()}\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                yield f"data: [ERROR] {str(e)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

# https://api.openai.com/v1/vector_stores
@app.post("/vector_stores", dependencies=[Depends(verify_static_token)])
def create_vector_stores(
    payload: CreateVectorStoreRequest,
) -> VectorStore:
    """
    Create a new vector store.
    """
    logger.info("[POST] /vector_stores")
    logger.debug(f"Creating vector store with payload: {payload}")
    
    index_name = payload.name
    
    try:
        vector_store = create_vector_store(index_name=index_name)
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to create vector store"})

    logger.info(f"Successfully created vector store with name: {index_name}")
    return VectorStore(
        id="vs_" + index_name,
        object="vector_store",
        name=index_name,
        metadata=payload.metadata,
        created_at=int(time.time()),
        index_name=index_name,
    )
    
@app.get("/vector_stores", dependencies=[Depends(verify_static_token)])
def list_vector_stores(
    limit: int = 20,
    order: str = "desc",
    after: Optional[str] = None,
    before: Optional[str] = None
) -> dict:
    """
    List all vector stores.
    """
    logger.info("[GET] /vector_stores")
    logger.info(f"Listing vector stores with limit: {limit}, order: {order}, after: {after}, before: {before}")
    
    indices = get_list_indices(VECTOR_STORE)
    
    # Apply limit and order
    limit = max(1, min(100, limit))
    indices = sorted(indices, reverse=(order == "desc"))
    indices = indices[:limit]

    vector_stores = [VectorStore(
        id="vs_" + index,
        object="vector_store",
        name=index,
        metadata={},
        created_at=int(time.time()),
        index_name=index,
    ) for index in indices]

    logger.info(f"Number of vector stores found: {len(vector_stores)}")
    
    return {
        "object": "list",
        "data": vector_stores,
        "first_id": vector_stores[0].id if vector_stores else None,
        "last_id": vector_stores[-1].id if vector_stores else None,
        "has_more": len(vector_stores) >= limit
    }
    
@app.post("/files", dependencies=[Depends(verify_static_token)])
async def upload_file(file: UploadFile = File(...), purpose: str = Form(...)) -> FileResponse:
    """
    Upload a file for use in vector stores.
    """
    logger.info("[POST] /files")
    logger.debug(f"Uploading file with purpose: {purpose}")
    
    if purpose not in ["vector_store", "fine_tuning", "assistants", "batch", "vision", "user_data", "evals"]:
        return JSONResponse(status_code=400, content={"message": "Invalid purpose, must be one of ['vector_store', 'fine_tuning', 'assistants', 'batch', 'vision', 'user_data', 'evals']"})

    file_id = str(uuid.uuid4())
    file_path = f"{FILE_STORAGE_PATH}/{file_id}.{file.filename.split('.')[-1]}"
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"Successfully uploaded file with ID: {file_id}")
        return FileResponse(
            id=file_id,
            object="file",
            created_at=int(time.time()),
            filename=file.filename,
            purpose=purpose,
            bytes=len(content),
        )
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return JSONResponse(status_code=500, content={"message": "Failed to upload file"})
    
@app.post("/vector_stores/{vector_store_id}/file_batches", dependencies=[Depends(verify_static_token)])
async def create_vector_store_file_batch(
    vector_store_id: str,
    payload: CreateVectorStoreFileBatchRequest,
    background_tasks: BackgroundTasks
) -> VectorStoreFileBatch:
    """
    Create a new file batch for the vector store.
    """
    logger.info("[POST] /vector_stores/{vector_store_id}/file_batches")
    logger.debug(f"Creating file batch for vector store {vector_store_id} with payload: {payload}")
    
    if not vector_store_id.startswith("vs_"):
        return JSONResponse(status_code=400, content={"message": "Invalid vector store ID format"})

    batch_id = str(uuid.uuid4())
    
    # background_tasks.add_task(
    #     process_vector_store_file_batch,
    #     vector_store_id=vector_store_id,
    #     files=payload.files,
    #     metadata=payload.metadata
    # )
    
    file_batch = VectorStoreFileBatch(
        id=batch_id,
        created_at=int(time.time()),
        status="in_progress",
        vector_store_id=vector_store_id,
        file_counts={
            "in_progress": len(payload.file_ids),
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "total": len(payload.file_ids)
        },
        chunking_strategy=payload.chunking_strategy or {"strategy": "auto"}
    )

    logger.info(f"Successfully created file batch with ID: {file_batch.id} for vector store {vector_store_id}")
    return file_batch
