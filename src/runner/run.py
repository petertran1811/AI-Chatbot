from langchain_core.messages import HumanMessage

async def run_async(app, user_input, character, thread_id, run_id, suggestions: bool = False):
    """Thực thi gọi app graph, lấy kết quả conversation + trạng thái."""
    
    if isinstance(user_input, str):
        inputs = [HumanMessage(content=user_input)]
    else:
        inputs = [user_input]

    graph_config = {
        "configurable": {"thread_id": f"{thread_id}"},
        "metadata": {
            "run_id": run_id,
            "character_id": character,
            "suggestions": suggestions,
        },
        "tags": ["chatbot_interaction", character],
    }
    
    input_configs = {
        "messages": inputs,
        "character": character,
    }

    model_out = ""
    async for msg, metadata in app.with_config(graph_config).astream(input_configs, stream_mode="messages"):
        if msg.content and not isinstance(msg, HumanMessage) and metadata['langgraph_node'] == "conversation":
            state = await app.aget_state(graph_config)
            yield msg.content, state
    state = await app.aget_state(graph_config)

    yield model_out, state


def run_sync(app, user_input, character, thread_id, run_id, suggestions: bool = False):
    """Thực thi gọi app graph, lấy kết quả conversation + trạng thái."""
    
    if isinstance(user_input, str):
        inputs = [HumanMessage(content=user_input)]
    else:
        inputs = [user_input]

    # if graph_config is None:
    graph_config = {
        "configurable": {"thread_id": f"{thread_id}"},
        "metadata": {
            "run_id": run_id,
            "character_id": character,
            "suggestions": suggestions,
        },
        "tags": ["chatbot_interaction", character],
    }

    input_configs = {
        "messages": inputs,
        "character": character,
    }

    response = app.invoke(input=input_configs, config=graph_config)
    model_out = response['messages'][-1].content

    state = app.get_state(graph_config)

    return model_out, state
