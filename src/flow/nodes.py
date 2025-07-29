from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from configs.character_config import (
    CHARACTER_CONFIG, SYSTEM_PROMPT_CONFIG
)

from models.llms import MODEL, MODEL_STRUCTURED_OUTPUT, model_json_schema
from flow.state import State, get_messages
from flow.prompts import get_prompt


def call_model(state: State, config: RunnableConfig):
    character_lore = CHARACTER_CONFIG["character"][state['character']]['instruction']
    prompt = get_prompt(character_lore=character_lore)
    
    run_id = config["metadata"]["run_id"]

    summary = state.get("summary", "")
    summary_message = ""
    
    if summary:
        summary_message = "\n\n" + f"Summary of conversation earlier: {summary}"
        
    conversation, is_image = get_messages(state)
    
    suggestions = config["metadata"].get("suggestions", False)  

    if suggestions:
        prompt += f"""
        \n\nOutput JSON Schema format:
        ```
        {model_json_schema}
        ```
        """
        
    response = MODEL.invoke(
        input=[
            SystemMessage(content=prompt + summary_message),
        ] + conversation,
        config=config | {"run_id": run_id},
    )

    return {"messages": response}


def summarize_conversation(state: State):
    summary = state.get("summary", "")
    if summary:
        summary_message = SYSTEM_PROMPT_CONFIG["system_message_summary_extend"]["content"].format(
            summary=summary,
            name=state["character"]
        )
    else:
        summary_message = SYSTEM_PROMPT_CONFIG["system_message_summary_new"]["content"].format(
            name=state["character"]
        )

    messages = []
    for message in reversed(state["messages"]):
        if message.additional_kwargs.get("soft_deleted", False):
            continue

        if isinstance(message, AIMessage):
            content = message.content
            messages.append(AIMessage(content=content))
        else:
            messages.append(message)

    messages.reverse()

    messages = state["messages"] + [HumanMessage(content=summary_message)]

    response = MODEL.invoke(messages)

    for message in reversed(state["messages"][:-2]):
        if message.additional_kwargs.get("soft_deleted", False):
            break
        else:
            message.additional_kwargs = {"soft_deleted": True}

    return {"summary": response.content, "messages": state["messages"]}

def suggestion_for_user(state: State):
    """
    Generate suggestions for the user based on the current state of the conversation.
    """
    prompt = """
        Based on the conversation context, suggest 3 follow-up questions that the user could ask to continue the dialogue.
        The questions should be relevant, natural, and help drive the conversation forward.
        Output only 3 concise questions from the user's perspective.
    """
    
    conversation, is_image = get_messages(state)
    
    response = MODEL_STRUCTURED_OUTPUT.ainvoke(
        input=[
            SystemMessage(content=prompt),
        ] + conversation,
    )
    suggestions = response.suggestions

    return {"suggestions": suggestions}