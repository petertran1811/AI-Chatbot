from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


class State(MessagesState):
    summary: str
    character: str
    user_profile: str
    cancelled: bool = False
    suggestions: list[str]


def get_message_from_state(state: State) -> str:
    """Get the message from the state."""
    context = ""
    summary = state.get("summary", "")
    if summary:
        context += f"Summary of conversation earlier: {summary}\n\n"

    context += "Here is the context of the conversation:\n"
    recent_messages = []
    for message in reversed(state["messages"][:-1]):
        if message.additional_kwargs.get("soft_deleted", False):
            continue
        role = "human"
        if isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool result"
        
        recent_messages.append("{}: {}".format(role, message.content))

    recent_messages.reverse()

    context += "\n".join(recent_messages)

    context += """\n
      Human message is:"""
    
    for message in reversed(state["messages"]):
        if message.additional_kwargs.get("soft_deleted", False):
            continue
        if isinstance(message, HumanMessage):
            context += "{}: {}\n".format(message.type, message.content)
            break
    return context


def get_messages(state: State) -> str:
    """Get the summary from the state."""
    is_image = False
    recent_messages = []
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            for message_content in message.content:
                if isinstance(message_content, dict) :
                    if message_content["type"] == "image_url":
                        is_image = True
                        break
        
        if message.additional_kwargs.get("soft_deleted", False):
            continue
        recent_messages.append(message)

    recent_messages.reverse()

    return recent_messages, is_image

def get_revert_messages(state: State) -> str:
    """Get the summary from the state."""
    is_image = False
    recent_messages = []
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            for message_content in message.content:
                if isinstance(message_content, dict) :
                    if message_content["type"] == "image_url":
                        is_image = True
                        break
        
        if message.additional_kwargs.get("soft_deleted", False):
            continue
        
        if isinstance(message, HumanMessage): 
            recent_messages.append(AIMessage(content=message.content))
        elif isinstance(message, AIMessage):
            recent_messages.append(HumanMessage(content=message.content))

    recent_messages.reverse()

    return recent_messages, is_image