from typing import Any

from langgraph.graph import END
from langchain_core.messages import AIMessage, HumanMessage

from flow.state import State
from constants import SUMMARIZE_STRATEGIES, SUMMARIZE_TOKENS_THRESHOLD, SUMMARIZE_TURN_THRESHOLD
from models.llms import TOKENIZER

# note: summarize by tokens by currently not include system prompt
def should_continue(state: State) -> str | Any:
    total_tokens = 0
    total_turns = 0
    summary = state.get("summary", "")

    if summary:
        total_tokens += len(TOKENIZER.tokenize(summary))
        total_turns += 1

    number_images = 0
    messages = []
    for message in state["messages"]:
        if message.additional_kwargs.get("soft_deleted", False):
            continue
        try:
            if isinstance(message, HumanMessage):
                for message_content in message.content:
                    if isinstance(message_content, dict) :
                        if message_content["type"] == "image_url":
                            number_images += 1
    
            messages.append(message.content[0]['text'] if isinstance(message, HumanMessage) else message.content)
        except Exception as e:
            continue

    total_turns += len(messages)

    messages = ""
    
    total_tokens += len(TOKENIZER.tokenize(messages))
    total_tokens += number_images * 1024
   # print("CURRENT TOKEN COUNTS: {}".format(total_tokens))
    # print("CURRENT TURN COUNTS: {}".format(total_turns))

    if SUMMARIZE_STRATEGIES == "tokens":
        if total_tokens > SUMMARIZE_TOKENS_THRESHOLD:
    #        print("Summarizing conversation by tokens")
            return "summarize_conversation"
    elif SUMMARIZE_STRATEGIES == "turns":
        if total_turns > SUMMARIZE_TURN_THRESHOLD:
            return "summarize_conversation"
    else:
        if total_tokens > SUMMARIZE_TOKENS_THRESHOLD or total_turns > SUMMARIZE_TURN_THRESHOLD:
            return "summarize_conversation"
    return END
