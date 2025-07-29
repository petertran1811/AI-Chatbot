from langchain_openai import ChatOpenAI
import asyncio
import json

config = json.load(open("/shared/GiangNT/AI-Mini-ChatBot/src/bot_prompts.json", "r"))


MODEL_NAME = "Qwen/Qwen3-8B"
BASE_URL = "http://localhost:8001/v1"
API_KEY = "token-abc123"

llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0.9,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

from langgraph.graph import MessagesState

from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

SYSTEM_MESSAGE = SystemMessage(content=config["crush"])

async def call_model(state: MessagesState, config: RunnableConfig):
    # If there is summary, then we add it
    messages = state["messages"]
    
    response = await llm.ainvoke([SYSTEM_MESSAGE] + messages, config=config)
    return {"messages": response}

from langgraph.graph import StateGraph, START

workflow = StateGraph(MessagesState)
workflow.add_node("conversation", call_model)

workflow.add_edge(START, "conversation")

memory = MemorySaver()

graph = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1"}}

async def run_graph():
    while True:
        # Get user input
        user_input = input("User: ")
        
        input_message = HumanMessage(content=user_input)    
        print("AI: ", end=" ", flush=True)
        async for msg, metadata in graph.with_config(config).astream({"messages": [input_message]}, stream_mode="messages"):
            
            print(msg.content, end="", flush=True)
        
        print()



if __name__ == "__main__":
    asyncio.run(run_graph())