import asyncio
from langchain_core.messages import HumanMessage
from langchain.globals import set_debug
from runner.engine import get_async_app, get_sync_app
from ui.layout import create_ui
import traceback
from ui.handlers import (
    change_character
)

import uuid
set_debug(True)

user_profile = ""
async def async_main():
    async with get_async_app() as app:
        async def chatbot_fn(user_input, history, character, thread_id):
            inputs = [HumanMessage(content=user_input)]
            history = history + [[user_input, None]]
            model_out = ""
                
            # try:
            graph_config = {
                "configurable": {"thread_id": "{}_{}".format(thread_id, character)},
                "metadata": {
                        "user_id": thread_id,
                        "character": character,
                        "run_id": str(uuid.uuid4()),
                        "suggestions": False
                    },
                    "tags": ["chatbot_interaction", character] 
                }
            
            input_configs = {
                    "messages": inputs,
                    "character": character,
                    "user_profile": user_profile
                }
                
            async for msg, metadata in app.with_config(graph_config).astream(input_configs, stream_mode="messages"):
                if msg.content and not isinstance(msg, HumanMessage) and metadata['langgraph_node'] == "conversation":
                    model_out += msg.content
                    chat_out = model_out
                    history[-1][1] = chat_out
                    yield history, "", None, "", "", ""
                        
        demo = create_ui(
            chatbot_fn_wrapper=chatbot_fn,
            change_character_wrapper=change_character
            )
    
        print("--- Launching Gradio Interface ---")
        demo.queue(default_concurrency_limit=50)
        demo.launch(server_name="0.0.0.0", share=True, server_port=8005, show_api=True)

def sync_main():
    with get_sync_app() as app:
    
        def chatbot_fn(user_input, history, character, thread_id):
            inputs = [HumanMessage(content=user_input)]
            history = history + [[user_input, None]]
            model_out = ""
                
            try:
                graph_config = {
                    "configurable": {"thread_id": "{}_{}".format(thread_id, character)},
                    "metadata": {
                            "user_id": thread_id,
                            "character": character,
                            "run_id": str(uuid.uuid4()),
                            "suggestions": False
                        },
                        "tags": ["chatbot_interaction", character] 
                    }
                
                input_configs = {
                        "messages": inputs,
                        "character": character,
                    }
                
                chat_out = app.invoke(input=input_configs, config=graph_config)
                print(chat_out)
                history[-1][1] = chat_out['messages'][-1].content 
                state = app.get_state(graph_config)
                yield history, "", None, "", "", ""

            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                model_out = "Tôi không hiểu câu hỏi của bạn. I don't understand your question. मैं आपका प्रश्न नहीं समझ पाया।"
                history[-1][1] = model_out
                yield history, "", None, "", "", ""

        demo = create_ui(
            chatbot_fn_wrapper=chatbot_fn,
            change_character_wrapper=change_character,
            )

        print("--- Launching Gradio Interface ---")
        demo.queue(default_concurrency_limit=50)
        demo.launch(server_name="0.0.0.0", share=True, server_port=8004, show_api=True)
            
if __name__ == "__main__":
    try:
        asyncio.run(async_main())
        # sync_main()
    except KeyboardInterrupt:
        print("\n--- Application Interrupted by User ---")