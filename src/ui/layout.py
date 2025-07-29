import gradio as gr
from configs.character_config import CHARACTER_SELECTOR_LIST, DESCRIBE_CHARACTERS
from utils.image import get_image


def create_ui(chatbot_fn_wrapper,
              change_character_wrapper):
    with gr.Blocks() as demo:
        with gr.Tab("Chat"):
            gr.Markdown("<h1 style='text-align: center;'>AI Character</h1>")
            with gr.Row(variant="compact"):
                with gr.Column(scale=3):
                    character_selector = gr.Dropdown(
                        CHARACTER_SELECTOR_LIST, label="Select Character", value="general_bot")
                    describe_characters = gr.Textbox(
                        label="Character Description", interactive=False, value=DESCRIBE_CHARACTERS["study_mentor"], lines=5, visible=False)
                    thread_id_input = gr.Textbox(
                        label="User ID (give any unique ID)")
                    chatbot_history = gr.Chatbot(
                        label="Chat History", height=550)
                    message_input = gr.Textbox(
                        show_label=False, placeholder="Type a message...")

                with gr.Column(scale=1):
                    prompt = gr.Textbox(label="prompt", visible=False)
                    emotion = gr.Textbox(label="Emotion", visible=False)
                    affection = gr.Textbox(label="Affection", visible=False)
                    suggestion_next_response = gr.Textbox(
                        lines=5,
                        interactive=False,
                        label="Suggestion",
                        value="\n".join([
                            "Hey, how’s your day going?",
                            "You just popped up in my mind, so I had to say hi!",
                            "What’s something fun you did today?",
                            "I need a good movie recommendation. Any ideas?",
                            "Hey, I think we’d have a fun conversation. What do you say?"
                        ]))

                    character_image = gr.Image(
                        label="Character Image", elem_id="chracter-image", width=512, height=512, value=get_image("Sophia"), visible=False)

        # --- Event Handlers ---
            character_selector.change(
                fn=change_character_wrapper,
                inputs=[character_selector, thread_id_input],
                outputs=[chatbot_history, message_input,
                         describe_characters, character_image]
            )

            message_input.submit(
                fn=chatbot_fn_wrapper,
                inputs=[
                    message_input,
                    chatbot_history,
                    character_selector,
                    thread_id_input
                ],
                outputs=[
                    chatbot_history,
                    message_input,
                    emotion,
                    prompt,
                    affection,
                    suggestion_next_response
                ],
                show_progress=False
            )

    return demo
