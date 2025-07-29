import gradio as gr

from configs.character_config import CHARACTER_SELECTOR_LIST, DESCRIBE_CHARACTERS
from utils.image import get_image


def change_character(character, thread_id):
    image_character = get_image(character)
    describe = DESCRIBE_CHARACTERS.get(character, "No information")
    return [], "", describe, image_character
