from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

from constants import LORE_CONFIG_FILE_PATH, SYSTEMP_PROMPT_CONFIG_FILE_PATH
from db.manager import update_data_by_name, add_data, check_exist_table, get_all_data, delete_data_by_name, create_tables
from utils.io import load_yaml

logger = logging.getLogger(__name__)

def _load_config_file(loader, path: Path, description: str) -> Dict[str, Any]:
    try:
        return loader(path)
    except FileNotFoundError:
        logger.error(f"{description} not found at {path}.")
        return {}
    except Exception as e:
        logger.error(f"Error loading {description} from {path}: {e}")
        return {}
    
def load_all_configs() -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Loads all necessary configurations."""
    system_prompt_config = _load_config_file(load_yaml, SYSTEMP_PROMPT_CONFIG_FILE_PATH, "System prompt config")
    
    # delete_data_by_name(name="general_bot")
    # add_data(
    #     name="general_bot",
    #     instruction="""
    #           **Primary Goal**  
    #     Act as a knowledgeable and insightful assistant, capable of answering any question and addressing any concern with accuracy, clarity, and helpfulness.
        
    #     ## **Approach**  
    #     - Carefully analyze user requests and provide clear, well-structured responses.  
    #     - Deliver informative and valuable answers, leveraging reliable sources when necessary.  
    #     - Maintain a friendly, respectful, and unbiased attitude in all interactions.  
    #     - Support creative content generation, such as storytelling, poetry, and idea brainstorming, ensuring coherence and engagement.  
    #     - If the user provides an image without accompanying question, analyze and describe the image in detail instead of declining to respond.
    #     - After each response, **suggest a next action or follow-up question** for the user. This should:
    #         - Encourage deeper exploration of the topic  
    #         - Take the form of a yes/no question or a short, open-ended prompt (e.g., “Would you like to explore more examples?”, “Shall we try a different approach?”, or “Do you want me to explain how it works step by step?”)
        
    #     ## **Limitations**  
    #     - Avoid providing medical, legal, or financial advice without clear factual backing.  
    #     - Refrain from engaging in discussions that violate ethical or community standards.

    #     ## **Scope of Topics**  
    #     - Capable of answering questions across various domains, including science, technology, history, culture, philosophy, and general life inquiries, without restriction.
        
    #     ## **Handling Inappropriate Questions**  
    #     - Politely inform users when a question is inappropriate and guide them toward safe and meaningful discussions.
    #     """
    # )
    if check_exist_table() == False:
        create_tables()
        character_config = _load_config_file(load_yaml, LORE_CONFIG_FILE_PATH, "Lore config")
        
        for character in character_config["character"]:
            add_data(
                name=character,
                instruction=character_config["character"][character]["instruction"],
            )
    else:
        data = get_all_data()
        character_config = {
            "character": {}
        }
        
        for character in data:
            character_config["character"][character["name"]] = {
                "instruction": character["instruction"]
            }
        
    return character_config, system_prompt_config

def save_new_character(name: str, instruction: str) -> None:
    """Saves a new character to the database."""
    if not name or not instruction:
        logger.error("Name and instruction must be provided.")
        return
    
    if name not in CHARACTER_CONFIG:
        CHARACTER_CONFIG["character"][name] = {}
        add_data(
            name=name,
            instruction=instruction,
        )
        CHARACTER_CONFIG["character"][name]["instruction"] = instruction
        CHARACTER_SELECTOR_LIST.append(name)
        DESCRIBE_CHARACTERS[name] = instruction
        logger.info(f"New character '{name}' added successfully.")
    else:
        update_data_by_name(
            name=name,
            instruction=instruction,
        )
        DESCRIBE_CHARACTERS[name] = instruction
        logger.info(f"Character '{name}' updated successfully.")

def update_character(name: str, instruction: str) -> None:
    """Updates an existing character in the database."""
    if not name or not instruction:
        logger.error("Name and instruction must be provided.")
        return
    
    if name in CHARACTER_CONFIG["character"]:
        update_data_by_name(
            name=name,
            instruction=instruction,
        )
        DESCRIBE_CHARACTERS[name] = instruction
        logger.info(f"Character '{name}' updated successfully.")
    else:
        logger.error(f"Character '{name}' does not exist.")

def delete_character(name: str) -> None:
    """Deletes a character from the database."""
    if not name:
        logger.error("Name must be provided.")
        return
    
    if name in CHARACTER_CONFIG["character"]:
        CHARACTER_CONFIG["character"].pop(name, None)
        CHARACTER_SELECTOR_LIST.remove(name)
        DESCRIBE_CHARACTERS.pop(name, None)
        delete_data_by_name(name=name)
        logger.info(f"Character '{name}' deleted successfully.")
    else:
        logger.error(f"Character '{name}' does not exist.")
 
CHARACTER_CONFIG, SYSTEM_PROMPT_CONFIG = load_all_configs()
CHARACTER_SELECTOR_LIST = list(CHARACTER_CONFIG["character"].keys())
DESCRIBE_CHARACTERS = {}
for character in CHARACTER_SELECTOR_LIST:
    DESCRIBE_CHARACTERS[character] = CHARACTER_CONFIG["character"][character]["instruction"]


