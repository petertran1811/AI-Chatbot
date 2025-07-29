
SYSTEM_PROMPT_TEMPLATE = """
You are an helpful assistant.

{character_lore}
"""

def get_prompt(character_lore: str
) -> str:
    """Generates a prompt for the character."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        character_lore=character_lore
    )
