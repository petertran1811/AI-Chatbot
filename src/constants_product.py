import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

LORE_CONFIG_FILE_PATH = "configs/characters/lore.yaml"
SYSTEMP_PROMPT_CONFIG_FILE_PATH = "configs/characters/system_prompt.yaml"

DB_URI = os.getenv("POSTGRES_URI", "")
# "turns" or "tokens" or "all". Turn is the number of turns in the conversation. "User" and "AI" are counted as 2 turn.
SUMMARIZE_STRATEGIES = "tokens"
SUMMARIZE_TOKENS_THRESHOLD = 500
SUMMARIZE_TURN_THRESHOLD = 50

TEST_CONNECTION_TIME=30
# --- Model Settings ---
TYPE_MODEL = "chatopenai"  # "ollama" or "openai" or "chatopenai == VLLM"

# OLLAMA_MODEL_NAME = "yantien/gemma2-uncensored"
# OLLAMA_TEMPERATURE = 1
# OLLAMA_TOP_P = 0.95
# OLLAMA_TOP_K = 50
LLAMA3_TOKENIZER_ID = "configs/tokenizer"
# HF_TOKEN = os.getenv("HF_TOKEN")


# Config VLLM Serve
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# BASE_URL = "http://0.0.0.0:8001/v1"
MODEL_NAME = "apero_Qwen2.5-7B-instruct-sexy-style"
BASE_URL = "http://0.0.0.0:8000/v1"
API_KEY = os.getenv("LLM_API_KEY")

# --- Database Settings ---
DB_POOL_MAX_SIZE = 20
DB_CONNECTION_KWARGS = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# --- LangChain Debugging ---
# LANGCHAIN_DEBUG = True
# LANGCHAIN_VERBOSE = True

# --- Image Settings ---
# EMOTION_IMAGE_DIR = "img_expression"
IMAGE_CHARACTERS_DIR = "configs/images_characters"
