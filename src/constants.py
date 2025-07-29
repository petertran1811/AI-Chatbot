import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

LORE_CONFIG_FILE_PATH = "configs/characters/lore.yaml"
SYSTEMP_PROMPT_CONFIG_FILE_PATH = "configs/characters/system_prompt.yaml"

DB_URI = os.getenv("POSTGRES_URI", "")

print()
DB_CHATBOT = "MiniChatbot-Apero"
# DB_URI = "postgresql://postgres:abc123@66.42.43.42:5444/postgres?sslmode=disable"
# "turns" or "tokens" or "all". Turn is the number of turns in the conversation. "User" and "AI" are counted as 2 turn.
SUMMARIZE_STRATEGIES = "tokens"
SUMMARIZE_TOKENS_THRESHOLD = 3000
SUMMARIZE_TURN_THRESHOLD = 50

TEST_CONNECTION_TIME=10

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
# MODEL_NAME = "deepseek-r1-250120"
# MODEL_NAME = "deepseek-v3-241226"
# MODEL_NAME = "deepseek-v3-250324"
# BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
# API_KEY = os.getenv("ARK_API_KEY")

MODEL_NAME = "Qwen/Qwen3-8B"
BASE_URL = "http://localhost:8000/v1"
API_KEY = "token-abc123"

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


# ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://45.77.129.87:9200/")
# ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "notica-demo")
ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", None)
ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", None)


ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "https://my-elasticsearch-project-d340a1.es.us-east-1.aws.elastic.cloud:443")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "test-vector-store")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "NEVJSHlwY0IwZFVNaEE2UHQweFI6S3VXQjJVQmdlTFVWS2YzbzFUUXYwQQ==")

# ELASTICSEARCH_URL=""
# ELASTICSEARCH_API_KEY="NEVJSHlwY0IwZFVNaEE2UHQweFI6S3VXQjJVQmdlTFVWS2YzbzFUUXYwQQ=="
# ELASTICSEARCH_INDEX="test-vector-store"

TYPE_EMBED = "openai"  # "vllm" or "openai"
# EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_URL = "http://0.0.0.0:8000/v1/embeddings"

FILE_STORAGE_PATH = "/files"

os.environ['LANGCHAIN_TRACING_V2']='false'
os.environ['LANGSMITH_TRACING']='false'