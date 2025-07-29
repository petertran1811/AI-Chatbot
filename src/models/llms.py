import time
import logging

from transformers import AutoTokenizer
from pydantic import BaseModel, Field

from openai import OpenAI
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from constants import (
    # OLLAMA_MODEL_NAME,
    # OLLAMA_TEMPERATURE,
    TEST_CONNECTION_TIME,
    LLAMA3_TOKENIZER_ID,
    MODEL_NAME,
    BASE_URL,
    API_KEY,
    TYPE_MODEL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_tokenizers() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(LLAMA3_TOKENIZER_ID)

def count_tokens(text: str | list) -> int:
    """
    Count the number of tokens in a given text using the specified tokenizer.
    """
    if isinstance(text, list):
        text = ". ".join(text)  # Join list of strings into a single string
    text_tokens = TOKENIZER.tokenize(text)
    return len(text_tokens)

def get_chatopenai_llm() -> BaseChatModel:
    """Initializes and returns the OpenAI Chat LLM."""
    print("Init get_chatopenai_llm")
    print("MODEL_NAME: ", MODEL_NAME)

    chatopenai_llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=1,
        presence_penalty=1.2,
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    
    while True:
        time.sleep(TEST_CONNECTION_TIME)
        try:
            logger.info("Testing connection...")
            response = chatopenai_llm.invoke("ping")
            logger.info("Connection successful")
            return chatopenai_llm
        except Exception as e:
            logger.info("Failed to connect. Retrying in {} seconds...".format(TEST_CONNECTION_TIME))
            

def get_chatopenai_openai() -> BaseChatModel:
    """Initializes and returns the OpenAI Chat LLM."""
    chatopenai_llm = ChatOpenAI(
        model="gpt-4o-mini",
    )
    
    # chatopenai_llm = ChatOpenAI(
    #     model="Qwen/Qwen2.5-VL-7B-Instruct",
    #     temperature=1,
    #     api_key="token-abc123",
    #     base_url="https://547ca360c9f4.ngrok-free.app/v1"
    # )


    return chatopenai_llm

class FormatStructuredOutput(BaseModel):
    """
    Model for structured output from the LLM.
    """
    suggestions: list[str] = Field(default_factory=list, description="Three suggestions for user response.")

if TYPE_MODEL == "chatopenai":
    MODEL = get_chatopenai_llm()
    
logger.info("Starting model initialization")
MODEL = get_chatopenai_llm()
TOKENIZER = get_tokenizers()

model_json_schema = FormatStructuredOutput.model_json_schema()

MODEL_STRUCTURED_OUTPUT = MODEL.with_structured_output(FormatStructuredOutput)