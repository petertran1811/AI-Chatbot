import requests
import logging

from elasticsearch import Elasticsearch
from langchain.embeddings.base import Embeddings
from langchain_elasticsearch import DenseVectorStrategy
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from constants import (
    TYPE_EMBED,
    ELASTICSEARCH_URL,
    ELASTICSEARCH_INDEX,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_URL,
    ELASTICSEARCH_USERNAME,
    ELASTICSEARCH_PASSWORD,
    ELASTICSEARCH_API_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLLMEmbedding(Embeddings):
    def __init__(self, endpoint_url=EMBEDDING_URL, model=EMBEDDING_MODEL_NAME):
        self.endpoint_url = endpoint_url
        self.model = model

    def embed_documents(self, texts):
        res = requests.post(self.endpoint_url, json={"input": texts, "model": self.model})
        embeddings = res.json()["data"]
        return [e["embedding"] for e in embeddings]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def create_vector_store(index_name: str = None) -> ElasticsearchStore:
    if index_name == None:
        index_name = ELASTICSEARCH_INDEX
        
    if ELASTICSEARCH_USERNAME and ELASTICSEARCH_PASSWORD:
        vector_store = ElasticsearchStore(
            index_name=ELASTICSEARCH_INDEX, 
            embedding=VLLMEmbedding() if TYPE_EMBED == "vllm" else OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME), 
            es_url=ELASTICSEARCH_URL,
            es_user=ELASTICSEARCH_USERNAME,
            es_password=ELASTICSEARCH_PASSWORD,
            strategy=DenseVectorStrategy()
        )
        logger.info("Using Elasticsearch with username and password authentication.")
    else:
        vector_store = ElasticsearchStore(
            index_name=ELASTICSEARCH_INDEX, 
            embedding=VLLMEmbedding() if TYPE_EMBED == "vllm" else OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME), 
            es_url=ELASTICSEARCH_URL,
            es_api_key=ELASTICSEARCH_API_KEY,
            strategy=DenseVectorStrategy()
        )
        logger.info("Using Elasticsearch with API key authentication.")

    list_indices = get_list_indices(vector_store)
    if index_name not in list_indices:
        logger.info(f"Creating index: {index_name}")
        docs = [
            Document(page_content="Hello world", metadata={"source": "create-index"}),
        ]
        ids = ["1"]
        vector_store.add_documents(documents=docs, ids=ids)
        vector_store.delete(ids=ids)
    return vector_store

def get_list_indices(vector_store: ElasticsearchStore) -> list:
    """
    Get a list of indices in the Elasticsearch store.
    """
    es_client: Elasticsearch = vector_store.client 
    indices_info = es_client.indices.get_alias(index="*")
    indices = list(indices_info.keys())
    return indices

VECTOR_STORE = None
