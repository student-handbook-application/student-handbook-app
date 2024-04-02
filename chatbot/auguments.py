import os
from dotenv import load_dotenv

def load_auguments():
    load_dotenv(".env")
    hf_embedding_model = os.environ.get("hf_embedding_model")
    url_database = os.environ.get("QDRANT_HOST")
    api_key_database = os.environ.get("QDRANT_API_KEY")
    return hf_embedding_model, url_database, api_key_database