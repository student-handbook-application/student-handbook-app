import os
from dotenv import load_dotenv
from qdrant_client import qdrant_client
from qdrant_client import QdrantClient

client = QdrantClient(
url="https://caee73b7-2e65-46e5-aac0-2024a527fa90.us-east4-0.gcp.cloud.qdrant.io:6333",
api_key="0rq8304tMvk-BnYLpi37-eYaJ1k1qLMywHWiyYoy7Uf0fD4FY4J1YA",
)