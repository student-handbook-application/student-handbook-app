import os
from dotenv import load_dotenv
from qdrant_client import qdrant_client
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Qdrant
from utils import create_doc_db
load_dotenv()

model = SentenceTransformer('keepitreal/vietnamese-sbert')
client = qdrant_client.QdrantClient(
            url = os.getenv("QDRANT_HOST"),
            api_key= os.getenv("QDRANT_API_KEY")
        )
print("Qrant client is connected.")

#check the database Pengi-FAQ if is not exsit
if qdrant_client.collection_exists("Pengi-FAQ") == False:
    client.recreate_collection(
    collection_name = "Pengi-FAQ",
    vectors_config = model.VectorParams(
        size = 768,
        distance = model.Distance.COSINE,
    )
)
    print("Pengi-FAQ collection is created.")
else:
    print("Pengi-FAQ collection is exsit.")


#check the database Pengi-Doc if is not exsit
if qdrant_client.collection_exists("Pengi-Doc") == False:
    client.recreate_collection(
    collection_name = "Pengi-Doc",
    vectors_config = model.VectorParams(
        size = 768,
        distance = model.Distance.COSINE,
    )
)
    print("Pengi-Doc collection is created.")
else:
    print("Pengi-Doc collection is exsit.")  