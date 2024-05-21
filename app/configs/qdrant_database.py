import os
from dotenv import load_dotenv
from qdrant_client import qdrant_client
from sentence_transformers import SentenceTransformer
load_dotenv()

model = SentenceTransformer('keepitreal/vietnamese-sbert')
client = qdrant_client.QdrantClient(
            url = os.getenv("QDRANT_HOST"),
            api_key= os.getenv("QDRANT_API_KEY")
        )
print("Qrant client is connected.")

# # print(type(client.get_collection("Pengi-FAQ")))
# if client.get_collection("Pengi-FAQ") == None:
#     client.recreate_collection(
#     collection_name = "Pengi-FAQ",
#     vectors_config = model.VectorParams(
#         size = 768,
#         distance = model.Distance.COSINE,
#     )
# )
#     print("Pengi-FAQ collection is created.")
# else:
#     print("Pengi-FAQ collection is exsit.")

# # #check the database Pengi-Doc if is not exsit
# if client.get_collection("Pengi-Doc") == None:

#     # docs = create_doc_db(pdf_path="../data/pdf_dataset/")
#     # qdrant = Qdrant.from_documents(
#     #     docs,
#     #     model,
#     #     url=os.getenv("QDRANT_HOST"),
#     #     api_key= os.getenv("QDRANT_API_KEY"))
#     print("Pengi-Doc collection is created.")
# else:
#     print("Pengi-Doc collection is exsit.")

# # #check the database Pengi-Feedback if is not exsit
# if client.get_collection("Pengi-Feedback") == None:
#     client.recreate_collection(
#     collection_name = "Pengi-Feedback",
#     vectors_config = model.VectorParams(
#         size = 768,
#         distance = model.Distance.COSINE,
#     )
# )
#     print("Pengi-Feedback collection is created.")
# else:
#     print("Pengi-Feedback collection is exsit.")