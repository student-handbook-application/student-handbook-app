import os
from app.configs.qdrant_database import client
from langchain_community.vectorstores import Qdrant
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

#load model embeddings
model_embeddings = SentenceTransformer('keepitreal/vietnamese-sbert')

# #load the pdf file
# loader = DirectoryLoader(path="../data/pdf_dataset" , glob="**/*.pdf", loader_cls=UnstructuredPDFLoader)
# pages = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                                chunk_overlap=100,
#                                                separators=["\n"],
#                                                length_function=len)
# docs = text_splitter.split_documents(pages)

# doc_store = Qdrant.from_documents(
#     docs, model_embeddings,
#     url=os.getenv("QDRANT_HOST"),
#     api_key=os.getenv("QDRANT_API_KEY"),
#     collection_name="Pengi-Doc"
# )

#call a docs database
def doc_db(collections_name:str):
    doc_db = Qdrant(
        client=client, collection_name=collections_name,
        embeddings=model_embeddings,
    )
    return doc_db