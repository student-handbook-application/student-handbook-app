import os
from qdrant_db import client
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

load_dotenv()

def create_doc_db(pdf_path:str):
    model = OpenAIEmbeddings()

    loader = DirectoryLoader(path=pdf_path , glob="**/*.pdf", loader_cls=UnstructuredPDFLoader)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400,
                                                   chunk_overlap=100,
                                                   separators=["\n\n","\n"],
                                                   length_function=len)
    docs = text_splitter.split_documents(pages)
    doc_store = Qdrant.from_documents(
        docs, model,
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="Pengi-Doc"
    )
    return doc_store