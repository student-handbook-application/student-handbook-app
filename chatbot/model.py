import langchain 
import torch
import os
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from qdrant_client import qdrant_client
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from chatbot.auguments import *
# class Model:
#     def __init__(self, model_path: str) -> None:
#         """
#         Auguments
#         model_id: the folder path of model that you choose in Hugging Face
#         token_id: the token api of hugging face account
#         temperatures: this auguments make model will creative when it is generating
#         """

#         self.model_path = model_path
#         self.model_type = "llama"
#         self.temperature = 0.01

        

#     def load_model(self):
#         config = {'context_length' : 2048}
#         llm  = CTransformers(model = self.model_path,
#                              model_type =self.model_type, 
#                              max_new_token=1024,
#                              temperature = self.temperature,
#                              config = config)
#         return llm
    
def load_model(model_path):
    config = {'context_length' : 2048}
    llm = CTransformers(model = model_path,model_type ="llama", max_new_token=1024,temperature = 0.01,config= config)
    return llm


def create_prompt(templates: None) :
    qa_chain_prompt = PromptTemplate.from_template(templates)
    return qa_chain_prompt


def create_qa_chain(llm: any, prompt: any) -> langchain.chains.retrieval_qa.base.RetrievalQA:
    hf_embedding_model, url_database, api_key_database= load_auguments()

    client = qdrant_client.QdrantClient(
            url = url_database,
            api_key= api_key_database
        )
    
    embeddings = HuggingFaceEmbeddings(model_name = hf_embedding_model)
    doc_store = Qdrant(
        client=client, collection_name="Pengi-Doc", 
        embeddings=embeddings,
    )


    # memory = ConversationBufferMemory(
    #         memory_key='chat_history', return_messages=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  # Change 'stuff' to a valid chain type
        retriever=  doc_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents = False, #trả về src trả lời
        # memory = memory,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt
        }
    )
    return qa_chain

def load_FAQ(msg):
    hf_embedding_model, url_database, api_key_database= load_auguments()
    model_embedings = SentenceTransformer(hf_embedding_model)
    query_vector = model_embedings.encode(msg).tolist()


    client = qdrant_client.QdrantClient(
            url = url_database,
            api_key= api_key_database
        )
    hits = client.search(
    collection_name = "Pengi-FAQ",
    query_vector = query_vector,
    limit = 1,
    )

    return hits