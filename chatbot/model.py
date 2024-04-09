import torch
import langchain
import langchain_core
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from chatbot.auguments import load_auguments
from langchain_community.vectorstores import Qdrant
from qdrant_client import qdrant_client
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.llms import CTransformers

def load_doc():
    _,hf_embedding_model, _, url_database, api_key_database = load_auguments()
    client = qdrant_client.QdrantClient(
            url = url_database,
            api_key= api_key_database
        )
    
    embeddings = HuggingFaceEmbeddings(model_name = hf_embedding_model)
    doc_store = Qdrant(
        client=client, collection_name="Pengi-Doc", 
        embeddings=embeddings,
    )

    return doc_store

def load_FAQ(msg):
    _,hf_embedding_model, _, url_database, api_key_database = load_auguments()
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

def load_model(model_path):
    config = {'context_length' : 2048}
    llm = CTransformers(model = model_path,model_type ="llama", max_new_token=1024,temperature = 0.01,config= config)
    return llm

def create_prompt(templete):
    prompt = PromptTemplate(template= templete,
                            input_variables=["context","question"])
    return prompt


def create_qa_chain(prompt, llm):
    db = load_doc()
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kawrgs={"k": 3},
                                    max_token_limit = 1024),
        return_source_documents = False,
        chain_type_kwargs = {'prompt': prompt,
                             "verbose": True
                    }
    )

    return qa_chain
