import torch
import langchain
import langchain_core
from langchain_google_genai import ChatGoogleGenerativeAI
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
class Model:
    def __init__(self, model_id: str, token_id: str, temperatures: float) -> None:
        """
        Auguments
        model_id: the folder path of model that you choose in Hugging Face
        token_id: the token api of hugging face account
        temperatures: this auguments make model will creative when it is generating
        """

        self.model_id = model_id
        self.token_id = token_id
        self.temperature = temperatures

        

    def load_model(self):
        model = ChatGoogleGenerativeAI(google_api_key=self.token_id, 
                                   model=self.model_id,
                                   temperature=self.temperature,
                                   max_output_tokens=1024)
        return model
    

def create_prompt(templates: None) -> langchain_core.prompts.prompt.PromptTemplate:
    qa_chain_prompt = PromptTemplate.from_template(templates)
    return qa_chain_prompt


def create_qa_chain(llm: any, prompt: any) -> langchain.chains.retrieval_qa.base.RetrievalQA:
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


    memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  # Change 'stuff' to a valid chain type
        retriever=  doc_store.as_retriever(search_kwargs={"k": 3,"score_threshold": 0.3}),
        return_source_documents = False, #trả về src trả lời
        # memory = memory,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt
        }
    )
    return qa_chain

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

def save_conversation_history(conversation_logs):
    with open("data/conversation_logs.txt", "a",encoding="utf-8") as file:
        for log in conversation_logs:
            file.write(f"User: {log['user_message']}\n")
            file.write(f"AI: {log['ai_response']}\n")
        file.write("\n")