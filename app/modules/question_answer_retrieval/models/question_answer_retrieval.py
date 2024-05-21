import langchain
import langchain_core
# from app.configs.qdrant_database import client
from app.utils.doc_vectorstore import doc_db
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Qdrant

def create_prompt(templates: None) -> langchain_core.prompts.prompt.PromptTemplate:
    qa_chain_prompt = PromptTemplate.from_template(templates)
    return qa_chain_prompt


def create_qa_chain(llm: any, prompt: any) -> langchain.chains.retrieval_qa.base.RetrievalQA:

    doc_store = doc_db("Pengi-Doc") 

    # memory = ConversationBufferMemory(
    #         memory_key='chat_history', return_messages=True)
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