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



def load_model(model_path):
    config = {'context_length' : 2048}
    llm = CTransformers(model = model_path,model_type ="llama", max_new_token=1024,temperature = 0.01,config= config)
    return llm


def create_prompt(templete):
    prompt = PromptTemplate(template= templete,
                            input_variables=["context","question"])
    return prompt
    

def create_qa_chain(prompt, llm, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kawrgs={"k": 3},
                                    max_token_limit = 1024),
        return_source_documents = False,
        chain_type_kwargs = {'prompt': prompt,
                             "verbose": True}
    )

    return qa_chain


def load_doc(api_key, url_database, hf_embedding_model):
    client = qdrant_client.QdrantClient(
            url = url_database,
            api_key= api_key
        )
    
    embeddings = HuggingFaceEmbeddings(model_name = hf_embedding_model)
    doc_store = Qdrant(
        client=client, collection_name="Pengi-Doc", 
        embeddings=embeddings,
    )

    return doc_store


def main():
    model_path = "model/vinallama-7b-chat_q5_0.gguf"
    _,hf_embedding_model, _, url_database, api_key_database = load_auguments()
    db = load_doc(api_key_database,url_database,hf_embedding_model)
    llm = load_model(model_path)

    template = """<|im_start|>system\nBạn là một trợ lý AI hữu ích. Bạn chỉ sử dụng những thông tin mà bạn được cung cấp để trả lời các câu hỏi,
    hãy trả lời câu hỏi một cách ngắn gọn, trung thực và chính xác. Tránh trả lời các câu hỏi không có trong thông tin mà bạn được cung cấp.
    Nếu câu hỏi không liên quan đến nội dung mà bạn được cung cấp, bạn hãy trả lời rằng bạn không biết câu trả lời.Tuyệt đối không được bịa ra
    câu trả lời.\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    
    prompt = create_prompt(template)

    qa_chain = create_qa_chain(prompt,llm,db)

    question = "Hiệu trưởng trường đại học FPT là ai?"
    response =  qa_chain.invoke({"query":question})
    print(response)


if __name__ == "__main__":
    main()


