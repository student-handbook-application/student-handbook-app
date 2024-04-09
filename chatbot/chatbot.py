from qdrant_client import qdrant_client
import os
import torch
from chatbot.auguments import load_auguments
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from chatbot.model import *

# def load_llm():
    
#     hf_api ,_, model_id, _, _ = load_auguments()

#     model = Model(model_id, hf_api, 0.01)
#     llm = model.load_model()

#     return llm

def Chatbot(msg,llm):
    _,hf_embedding_model, _, url_database, api_key_database = load_auguments()
    template = """<|im_start|>system\nChỉ sử dụng thông tin sau đây để trả lời câu hỏi. Nếu câu hỏi không liên quan dến nội dung sau đây, hãy trả lời rằng bạn không biết câu trả lời, đừng cố sinh thêm thông tin để trả lời\n
{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assitant"""

    # #load database documents
    # embeddings = HuggingFaceEmbeddings(model_name = hf_embedding_model)
    # doc_store = Qdrant(
    #     client=client, collection_name="Pengi-Doc", 
    #     embeddings=embeddings,
    # )


    # #load database FAQ
    # model_embedings = SentenceTransformer(hf_embedding_model)
    # query_vector = model_embedings.encode(msg).tolist()

    # hits = client.search(
    # collection_name = "Pengi-FAQ",
    # query_vector = query_vector,
    # limit = 1,
    # )

    hits = load_FAQ(msg)

    #run result
    for hit in hits:
        if hit.score > 0.7:
            return f"{hit.payload['Answers']}"
        else: 
            prompt = create_prompt(template)
            qa_chain = create_qa_chain(llm, prompt)
            return qa_chain.invoke(msg)