from qdrant_client import qdrant_client
import os
import torch
from chatbot.auguments import load_auguments
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from chatbot.chatbot import Model, create_prompt, create_qa_chain


def Chatbot(msg):
    torch.cuda.empty_cache()
    hf_api ,hf_embedding_model, model_id, url_database, api_key_database = load_auguments()

    template = """<|im_start|>system\nChỉ sử dụng thông tin sau đây để trả lời câu hỏi. Nếu câu hỏi không liên quan dến nội dung sau đây, hãy trả lời rằng bạn không biết câu trả lời, đừng cố sinh thêm thông tin để trả lời\n
{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assitant"""
            


    client = qdrant_client.QdrantClient(
    url = url_database,
    api_key= api_key_database
    )
    embeddings = HuggingFaceEmbeddings(model_name = hf_embedding_model)

    doc_store = Qdrant(
        client=client, collection_name="Pengi-Doc", 
        embeddings=embeddings,
    )

    model_embedings = SentenceTransformer(hf_embedding_model)
    query_vector = model_embedings.encode(msg).tolist()

    hits = client.search(
    collection_name = "Pengi-FAQ",
    query_vector = query_vector,
    limit = 1,
    )

    # print('da chay')
    for hit in hits:
        if hit.score > 0.49:
            print('da chay 1')
            print(hit.score)
            return f"{hit.payload['Answers']}"

        else: 
            print('da chay 2')
            model = Model(model_id, hf_api, 0.01)
            llm = model.load_model()
            prompt = create_prompt(template)
    
            qa_chain = create_qa_chain(llm, doc_store, prompt)
            return qa_chain.invoke(msg)
