import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from app.configs.qdrant_database import client

model = SentenceTransformer('keepitreal/vietnamese-sbert')


def read_file(path: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", header= None , on_bad_lines='skip')
    df = df.rename(columns={0:"Question", 1:"Answers"})
    questions = df["Question"].apply(lambda x: x.lower() if isinstance(x, str) else x)
    answers = df["Answers"]
    return questions , answers


def embedding(questions: pd.Series, answers: pd.Series) -> np.ndarray:
    Series_embeddings = questions + answers
    embeddings = model.encode(Series_embeddings)
    return embeddings

def load_FAQ(msg):
    query_vector = model.encode(msg).tolist()

    hits = client.search(
    collection_name = "Pengi-FAQ",
    query_vector = query_vector,
    limit = 1,
    )

    return hits


# if __name__ == "__main__":
#     questions, answers = read_file("../data/faq_dataset/fqt_QA.txt")
#     vector_embedding = embedding(questions, answers)

#     df = pd.DataFrame({'Questions': questions, 'Answers': answers})
#     payload = df.to_dict(orient='records')
#     index = list(range(1,139))

#     client.upsert(
#         collection_name = "Pengi-FAQ",
#         points = models.Batch(
#                 ids = index,
#                 vectors= vector_embedding,
#                 payloads = payload
#                 )
#     )