import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader , DirectoryLoader

def read_file(path: str) -> pd.Series:
    """
    Augment:
    path: the path of data you need to provide

    return:
    this function will return the questions, answers in data
    """
    df = pd.read_csv(path, sep="\t", header= None , on_bad_lines='skip')
    df = df.rename(columns={0:"Question", 1:"Answers"})
    questions = df["Question"]
    answers = df["Answers"]
    return questions , answers

def read_pdf_file(path: str) -> None:

    loader = DirectoryLoader(path=path , glob="**/*.pdf", loader_cls=UnstructuredPDFLoader)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=100,
                                                separators=["\n"],
                                                length_function=len)
    
    docs = text_splitter.split_documents(pages)
    return docs

def save_vector_data(questions: pd.Series,  answers: pd.Series):
    """
    Augment:
    questions: the Series of question after you extract from the 
                csv file
    answers: the Series of answers after you extract from the 
                csv file
    embedding: the vector embedding question after going through model embedding
    """
    
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    embeddings = model.encode(questions)

    df_2 = pd.DataFrame({
    "questions": questions,
    "answers": answers,
    "vector_embedding": embeddings.tolist()
    })
    return df_2.to_csv("../data/csv_dataset/fqt_QA_data.csv",index = False)