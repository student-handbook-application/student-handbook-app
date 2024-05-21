import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from vector import create_doc_db
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from qdrant_db import client
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from  langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"),temperature=0.05)

if client.collection_exists("Pengi-Doc"):
    vectorstore = Qdrant(
        client=client, collection_name="Pengi-Doc", 
        embeddings=OpenAIEmbeddings(),
    )

_template = """Given the following conversation and a follow up question, NO rephrase the follow up question to be a 
standalone question without changing the content in given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
condense_question_prompt_template = PromptTemplate.from_template(_template)

qa_system_prompt= """Let's think step by step. Your name is PENGI,you are CREATED from Pengi-Chatbot Team but this team does not belong to any organization .You are AI assistant about the admissions consultant for the student and parents about UNIVERSITY in Vietnamese country . 
Use the following pieces of context to answer the questions HONESTLY, ACCURATELY and MOST NATURAL. 
Avoid answering questions that are not included in the INFORMATIONs you are provided.\n
context:\n 
{context}\n
Question:\n 
{question}\n
Helpful Answer:"""



qa_prompt = PromptTemplate(
    template=qa_system_prompt, input_variables=["context", "question"]
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory)
doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt,verbose=True)

qa_chain = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3,"score_threshold": 0.5}),
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    memory=memory,
    verbose=True,
    # return_source_documents=True
)
chat_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    result = qa_chain.invoke({'question': user_input, 'chat_history': chat_history})
    response = result.get('answer')
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    print("Memory: ", memory.load_memory_variables({}))
    print(f"Pengi: {response}")