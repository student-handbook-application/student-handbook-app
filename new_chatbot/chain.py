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
from langchain.schema import Document
load_dotenv()

def create_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"),temperature=0.05)



    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm,vectorstore.as_retriever(search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.3,"k":3}), contextualize_q_prompt
    )


    ### Answer question ###
    qa_system_prompt = """Let's think step by step. Your name is PENGI,you are CREATED from Pengi-Chatbot Team but this team does not belong to any organization .You are AI assistant about the admissions consultant for the student and parents about UNIVERSITY in Vietnamese country . 
    Use the following pieces of context to answer the questions HONESTLY, ACCURATELY and MOST NATURAL. 
    Avoid answering questions that are not included in the INFORMATIONs you are provided.\n

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### Statefully manage chat history ###
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def process_chain(chain,user_input,chat_history):
    response = chain.invoke({'input': user_input, 'chat_history': chat_history},config={"configurable": {"session_id": "abc123"}})
    return response


if __name__ == "__main__":
    if client.collection_exists("Pengi-Doc"):
        vectorstore = Qdrant(
            client=client, collection_name="Pengi-Doc", 
            embeddings=OpenAIEmbeddings(),
        )
    chain = create_chain(vectorstore)
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        result =  process_chain(chain,user_input,chat_history)
        response = result.get('answer')
        documents = result.get("context")
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print(f"Pengi: {response}")
        print(documents)