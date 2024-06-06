from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from qdrant_db import client
from langchain_community.vectorstores import Qdrant
load_dotenv()
def create_agent(vectorstore):
    search = TavilySearchResults(max_results=3, search_type="web", search_kwargs={"lang": "vi"})
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.86,"k":3})
    
    # tool = []
    # if retriever is None:
    #     tool = [search]
    # else:
    retriever_tool = create_retriever_tool(
        retriever,
        "pengi_search",
        "Use this tool when searching for information about the UNIVERSITY in VIETNAMESE.",
    )
    #     tool = [retriever_tool]
        
    tools = [retriever_tool,search]

    # template ="""Let's think step by step. Your name is PENGI,you are CREATED from Pengi-Chatbot Team but this team does not belong to any organization .You are AI assistant about the admissions consultant for the student and parents about UNIVERSITY in Vietnamese country . 
    # Use the following pieces of context to answer the questions HONESTLY, ACCURATELY and AS NATURAL AS HUMAN BEING. 
    # Avoid answering QUESTIONS that are not included in the INFORMATIONs you are provided."""
    
    template ="""Let's think step by step. Your name is PENGI,you are CREATED from Pengi-Chatbot Team but this team does not belong to any organization .You are AI assistant about the admissions consultant for the student and parents about UNIVERSITY in Vietnamese country . 
    Use the following pieces of context to answer the questions HONESTLY, ACCURATELY and AS NATURAL AS HUMAN BEING. 
    If the question is not covered by the information you are given, do an internet search. And remember, the information you find must be ACCURATE, do not use UNVERIFIED information. After finding the right information, compare it with the INFORMATION THAT YOU ARE PROVIDED and GIVE THE MOST ACCURATE ANSWER.
    DO NOT USE YOUR KNOWLEDGE TO ANSWER THE QUESTION."""
    


    
    prompt = ChatPromptTemplate.from_messages([
        ("system",template ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

def process_chat(agent_executor, user_input, chat_history):
    response=agent_executor.invoke({
        "input":user_input,
        "chat_history":chat_history})
    return response['output']


if __name__ == '__main__':
    if client.collection_exists("Pengi-Doc"):
        vectorstore = Qdrant(
            client=client, collection_name="Pengi-Doc", 
            embeddings=OpenAIEmbeddings(),
        )
    agent_executor = create_agent(vectorstore)
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agent_executor, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Pengi:", response)
