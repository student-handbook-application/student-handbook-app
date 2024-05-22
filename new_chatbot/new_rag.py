import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os, tempfile, glob, random
from pathlib import Path
from IPython.display import Markdown
from PIL import Image
from getpass import getpass
import numpy as np
from itertools import combinations
from dotenv import load_dotenv

# LLM: openai and google_genai
import openai
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# LLM: HuggingFace
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# langchain prompts, memory, chains...
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.schema import Document, format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string

# Document loaders
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredPDFLoader
)

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Chroma: vectorstore
from qdrant_db import client
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Qdrant


# Contextual Compression
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter,LongContextReorder
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.memory import ConversationSummaryBufferMemory

# Cohere
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere
import cohere
load_dotenv()

TMP_DIR = Path("./data").resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = Path("./data").resolve().parent.joinpath("data", "vector_stores")


def langchain_document_loader(TMP_DIR):
    """
    Load files from TMP_DIR (temporary directory) as documents. Files can be in txt, pdf, CSV or docx format.
    https://python.langchain.com/docs/modules/data_connection/document_loaders/file_directory
    """

    documents = []

    txt_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        TMP_DIR, glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
        loader_kwargs={"encoding":"utf8"}
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        TMP_DIR,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents

def select_embeddings_model(LLM_service="OpenAI"):
    """Connect to the embeddings API endpoint by specifying the name of the embedding model."""
    if LLM_service == "OpenAI":
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
            )
    
    if LLM_service == "Google":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    return embeddings

def create_vectorstore(embeddings, documents, vectorstore_name):
    vector_store = Qdrant.from_documents(
        documents, embeddings,
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=vectorstore_name
    )

    return vector_store, vectorstore_name


def print_documents(docs,search_with_score=False):
    """helper function to print documents."""
    if search_with_score:
        # used for similarity_search_with_score
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + doc[0].page_content +"\n\nscore:"+str(round(doc[-1],3))+"\n" 
                 for i, doc in enumerate(docs)]
            )
        )
    else:
        # used for similarity_search or max_marginal_relevance_search
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + doc.page_content 
                 for i, doc in enumerate(docs)]
            )
        )


def Vectorstore_backed_retriever(vectorstore,search_type="similarity",k=4,score_threshold=None):
    """create a vectorsore-backed retriever
    Parameters: 
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4) 
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs={}
    if k is not None:
        search_kwargs['k'] = k
    if score_threshold is not None:
        search_kwargs['score_threshold'] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    return retriever

def create_compression_retriever(embeddings, base_retriever, chunk_size=500, k=16, similarity_threshold=None):
    """Build a ContextualCompressionRetriever.
    We wrap the the base_retriever (a vectorstore-backed retriever) into a ContextualCompressionRetriever.
    The compressor here is a Document Compressor Pipeline, which splits documents
    into smaller chunks, removes redundant documents, filters out the most relevant documents,
    and reorder the documents so that the most relevant are at the top and bottom of the list.
    
    Parameters:
        embeddings: OpenAIEmbeddings, GoogleGenerativeAIEmbeddings or HuggingFaceInferenceAPIEmbeddings.
        base_retriever: a vectorstore-backed retriever.
        chunk_size (int): Documents will be splitted into smaller chunks using a CharacterTextSplitter with a default chunk_size of 500. 
        k (int): top k relevant chunks to the query are filtered using the EmbeddingsFilter. default =16.
        similarity_threshold : minimum relevance threshold used by the EmbeddingsFilter.. default =None.
    """
    
    # 1. splitting documents into smaller chunks
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")
    
    # 2. removing redundant documents
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

    # 3. filtering based on relevance to the query    
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold) # similarity_threshold and top K

    # 4. Reorder the documents 
    
    # Less relevant document will be at the middle of the list and more relevant elements at the beginning or end of the list.
    # Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/long_context_reorder
    reordering = LongContextReorder()

    # 5. Create compressor pipeline and retriever
    
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]  
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, 
        base_retriever=base_retriever
    )

    return compression_retriever

def CohereRerank_retriever(
    base_retriever, 
    cohere_api_key, 
    cohere_model="rerank-multilingual-v2.0", 
    top_n=8
):
    """Build a ContextualCompressionRetriever using Cohere Rerank endpoint to reorder the results based on relevance.
    Parameters:
       base_retriever: a Vectorstore-backed retriever
       cohere_api_key: the Cohere API key
       cohere_model: The Cohere model can be either 'rerank-english-v2.0' or 'rerank-multilingual-v2.0', with the latter being the default.
       top_n: top n results returned by Cohere rerank, default = 8.
    """
    
    # Initialize the CohereRerank compressor
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, 
        model=cohere_model, 
        top_n=top_n
    )

    # Initialize the ContextualCompressionRetriever with the compressor and base retriever
    retriever_Cohere = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return retriever_Cohere


def retrieval_blocks(
    create_vectorstore=True,# if True a Chroma vectorstore is created, else the Chroma vectorstore will be loaded
    LLM_service="Google",
    vectorstore_name="Pengi",
    chunk_size = 1600, chunk_overlap=200, # parameters of the RecursiveCharacterTextSplitter
    retriever_type="Vectorstore_backed_retriever",
    base_retriever_search_type="similarity", base_retriever_k=10, base_retriever_score_threshold=None,
    compression_retriever_k=16,
    cohere_api_key="***", cohere_model="rerank-multilingual-v2.0", cohere_top_n=8,
):
    """
    Rertieval includes: document loaders, text splitter, vectorstore and retriever. 
    
    Parameters: 
        create_vectorstore (boolean): If True, a new Chroma vectorstore will be created. Otherwise, an existing vectorstore will be loaded.
        LLM_service: OpenAI, Google or HuggingFace.
        vectorstore_name (str): the name of the vectorstore.
        chunk_size and chunk_overlap: parameters of the RecursiveCharacterTextSplitter, default = (1600,200).
        
        retriever_type (str): in [Vectorstore_backed_retriever,Contextual_compression,Cohere_reranker]
        
        base_retriever_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retriever_k: The most similar vectors to retrieve (default k = 10).  
        base_retriever_score_threshold: score_threshold used by the base retriever, default = None.

        compression_retriever_k: top k documents returned by the compression retriever, default=16
        
        cohere_api_key: Cohere API key
        cohere_model (str): The Cohere model can be either 'rerank-english-v2.0' or 'rerank-multilingual-v2.0', with the latter being the default.
        cohere_top_n: top n results returned by Cohere rerank, default = 8.
   
    Output:
        retriever.
    """
    try:
        # Create new Vectorstore (Chroma index)
        if create_vectorstore: 
            # 1. load documents
            documents = langchain_document_loader(TMP_DIR)
            
            # 2. Text Splitter: split documents to chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators = ["\n\n", "\n", " ", ""],    
                chunk_size = chunk_size,
                chunk_overlap= chunk_overlap
            )
            chunks = text_splitter.split_documents(documents=documents)
            
            # 3. Embeddings
            embeddings = select_embeddings_model(LLM_service=LLM_service)
        
            # 4. Vectorsore: create Chroma index
            vector_store = create_vectorstore(
                embeddings=embeddings,
                documents = chunks,
                vectorstore_name=vectorstore_name,
            )
    
        # 5. Load a Vectorstore (Chroma index)
        else: 
            embeddings = select_embeddings_model(LLM_service=LLM_service)        
            vector_store = Qdrant(
                client=client, collection_name="Pengi-documents-google-embeddings", 
                embeddings=embeddings_OpenAI,
        )
            
        # 6. base retriever: Vector store-backed retriever 
        base_retriever = Vectorstore_backed_retriever(
            vector_store,
            search_type=base_retriever_search_type,
            k=base_retriever_k,
            score_threshold=base_retriever_score_threshold
        )
        retriever = None
        if retriever_type=="Vectorstore_backed_retriever": 
            retriever = base_retriever
    
        # 7. Contextual Compression Retriever
        if retriever_type=="Contextual_compression":    
            retriever = create_compression_retriever(
                embeddings=embeddings,
                base_retriever=base_retriever,
                k=compression_retriever_k,
            )
    
        # 8. CohereRerank retriever
        if retriever_type=="Cohere_reranker":
            retriever = CohereRerank_retriever(
                base_retriever=base_retriever, 
                cohere_api_key=cohere_api_key, 
                cohere_model=cohere_model, 
                top_n=cohere_top_n
            )
    
        print(f"\n{retriever_type} is created successfully!")
        print(f"Relevant documents will be retrieved from vectorstore ({vectorstore_name}) which uses {LLM_service} embeddings")
        
        return retriever
    except Exception as e:
        print(e)

def instantiate_LLM(LLM_provider,api_key,temperature=0.5,model_name='gpt-3.5-turbo'):
    """Instantiate LLM in Langchain.
    Parameters:
        LLM_provider (str): the LLM provider; in ["OpenAI","Google","HuggingFace"]
        model_name (str): in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-turbo-preview", 
            "gemini-pro", "mistralai/Mistral-7B-Instruct-v0.2"].            
        api_key (str): google_api_key or openai_api_key or huggingfacehub_api_token 
        temperature (float): Range: 0.0 - 1.0; default = 0.5
        top_p (float): : Range: 0.0 - 1.0; default = 1.
    """
    if LLM_provider == "OpenAI":
        llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            # model_kwargs={
            #     "top_p": top_p
            # }
        )
    return llm



def create_memory(model_name='gpt-3.5-turbo',memory_max_token=None):
    """Creates a ConversationSummaryBufferMemory for gpt-3.5-turbo
    Creates a ConversationBufferMemory for the other models."""
    
    if model_name=="gpt-3.5-turbo":
        if memory_max_token is None:
            memory_max_token = 1024 # max_tokens for 'gpt-3.5-turbo' = 4096
        memory = ConversationSummaryBufferMemory(
            max_token_limit=memory_max_token,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0.1),
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question"
        )
    else:
        memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question",
        )  
    return memory

def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context 
    to the `LLM` wihch will answer"""
    
    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template

def create_ConversationalRetrievalChain(
    llm,condense_question_llm,
    retriever,
    chain_type= 'stuff',
    language="english",
    model_name='gpt-3.5-turbo'
):
    """Create a ConversationalRetrievalChain.
    First, it passes the follow-up question along with the chat history to an LLM which rephrases 
    the question and generates a standalone query. 
    This query is then sent to the retriever, which fetches relevant documents (context) 
    and passes them along with the standalone question and chat history to an LLM to answer.
    """
    
    # 1. Define the standalone_question prompt. 
    # Pass the follow-up question along with the chat history to the `condense_question_llm`
    # which rephrases the question and generates a standalone question.

    standalone_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'], 
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""")

    # 2. Define the answer_prompt
    # Pass the standalone question + the chat history + the context (retrieved documents) to the `LLM` wihch will answer
    
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    
    memory = create_memory(model_name)

    # 4. Create the ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=standalone_question_prompt,
        combine_docs_chain_kwargs={'prompt': answer_prompt},
        condense_question_llm=condense_question_llm,

        memory=memory,
        retriever = retriever,
        llm=llm,

        chain_type= chain_type,
        verbose= False,
        return_source_documents=True    
    )

    print("Conversational retriever chain created successfully!")
    
    return chain,memory

def _combine_documents(docs, document_prompt, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context (retrieved documents) to the `LLM` to get an answer."""
    
    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template
    
def _combine_documents(docs, document_prompt, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def custom_ConversationalRetrievalChain(
    llm,condense_question_llm,
    retriever,
    language="english",
    llm_provider="OpenAI",
    model_name='gpt-3.5-turbo',
):
    """Create a ConversationalRetrievalChain step by step.
    """
    ##############################################################
    # Step 1: Create a standalone_question chain
    ##############################################################
    
    # 1. Create memory: ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    
    memory = create_memory(model_name)
    # memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", input_key="question",return_messages=True)

    # 2. load memory using RunnableLambda. Retrieves the chat_history attribute using itemgetter.
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )

    # 3. Pass the follow-up question along with the chat history to the LLM, and parse the answer (standalone_question).

    condense_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'], 
        template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in the same language as the follow up question.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:"""        
)
        
    standalone_question_chain = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | condense_question_prompt
        | condense_question_llm
        | StrOutputParser(),
    }
    
    # 4. Combine load_memory and standalone_question_chain
    chain_question = loaded_memory | standalone_question_chain
    
    ####################################################################################
    #   Step 2: Retrieve documents, pass them to the LLM, and return the response.
    ####################################################################################

    # 5. Retrieve relevant documents
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    
    # 6. Get variables ['chat_history', 'context', 'question'] that will be passed to `answer_prompt`
    
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language)) 
    # 3 variables are expected ['chat_history', 'context', 'question'] by the ChatPromptTemplate   
    answer_prompt_variables = {
        "context": lambda x: _combine_documents(docs=x["docs"],document_prompt=DEFAULT_DOCUMENT_PROMPT),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history") # get it from `loaded_memory` variable
    }
    
    # 7. Load memory, format `answer_prompt` with variables (context, question and chat_history) and pass the `answer_prompt to LLM.
    # return answer, docs and standalone_question
    
    chain_answer = {
        "answer": loaded_memory | answer_prompt_variables | answer_prompt | llm,
        # return only page_content and metadata 
        "docs": lambda x: [Document(page_content=doc.page_content,metadata=doc.metadata) for doc in x["docs"]],
        "standalone_question": lambda x:x["question"] # return standalone_question
    }

    # 8. Final chain
    conversational_retriever_chain = chain_question | retrieved_documents | chain_answer

    print("Conversational retriever chain created successfully!")

    return conversational_retriever_chain,memory


if __name__ == "__main__":
    documents = langchain_document_loader(r"data/")
    text_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " ", ""],    
    chunk_size = 1600,
    chunk_overlap= 200
    )

    # Text splitting
    chunks = text_splitter.split_documents(documents=documents)
    print(f"number of chunks: {len(chunks)}")

    #embedding
    embeddings_OpenAI = select_embeddings_model(LLM_service="Google")

    #vector database
    vector_store_OpenAI,_ = create_vectorstore(
        embeddings=embeddings_OpenAI,
        documents = chunks,
        vectorstore_name="Pengi-documents-google-embeddings",
    )

    #load Chroma vector database
    vector_store_OpenAI = Qdrant(
        client=client, collection_name="Pengi-documents-google-embeddings", 
        embeddings=embeddings_OpenAI,
    )
    # print("vector_store_OpenAI:",vector_store_OpenAI.get_collection.count(),"chunks.")

    #test similarity_search
    query = 'What is Diffuse to choose?'
    
    docs_withScores = vector_store_OpenAI.similarity_search_with_score(query,k=4)
    print_documents(docs_withScores,search_with_score=True)

    #calculate dot product between query vector and documents vector
    query_embeddings = embeddings_OpenAI.embed_query(query)
    docs_embeddings = embeddings_OpenAI.embed_documents(
        [docs_withScores[i][0].page_content 
        for i in range(len(docs_withScores))
        ]
    )

    for i in range(len(docs_embeddings)):
        dot_product = round(np.dot(query_embeddings, docs_embeddings[i]),4)
        print(f"Similarty of document_{i} to the query: {dot_product}")
    

    # similarity search and similarity_score_threshold
    base_retriever_OpenAI = Vectorstore_backed_retriever(vector_store_OpenAI,"similarity",k=10)
    # base_retriever_OpenAI = Vectorstore_backed_retriever(vector_store_OpenAI,"similarity_score_threshold",score_threshold=0.5,k=10)

    #get relevant documents
    relevant_docs = base_retriever_OpenAI.get_relevant_documents(query)
    print_documents(relevant_docs)


    #CONTEXTUAL COMPRESSION
    compression_retriever_OpenAI = create_compression_retriever(
        embeddings=embeddings_OpenAI,
        base_retriever=base_retriever_OpenAI,
        k=16
    )
    compressed_docs = compression_retriever_OpenAI.get_relevant_documents(query)
    print_documents(compressed_docs)

    #COHERE RERANK
    query = 'what is Diffuse to Choose?'
    retriever_Cohere_OpenAI = CohereRerank_retriever(
        base_retriever=base_retriever_OpenAI, 
        cohere_api_key=os.getenv("COHERE_API_KEY"),  
        top_n=8
    )
    docs_cohere = retriever_Cohere_OpenAI.get_relevant_documents(query)
    print("Reranking using Cohere. Vcetorstore using OpenAI Embeddings:\n")
    print_documents(docs_cohere)

    #TEST MEMORY 
    memory = create_memory(model_name='gpt-3.5-turbo',memory_max_token=20)

    # save context
    print("Show how older messages are summarized")
    memory.save_context(
        inputs={"question":"what does DTC stand for?"},
        outputs={"answer":"""Diffuse to Choose (DTC) is a novel diffusion inpainting approach designed for the Vit-All application, 
        which allows users to virtually place any e-commerce item in any setting, ensuring detailed, semantically coherent blending with realistic 
        lighting and shadows. It effectively incorporates fine-grained cues from the reference image into the main U-Net decoder 
        using a secondary U-Net encoder.
        DTC can handle a variety of e-commerce products and can generate images using in-the-wild images & references. 
        It is superior to existing zero-shot personalization methods, especially in preserving the fine-grained details of items."""}
    )
    memory.save_context(
        inputs={"question":"what does Vit-all stand for?"},
        outputs={"answer":"Virtual Try-All"}
    )

    memory.load_memory_variables({})

    #PROMPT TEMPLATE
    # Prompt templates generate prompts for the LLM. For our RAG chatbot, we need two templates:
    # The first template asks the LLM to generate a standalone question given the chat history and a follow-up question. The PromptTemplate uses this template to return a string prompt when invoked.
    # The second template includes a placeholder for the context(i.e. retrieved documents), chat history, and the user's question. It instructs the LLM to answer the question based solely on the provided context. The ChatPromptTemplate uses this template to return a list of chat messages.

    standalone_question_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:"""

    standalone_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'], 
        template=standalone_question_template
    )
    memory = create_memory(model_name='gpt-3.5-turbo')
    memory.save_context(
        inputs={"question":"what does DTC stand for?"}, 
        outputs={"answer":"DTC stands for Diffuse to choose."}
    )

    standalone_question_prompt.invoke(
        {"question":"plaese give more details about it, including its use cases and implementation.",
        "chat_history":memory.chat_memory})
    

    #TEST CHAT PROMPT TEMPLATE
    answer_prompt = ChatPromptTemplate.from_template(answer_template())

    # invoke the ChatPromptTemplate
    answer_prompt.invoke(
        {"question":"plaese give more details about DTC, including its use cases and implementation.",
        "context":[Document(page_content="DTC use cases include...")], # the context is a list of retrieved documents.
        "chat_history":memory.chat_memory}
    )

    #example of ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
    condense_question_prompt=standalone_question_prompt,
    combine_docs_chain_kwargs={'prompt': answer_prompt},
    condense_question_llm=instantiate_LLM(
        LLM_provider="OpenAI",api_key=os.getenv('OPENAI_API_KEY'),temperature=0.1,
        model_name="gpt-3.5-turbo"
    ),

    memory=create_memory("gpt-3.5-turbo"),
    retriever = base_retriever_OpenAI, 
    llm=instantiate_LLM(
        LLM_provider="OpenAI",api_key=os.getenv('OPENAI_API_KEY'),temperature=0.5,
        model_name="gpt-3.5-turbo"),
        chain_type= "stuff",
        verbose= True,
        return_source_documents=True   
    )

    response = chain.invoke({"question":query})
    print(response)
    print(chain.memory.load_memory_variables({})) # get chat history of the chain's memory
    
    follow_up_question = "plaese give more details about it, including its use cases and implementation."
    response = chain.invoke({"question":follow_up_question})['answer']

    #STEP BY STEP APPROACH TO CONVERSATIONAL RETRIEVAL CHAIN

    #We will use a simple `ConversationBufferMemory' here for simplicity.
    memory = ConversationBufferMemory(memory_key="chat_history",output_key="answer", input_key="question",return_messages=True)
    print(memory.load_memory_variables({}))

    #STEP 1.tải bộ nhớ bằng RunnableLambda. Truy xuất thuộc tính chat_history bằng itemgetter.
    # `RunnablePassthrough.sign` thêm chat_history vào hàm asign
    loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("chat_history"),
    )
    print(loaded_memory.invoke({}))
    print(loaded_memory.invoke({"x":1}))

    #1.2:Chuyển câu hỏi tiếp theo cùng với lịch sử trò chuyện tới LLM và phân tích câu trả lời (standalone_question).
    condense_question_prompt = PromptTemplate(
        input_variables=['chat_history', 'question'], 
        template="""Given the following conversation and a follow up question (at the end), 
    rephrase the follow up question to be a standalone question, in the same language as the follow up question.\n\n
    Chat History:\n{chat_history}\n
    Follow up question: {question}\n
    Standalone question:"""
    
    )
    condense_question_llm = instantiate_LLM(
        LLM_provider="OpenAI",api_key=os.getenv('OPENAI_API_KEY'),temperature=0.1
        )
    
    standalone_question_chain = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | condense_question_prompt
        | condense_question_llm
        | StrOutputParser(),
    }
    #1.3. Combine load_memory and standalone_question_chain
    chain_question = loaded_memory | standalone_question_chain


    #Example of how to invoke the chain_question
    memory.clear()
    memory.save_context(
        {"question": "What does DTC stand for?"},
        {"answer": "Diffuse to Choose."}
    )

    print("Chat history:\n",memory.load_memory_variables({}))
    print("\nFollow-up question:\n","plaese give more details about it, including its use cases and implementation.")

    # invoke chain_question
    response = chain_question.invoke({"question":"plaese give more details about it, including its use cases and implementation."})['standalone_question']
    print("\nStandalone_question:\n",response)

    #STEP 2: TRUY XUẤT DOCUMENTS, BỎ CHÚNG QUA LLM cùng với câu hỏi độc lập và lịch sử trò chuyện, đồng thời phân tích câu trả lời.
    retriever_OpenAI = retrieval_blocks(
        create_vectorstore=False,
        LLM_service="OpenAI",
        vectorstore_name="Pengi",
        retriever_type="Cohere_reranker",
        base_retriever_search_type="similarity", base_retriever_k=12,
        compression_retriever_k=16,
        cohere_api_key=os.getenv('COHERE_API_KEY'),cohere_top_n=10,
    )

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever_OpenAI,
        "question": lambda x: x["standalone_question"],
    }
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
    answer_prompt = ChatPromptTemplate.from_template(answer_template()) # 3 variables are expected ['chat_history', 'context', 'question']

    answer_prompt_variables = {
        "context": lambda x: _combine_documents(docs=x["docs"],document_prompt=DEFAULT_DOCUMENT_PROMPT),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history") # get chat_history from `loaded_memory` variable
    }
    llm = instantiate_LLM(
        LLM_provider="OpenAI",api_key=os.getenv('OPENAI_API_KEY'),temperature=0.5,
        model_name="gpt-3.5-turbo"
    )
    chain_answer = {
        "answer": loaded_memory | answer_prompt_variables | answer_prompt | llm,
        "docs": lambda x: [
            Document(page_content=doc.page_content,metadata=doc.metadata) # return only page_content and metadata (_DocumentWithState returns them + state.)
            for doc in x["docs"]
        ], 
        "standalone_question": lambda x:x["question"] # return standalone_question
    }
    conversational_retriever_chain = chain_question | retrieved_documents | chain_answer
    print(memory.load_memory_variables({}))
    follow_up_question = "plaese give more details about it, including its use cases and implementation."

    response = conversational_retriever_chain.invoke({"question":follow_up_question})
    Markdown(response['answer'].content)

    print(memory.load_memory_variables({}))
    memory.save_context( {"question": follow_up_question}, {"answer": response['answer'].content} ) # update memory

    print(memory.load_memory_variables({}))

    retriever_OpenAI = retrieval_blocks(
    create_vectorstore=False,
    LLM_service="Google",
    vectorstore_name="Vit_All_OpenAI_Embeddings",
    retriever_type="Cohere_reranker",
    base_retriever_search_type="similarity", base_retriever_k=10,
    compression_retriever_k=16,
    cohere_api_key=os.getenv('COHERE_API_KEY'),cohere_top_n=8,
)

    chain_openAI,memory_openAI = custom_ConversationalRetrievalChain(
        llm = instantiate_LLM(
            LLM_provider="OpenAI",model_name="gpt-3.5-turbo",api_key=os.getenv('OPENAI_API_KEY'),temperature=0.5
        ),
        condense_question_llm = instantiate_LLM(
            LLM_provider="OpenAI",model_name="gpt-3.5-turbo",api_key=os.getenv('OPENAI_API_KEY'),temperature=0.1
        ) ,
        retriever=retriever_OpenAI,
        language="english",
        llm_provider="OpenAI",
        model_name="gpt-3.5-turbo-0125"
    )
    memory_openAI.clear()

    questions = ["what does DTC stands for?",
             "plaese give more details about it, including its use cases and implementation.",
             "does it outperform other diffusion-based models? explain in details.",
             "what is Langchain?"]
    responses = []
    for i,question in enumerate(questions):
        response = chain_openAI.invoke({"question":question})
        responses.append(response)
        
        answer = response['answer'].content
        print(f"Question[{i}]:",question)
        print("Standalone_question:",response['standalone_question'])
        print("Answer:\n",answer,f"\n\n{'-' * 100}\n")
        
        memory_openAI.save_context( {"question": question}, {"answer": answer} )
