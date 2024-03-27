import torch
import langchain
import langchain_core
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class Model:
    def __init__(self, model_id: str, token_id: str, temperatures: float) -> None:
        """
        Auguments
        model_id: the folder path of model that you choose in Hugging Face
        token_id: the token api of hugging face account
        temperatures: this auguments make model will creative when it is generating
        """

        self.model_id = model_id
        self.token_id = token_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token= self.token_id)
        self.temperature = temperatures

        

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id,
                                             device_map='cuda:0',
                                             torch_dtype=torch.bfloat16,
                                             use_auth_token=self.token_id,
                                            #  load_in_4bit=True,
                                             load_in_8bit=True,
                        
        )

        pipe = pipeline("text-generation",
                        model= model,
                        tokenizer= self.tokenizer,
                        torch_dtype= torch.bfloat16,
                        device_map= "auto",
                        max_new_tokens = 512,
                        do_sample = True,
                        top_k = 15,
                        num_return_sequences=1,
                        eos_token_id=self.tokenizer.eos_token_id)
        
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": self.temperature
                                                               })
        return llm
    

def create_prompt(templates: None) -> langchain_core.prompts.prompt.PromptTemplate:
    qa_chain_prompt = PromptTemplate.from_template(templates)
    return qa_chain_prompt


def create_qa_chain(llm: any, database_vector: any, prompt: any) -> langchain.chains.retrieval_qa.base.RetrievalQA:
    memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  # Change 'stuff' to a valid chain type
        retriever=  database_vector.as_retriever(search_kwargs={"k": 3}),
        return_source_documents = False, #trả về src trả lời
        # memory = memory,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt
        }
    )
    return qa_chain
