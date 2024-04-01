import os
import csv
import time
import torch

from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from qdrant_client import qdrant_client
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def API_key() -> None:
    os.environ["QDRANT_HOST"] = "https://ef3d190d-4471-4044-b8a5-6aba8c656aa8.us-east4-0.gcp.cloud.qdrant.io:6333"
    os.environ["QDRANT_API_KEY"] = "EPOl1GIG9WTr2joExRzwjRkk5LuLWNvImfgU4y2GPaN_Nu5kWcs51w"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"



def generate_prompte_template():
    # template="""
    # <|im_start|>system\nBạn là một trợ lý AI hữu ích. Bạn chỉ sử dụng những thông tin mà bạn được cung cấp để trả lời các câu hỏi,
    # hãy trả lời câu hỏi một cách ngắn gọn, trung thực và chính xác. Tránh trả lời các câu hỏi không có trong thông tin mà bạn được cung cấp.
    # Nếu câu hỏi không liên quan đến nội dung mà bạn được cung cấp, bạn hãy trả lời rằng bạn không biết câu trả lời. Không được sử dụng những kiến thức
    # mà bạn đã học để tạo ra câu trả lời\n{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assitant"""


    # template = """
    # <|im_start|>system\nBạn là một trợ lý AI hữu ích. Bạn chỉ sử dụng những thông tin mà bạn được cung cấp để trả lời các câu hỏi,
    # hãy trả lời câu hỏi một cách ngắn gọn, trung thực và chính xác. Tránh trả lời các câu hỏi không có trong thông tin mà bạn được cung cấp.
    # Nếu câu hỏi không liên quan đến nội dung mà bạn được cung cấp, bạn hãy trả lời rằng "Tôi không biết câu trả lời".Tuyệt đối không được sử dụng những kiến thức
    # mà bạn đã học để tạo ra câu trả lời\n{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assitant"""


    # template = """
    # <|im_start|>system\nBạn là một bậc thầy trong công việc làm trợ lý ảo. Bạn chỉ được sử dụng những thông tin mà bạn được cung cấp để trả lời các câu hỏi,
    # hãy trả lời câu hỏi một cách ngắn gọn, trung thực và chính xác. Tránh trả lời các câu hỏi không có trong thông tin mà bạn được cung cấp.
    # Nếu câu hỏi không liên quan đến nội dung mà bạn được cung cấp, bạn hãy trả lời rằng "Tôi không biết câu trả lời". Bỏ qua những tìm kiếm ngoại lệ không liên quan đến câu hỏi.
    # Không được sử dụng những kiến thức mà bạn đã học để bịa ra câu trả lời\n{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assitant"""
    
    return template


def chatbot() -> dict:
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    embeddings = HuggingFaceEmbeddings(model_name='keepitreal/vietnamese-sbert')
    doc_store = Qdrant(
        client=client, collection_name="Pengi-Doc",
        embeddings=embeddings,
    )

    tokenizer = AutoTokenizer.from_pretrained("vilm/vinallama-7b-chat", token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained("vilm/vinallama-7b-chat",
                                                 device_map='cuda:0',
                                                 torch_dtype=torch.float16,
                                                 load_in_8bit=True,
                                                 use_auth_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=512,
                    do_sample=True,
                    top_k=15,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id)

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temperature": 0.05})

#     template = """<|im_start|>system\nChỉ sử dụng thông tin sau đây để trả lời câu hỏi. Nếu câu hỏi không liên quan dến nội dung sau đây, hãy trả lời rằng bạn không biết câu trả lời, đừng cố sinh thêm thông tin để trả lời\n
# {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assitant"""
    template = generate_prompte_template()
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',  
        retriever=doc_store.as_retriever(search_kwargs={"k": 3}),
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": QA_CHAIN_PROMPT
        }
    )

    return qa_chain

def run_and_save(qa_chain: dict, input_file_path: str, save_csv: str, wait_time: float = 5) -> None:
    """
    yêu cầu bộ data phải chuẩn format, cuối mỗi câu hỏi phải có dấu chấm hỏi
    không khoảng trắng so với kí tự cuối, mỗi câu không cần có enter xuống dòng
    """
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name()
    
    with open(save_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'result', 'inference_time', 'device']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        with open(input_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    query = parts[0]  
                    start_time = time.time()
                    result = qa_chain.invoke(query)
                    inference_time = time.time() - start_time

                    result_text = result['result'] if 'result' in result else ''


                    writer.writerow({'query': query, 'result': result_text, 'inference_time': inference_time, 'device': device})

                    #time.sleep(wait_time)

def main() -> None:
    data_path = "data\Test FAQ.txt"
    result_path = "data\KhanhDB_result_1.csv"
    API_key()
    qa_chain = chatbot()
    run_and_save(qa_chain,data_path, result_path)

if __name__ == "__main__":
    main()
