from chatbot.model import *

def Chatbot(msg,llm):
    template = """<|im_start|>system\nBạn là một trợ lý AI hữu ích. Bạn chỉ sử dụng những thông tin mà bạn được cung cấp để trả lời các câu hỏi,
    hãy trả lời câu hỏi một cách ngắn gọn, trung thực và chính xác. Tránh trả lời các câu hỏi không có trong thông tin mà bạn được cung cấp.
    Nếu câu hỏi không liên quan đến nội dung mà bạn được cung cấp, bạn hãy trả lời rằng bạn không biết câu trả lời.Tuyệt đối không được bịa ra
    câu trả lời.\n{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assitant"""

    #load FAQ database
    hits = load_FAQ(msg)

    #run result
    for hit in hits:
        if hit.score > 0.49:
            return f"{hit.payload['Answers']}"
        else:
            prompt = create_prompt(template)
            qa_chain = create_qa_chain(llm, prompt)
            return qa_chain.invoke({"query": msg})