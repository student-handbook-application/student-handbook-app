from chatbot.model import *
def Chatbot(msg,llm):
    template = """You are an AI assistant and admissions consultant. Use the following pieces of context to answer the questions HONESTLY, ACCURATELY and MOST NATURAL. Avoid answering questions that are not included in the information you are provided. If the question is not related to the content you are provided, answer with TÔI KHÔNG BIẾT CÂU TRẢ LỜI. ABSOLUTELY not allowed. make up an answer.\n
{context}
Question: {question}
Helpful Answer:"""

    hits = load_FAQ(msg)

    #run result
    for hit in hits:
        if hit.score > 0.6:
            conversation_logs = [{"user_message": msg, "ai_response": hit.payload['Answers']}]
            save_conversation_history(conversation_logs)
            return f"{hit.payload['Answers']}"
        else: 
            prompt = create_prompt(template)
            qa_chain = create_qa_chain(llm, prompt)
            result = qa_chain.invoke({"query": msg})['result']
            conversation_logs = [{"user_message": msg, "ai_response": result}]
            save_conversation_history(conversation_logs)
            return result
        