from app.utils.chatbot_template import template
from app.utils.faq_vectorstore import load_FAQ
from app.modules.save_history.models.save_history import save_conversation_history
from app.modules.question_answer_retrieval.models.question_answer_retrieval import create_prompt, create_qa_chain
# from app.configs.llm_models import llm


def Chatbot(msg,llm):
    hits = load_FAQ(msg)
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