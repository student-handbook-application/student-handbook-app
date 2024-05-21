def save_conversation_history(conversation_logs):
    with open("data/conversation_logs.txt", "a",encoding="utf-8") as file:
        for log in conversation_logs:
            file.write(f"User: {log['user_message']}\n")
            file.write(f"AI: {log['ai_response']}\n")
        file.write("\n")