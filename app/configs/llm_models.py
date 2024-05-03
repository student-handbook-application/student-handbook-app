import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True, api_key=GOOGLE_API_KEY, request_timeout=120)