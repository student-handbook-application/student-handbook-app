from fastapi import APIRouter

from app.modules.chatbot.model.chatbot import Chatbot

chatbot_router = APIRouter(prefix="/generate", tags=["generate"])

@chatbot_router.get("/")
async def index():
    return {"message": "Welcome to generate"}

