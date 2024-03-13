from fastapi import APIRouter
from app.modules import chatbot


modules_router = APIRouter(prefix="/modules", tags=["modules"])
modules_router.include_router(chatbot)

@modules_router.get("/")
async def index():
    return {"message": "Welcome to modules page"}