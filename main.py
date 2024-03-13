import uvicorn
from  fastapi import FastAPI
from app.modules import modules_router

app = FastAPI()
app.include_router(modules_router)

@app.get("/")
async def index():
    return {"message": "Simple Question API Services"}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=7860)