from fastapi import FastAPI
from routes import chat, batch
from utils.authorization import auth_middleware
from utils.vllm_queue import start_vllm_consumer

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    start_vllm_consumer()

app.middleware("http")(auth_middleware)
app.include_router(chat.router)
app.include_router(batch.router)
