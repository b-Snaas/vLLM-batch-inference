from fastapi import FastAPI
from routes import chat, batch
from utils.authorization import auth_middleware
from utils.vllm_queue import start_vllm_consumer

app = FastAPI()

NUM_VLLM_WORKERS = 4

@app.on_event("startup")
async def startup_event():
    for i in range(NUM_VLLM_WORKERS):
        start_vllm_consumer(worker_id=i)

app.middleware("http")(auth_middleware)
app.include_router(chat.router)
app.include_router(batch.router)
