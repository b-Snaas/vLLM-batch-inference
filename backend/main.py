from fastapi import FastAPI
from routes import chat, batch
from utils.authorization import auth_middleware
from utils.vllm_queue import start_vllm_consumer, interactive_queue, batch_queue

app = FastAPI()


INTERACTIVE_WORKERS = 4
INTERACTIVE_BATCH_SIZE = 1
INTERACTIVE_WAIT_TIME = 0.01

BATCH_WORKERS = 2
BATCH_BATCH_SIZE = 128
BATCH_WAIT_TIME = 0.1

@app.on_event("startup")
async def startup_event():
    # Start consumers for the interactive queue
    for i in range(INTERACTIVE_WORKERS):
        start_vllm_consumer(
            worker_id=i, 
            queue=interactive_queue, 
            batch_size=INTERACTIVE_BATCH_SIZE, 
            wait_time=INTERACTIVE_WAIT_TIME
        )
    
    # Start consumers for the batch queue
    for i in range(BATCH_WORKERS):
        start_vllm_consumer(
            worker_id=i + INTERACTIVE_WORKERS, 
            queue=batch_queue, 
            batch_size=BATCH_BATCH_SIZE, 
            wait_time=BATCH_WAIT_TIME
        )

app.middleware("http")(auth_middleware)
app.include_router(chat.router)
app.include_router(batch.router)
