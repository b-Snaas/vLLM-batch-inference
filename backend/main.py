from fastapi import FastAPI
from routes import chat, batch
from utils.authorization import auth_middleware

app = FastAPI()

app.middleware("http")(auth_middleware)
app.include_router(chat.router)
app.include_router(batch.router)
