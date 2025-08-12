import os
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000")
API_TOKEN = os.getenv("API_TOKEN")
