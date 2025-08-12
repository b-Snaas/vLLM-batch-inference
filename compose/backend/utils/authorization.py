from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.status import HTTP_401_UNAUTHORIZED

from .config import API_TOKEN

async def auth_middleware(request: Request, call_next):
    if API_TOKEN:
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {API_TOKEN}":
            return JSONResponse(
                status_code=HTTP_401_UNAUTHORIZED,
                content={"error": "Unauthorized"},
            )
    response = await call_next(request)
    return response
