import aiohttp
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from utils.schemas import ChatCompletionRequest
from utils.truncation import truncate_messages, MAX_INPUT_LENGTH
from utils.config import VLLM_URL

router = APIRouter()

async def send_request_with_retry(session, request: ChatCompletionRequest):
    """Sends a request to vLLM, truncating and retrying if the context is too long."""
    vllm_endpoint = f"{VLLM_URL}/v1/chat/completions"
    payload = request.model_dump(exclude_none=True)

    try:
        resp = await session.post(vllm_endpoint, json=payload, timeout=180)
        if resp.status == 200:
            return resp

        # Check for context length error and retry if needed
        if resp.status == 400:
            error_details = await resp.json()
            if "too long" in error_details.get("message", ""):
                request.messages = truncate_messages(request.messages, MAX_INPUT_LENGTH)
                payload = request.model_dump(exclude_none=True)
                return await session.post(vllm_endpoint, json=payload, timeout=180)

        return resp

    except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
        raise


async def stream_vllm_response(request: ChatCompletionRequest):
    """Proxy streaming requests to vLLM."""
    
    async with aiohttp.ClientSession() as session:
        try:
            resp = await send_request_with_retry(session, request)
            
            if resp.status != 200:
                error_text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"vLLM Error: {error_text}")
            
            async for chunk in resp.content.iter_any():
                yield chunk

        except aiohttp.ClientConnectorError:
            raise HTTPException(status_code=503, detail="Could not connect to vLLM service.")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request to vLLM timed out.")


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    
    if request.stream:
        return StreamingResponse(
            stream_vllm_response(request),
            media_type="text/event-stream"
        )
    else:
        async with aiohttp.ClientSession() as session:
            try:
                resp = await send_request_with_retry(session, request)
                response_json = await resp.json()
                return JSONResponse(content=response_json, status_code=resp.status)

            except aiohttp.ClientConnectorError:
                raise HTTPException(status_code=503, detail="Could not connect to vLLM service.")
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Request to vLLM timed out.")
