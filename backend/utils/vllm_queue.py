import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, List, Dict
import aiohttp
import logging

from .config import VLLM_URL


@dataclass
class VLLMRequest:
    """Represents a request to the VLLM engine."""
    request_body: Dict[str, Any]
    future: asyncio.Future = field(default_factory=asyncio.Future)
    vllm_endpoint: str = "/v1/chat/completions"
    custom_id: str = None

interactive_queue = asyncio.Queue()
batch_queue = asyncio.Queue()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def vllm_consumer(worker_id: int, queue: asyncio.Queue, batch_size: int, wait_time: float):
    """
    A consumer that pulls requests from a given queue, batches them, and sends them to vLLM.
    """
    logger.info(f"vLLM consumer worker-{worker_id} started for queue: {queue.__class__.__name__}.")
    while True:
        requests_batch: List[VLLMRequest] = []
        
        start_time = time.time()
        while time.time() - start_time < wait_time and len(requests_batch) < batch_size:
            try:
                request = queue.get_nowait()
                requests_batch.append(request)
                queue.task_done()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.01)
                if not requests_batch:  # If no requests were in the batch, break inner loop to avoid waiting
                    break
        
        if not requests_batch:
            await asyncio.sleep(0.01)
            continue

        logger.info(f"Worker-{worker_id}: Processing batch of {len(requests_batch)} requests.")
        endpoint = requests_batch[0].vllm_endpoint
        vllm_full_url = f"{VLLM_URL}{endpoint}"

        async with aiohttp.ClientSession() as session:
            tasks = []
            for req in requests_batch:
                task = session.post(vllm_full_url, json=req.request_body, timeout=180)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for request, response in zip(requests_batch, responses):
            try:
                if isinstance(response, Exception):
                    logger.error(f"Worker-{worker_id}: Request {request.custom_id} failed with exception: {response}")
                    result = {
                        "status_code": 500,
                        "body": {"error": str(response)}
                    }
                    request.future.set_result(result)
                else:
                    response_body = await response.json()
                    result = {
                        "status_code": response.status,
                        "body": response_body
                    }
                    if response.status != 200:
                         logger.warning(f"Worker-{worker_id}: Request {request.custom_id} received non-200 status: {response.status}")
                    
                    request.future.set_result(result)

            except Exception as e:
                logger.error(f"Worker-{worker_id}: Error processing response for request {request.custom_id}: {e}")
                error_result = {
                    "status_code": 500,
                    "body": {"error": f"Internal server error processing response: {e}"}
                }
                if not request.future.done():
                    request.future.set_result(error_result)


def start_vllm_consumer(worker_id: int, queue: asyncio.Queue, batch_size: int, wait_time: float):
    """
    Starts the vLLM consumer as a background task for a specific queue.
    """
    asyncio.create_task(vllm_consumer(worker_id, queue, batch_size, wait_time))
