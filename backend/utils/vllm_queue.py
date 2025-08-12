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

high_priority_queue = asyncio.Queue()
low_priority_queue = asyncio.Queue()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def vllm_consumer(worker_id: int, batch_size: int = 128, wait_time: float = 0.1):
    """
    A consumer that pulls requests from queues, batches them, and sends them to vLLM.
    """
    logger.info(f"vLLM consumer worker-{worker_id} started.")
    while True:
        requests_batch: List[VLLMRequest] = []
        
        # 1. Collect requests, prioritizing the high-priority queue
        start_time = time.time()
        while time.time() - start_time < wait_time and len(requests_batch) < batch_size:
            # Prioritize high-priority queue
            while not high_priority_queue.empty() and len(requests_batch) < batch_size:
                request = await high_priority_queue.get()
                requests_batch.append(request)
                high_priority_queue.task_done()
            
            # If there's space, check low-priority queue
            if len(requests_batch) < batch_size:
                 while not low_priority_queue.empty() and len(requests_batch) < batch_size:
                    request = await low_priority_queue.get()
                    requests_batch.append(request)
                    low_priority_queue.task_done()

            # If no requests were found, sleep briefly to prevent a tight loop
            if not requests_batch:
                await asyncio.sleep(0.01)
                break # Exit inner while to re-check timer
        
        if not requests_batch:
            continue

        # 2. Process the batch
        logger.info(f"Worker-{worker_id}: Processing batch of {len(requests_batch)} requests.")
        endpoint = requests_batch[0].vllm_endpoint
        vllm_full_url = f"{VLLM_URL}{endpoint}"

        async with aiohttp.ClientSession() as session:
            tasks = []
            for req in requests_batch:
                task = session.post(vllm_full_url, json=req.request_body, timeout=180)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Set results for each future
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


def start_vllm_consumer(worker_id: int):
    """
    Starts the vLLM consumer as a background task.
    """
    asyncio.create_task(vllm_consumer(worker_id))
