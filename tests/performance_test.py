import asyncio
import aiohttp
import time
import os
import json

API_BASE_URL = "http://127.0.0.1:8000"
API_KEY = "your_api_key"
DATASET_PATH = "dataset.jsonl"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

async def upload_dataset(session: aiohttp.ClientSession) -> str:
    """Uploads the dataset file and returns the file ID."""
    print("1. Uploading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset file not found at '{DATASET_PATH}'")
        return None

    with open(DATASET_PATH, "rb") as f:
        data = aiohttp.FormData()
        data.add_field('file', f, filename=DATASET_PATH, content_type='application/octet-stream')
        data.add_field('purpose', 'batch')

        async with session.post(f"{API_BASE_URL}/v1/files", data=data, headers=HEADERS) as resp:
            if resp.status == 200:
                response_json = await resp.json()
                print(f"   - File uploaded successfully. File ID: {response_json['id']}")
                return response_json['id']
            else:
                print(f"   - Error uploading file: {resp.status} {await resp.text()}")
                return None

async def create_batch(session: aiohttp.ClientSession, file_id: str) -> str:
    """Creates a batch job and returns the batch ID."""
    print("2. Creating batch job...")
    payload = {
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    }
    async with session.post(f"{API_BASE_URL}/v1/batches", json=payload, headers=HEADERS) as resp:
        if resp.status == 201:
            response_json = await resp.json()
            print(f"   - Batch job created successfully. Batch ID: {response_json['id']}")
            return response_json['id']
        else:
            print(f"   - Error creating batch: {resp.status} {await resp.text()}")
            return None

async def monitor_batch_status(session: aiohttp.ClientSession, batch_id: str):
    """Monitors the status of the batch until it's completed."""
    print("3. Monitoring batch status...")
    start_time = time.time()
    while True:
        async with session.get(f"{API_BASE_URL}/v1/batches/{batch_id}", headers=HEADERS) as resp:
            if resp.status == 200:
                batch_info = await resp.json()
                status = batch_info['status']
                print(f"   - Batch status: {status} (Elapsed: {time.time() - start_time:.2f}s)")
                if status in ["completed", "failed", "cancelled"]:
                    end_time = time.time()
                    print(f"   - Batch finished with status: {status}")
                    print(f"   - Total batch processing time: {end_time - start_time:.2f} seconds")
                    print(f"   - Request counts: {batch_info['request_counts']}")
                    break
            else:
                print(f"   - Error fetching batch status: {resp.status}")
                break
        await asyncio.sleep(5)

async def run_single_completions(session: aiohttp.ClientSession, stop_event: asyncio.Event):
    """Continuously sends single chat completion requests."""
    print("4. Running single-user completions on the side...")
    latencies = []
    payload = {
        "model": "your-model-name", # Replace with your model name
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50
    }
    
    while not stop_event.is_set():
        start_time = time.time()
        try:
            async with session.post(f"{API_BASE_URL}/v1/chat/completions", json=payload, headers=HEADERS) as resp:
                if resp.status == 200:
                    await resp.json()
                    latency = time.time() - start_time
                    latencies.append(latency)
                else:
                    print(f"   - Single completion error: {resp.status} {await resp.text()}")
        except aiohttp.ClientError as e:
            print(f"   - Single completion connection error: {e}")
            
        await asyncio.sleep(1) # Send a request every second

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print("\n--- Single-User Completion Results ---")
        print(f"   - Average latency: {avg_latency:.4f} seconds")
        print(f"   - Total requests sent: {len(latencies)}")

async def main():
    async with aiohttp.ClientSession() as session:
        # Step 1: Upload dataset
        file_id = await upload_dataset(session)
        if not file_id:
            return

        # Step 2: Create batch
        batch_id = await create_batch(session, file_id)
        if not batch_id:
            return

        # Step 3 & 4: Monitor batch and run single completions concurrently
        stop_event = asyncio.Event()
        
        monitor_task = asyncio.create_task(monitor_batch_status(session, batch_id))
        completions_task = asyncio.create_task(run_single_completions(session, stop_event))

        await monitor_task
        stop_event.set()
        await completions_task

if __name__ == "__main__":
    asyncio.run(main())
