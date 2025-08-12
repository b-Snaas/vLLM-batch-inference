import asyncio
import aiohttp
import time
import os
import json
import statistics

API_BASE_URL = "http://127.0.0.1:3000"
API_KEY = "123"
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
                    processing_time = end_time - start_time
                    
                    print(f"   - Batch finished with status: {status}")
                    print(f"   - Total batch processing time: {processing_time:.2f} seconds")

                    if 'request_counts' in batch_info:
                        completed_requests = batch_info['request_counts'].get('completed', 0)
                        if processing_time > 0 and completed_requests > 0:
                            throughput = completed_requests / processing_time
                            print(f"   - Batch throughput: {throughput:.2f} req/s")
                        print(f"   - Request counts: {batch_info['request_counts']}")

                    if status == "completed" and 'usage' in batch_info and batch_info['usage'] is not None:
                        input_tokens = batch_info['usage'].get('prompt_tokens', 0)
                        output_tokens = batch_info['usage'].get('completion_tokens', 0)
                        total_tokens = input_tokens + output_tokens

                        print(f"   - Token usage:")
                        print(f"     - Input tokens: {input_tokens}")
                        print(f"     - Output tokens: {output_tokens}")
                        print(f"     - Total tokens: {total_tokens}")

                        if processing_time > 0:
                            input_tps = input_tokens / processing_time
                            output_tps = output_tokens / processing_time
                            print(f"   - Throughput (input tokens/s): {input_tps:.2f}")
                            print(f"   - Throughput (output tokens/s): {output_tps:.2f}")

                    break
            else:
                print(f"   - Error fetching batch status: {resp.status}")
                break
        await asyncio.sleep(5)

async def single_request_worker(session: aiohttp.ClientSession, stop_event: asyncio.Event, results: dict):
    """A worker that continuously sends single chat completion requests."""
    payload = {
        "model": "qwen3-4b",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    while not stop_event.is_set():
        start_time = time.time()
        ttft = -1
        prompt_tokens = 0
        completion_tokens = 0
        try:
            async with session.post(f"{API_BASE_URL}/v1/chat/completions", json=payload, headers=HEADERS) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        line = line.strip()
                        if not line:
                            continue
                        if ttft == -1:
                            ttft = time.time() - start_time
                        
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data:"):
                            data_str = line_str[5:].strip()
                            if data_str != "[DONE]":
                                try:
                                    json_data = json.loads(data_str)
                                    if 'usage' in json_data and json_data['usage'] is not None:
                                        prompt_tokens = json_data['usage'].get('prompt_tokens', 0)
                                        completion_tokens = json_data['usage'].get('completion_tokens', 0)
                                except json.JSONDecodeError:
                                    pass

                    latency = time.time() - start_time
                    results['latencies'].append(latency)
                    results['ttfts'].append(ttft if ttft != -1 else 0)
                    if prompt_tokens > 0:
                        results['prompt_tokens'].append(prompt_tokens)
                    if completion_tokens > 0:
                        results['completion_tokens'].append(completion_tokens)
                else:
                    results['errors'] += 1
        except aiohttp.ClientError:
            results['errors'] += 1
        
        await asyncio.sleep(0.1)

def print_stats(label: str, latencies: list, ttfts: list, prompt_tokens: list, completion_tokens: list, duration: float):
    """Calculates and prints performance statistics."""
    if not latencies:
        print(f"\n--- {label} ---")
        print("   - No successful requests recorded.")
        return

    throughput_reqs = len(latencies) / duration
    total_prompt_tokens = sum(prompt_tokens)
    total_completion_tokens = sum(completion_tokens)
    throughput_prompt_tps = total_prompt_tokens / duration if duration > 0 else 0
    throughput_completion_tps = total_completion_tokens / duration if duration > 0 else 0
    
    print(f"\n--- {label} (Duration: {duration:.2f}s) ---")
    print(f"   - Successful requests: {len(latencies)}")
    print(f"   - Throughput (req/s): {throughput_reqs:.2f}")
    print(f"   - Throughput (prompt tokens/s): {throughput_prompt_tps:.2f}")
    print(f"   - Throughput (completion tokens/s): {throughput_completion_tps:.2f}")
    
    for metric_name, data in [("Latency", latencies), ("TTFT", ttfts)]:
        if not data: continue
        print(f"\n   {metric_name} Stats:")
        print(f"     - Average: {statistics.mean(data):.4f}s")
        print(f"     - Median (p50): {statistics.median(data):.4f}s")
        if len(data) > 1:
            data.sort()
            p95_index = int(len(data) * 0.95)
            p99_index = int(len(data) * 0.99)
            print(f"     - p95: {data[p95_index]:.4f}s")
            print(f"     - p99: {data[p99_index]:.4f}s")
        print(f"     - Min: {min(data):.4f}s")
        print(f"     - Max: {max(data):.4f}s")


async def run_single_completions(duration_seconds: int, concurrency: int):
    """Runs a single-user completion test for a fixed duration."""
    print(f"\n--- Running Single-User Test (Concurrency: {concurrency}, Duration: {duration_seconds}s) ---")
    
    results = {'latencies': [], 'ttfts': [], 'errors': 0, 'prompt_tokens': [], 'completion_tokens': []}
    stop_event = asyncio.Event()
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Start worker tasks
        worker_tasks = [
            asyncio.create_task(single_request_worker(session, stop_event, results))
            for _ in range(concurrency)
        ]
        
        # Wait for the specified duration
        await asyncio.sleep(duration_seconds)
        
        # Stop workers
        stop_event.set()
        await asyncio.gather(*worker_tasks)
        
        end_time = time.time()

    print_stats(f"Single-User Results (Concurrency: {concurrency})", results['latencies'], results['ttfts'], end_time - start_time)


async def run_mixed_workload_test(concurrency: int):
    """Runs a batch job alongside a single-user load test."""
    print(f"\n--- Running Mixed-Workload Test (Batch + {concurrency} Concurrent Users) ---")
    
    results = {'latencies': [], 'ttfts': [], 'errors': 0, 'prompt_tokens': [], 'completion_tokens': []}
    stop_event = asyncio.Event()

    async with aiohttp.ClientSession() as session:
        # Step 1: Upload dataset
        file_id = await upload_dataset(session)
        if not file_id: return

        # Step 2: Create batch
        batch_id = await create_batch(session, file_id)
        if not batch_id: return
        
        test_start_time = time.time()
        
        # Start batch monitoring and single-user workers concurrently
        monitor_task = asyncio.create_task(monitor_batch_status(session, batch_id))
        worker_tasks = [
             asyncio.create_task(single_request_worker(session, stop_event, results))
             for _ in range(concurrency)
        ]

        # Wait for the batch job to finish
        await monitor_task
        
        # Stop the single-user workers
        stop_event.set()
        await asyncio.gather(*worker_tasks)

        test_end_time = time.time()

    print_stats(f"Mixed-Workload Single-User Results (Concurrency: {concurrency})", results['latencies'], results['ttfts'], test_end_time - test_start_time)


async def run_batch_only_test():
    """Runs the performance test for a batch job without other requests."""
    print("\n" + "="*80)
    print("--- Running Test Condition 1: Batch-Only Performance ---")
    print("="*80)
    async with aiohttp.ClientSession() as session:
        file_id = await upload_dataset(session)
        if not file_id: return
        batch_id = await create_batch(session, file_id)
        if not batch_id: return
        await monitor_batch_status(session, batch_id)


async def main():
    # Test 1: Batch-only
    await run_batch_only_test()

    # Test 2: Isolated single-user load tests
    print("\n" + "="*80)
    print("--- Running Test Condition 2: Isolated Single-User Performance ---")
    print("="*80)
    await run_single_completions(duration_seconds=30, concurrency=1)
    await run_single_completions(duration_seconds=30, concurrency=10)
    
    # Test 3: Mixed workload tests
    print("\n" + "="*80)
    print("--- Running Test Condition 3: Mixed-Workload Performance ---")
    print("="*80)
    await run_mixed_workload_test(concurrency=1)
    await run_mixed_workload_test(concurrency=10)


if __name__ == "__main__":
    asyncio.run(main())
