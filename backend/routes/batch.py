import asyncio
import json
import uuid
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse

from utils.schemas import Batch, FileObject, BatchCreate
from utils.config import VLLM_URL
from utils.vllm_queue import batch_queue, VLLMRequest


router = APIRouter()

batches_db = {}
files_db = {}

os.makedirs("batch_files", exist_ok=True)
FILES_DIR = "batch_files"

@router.post("/v1/files", response_model=FileObject)
async def upload_file(file: UploadFile = File(...), purpose: str = Form(...)):
    if purpose != "batch":
        raise HTTPException(status_code=400, detail="Purpose must be 'batch'")

    file_id = f"file-{uuid.uuid4()}"
    file_path = os.path.join(FILES_DIR, file_id)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    file_size = os.path.getsize(file_path)

    file_object = FileObject(
        id=file_id,
        bytes=file_size,
        created_at=int(datetime.now().timestamp()),
        filename=file.filename,
        purpose=purpose,
    )
    files_db[file_id] = file_object
    return file_object

async def process_batch_in_background(batch_id: str):
    """
    The background task for processing a batch.
    """
    batch = batches_db.get(batch_id)
    if not batch:
        return

    batch.status = "in_progress"
    batch.in_progress_at = int(datetime.now().timestamp())
    batch.expires_at = int((datetime.now() + timedelta(hours=24)).timestamp())
    if getattr(batch, "usage", None) is None:
        batch.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    input_file_path = os.path.join(FILES_DIR, batch.input_file_id)
    output_file_id = f"file-{uuid.uuid4()}"
    error_file_id = f"file-{uuid.uuid4()}"
    output_file_path = os.path.join(FILES_DIR, output_file_id)
    error_file_path = os.path.join(FILES_DIR, error_file_id)

    requests_to_process = []
    try:
        with open(input_file_path, "r") as f_in:
            for i, line in enumerate(f_in):
                try:
                    request_data = json.loads(line)
                    messages = request_data.get("messages", [])
                    system_message = next((msg for msg in messages if msg.get("role") == "system"), None)
                    user_message = next((msg for msg in messages if msg.get("role") == "user"), None)

                    if not system_message or not user_message:
                        raise ValueError("Missing system or user message in the input data.")

                    template = system_message.get("content", "")
                    data = user_message.get("content", "")
                    
                    final_content = template.replace("<user_profile>", data).replace("<system_info>", "")
                    final_message = {"role": "system", "content": final_content}
                    
                    custom_id = f"request-{i+1}"
                    request_body = {
                        "model": "qwen3-4b",
                         "messages": [final_message],
                         "max_tokens": 256
                    }
                    
                    vllm_request = VLLMRequest(
                        custom_id=custom_id,
                        request_body={
                            **request_body,
                            "priority": 10
                        },
                        vllm_endpoint=batch.endpoint
                    )
                    requests_to_process.append(vllm_request)

                except (json.JSONDecodeError, ValueError) as e:
                    batch.request_counts.failed += 1
                    with open(error_file_path, "a") as f_err:
                        error_result = {"error": f"Error processing line {i+1}: {e}"}
                        f_err.write(json.dumps(error_result) + "\n")

    except Exception as e:
        batch.status = "failed"
        batch.failed_at = int(datetime.now().timestamp())
        batch.errors = {"code": "500", "message": f"Failed to read or parse input file: {e}"}
        return


    batch.request_counts.total = len(requests_to_process)

    for req in requests_to_process:
        await batch_queue.put(req)

    tasks = [req.future for req in requests_to_process]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    with open(output_file_path, "w") as f_out, open(error_file_path, "a") as f_err:
        for i, result in enumerate(results):
            req = requests_to_process[i]

            if batch.status == "cancelling":
                break
            
            if isinstance(result, Exception):
                batch.request_counts.failed += 1
                error_entry = {
                    "custom_id": req.custom_id,
                    "response": {"status_code": 500, "body": f"An unexpected error occurred: {str(result)}"}
                }
                f_err.write(json.dumps(error_entry) + "\n")
            else:
                status_code = result.get("status_code")
                body = result.get("body")

                if status_code == 200:
                    response_entry = {
                        "custom_id": req.custom_id,
                        "response": {"status_code": status_code, "body": body}
                    }
                    f_out.write(json.dumps(response_entry) + "\n")
                    batch.request_counts.completed += 1

                    # Aggregate token usage if provided by vLLM
                    if isinstance(body, dict):
                        usage = body.get("usage") or {}
                        if isinstance(usage, dict):
                            batch.usage["prompt_tokens"] = batch.usage.get("prompt_tokens", 0) + int(usage.get("prompt_tokens", 0))
                            batch.usage["completion_tokens"] = batch.usage.get("completion_tokens", 0) + int(usage.get("completion_tokens", 0))
                else:
                    error_entry = {
                        "custom_id": req.custom_id,
                        "response": {"status_code": status_code, "body": body}
                    }
                    f_err.write(json.dumps(error_entry) + "\n")
                    batch.request_counts.failed += 1

    if batch.status == "cancelling":
        batch.status = "cancelled"
        batch.cancelled_at = int(datetime.now().timestamp())
    else:
        batch.status = "completed"
        batch.completed_at = int(datetime.now().timestamp())
        
    batch.output_file_id = output_file_id
    
    if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
        files_db[output_file_id] = FileObject(
            id=output_file_id,
            bytes=os.path.getsize(output_file_path),
            created_at=int(datetime.now().timestamp()),
            filename=f"{batch_id}_output.jsonl",
            purpose="batch_output"
        )
    else:
        batch.output_file_id = None
        if os.path.exists(output_file_path):
             os.remove(output_file_path)

    if os.path.exists(error_file_path) and os.path.getsize(error_file_path) > 0:
        batch.error_file_id = error_file_id
        files_db[error_file_id] = FileObject(
            id=error_file_id,
            bytes=os.path.getsize(error_file_path),
            created_at=int(datetime.now().timestamp()),
            filename=f"{batch_id}_errors.jsonl",
            purpose="batch_output"
        )
    else:
        batch.error_file_id = None
        if os.path.exists(error_file_path):
            os.remove(error_file_path)


@router.post("/v1/batches", response_model=Batch, status_code=201)
async def create_batch(batch_create: BatchCreate, background_tasks: BackgroundTasks):
    batch_id = f"batch_{uuid.uuid4()}"
    
    new_batch = Batch(
        id=batch_id,
        input_file_id=batch_create.input_file_id,
        endpoint=batch_create.endpoint,
        completion_window=batch_create.completion_window,
        status="pending",
        created_at=int(datetime.now().timestamp())
    )
    
    batches_db[batch_id] = new_batch
    background_tasks.add_task(process_batch_in_background, batch_id)
    
    return new_batch

@router.get("/v1/batches/{batch_id}", response_model=Batch)
async def retrieve_batch(batch_id: str):
    if batch_id not in batches_db:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batches_db[batch_id]

@router.post("/v1/batches/{batch_id}/cancel", response_model=Batch)
async def cancel_batch(batch_id: str):
    if batch_id not in batches_db:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = batches_db[batch_id]
    
    if batch.status in ["cancelling", "cancelled", "completed", "failed", "expired"]:
        raise HTTPException(status_code=400, detail=f"Batch is already in a terminal state: {batch.status}")

    batch.status = "cancelling"
    batch.cancelling_at = int(datetime.now().timestamp())
    
    return batch
