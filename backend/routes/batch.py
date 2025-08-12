import asyncio
import aiohttp
import json
import uuid
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse

from utils.schemas import Batch, FileObject, BatchCreate
from utils.config import VLLM_URL


router = APIRouter()

# In-memory storage for simplicity
batches_db = {}
files_db = {}

# Directory to store uploaded and generated files
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

    # 1. Update batch status
    batch.status = "in_progress"
    batch.in_progress_at = int(datetime.now().timestamp())
    batch.expires_at = int((datetime.now() + timedelta(hours=24)).timestamp())

    input_file_path = os.path.join(FILES_DIR, batch.input_file_id)
    output_file_id = f"file-{uuid.uuid4()}"
    error_file_id = f"file-{uuid.uuid4()}"
    output_file_path = os.path.join(FILES_DIR, output_file_id)
    error_file_path = os.path.join(FILES_DIR, error_file_id)

    # 2. Read input file and process requests
    try:
        with open(input_file_path, "r") as f_in, \
             open(output_file_path, "w") as f_out, \
             open(error_file_path, "w") as f_err:

            async with aiohttp.ClientSession() as session:
                for line in f_in:
                    # Check for cancellation
                    if batch.status == "cancelling":
                        batch.status = "cancelled"
                        batch.cancelled_at = int(datetime.now().timestamp())
                        # Clean up empty files if no processing happened
                        if batch.request_counts.total == 0:
                            os.remove(output_file_path)
                            os.remove(error_file_path)
                        return

                    batch.request_counts.total += 1
                    try:
                        request_data = json.loads(line)
                        
                        # Extract messages and find the template and data
                        messages = request_data.get("messages", [])
                        system_message = next((msg for msg in messages if msg.get("role") == "system"), None)
                        user_message = next((msg for msg in messages if msg.get("role") == "user"), None)

                        if not system_message or not user_message:
                            raise ValueError("Missing system or user message in the input data.")

                        # Extract placeholders and their values
                        template = system_message.get("content", "")
                        data = user_message.get("content", "")
                        
                        # Simple string replacement for placeholders
                        final_content = template.replace("<user_profile>", data).replace("<system_info>", "")

                        # Construct the new message
                        final_message = {"role": "system", "content": final_content}
                        
                        custom_id = f"request-{batch.request_counts.total}"
                        
                        # Construct the request body for the chat completion
                        request_body = {
                            "model": "qwen3-4b",
                            "messages": [final_message]
                        }
                        
                        endpoint = batch.endpoint

                        # 3. Send request to vLLM
                        vllm_full_url = f"{VLLM_URL}{endpoint}"
                        async with session.post(vllm_full_url, json=request_body, timeout=180) as resp:
                            response_body = await resp.json()
                            if resp.status == 200:
                                # 4. Write successful response
                                result = {
                                    "custom_id": custom_id,
                                    "response": {
                                        "status_code": resp.status,
                                        "body": response_body
                                    }
                                }
                                f_out.write(json.dumps(result) + "\n")
                                batch.request_counts.completed += 1
                            else:
                                # 5. Write error response
                                error_result = {
                                    "custom_id": custom_id,
                                    "response": {
                                        "status_code": resp.status,
                                        "body": response_body
                                    }
                                }
                                f_err.write(json.dumps(error_result) + "\n")
                                batch.request_counts.failed += 1

                    except json.JSONDecodeError as e:
                        batch.request_counts.failed += 1
                        error_result = {"error": f"JSON decode error: {e}"}
                        f_err.write(json.dumps(error_result) + "\n")
                    except Exception as e:
                        batch.request_counts.failed += 1
                        error_result = {"error": f"An unexpected error occurred: {e}"}
                        f_err.write(json.dumps(error_result) + "\n")

        # 6. Finalize batch
        batch.status = "completed"
        batch.completed_at = int(datetime.now().timestamp())
        batch.output_file_id = output_file_id
        
        # Create FileObject for output file
        files_db[output_file_id] = FileObject(
            id=output_file_id,
            bytes=os.path.getsize(output_file_path),
            created_at=int(datetime.now().timestamp()),
            filename=f"{batch_id}_output.jsonl",
            purpose="batch_output"
        )
        
        if batch.request_counts.failed > 0:
            batch.error_file_id = error_file_id
            files_db[error_file_id] = FileObject(
                id=error_file_id,
                bytes=os.path.getsize(error_file_path),
                created_at=int(datetime.now().timestamp()),
                filename=f"{batch_id}_errors.jsonl",
                purpose="batch_output"
            )
        else:
             # If there are no errors, remove the empty error file
            os.remove(error_file_path)

    except Exception as e:
        batch.status = "failed"
        batch.failed_at = int(datetime.now().timestamp())
        batch.errors = {"code": "500", "message": f"Batch processing failed: {e}"}
        # Clean up files on catastrophic failure
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        if os.path.exists(error_file_path):
            os.remove(error_file_path)


@router.post("/v1/batches", response_model=Batch, status_code=201)
async def create_batch(batch_create: BatchCreate, background_tasks: BackgroundTasks):
    # This is a simplified implementation.
    # We will expand on this in the next steps.
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
