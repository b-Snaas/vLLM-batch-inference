#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B-FP8}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_TOKENS=${MAX_TOKENS:-256}
BATCH_SIZE=${BATCH_SIZE:-128}
QUANTIZATION=${QUANTIZATION:-fp8}
DTYPE=${DTYPE:-half}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Starting vLLM server..."
echo "Model: ${MODEL_NAME} | dtype: ${DTYPE} | max_model_len: ${MAX_MODEL_LEN} | max_tokens: ${MAX_TOKENS} | batch_size: ${BATCH_SIZE}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --dtype "${DTYPE}" \
  --quantization="${QUANTIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${BATCH_SIZE}" \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 1 \
  --enable-chunked-prefill \
  --port "${PORT}" \
  --served-model-name qwen3-4b \
  --trust-remote-code \
  --response-role assistant

