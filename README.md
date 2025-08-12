# vLLM Batched Inference

## Setup

To build the docker image:
```bash
docker build -t vllm-batched-inference -f backend/docker/Dockerfile .
```
To run the image:
```bash
docker run --rm -it --gpus all -p 8000:8000 -p 3000:3000 --env-file .env vllm-batched-inference
```
To run the benchmark:
```bash
python tests/performance_test.py
```

---

## Batch Inference Benchmark Results

### Test Condition 1: Batch-Only Performance

-   **Token usage (batch-wide):**
    -   Prompt tokens: total=13704028
    -   Completion tokens: total=1253888
-   **Token throughput:**
    -   Prompt tokens/sec: 34649.87
    -   Completion tokens/sec: 3170.39
    -   Total tokens/sec: 37820.25

---

### Test Condition 2: Isolated Single-User Performance

#### Single-User Test (Concurrency: 1, Duration: 30s)

-   **Successful requests:** 77
-   **Throughput:** 2.55 req/s

| Latency Stats | Value |
| :--- | :--- |
| Average | 0.2919s |
| Median (p50) | 0.2909s |
| p95 | 0.2997s |
| p99 | 0.3000s |
| Min | 0.2889s |
| Max | 0.3000s |

| TTFT Stats | Value |
| :--- | :--- |
| Average | 0.0125s |
| Median (p50) | 0.0123s |
| p95 | 0.0135s |
| p99 | 0.0195s |
| Min | 0.0110s |
| Max | 0.0195s |


#### Single-User Test (Concurrency: 10, Duration: 30s)

-   **Successful requests:** 736
-   **Throughput:** 24.29 req/s

| Latency Stats | Value |
| :--- | :--- |
| Average | 0.3082s |
| Median (p50) | 0.3088s |
| p95 | 0.3154s |
| p99 | 0.3309s |
| Min | 0.2976s |
| Max | 0.3318s |

| TTFT Stats | Value |
| :--- | :--- |
| Average | 0.0192s |
| Median (p50) | 0.0197s |
| p95 | 0.0242s |
| p99 | 0.0297s |
| Min | 0.0113s |
| Max | 0.0302s |

---

### Test Condition 3: Mixed-Workload Performance

#### Mixed-Workload Test (Batch + 1 Concurrent Users)

-   **Token usage (batch-wide):**
    -   Prompt tokens: total=13704028
    -   Completion tokens: total=1253888
-   **Token throughput:**
    -   Prompt tokens/sec: 34652.37
    -   Completion tokens/sec: 3170.61
    -   Total tokens/sec: 37822.98

**Single-User Results (Concurrency: 1, Duration: 395.66s)**
-   **Successful requests:** 182
-   **Throughput:** 0.46 req/s

| Latency Stats | Value |
| :--- | :--- |
| Average | 2.0738s |
| Median (p50) | 1.0119s |
| p95 | 8.7602s |
| p99 | 9.0080s |
| Min | 0.2916s |
| Max | 9.1149s |

| TTFT Stats | Value |
| :--- | :--- |
| Average | 0.4098s |
| Median (p50) | 0.0372s |
| p95 | 3.4477s |
| p99 | 3.8015s |
| Min | 0.0119s |
| Max | 3.8520s |


#### Mixed-Workload Test (Batch + 10 Concurrent Users)

-   **Token usage (batch-wide):**
    -   Prompt tokens: total=13704028
    -   Completion tokens: total=1253888
-   **Token throughput:**
    -   Prompt tokens/sec: 33387.83
    -   Completion tokens/sec: 3054.91
    -   Total tokens/sec: 36442.74

**Single-User Results (Concurrency: 10, Duration: 410.80s)**
-   **Successful requests:** 2327
-   **Throughput:** 5.66 req/s

| Latency Stats | Value |
| :--- | :--- |
| Average | 1.6642s |
| Median (p50) | 1.3540s |
| p95 | 4.6418s |
| p99 | 5.7095s |
| Min | 0.3000s |
| Max | 7.7352s |

| TTFT Stats | Value |
| :--- | :--- |
| Average | 0.2447s |
| Median (p50) | 0.0838s |
| p95 | 0.9277s |
| p99 | 1.8339s |
| Min | 0.0113s |
| Max | 3.1654s |