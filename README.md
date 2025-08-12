To build the docker image:
```
docker build -t vllm-batched-inference -f backend/docker/Dockerfile .
```
To run the image:
```
docker run --rm -it --gpus all -p 8000:8000 -p 3000:3000 --env-file .env vllm-batched-inference
```