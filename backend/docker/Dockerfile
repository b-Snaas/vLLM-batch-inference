# Use the vLLM base image
FROM vllm/vllm-openai:latest

# Set the working directory
WORKDIR /app

# Copy the backend application code
COPY backend/ .

# Install Python dependencies for the backend
RUN pip install --no-cache-dir -r /app/requirements.txt

# Make the startup script executable
RUN chmod +x /app/docker/start.sh

# Expose ports for vLLM (8000) and the FastAPI backend (3000)
EXPOSE 8000
EXPOSE 3000

# Set the URL for the vLLM service, which will be running locally inside the container
ENV VLLM_URL=http://localhost:8000

# Set the default command to run our combined startup script
CMD ["/app/docker/start.sh"]

