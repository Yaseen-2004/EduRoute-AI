# EduRoute AI app image (the LLMs run in the separate Ollama container).
# Built by deploy/gpu/docker-compose.yml from the project root.
FROM python:3.12-slim

ARG TORCH_INDEX=cu124
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/models/hf

RUN apt-get update && apt-get install -y --no-install-recommends curl espeak-ng && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements-gpu.txt
# cuDNN/cuBLAS from the nvidia-* wheels, for faster-whisper on the GPU
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib

COPY . .
EXPOSE 8501
HEALTHCHECK CMD curl -fs http://127.0.0.1:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
