FROM python:3.11-slim

# libgomp1 is required at runtime by faiss-cpu (OpenMP).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .

# torch is pulled in transitively by sentence-transformers. Installing it
# from PyPI directly resolves a CUDA-enabled build (multiple GB of NVIDIA
# driver packages) even on this CPU-only, non-x86 image — installing the
# official CPU-only wheel first keeps the image an order of magnitude
# smaller, and pip then leaves it alone when it sees the requirement is
# already satisfied while installing the rest.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Default to running the API server. The ingestion program is invoked as
# an override command against the same image, e.g.:
#   docker run --rm -v $(pwd)/data:/app/data <image> python -m app.ingest data/raw_pdfs
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
