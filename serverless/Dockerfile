# Serverless worker image for the vanKonijnenburg analysis pipeline. Built once by CI (see
# .github/workflows/build-serverless.yml) and pushed to GHCR -- every RunPod Serverless worker
# just pulls this image, no per-worker pip install / venv-on-volume dance like the old Pod
# approach needed. The base model (google/gemma-4-31B-it, ~59GB) is NOT baked in here -- that
# would make the image itself enormous and slow to pull -- it's downloaded from Hugging Face at
# container start instead, same as it always was for a cold worker.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# torch 2.4.1 (the base image's default) doesn't know the "gemma4" architecture at all, and the
# transformers version that DOES know it hard-requires torch>=2.5 -- diagnosed live earlier this
# project, see model_server.py's history. torchvision/torchaudio must move in lockstep with torch
# or unrelated transformers image-processing imports break with a torchvision::nms error.
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -U transformers peft accelerate bitsandbytes runpod

# eval_extract_compose_gemma.py supplies the frozen, already-validated EXTRACT_SYSTEM/
# COMPOSE_SYSTEM prompts (handler.py execs its top portion) -- kept as a single source of truth
# instead of copy-pasting the prompts, so this can never silently drift from what was tested
# against FACTS-860 + HaluEval.
COPY eval_extract_compose_gemma.py /app/eval_extract_compose_gemma.py
# Adapter weights (~200MB) are small enough to bake straight into the image -- downloaded from
# the private hallucheck-scripts GitHub Release by the CI workflow before this build runs (see
# workflow file), landing in ./adapter relative to this Dockerfile.
COPY adapter/ /app/adapter/
COPY handler.py /app/handler.py

ENV EVAL_SCRIPT_PATH=/app/eval_extract_compose_gemma.py
ENV ADAPTER_PATH=/app/adapter

CMD ["python3", "-u", "/app/handler.py"]
