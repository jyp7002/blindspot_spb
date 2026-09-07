# Environment-neutral run image for the v11 scale-up panels.
#
# The contract: whatever the host is (SLURM node, rented 8xH100 box, on-demand
# cloud instance), the panel runs inside THIS image, so the stack that
# PREREGISTRATION.md L545-547 pins is the stack that actually executes.
#
#   docker build -t blindspot:v11 -f env/Dockerfile .
#   docker run --gpus all --rm \
#       -v $HF_HOME:/hf -v $PWD/results:/app/results \
#       -e HF_HOME=/hf -e HF_TOKEN=$HF_TOKEN \
#       blindspot:v11 python3 scripts/preflight.py
#
# The HF cache is a MOUNT, never baked in: it is 178 GB on the analysis box and
# re-downloading it per node is the single biggest avoidable cost in a rented
# environment.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first, from the cu128 index, so the pinned build is the one that sticks.
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

COPY requirements-run.txt requirements-analysis.txt /app/
RUN pip3 install -r requirements-run.txt -r requirements-analysis.txt

COPY . /app

# Fail loudly at container start if the stack drifted from the pins.
RUN python3 scripts/preflight.py --no-gpu --no-cache || \
    (echo "PREFLIGHT FAILED AT BUILD TIME" && exit 1)

CMD ["python3", "scripts/preflight.py"]
