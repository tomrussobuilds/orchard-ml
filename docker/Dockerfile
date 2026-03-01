# =====================================================
# CUDA-enabled image (works on GPU, falls back to CPU)
# =====================================================
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Reproducibility defaults (overridable at runtime with -e)
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Create non-root user for runtime (UID 1000 for host-mount compatibility).
# Override at runtime with: docker run -u $(id -u):$(id -g) ...
RUN groupadd -g 1000 appuser && useradd -m -u 1000 -g appuser appuser

# Headless rendering and cache directories
ENV HOME=/home/appuser
ENV USER=appuser
ENV IN_DOCKER=TRUE
ENV MPLCONFIGDIR=/home/appuser/.cache/matplotlib
ENV TORCH_HOME=/home/appuser/.cache/torch

# Merge system dependencies and pip upgrade in one RUN layer
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        git \
        procps \
    && pip3 install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (rebuilds only when requirements.txt changes)
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy project files (changes most frequently — last layer)
COPY . .

# Install the package (editable for local dev, baked for production)
RUN pip3 install -e .

# Hand ownership to non-root user
RUN chown -R appuser:appuser /app /home/appuser

USER appuser

# Default command
ENTRYPOINT ["orchard", "run"]
CMD []
