# =====================================================
# CUDA-enabled image (works on GPU, falls back to CPU)
# =====================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Reproducibility defaults (overridable at runtime with -e)
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Headless rendering and cache directories
# When running with -u $(id -u):$(id -g), the host UID does not exist in
# /etc/passwd. HOME and USER must be set explicitly so that PyTorch's
# cache_dir() (which calls getpass.getuser()) and other libs work correctly.
ENV HOME=/tmp
ENV USER=appuser
ENV IN_DOCKER=TRUE
ENV MPLCONFIGDIR=/tmp/matplotlib_cache
ENV TORCH_HOME=/tmp/torch_cache

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

# Default command
ENTRYPOINT ["orchard", "run"]
CMD []
